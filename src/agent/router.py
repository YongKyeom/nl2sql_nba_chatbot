from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from src.metrics.registry import MetricsRegistry
from src.prompt.router import ROUTER_PROMPT
from src.prompt.system import SYSTEM_PROMPT


class Route(str, Enum):
    """
    라우팅 타입.
    """

    GENERAL = "general"
    DIRECT = "direct"
    REUSE = "reuse"
    SQL_REQUIRED = "sql_required"


@dataclass(frozen=True)
class RouterResult:
    """
    라우터 출력 구조.

    Args:
        route: 라우팅 결과.
        metric_name: 감지된 메트릭 이름.
        reason: 라우팅 근거.
    """

    route: Route
    metric_name: str | None
    reason: str


@dataclass(frozen=True)
class RoutingContext:
    """
    라우팅 컨텍스트.

    Args:
        user_message: 사용자 입력.
        has_previous: 이전 결과 존재 여부.
        last_result_schema: 이전 결과 컬럼 목록.
        last_sql: 직전 SQL.
        last_slots: 직전 슬롯.
        available_metrics: 레지스트리 메트릭 이름/별칭 리스트.
    """

    user_message: str
    has_previous: bool
    last_result_schema: list[str] | None
    last_sql: str | None
    last_slots: dict[str, Any] | None
    available_metrics: list[str]


@dataclass(frozen=True)
class ReuseResult:
    """
    이전 결과 재사용 결과.

    Args:
        dataframe: 후처리된 데이터프레임.
        summary: 변경 요약.
    """

    dataframe: pd.DataFrame
    summary: str


DEFINITION_KEYWORDS = ["정의", "뭐야", "무엇", "설명", "기준", "공식", "뜻", "의미"]
REUSE_KEYWORDS = ["방금", "이전", "정렬", "상위", "내림", "오름", "제외", "필터", "바꿔", "줄여", "늘려"]
PREVIOUS_REF_KEYWORDS = ["방금", "이전", "그 결과", "그거", "다시", "재정렬", "필터", "제외", "바꿔", "줄여", "늘려"]
GENERAL_KEYWORDS = [
    "안녕",
    "ㅎㅇ",
    "ㅎㅇㅇ",
    "안뇽",
    "하이",
    "hello",
    "hi",
    "도움",
    "도움말",
    "도와",
    "도움줘",
    "사용법",
    "사용 방법",
    "가이드",
    "안내",
    "기능",
    "할 수",
    "할수",
    "할 수 있",
    "할수있",
    "가능해",
    "가능한",
    "무슨 일",
    "무엇을 할 수",
    "어떤 일",
    "어떤 걸",
    "누구",
    "누구야",
    "누구냐",
    "너는",
    "너 뭐",
    "너 뭐야",
    "알려줄 수",
    "해줄 수",
    "무슨 데이터",
    "어떤 데이터",
    "데이터 알려",
    "데이터 제공",
    "제공 데이터",
    "제공 가능한",
    "사용 가능한",
    "데이터셋",
    "데이터 종류",
    "데이터 범위",
    "데이터가",
    "what data",
    "data can you",
    "지원",
    "capability",
    "capabilities",
]


class RouterLLM:
    """
    LLM 기반 라우터.

    키워드 규칙 대신 맥락을 보고 direct/reuse/sql_required를 판단한다.
    """

    def __init__(self, *, model: str, temperature: float = 0.0) -> None:
        """
        RouterLLM 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
        """

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def route(self, context: RoutingContext, registry: MetricsRegistry) -> RouterResult:
        """
        라우팅을 결정한다.

        Args:
            context: 라우팅 컨텍스트.
            registry: 메트릭 레지스트리.

        Returns:
            RouterResult.
        """

        metric_name = _detect_metric(context.user_message, registry)
        if _is_general_question(context.user_message):
            return RouterResult(route=Route.GENERAL, metric_name=None, reason="일반 안내 질문 감지")

        try:
            prompt = ROUTER_PROMPT.format(
                user_message=context.user_message,
                has_previous=context.has_previous,
                last_result_schema=context.last_result_schema or [],
                last_sql=context.last_sql or "없음",
                last_slots=context.last_slots or {},
                available_metrics=context.available_metrics,
            )
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = _parse_router_json(content)
            route_value = parsed.get("route")
            reason = parsed.get("reason", "라우터 판단")

            if route_value not in {route.value for route in Route}:
                return _fallback_route(
                    context.user_message,
                    registry,
                    context.has_previous,
                    reason="라우터 응답이 유효하지 않아 폴백 처리",
                )

            route = Route(route_value)
        except Exception:  # noqa: BLE001
            return _fallback_route(
                context.user_message,
                registry,
                context.has_previous,
                reason="라우터 응답 파싱 실패로 폴백 처리",
            )

        if route == Route.REUSE and not context.has_previous:
            return RouterResult(route=Route.SQL_REQUIRED, metric_name=metric_name, reason="이전 결과 없음")

        if metric_name is None and _is_general_question(context.user_message):
            return RouterResult(route=Route.GENERAL, metric_name=None, reason="일반 안내 질문 감지")

        return RouterResult(route=route, metric_name=metric_name, reason=reason)


def route_message(user_message: str, registry: MetricsRegistry, has_previous: bool) -> RouterResult:
    """
    키워드 기반 라우팅(예외 상황 대비용).

    Args:
        user_message: 사용자 입력.
        registry: 메트릭 레지스트리.
        has_previous: 이전 결과 존재 여부.

    Returns:
        RouterResult.
    """

    return _fallback_route(user_message, registry, has_previous, reason="키워드 폴백")


def apply_reuse_rules(user_message: str, dataframe: pd.DataFrame) -> ReuseResult:
    """
    이전 결과에 후처리 규칙을 적용.

    Args:
        user_message: 사용자 입력.
        dataframe: 이전 결과.

    Returns:
        ReuseResult.
    """

    updated = dataframe.copy()
    summaries: list[str] = []

    sort_column, ascending = _extract_sort_rule(user_message, updated.columns)
    if sort_column:
        updated = updated.sort_values(by=sort_column, ascending=ascending)
        order_label = "오름차순" if ascending else "내림차순"
        summaries.append(f"{sort_column} 기준 {order_label} 정렬")

    exclude_token = _extract_exclude_token(user_message)
    if exclude_token:
        updated = _filter_token(updated, exclude_token, include=False)
        summaries.append(f"{exclude_token} 제외")

    include_token = _extract_include_token(user_message)
    if include_token:
        updated = _filter_token(updated, include_token, include=True)
        summaries.append(f"{include_token}만 필터")

    top_k = _extract_top_k(user_message)
    if top_k:
        updated = updated.head(top_k)
        summaries.append(f"상위 {top_k}개만 표시")

    summary_text = " / ".join(summaries) if summaries else "이전 결과를 그대로 유지"
    return ReuseResult(dataframe=updated, summary=summary_text)


def _fallback_route(user_message: str, registry: MetricsRegistry, has_previous: bool, reason: str) -> RouterResult:
    """
    키워드 기반 폴백 라우팅.

    Args:
        user_message: 사용자 입력.
        registry: 메트릭 레지스트리.
        has_previous: 이전 결과 존재 여부.
        reason: 폴백 사유.

    Returns:
        RouterResult.
    """

    metric_name = _detect_metric(user_message, registry)

    if _is_general_question(user_message):
        return RouterResult(route=Route.GENERAL, metric_name=None, reason=reason)

    if _is_definition_question(user_message):
        return RouterResult(route=Route.DIRECT, metric_name=metric_name, reason=reason)

    if has_previous and _is_reuse_question(user_message):
        if metric_name and not _is_previous_reference(user_message):
            return RouterResult(route=Route.SQL_REQUIRED, metric_name=metric_name, reason=reason)
        return RouterResult(route=Route.REUSE, metric_name=metric_name, reason=reason)

    return RouterResult(route=Route.SQL_REQUIRED, metric_name=metric_name, reason=reason)


def _parse_router_json(text: str) -> dict[str, Any]:
    """
    라우터 JSON 응답을 파싱.

    Args:
        text: LLM 응답 문자열.

    Returns:
        JSON 딕셔너리.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(cleaned[start : end + 1])


def _detect_metric(text: str, registry: MetricsRegistry) -> str | None:
    """
    메트릭 이름을 감지.

    Args:
        text: 사용자 입력.
        registry: 메트릭 레지스트리.

    Returns:
        메트릭 이름 또는 None.
    """

    results = registry.search(text, limit=1)
    if results:
        return results[0].name
    return None


def _is_definition_question(text: str) -> bool:
    """
    정의/설명 질문 여부를 판별.

    Args:
        text: 사용자 입력.

    Returns:
        정의/설명 질문이면 True.
    """

    return any(keyword in text for keyword in DEFINITION_KEYWORDS)


def _is_general_question(text: str) -> bool:
    """
    일반 안내/인사 질문 여부를 판별.

    Args:
        text: 사용자 입력.

    Returns:
        일반 질문이면 True.
    """

    lowered = text.lower()
    return any(keyword in text or keyword in lowered for keyword in GENERAL_KEYWORDS)


def _is_reuse_question(text: str) -> bool:
    """
    이전 결과 재사용 의도를 판별.

    Args:
        text: 사용자 입력.

    Returns:
        재사용 의도면 True.
    """

    return any(keyword in text for keyword in REUSE_KEYWORDS)


def _is_previous_reference(text: str) -> bool:
    """
    이전 결과를 지칭하는 표현이 있는지 확인.

    Args:
        text: 사용자 입력.

    Returns:
        이전 결과 지칭 여부.
    """

    return any(keyword in text for keyword in PREVIOUS_REF_KEYWORDS)


def _extract_top_k(text: str) -> int | None:
    """
    상위 N 값을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        상위 N 또는 None.
    """

    match = re.search(r"(?:상위|top)\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_sort_rule(text: str, columns: pd.Index) -> tuple[str | None, bool]:
    """
    정렬 규칙을 추출.

    Args:
        text: 사용자 입력.
        columns: 결과 컬럼 목록.

    Returns:
        (정렬 컬럼, 오름차순 여부)
    """

    ascending = "오름" in text or "낮" in text
    descending = "내림" in text or "높" in text
    column_name = None

    for col in columns:
        if col.lower() in text.lower():
            column_name = col
            break

    if column_name is None:
        return None, True

    if descending and not ascending:
        return column_name, False
    return column_name, True


def _extract_exclude_token(text: str) -> str | None:
    """
    제외 토큰을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        제외 토큰 또는 None.
    """

    match = re.search(r"([A-Za-z]{2,3})\s*제외", text)
    if match:
        return match.group(1).upper()
    return None


def _extract_include_token(text: str) -> str | None:
    """
    포함 토큰을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        포함 토큰 또는 None.
    """

    match = re.search(r"([A-Za-z]{2,3})\s*만", text)
    if match:
        return match.group(1).upper()
    return None


def _filter_token(dataframe: pd.DataFrame, token: str, *, include: bool) -> pd.DataFrame:
    """
    데이터프레임에서 토큰을 포함/제외 필터링.

    Args:
        dataframe: 원본 데이터프레임.
        token: 필터 토큰.
        include: True면 포함, False면 제외.

    Returns:
        필터링된 데이터프레임.
    """

    mask = dataframe.astype(str).apply(lambda row: token in " ".join(row.values), axis=1)
    return dataframe[mask] if include else dataframe[~mask]


if __name__ == "__main__":
    # 1) 레지스트리 로드
    registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
    registry.ensure_loaded()

    # 2) 폴백 라우팅 테스트
    questions = [
        "안녕? 넌 어떤 일을 할 수 있어?",
        "승률 정의 알려줘",
        "상위 3개만",
        "최근 리그에서 승률 상위 5개 팀 알려줘",
    ]

    for question in questions:
        result = route_message(question, registry, has_previous=False)
        print(question, "->", result.route, result.reason)
