from __future__ import annotations

import asyncio
import re
import json
from pathlib import Path
from typing import Any, Literal

from src.model.llm_operator import LLMOperator
from pydantic import BaseModel, Field

from src.metrics.registry import MetricsRegistry
from src.prompt.planner import PLANNER_PROMPT
from src.prompt.system import SYSTEM_PROMPT
from src.agent.tools.metric_selector import MetricSelector
from src.agent.tools.entity_resolver import EntityResolver


class PlannerSlots(BaseModel):
    """
    플래너 슬롯 구조.

    Args:
        entity_type: 대상 엔티티 유형.
        season: 시즌 식별자(예: 2023-24).
        date_range: 기간 표현(예: 최근, 전체).
        metric: 메트릭 이름.
        top_k: 상위 N.
        filters: 추가 필터.
    """

    entity_type: Literal["player", "team", "game"] = "team"
    season: str | None = None
    date_range: str | None = "전체"
    metric: str | None = None
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(BaseModel):
    """
    플래너 출력 구조.

    Args:
        slots: 추출된 슬롯.
        clarify_question: 확인 질문(없으면 None).
        metric_candidates: 메트릭 후보 목록.
        entity_resolution: 엔티티 보강 결과.
    """

    slots: PlannerSlots
    clarify_question: str | None = None
    metric_candidates: list[dict[str, Any]] = Field(default_factory=list)
    entity_resolution: dict[str, Any] | None = None
    metric_tool_used: bool = False
    entity_tool_used: bool = False


class Planner:
    """
    사용자 질의에서 슬롯을 추출하고 확인 질문을 생성.

    멀티턴에서는 이전 슬롯을 재사용하고, 사용자가 변경한 항목만 덮어쓴다.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        metric_selector: MetricSelector | None = None,
        entity_resolver: EntityResolver | None = None,
        enable_tools: bool = False,
        max_tool_calls: int = 3,
    ) -> None:
        """
        Planner 초기화.

        Args:
            registry: 메트릭 레지스트리.
            model: 사용할 LLM 모델명(툴 호출용).
            temperature: LLM temperature.
            metric_selector: 메트릭 셀렉터 도구.
            entity_resolver: 엔티티 리졸버 도구.
            enable_tools: LLM 툴 호출 여부.
            max_tool_calls: 최대 툴 호출 횟수.
        """

        self._registry = registry
        self._model = model
        self._temperature = temperature
        self._metric_selector = metric_selector
        self._entity_resolver = entity_resolver
        self._enable_tools = enable_tools and model and metric_selector and entity_resolver
        self._max_tool_calls = max(1, max_tool_calls)
        self._client = LLMOperator() if self._enable_tools else None

    async def plan(
        self,
        user_message: str,
        previous_slots: PlannerSlots | None,
        *,
        conversation_context: str | None = None,
    ) -> PlannerOutput:
        """
        질의에서 슬롯을 추출한다.

        Args:
            user_message: 사용자 입력.
            previous_slots: 직전 슬롯(없으면 None).
            conversation_context: 최근 대화 컨텍스트.

        Returns:
            PlannerOutput.
        """

        return self._plan_rule_based(user_message, previous_slots, conversation_context=conversation_context)

    def _plan_rule_based(
        self,
        user_message: str,
        previous_slots: PlannerSlots | None,
        *,
        conversation_context: str | None = None,
    ) -> PlannerOutput:
        """
        규칙 기반으로 슬롯을 추출한다.

        Args:
            user_message: 사용자 입력.
            previous_slots: 직전 슬롯.
            conversation_context: 최근 대화 컨텍스트.

        Returns:
            PlannerOutput.
        """

        base_slots = previous_slots.model_dump() if previous_slots else PlannerSlots().model_dump()
        updated_slots = base_slots.copy()

        # 1) 엔티티 유형 추출
        entity_type = _extract_entity_type(user_message)
        if entity_type:
            updated_slots["entity_type"] = entity_type

        # 2) 시즌/기간 추출
        season = _extract_season(user_message)
        if season:
            updated_slots["season"] = season
        date_range = _extract_date_range(user_message)
        if date_range:
            updated_slots["date_range"] = date_range

        # 3) 메트릭 추출
        metric = _extract_metric(user_message, self._registry)
        if metric:
            updated_slots["metric"] = metric

        # 4) Top-K 추출
        top_k = _extract_top_k(user_message)
        if top_k:
            updated_slots["top_k"] = top_k

        # 5) 필터 추출
        filters = dict(updated_slots.get("filters", {}))
        filters.update(_extract_filters(user_message))
        updated_slots["filters"] = filters

        slots = PlannerSlots(**updated_slots)
        clarify_question = _build_clarify_question(user_message, slots, self._registry)

        return PlannerOutput(
            slots=slots,
            clarify_question=clarify_question,
            metric_tool_used=False,
            entity_tool_used=False,
        )

    async def plan_with_retry(
        self,
        user_message: str,
        previous_slots: PlannerSlots | None,
        *,
        previous_entities: dict[str, list[str]] | None = None,
        conversation_context: str | None = None,
        max_retries: int = 5,
    ) -> PlannerOutput:
        """
        메트릭이 누락/오류일 때 최대 N회 보정하며 플래닝한다.

        Args:
            user_message: 사용자 입력.
            previous_slots: 직전 슬롯(없으면 None).
            max_retries: 최대 보정 시도 횟수.
            conversation_context: 최근 대화 컨텍스트.

        Returns:
            PlannerOutput.
        """

        if max_retries < 1:
            return await self.plan(user_message, previous_slots, conversation_context=conversation_context)

        tool_output: PlannerOutput | None = None
        metric_tool_used = False
        entity_tool_used = False
        metric_candidates: list[dict[str, Any]] = []
        entity_resolution: dict[str, Any] | None = None

        if self._enable_tools:
            try:
                tool_output = await self._plan_with_tools(
                    user_message,
                    previous_slots,
                    previous_entities,
                    conversation_context=conversation_context,
                )
            except Exception:
                tool_output = None

            if tool_output:
                metric_tool_used = tool_output.metric_tool_used
                entity_tool_used = tool_output.entity_tool_used
                metric_candidates = tool_output.metric_candidates or []
                entity_resolution = tool_output.entity_resolution

            if tool_output and tool_output.slots.metric and self._registry.get(tool_output.slots.metric):
                return tool_output

        output = self._plan_rule_based(user_message, previous_slots, conversation_context=conversation_context)
        output.metric_tool_used = metric_tool_used
        output.entity_tool_used = entity_tool_used
        if metric_candidates:
            output.metric_candidates = metric_candidates
        if entity_resolution:
            output.entity_resolution = entity_resolution
        metric_name = output.slots.metric

        # 후보 메트릭을 최대 N개까지 확보해 강제로 보정한다.
        candidates = self._registry.search(user_message, limit=max_retries)
        if metric_name and self._registry.get(metric_name):
            if not candidates or metric_name == candidates[0].name:
                return output

        for candidate in candidates:
            forced_slots = output.slots.model_dump()
            forced_slots["metric"] = candidate.name
            slots = PlannerSlots(**forced_slots)
            clarify_question = _build_clarify_question(user_message, slots, self._registry)
            metric_name = slots.metric
            if metric_name and self._registry.get(metric_name):
                return PlannerOutput(slots=slots, clarify_question=clarify_question)

        if previous_slots is not None:
            # 이전 슬롯이 영향을 줬을 가능성이 있어 빈 상태로 재시도한다.
            for _ in range(max_retries):
                retry_output = await self.plan(user_message, None, conversation_context=conversation_context)
                metric_name = retry_output.slots.metric
                if metric_name and self._registry.get(metric_name):
                    return retry_output

            retry_output.metric_tool_used = metric_tool_used
            retry_output.entity_tool_used = entity_tool_used
            if metric_candidates:
                retry_output.metric_candidates = metric_candidates
            if entity_resolution:
                retry_output.entity_resolution = entity_resolution
            return retry_output

        return output

    async def _plan_with_tools(
        self,
        user_message: str,
        previous_slots: PlannerSlots | None,
        previous_entities: dict[str, list[str]] | None,
        *,
        conversation_context: str | None = None,
    ) -> PlannerOutput:
        """
        LLM과 Tool을 활용해 슬롯을 보강한다.

        Args:
            user_message: 사용자 입력.
            previous_slots: 직전 슬롯.
            previous_entities: 직전 엔티티 정보.
            conversation_context: 최근 대화 컨텍스트.

        Returns:
            PlannerOutput.
        """

        baseline = self._plan_rule_based(user_message, previous_slots, conversation_context=conversation_context)

        metric_candidates: list[dict[str, Any]] = []
        entity_resolution: dict[str, Any] | None = None
        metric_tool_used = False
        entity_tool_used = False

        prompt = PLANNER_PROMPT.format(
            user_question=user_message,
            conversation_context=conversation_context or "없음",
            previous_slots=json.dumps(previous_slots.model_dump() if previous_slots else {}, ensure_ascii=False),
            baseline_slots=json.dumps(baseline.slots.model_dump(), ensure_ascii=False),
            last_entities=json.dumps(previous_entities or {}, ensure_ascii=False),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        tools = [
            self._metric_selector.tool_schema(),
            self._entity_resolver.tool_schema(),
        ]
        for _ in range(self._max_tool_calls):
            response = await self._client.invoke(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            message = response.choices[0].message
            if message.tool_calls:
                messages.append(_build_tool_call_message(message))
                for call in message.tool_calls:
                    args = _safe_json(call.function.arguments)
                    if call.function.name == "metric_selector":
                        result = await self._metric_selector.select(
                            query=str(args.get("query") or user_message),
                            top_k=int(args.get("top_k") or 5),
                        )
                        metric_candidates = result.get("candidates", [])
                        metric_tool_used = True
                    elif call.function.name == "entity_resolver":
                        result = await self._entity_resolver.resolve(
                            query=str(args.get("query") or user_message),
                            previous_entities=previous_entities,
                            top_k=int(args.get("top_k") or 3),
                        )
                        entity_resolution = result
                        entity_tool_used = True
                    else:
                        result = {"error": "unknown_tool"}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue

            content = (message.content or "").strip()
            parsed = _parse_planner_json(content)
            if not parsed:
                break

            slots_payload = parsed.get("slots")
            clarify_question = parsed.get("clarify_question")
            if not isinstance(slots_payload, dict):
                break

            try:
                slots = PlannerSlots(**slots_payload)
            except Exception:
                break

            if not slots.metric and baseline.slots.metric:
                slots = PlannerSlots(**{**slots.model_dump(), "metric": baseline.slots.metric})

            return PlannerOutput(
                slots=slots,
                clarify_question=str(clarify_question) if clarify_question else None,
                metric_candidates=metric_candidates,
                entity_resolution=entity_resolution,
                metric_tool_used=metric_tool_used,
                entity_tool_used=entity_tool_used,
            )

        return PlannerOutput(
            slots=baseline.slots,
            clarify_question=baseline.clarify_question,
            metric_candidates=metric_candidates,
            entity_resolution=entity_resolution,
            metric_tool_used=metric_tool_used,
            entity_tool_used=entity_tool_used,
        )


def _extract_entity_type(text: str) -> Literal["player", "team", "game"] | None:
    """
    엔티티 유형을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        엔티티 유형 또는 None.
    """

    lowered = text.lower()
    if "팀" in text or "team" in lowered:
        return "team"
    if _has_player_hint(text):
        return "player"
    if "경기" in text or "game" in lowered:
        return "game"
    return None


def _extract_season(text: str) -> str | None:
    """
    시즌 문자열을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        시즌 문자열 또는 None.
    """

    match = re.search(r"(20\d{2}\s*[-/\.]\s*\d{2})", text)
    if match:
        return match.group(1).replace(" ", "")

    short_match = re.search(r"\b(\d{2})\s*[-/\.]\s*(\d{2})\b", text)
    if short_match:
        start_year = int(short_match.group(1))
        end_year = int(short_match.group(2))
        if 0 <= start_year <= 99 and 0 <= end_year <= 99:
            return f"20{start_year:02d}-{end_year:02d}"
    return None


def _extract_date_range(text: str) -> str | None:
    """
    기간 표현을 추출.

    Args:
        text: 사용자 입력.

    Returns:
        기간 표현 또는 None.
    """

    if "최근" in text or "last" in text.lower():
        return "최근"
    if "전체" in text:
        return "전체"
    return None


def _extract_metric(text: str, registry: MetricsRegistry) -> str | None:
    """
    메트릭 이름을 추출.

    Args:
        text: 사용자 입력.
        registry: 메트릭 레지스트리.

    Returns:
        메트릭 이름 또는 None.
    """

    # 1) 레지스트리 검색
    results = registry.search(text, limit=1)
    if results:
        return results[0].name

    # 2) 키워드 기반 보정
    if "최근" in text and "경기" in text:
        return "last_n_games"
    if "연승" in text or "연패" in text:
        return "streak"

    return None


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
    match = re.search(r"(\d+)\s*개", text)
    if match:
        return int(match.group(1))
    return None


def _safe_json(text: str) -> dict[str, Any]:
    """
    JSON 문자열을 안전하게 파싱한다.

    Args:
        text: JSON 문자열.

    Returns:
        파싱 결과(실패 시 빈 딕셔너리).
    """

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _build_tool_call_message(message: Any) -> dict[str, Any]:
    """
    tool_calls가 포함된 assistant 메시지를 직렬화한다.

    Args:
        message: LLM 응답 메시지.

    Returns:
        직렬화된 메시지.
    """

    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    payload = {"role": "assistant", "tool_calls": tool_calls}
    if message.content:
        payload["content"] = message.content
    return payload


def _parse_planner_json(text: str) -> dict[str, Any] | None:
    """
    플래너 JSON 응답을 파싱한다.

    Args:
        text: LLM 응답 문자열.

    Returns:
        파싱 결과(실패 시 None).
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return None
    return None


def _extract_filters(text: str) -> dict[str, Any]:
    """
    필터 정보를 추출.

    Args:
        text: 사용자 입력.

    Returns:
        필터 딕셔너리.
    """

    filters: dict[str, Any] = {}

    if "홈" in text:
        filters["home_away"] = "home"
    if "원정" in text or "away" in text.lower():
        filters["home_away"] = "away"

    if "승리" in text or ("승" in text and "승률" not in text):
        filters["result"] = "win"
    if "패배" in text or "졌" in text:
        filters["result"] = "loss"

    min_games_match = re.search(r"(\d+)\s*경기", text)
    if min_games_match:
        filters["min_games"] = int(min_games_match.group(1))

    min_minutes_match = re.search(r"(\d+)\s*분", text)
    if min_minutes_match:
        filters["min_minutes"] = int(min_minutes_match.group(1))

    opponent_match = re.search(r"상대\s*([A-Za-z]{2,3})", text)
    if opponent_match:
        filters["opponent"] = opponent_match.group(1).upper()

    # 팀 맞대결 표기를 감지해 팀 약어를 추출한다.
    if any(keyword in text.lower() for keyword in ["vs", "맞대결", "대결", "상대전", "vs."]):
        tokens = re.findall(r"[A-Za-z]{2,3}", text)
        team_tokens = [token.upper() for token in tokens if token.lower() not in {"vs", "v"}]
        if len(team_tokens) >= 1:
            filters["team_abbreviation"] = team_tokens[0]
        if len(team_tokens) >= 2:
            filters["team_a"] = team_tokens[0]
            filters["team_b"] = team_tokens[1]

    return filters


def _build_clarify_question(
    user_message: str,
    slots: PlannerSlots,
    registry: MetricsRegistry,
) -> str | None:
    """
    확인 질문을 생성한다.

    Args:
        user_message: 사용자 입력.
        slots: 현재 슬롯.

    Returns:
        확인 질문 또는 None.
    """

    if slots.entity_type == "player" and _needs_player_game_logs(user_message):
        return (
            "현재 데이터에는 선수 경기 로그(월별/경기별 득점·출전시간·슛 성공률)가 없습니다. "
            "선수 프로필/드래프트/컴바인 정보나 팀 단위 트렌드로 안내해드릴까요?"
        )

    if slots.metric is None:
        return "어떤 지표를 조회할까요? 예: 승률, 득실차, TS%"

    if slots.entity_type == "player" and not _is_player_metric(slots.metric, registry):
        return (
            "선수 단위는 프로필/드래프트/컴바인 위주로 제공됩니다. "
            "팀 기준 지표로 전환할까요, 아니면 선수 정보 범위로 조회할까요?"
        )

    return None


def _is_player_metric(metric_name: str | None, registry: MetricsRegistry) -> bool:
    """
    선수 단위로 처리 가능한 메트릭인지 판단.

    Args:
        metric_name: 메트릭 이름.
        registry: 메트릭 레지스트리.

    Returns:
        선수 메트릭이면 True.
    """

    if not metric_name:
        return False

    metric = registry.get(metric_name)
    if metric is None:
        return False

    player_tables = {"common_player_info", "draft_history", "draft_combine_stats", "player", "inactive_players"}
    return any(table in player_tables for table in metric.required_tables)


def _has_player_hint(text: str) -> bool:
    """
    선수 질의를 암시하는 키워드가 있는지 확인한다.

    Args:
        text: 사용자 입력.

    Returns:
        선수 관련 힌트가 있으면 True.
    """

    lowered = text.lower()
    keywords = [
        "선수",
        "프로필",
        "드래프트",
        "컴바인",
        "포지션",
        "키",
        "체중",
        "나이",
        "커리어",
        "출전시간",
        "출전 시간",
        "슛 성공률",
        "fg%",
        "ts%",
        "efg%",
        "필드골",
    ]
    english_keywords = ["player", "profile", "draft", "combine", "minutes", "shooting", "field goal"]
    return any(keyword in text for keyword in keywords) or any(keyword in lowered for keyword in english_keywords)


def _needs_player_game_logs(text: str) -> bool:
    """
    선수 경기 로그 기반 요청인지 판단한다.

    Args:
        text: 사용자 입력.

    Returns:
        경기 로그 기반이면 True.
    """

    lowered = text.lower()
    keywords = [
        "월별",
        "트렌드",
        "추이",
        "변화",
        "정규시즌",
        "플레이오프",
        "출전시간",
        "출전 시간",
        "평균득점",
        "평균 득점",
        "슛 성공률",
        "경기별",
        "게임 로그",
    ]
    english_keywords = ["trend", "monthly", "playoffs", "regular season", "game log"]
    return any(keyword in text for keyword in keywords) or any(keyword in lowered for keyword in english_keywords)


if __name__ == "__main__":
    async def _main() -> None:
        # 1) 레지스트리 로드
        registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
        registry.ensure_loaded()
        planner = Planner(registry)

        # 2) 샘플 질의 실행
        queries = [
            "최근 리그에서 승률 상위 5개 팀 알려줘",
            "2018 드래프트 전체 픽 리스트 보여줘",
            "LAL과 BOS 맞대결 최근 기록 알려줘",
        ]

        # 3) 슬롯 추출 결과 확인
        for query in queries:
            result = await planner.plan(query, None)
            print(query)
            print(result.slots.model_dump())

    asyncio.run(_main())
