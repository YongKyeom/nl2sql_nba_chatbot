from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

STOPWORDS = {
    "상위",
    "순위",
    "랭킹",
    "리더",
    "top",
    "leaders",
    "leader",
    "리그",
    "최근",
    "전체",
    "보기",
    "보여줘",
    "알려줘",
    "몇",
    "개",
    "만",
    "팀",
    "선수",
}


@dataclass(frozen=True)
class MetricDefinition:
    """
    메트릭 정의 구조.

    Args:
        name: 메트릭 식별자.
        aliases: 사용자 질의 매칭을 위한 별칭.
        description_ko: 한글 설명(정의/해석/주의점 요약).
        formula_ko: 한글 계산식/규칙.
        required_tables: 필요한 테이블 목록.
        required_columns: 필요한 컬럼 목록.
        sql_template: SQL 템플릿(가능한 경우).
        cut_rules: 컷 규칙(경기 수, 분 등).
    """

    name: str
    aliases: list[str]
    description_ko: str
    formula_ko: str | None
    required_tables: list[str]
    required_columns: list[str]
    sql_template: str | None
    cut_rules: dict[str, Any]

    def all_names(self) -> list[str]:
        """
        이름과 별칭을 모두 반환.

        Returns:
            이름/별칭 리스트.
        """

        return [self.name, *self.aliases]


class MetricsRegistry:
    """
    metrics.yaml을 로드하고 검색/설명 생성을 담당.

    비즈니스 로직이 YAML로 주입되므로, SQL 생성/설명 단계는 반드시
    이 레지스트리를 통해 정의를 조회한다.
    """

    def __init__(self, metrics_path: Path) -> None:
        """
        MetricsRegistry 초기화.

        Args:
            metrics_path: metrics.yaml 경로.
        """

        self._metrics_path = metrics_path
        self._metrics: list[MetricDefinition] = []

    def load(self) -> None:
        """
        metrics.yaml을 로드.

        Raises:
            FileNotFoundError: metrics.yaml이 없을 때.
            ValueError: YAML 형식이 잘못되었을 때.
        """

        if not self._metrics_path.exists():
            raise FileNotFoundError(f"metrics.yaml을 찾을 수 없습니다: {self._metrics_path}")

        payload = yaml.safe_load(self._metrics_path.read_text(encoding="utf-8"))
        if not payload or "metrics" not in payload:
            raise ValueError("metrics.yaml 형식이 올바르지 않습니다. 'metrics' 키를 확인하세요.")

        self._metrics = [_build_metric(item) for item in payload["metrics"]]

    def ensure_loaded(self) -> None:
        """
        레지스트리 로드 여부를 확인하고 필요 시 로드.

        Returns:
            None
        """

        if not self._metrics:
            self.load()

    def list_metrics(self) -> list[MetricDefinition]:
        """
        전체 메트릭 목록을 반환.

        Returns:
            메트릭 정의 리스트.
        """

        self.ensure_loaded()
        return list(self._metrics)

    def get(self, name: str) -> MetricDefinition | None:
        """
        이름/별칭으로 메트릭을 조회.

        Args:
            name: 메트릭 이름 또는 별칭.

        Returns:
            메트릭 정의(없으면 None).
        """

        self.ensure_loaded()
        normalized = name.strip().lower()
        for metric in self._metrics:
            if normalized == metric.name.lower():
                return metric
            if any(normalized == alias.lower() for alias in metric.aliases):
                return metric
        return None

    def search(self, keyword: str, limit: int = 5) -> list[MetricDefinition]:
        """
        키워드로 메트릭을 검색.

        Args:
            keyword: 검색 키워드.
            limit: 최대 반환 개수.

        Returns:
            검색 결과 메트릭 리스트.
        """

        self.ensure_loaded()
        query = keyword.strip().lower()
        tokens = _tokenize(query)
        if not tokens and not query:
            return []

        scored: list[tuple[int, MetricDefinition]] = []
        for metric in self._metrics:
            names = metric.all_names()
            alias_score = _alias_match_score(query, names)
            haystack = " ".join(names + [metric.description_ko or "", metric.formula_ko or ""]).lower()
            score = alias_score + _match_score(haystack, query, tokens)
            if score > 0:
                scored.append((score, metric))

        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [metric for _, metric in scored[:limit]]

    def build_direct_answer(self, metric: MetricDefinition) -> str:
        """
        Direct Answer용 설명 텍스트를 생성.

        Args:
            metric: 메트릭 정의.

        Returns:
            한글 설명 문자열.
        """

        formula = metric.formula_ko or "정의에 계산식이 명시되어 있지 않습니다."
        example = _build_example(metric)
        caution = _build_caution(metric)

        return "\n".join(
            [
                f"정의: {metric.description_ko}",
                f"계산/규칙: {formula}",
                f"예시: {example}",
                f"주의점: {caution}",
            ]
        )

    def build_sql_context(self, metric: MetricDefinition) -> dict[str, Any]:
        """
        SQL 생성에 필요한 메트릭 컨텍스트를 반환.

        Args:
            metric: 메트릭 정의.

        Returns:
            SQL 생성 컨텍스트 딕셔너리.
        """

        return {
            "name": metric.name,
            "aliases": metric.aliases,
            "description_ko": metric.description_ko,
            "formula_ko": metric.formula_ko,
            "required_tables": metric.required_tables,
            "required_columns": metric.required_columns,
            "sql_template": metric.sql_template,
            "cut_rules": metric.cut_rules,
        }


def _build_metric(raw: dict[str, Any]) -> MetricDefinition:
    """
    YAML 로우 데이터를 MetricDefinition으로 변환.

    Args:
        raw: YAML에서 읽은 메트릭 딕셔너리.

    Returns:
        MetricDefinition 객체.

    Raises:
        ValueError: 필수 필드가 누락된 경우.
    """

    required_fields = ["name", "description_ko", "required_tables", "required_columns"]
    missing = [field for field in required_fields if field not in raw]
    if missing:
        raise ValueError(f"메트릭 정의에 필수 필드가 없습니다: {missing}")

    return MetricDefinition(
        name=str(raw["name"]),
        aliases=list(raw.get("aliases", [])),
        description_ko=str(raw["description_ko"]),
        formula_ko=raw.get("formula_ko"),
        required_tables=list(raw.get("required_tables", [])),
        required_columns=list(raw.get("required_columns", [])),
        sql_template=raw.get("sql_template"),
        cut_rules=raw.get("cut_rules", {}),
    )


def _match_score(text: str, keyword: str, tokens: list[str]) -> int:
    """
    간단한 키워드 매칭 점수를 계산.

    Args:
        text: 검색 대상 문자열.
        keyword: 검색 키워드.
        tokens: 토큰 리스트.

    Returns:
        매칭 점수(0이면 불일치).
    """

    score = 0
    if keyword and keyword in text:
        score += 3

    for token in tokens:
        if len(token) < 2:
            continue
        if token in text:
            score += 1

    return score


def _tokenize(text: str) -> list[str]:
    """
    검색용 토큰을 추출.

    Args:
        text: 원본 문자열.

    Returns:
        토큰 리스트.
    """

    raw_tokens = re.findall(r"[A-Za-z%]+|[가-힣]+|\d+", text.lower())
    normalized: list[str] = []
    for token in raw_tokens:
        normalized.append(token)
        if re.fullmatch(r"[가-힣]+", token):
            stripped = _strip_korean_particle(token)
            if stripped and stripped != token:
                normalized.append(stripped)

    filtered = [token for token in normalized if token not in STOPWORDS]
    return _dedupe(filtered)


def _strip_korean_particle(token: str) -> str:
    """
    한국어 조사/어미를 간단히 제거.

    Args:
        token: 원본 토큰.

    Returns:
        조사/어미가 제거된 토큰.
    """

    particles = (
        "으로부터",
        "로부터",
        "에서",
        "에게",
        "한테",
        "까지",
        "부터",
        "마다",
        "처럼",
        "으로",
        "로",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "의",
        "과",
        "와",
        "도",
        "만",
    )

    for particle in particles:
        if token.endswith(particle) and len(token) > len(particle):
            return token[: -len(particle)]
    return token


def _dedupe(tokens: list[str]) -> list[str]:
    """
    토큰 리스트를 순서대로 중복 제거.

    Args:
        tokens: 토큰 리스트.

    Returns:
        중복 제거된 토큰 리스트.
    """

    seen: set[str] = set()
    output: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _alias_match_score(query: str, names: list[str]) -> int:
    """
    별칭 문자열이 질의에 포함되는지 점수화.

    Args:
        query: 검색 문자열.
        names: 메트릭 이름/별칭 리스트.

    Returns:
        매칭 점수.
    """

    score = 0
    for name in names:
        alias = name.strip().lower()
        if not alias:
            continue
        if alias in query:
            score += 5
    return score


def _build_example(metric: MetricDefinition) -> str:
    """
    메트릭 설명에 맞는 예시 문장을 생성.

    Args:
        metric: 메트릭 정의.

    Returns:
        예시 문장.
    """

    if metric.name in {"top_scorers", "top_rebounders", "top_assisters"}:
        return "예를 들어 시즌 2023-24 기준 상위 5개 팀을 비교할 때 사용한다."
    if metric.name in {"last_n_games", "streak"}:
        return "예를 들어 팀 약어를 LAL로 두고 최근 5경기를 조회해 흐름을 확인한다."
    if metric.name in {"attendance_top_games"}:
        return "예를 들어 관중 수 상위 10경기를 조회해 흥행 경향을 확인한다."
    return "예를 들어 특정 시즌/팀 조건을 지정해 비교 지표로 활용한다."


def _build_caution(metric: MetricDefinition) -> str:
    """
    메트릭 사용 시 주의사항을 생성.

    Args:
        metric: 메트릭 정의.

    Returns:
        주의 문장.
    """

    cut_rules = metric.cut_rules or {}
    min_games = cut_rules.get("min_games")

    if min_games:
        return f"표본이 너무 적으면 왜곡될 수 있으니 최소 {min_games}경기 이상 조건을 권장한다."
    return "필요한 테이블/컬럼이 데이터에 없으면 계산이 불가하다."
