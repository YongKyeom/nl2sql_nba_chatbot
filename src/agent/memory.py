from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.memory_store import InMemoryLongTermMemoryStore, LongTermMemoryStore, SQLiteLongTermMemoryStore


@dataclass(slots=True)
class ConversationTurn:
    """
    한 턴의 대화 기록.

    Args:
        user_message: 사용자가 입력한 원문.
        assistant_message: 에이전트가 최종으로 응답한 텍스트.
        route: 라우팅 결과(디버깅 용도).
        sql: 실행/생성된 SQL(있을 때만).
        planned_slots: 플래너 슬롯(있을 때만).
    """

    user_message: str
    assistant_message: str | None = None
    route: str | None = None
    sql: str | None = None
    planned_slots: dict[str, Any] | None = None


@dataclass(slots=True)
class ShortTermMemory:
    """
    세션 내에서만 유지되는 단기 메모리.

    설계 의도:
    - NL2SQL은 "직전 결과"를 기반으로 후속 질의(예: "상위 3개만")가 자주 발생한다.
    - 따라서 마지막 SQL/결과/슬롯은 구조화된 형태로 보관하고, 대화 로그는 짧게만 유지한다.
    """

    last_sql: str | None = None
    last_result: pd.DataFrame | None = None
    last_result_schema: list[str] | None = None
    last_slots: dict[str, Any] = field(default_factory=dict)
    last_route: str | None = None
    last_entities: dict[str, list[str]] = field(default_factory=dict)

    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 20

    def reset(self) -> None:
        """
        단기 메모리를 초기화한다.

        Returns:
            None
        """

        self.last_sql = None
        self.last_result = None
        self.last_result_schema = None
        self.last_slots = {}
        self.last_route = None
        self.last_entities = {}
        self.turns = []

    def start_turn(self, user_message: str) -> None:
        """
        새 사용자 입력을 턴으로 기록한다.

        Args:
            user_message: 사용자 입력.

        Returns:
            None
        """

        self.turns.append(ConversationTurn(user_message=user_message))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def finish_turn(
        self,
        *,
        assistant_message: str | None,
        route: str | None,
        sql: str | None,
        planned_slots: dict[str, Any] | None,
    ) -> None:
        """
        직전 턴에 에이전트 응답/메타 정보를 채운다.

        Args:
            assistant_message: 최종 응답.
            route: 라우팅 결과.
            sql: SQL 문자열.
            planned_slots: 플래너 슬롯.

        Returns:
            None
        """

        if not self.turns:
            self.turns.append(ConversationTurn(user_message="(unknown)"))

        self.turns[-1].assistant_message = assistant_message
        self.turns[-1].route = route
        self.turns[-1].sql = sql
        self.turns[-1].planned_slots = planned_slots

    def update_sql_result(self, sql: str, dataframe: pd.DataFrame, slots: dict[str, Any]) -> None:
        """
        SQL 실행 결과를 저장한다.

        Args:
            sql: 실행 SQL.
            dataframe: 결과 데이터프레임.
            slots: 플래너 슬롯.

        Returns:
            None
        """

        self.last_sql = sql
        self.last_result = dataframe
        self.last_result_schema = list(dataframe.columns)
        self.last_slots = slots
        self.last_entities = _extract_entities_from_dataframe(dataframe)

    def update_route(self, route: str) -> None:
        """
        마지막 라우트를 저장한다.

        Args:
            route: 라우팅 문자열.

        Returns:
            None
        """

        self.last_route = route

    def update_entities_from_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        결과 데이터프레임에서 엔티티를 추출해 저장한다.

        Args:
            dataframe: 결과 데이터프레임.

        Returns:
            None
        """

        self.last_entities = _extract_entities_from_dataframe(dataframe)

    def build_recent_dialogue(self, *, limit: int = 4, max_chars_per_message: int = 180) -> str:
        """
        최근 대화 일부를 프롬프트 주입용으로 요약한다.

        이 값은 LLM 토큰을 갉아먹기 쉬워서, "짧게/얕게"만 유지한다.

        Args:
            limit: 포함할 최근 턴 개수.
            max_chars_per_message: 메시지당 최대 글자 수.

        Returns:
            요약 문자열(없으면 "없음").
        """

        if not self.turns:
            return "없음"

        selected = self.turns[-limit:]
        lines: list[str] = []
        for turn in selected:
            user = _truncate(turn.user_message, max_chars_per_message)
            lines.append(f"- 사용자: {user}")

            if turn.assistant_message:
                assistant = _truncate(turn.assistant_message, max_chars_per_message)
                lines.append(f"  어시스턴트: {assistant}")

        return "\n".join(lines) if lines else "없음"


class LongTermMemory:
    """
    세션을 넘어 유지되는 장기 메모리.

    구현 방향:
    - 대화 원문을 그대로 저장하면 프라이버시/용량 문제가 생긴다.
    - 대신, "선호/기본값"처럼 실제 도움되는 신호만 구조화해 저장한다.
    """

    PROFILE_DEFAULT_SEASON = "default_season"
    PROFILE_PREFERRED_TEAM = "preferred_team"

    def __init__(self, store: LongTermMemoryStore) -> None:
        """
        LongTermMemory 초기화.

        Args:
            store: 장기 메모리 저장소.
        """

        self._store = store

    def observe(self, *, user_message: str, planned_slots: dict[str, Any] | None) -> None:
        """
        한 턴의 입력/슬롯에서 장기 메모리 신호를 학습한다.

        Args:
            user_message: 사용자 입력.
            planned_slots: 플래너 슬롯(없으면 None).

        Returns:
            None
        """

        try:
            # 1) 슬롯 기반(가장 신뢰도가 높음) 집계
            if planned_slots:
                season = planned_slots.get("season")
                if isinstance(season, str) and season:
                    self._store.increment("season", season)

                metric = planned_slots.get("metric")
                if isinstance(metric, str) and metric:
                    self._store.increment("metric", metric)

                filters = planned_slots.get("filters", {})
                if isinstance(filters, dict):
                    for key in ("team_abbreviation", "team_a", "team_b", "opponent"):
                        value = filters.get(key)
                        if isinstance(value, str) and value:
                            self._store.increment("team", value)

            # 2) 명시적 "기본값" 선언은 프로필로 승격
            explicit = _extract_explicit_profile(user_message)
            for key, value in explicit.items():
                self._store.set_profile(key, value)
        except Exception:
            # 장기 메모리는 "있으면 좋고 없어도 동작해야" 한다.
            # 파일 잠금/권한 이슈로 저장이 실패하더라도 대화 흐름을 깨지 않는다.
            return

    def get_default_season(self) -> str | None:
        """
        기본 시즌을 반환한다.

        Returns:
            시즌 문자열(없으면 None).
        """

        try:
            value = self._store.get_profile(self.PROFILE_DEFAULT_SEASON)
            if value:
                return value

            top = self._store.top("season", limit=1)
            return top[0].value if top else None
        except Exception:
            return None

    def get_preferred_team(self) -> str | None:
        """
        선호 팀 약어를 반환한다.

        Returns:
            팀 약어(없으면 None).
        """

        try:
            value = self._store.get_profile(self.PROFILE_PREFERRED_TEAM)
            if value:
                return value

            top = self._store.top("team", limit=1)
            return top[0].value if top else None
        except Exception:
            return None

    def build_hint(self) -> str:
        """
        프롬프트 주입용 장기 메모리 요약을 생성한다.

        Returns:
            요약 문자열(없으면 "없음").
        """

        try:
            parts: list[str] = []

            default_season = self.get_default_season()
            if default_season:
                parts.append(f"default_season={default_season}")

            preferred_team = self.get_preferred_team()
            if preferred_team:
                parts.append(f"preferred_team={preferred_team}")

            top_metrics = self._store.top("metric", limit=3)
            if top_metrics:
                metrics_text = ", ".join(item.value for item in top_metrics)
                parts.append(f"top_metrics=[{metrics_text}]")

            return "\n".join(parts) if parts else "없음"
        except Exception:
            return "없음"

    def apply_defaults(self, planned_slots: dict[str, Any]) -> None:
        """
        비어있는 슬롯을 장기 메모리 기본값으로 보강한다.

        Args:
            planned_slots: 플래너 슬롯.

        Returns:
            None
        """

        if planned_slots.get("season"):
            return

        default_season = self.get_default_season()
        if default_season:
            planned_slots["season"] = default_season
            filters = planned_slots.get("filters", {})
            if isinstance(filters, dict):
                filters["season_default"] = default_season
                planned_slots["filters"] = filters

    def clear(self) -> None:
        """
        장기 메모리를 초기화한다.

        Returns:
            None
        """

        try:
            self._store.clear()
        except Exception:
            return


@dataclass(slots=True)
class ConversationMemory:
    """
    단기/장기 메모리를 함께 다루는 메모리 파사드.

    기존 코드(체인)가 `memory.last_sql` 같은 필드를 직접 참조하고 있어서,
    외부에는 동일한 속성을 유지하고 내부에서만 구조를 나눈다.
    """

    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = field(default_factory=lambda: LongTermMemory(InMemoryLongTermMemoryStore()))

    @classmethod
    def persistent(cls, path: Path) -> ConversationMemory:
        """
        SQLite 기반 장기 메모리를 사용하는 메모리를 생성한다.

        Args:
            path: 장기 메모리 DB 파일 경로.

        Returns:
            ConversationMemory.
        """

        try:
            store = SQLiteLongTermMemoryStore(path)
        except Exception:
            return cls()
        return cls(long_term=LongTermMemory(store))

    @property
    def last_sql(self) -> str | None:  # noqa: D401 - 기존 인터페이스 유지
        return self.short_term.last_sql

    @last_sql.setter
    def last_sql(self, value: str | None) -> None:
        self.short_term.last_sql = value

    @property
    def last_result(self) -> pd.DataFrame | None:  # noqa: D401 - 기존 인터페이스 유지
        return self.short_term.last_result

    @last_result.setter
    def last_result(self, value: pd.DataFrame | None) -> None:
        self.short_term.last_result = value

    @property
    def last_result_schema(self) -> list[str] | None:  # noqa: D401 - 기존 인터페이스 유지
        return self.short_term.last_result_schema

    @last_result_schema.setter
    def last_result_schema(self, value: list[str] | None) -> None:
        self.short_term.last_result_schema = value

    @property
    def last_slots(self) -> dict[str, Any]:  # noqa: D401 - 기존 인터페이스 유지
        return self.short_term.last_slots

    @last_slots.setter
    def last_slots(self, value: dict[str, Any]) -> None:
        self.short_term.last_slots = value

    @property
    def last_route(self) -> str | None:  # noqa: D401 - 기존 인터페이스 유지
        return self.short_term.last_route

    def reset(self) -> None:
        """
        단기 메모리를 초기화한다.

        장기 메모리는 사용자의 선호/기본값을 담는 영역이라 기본 동작에서는 유지한다.

        Returns:
            None
        """

        self.short_term.reset()

    def clear_long_term(self) -> None:
        """
        장기 메모리를 초기화한다.

        Returns:
            None
        """

        self.long_term.clear()

    def reset_all(self) -> None:
        """
        단기/장기 메모리를 모두 초기화한다.

        Returns:
            None
        """

        self.short_term.reset()
        self.long_term.clear()

    def update_sql_result(self, sql: str, dataframe: pd.DataFrame, slots: dict[str, Any]) -> None:
        """
        SQL 실행 결과를 단기 메모리에 저장한다.

        Args:
            sql: 실행 SQL.
            dataframe: 결과 데이터프레임.
            slots: 플래너 슬롯.

        Returns:
            None
        """

        self.short_term.update_sql_result(sql, dataframe, slots)

    def update_entities_from_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        결과 데이터프레임에서 엔티티를 추출해 저장한다.

        Args:
            dataframe: 결과 데이터프레임.

        Returns:
            None
        """

        self.short_term.update_entities_from_dataframe(dataframe)

    def update_route(self, route: str) -> None:
        """
        마지막 라우트를 저장한다.

        Args:
            route: 라우팅 문자열.

        Returns:
            None
        """

        self.short_term.update_route(route)

    def start_turn(self, user_message: str) -> None:
        """
        새 턴을 시작한다.

        Args:
            user_message: 사용자 입력.

        Returns:
            None
        """

        self.short_term.start_turn(user_message)

    def finish_turn(
        self,
        *,
        assistant_message: str | None,
        route: str | None,
        sql: str | None,
        planned_slots: dict[str, Any] | None,
    ) -> None:
        """
        직전 턴을 마무리하고, 장기 메모리 학습까지 수행한다.

        Args:
            assistant_message: 최종 응답.
            route: 라우팅 결과.
            sql: 생성/실행 SQL.
            planned_slots: 플래너 슬롯.

        Returns:
            None
        """

        self.short_term.finish_turn(
            assistant_message=assistant_message,
            route=route,
            sql=sql,
            planned_slots=planned_slots,
        )
        self.long_term.observe(user_message=self.short_term.turns[-1].user_message, planned_slots=planned_slots)


def _truncate(text: str, max_chars: int) -> str:
    """
    텍스트를 max_chars 이내로 줄인다.

    Args:
        text: 원본 텍스트.
        max_chars: 최대 글자 수.

    Returns:
        줄인 텍스트.
    """

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _extract_entities_from_dataframe(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """
    데이터프레임에서 팀/선수 엔티티를 추출한다.

    Args:
        dataframe: 결과 데이터프레임.

    Returns:
        {"teams": [...], "players": [...]} 형태의 엔티티 딕셔너리.
    """

    if dataframe.empty:
        return {}

    team_candidates: set[str] = set()
    player_candidates: set[str] = set()

    team_cols = _find_team_columns(dataframe.columns)
    for col in team_cols:
        values = dataframe[col].dropna().astype(str).unique().tolist()
        for value in values:
            token = value.strip().upper()
            if re.fullmatch(r"[A-Z]{2,3}", token):
                team_candidates.add(token)

    player_cols = _find_player_columns(dataframe.columns)
    for col in player_cols:
        values = dataframe[col].dropna().astype(str).unique().tolist()
        for value in values:
            name = value.strip()
            if len(name) < 3:
                continue
            player_candidates.add(name)

    entities: dict[str, list[str]] = {}
    if team_candidates:
        entities["teams"] = sorted(team_candidates)
    if player_candidates:
        entities["players"] = sorted(player_candidates)
    return entities


def _find_team_columns(columns: list[str]) -> list[str]:
    """
    팀 약어 컬럼 후보를 찾는다.

    Args:
        columns: 데이터프레임 컬럼 목록.

    Returns:
        팀 컬럼 리스트.
    """

    team_cols: list[str] = []
    candidates = {
        "team_abbreviation",
        "team_abbreviation_home",
        "team_abbreviation_away",
        "team_a",
        "team_b",
        "opponent",
    }
    for col in columns:
        lowered = col.lower()
        if col in candidates or lowered.endswith("team_abbreviation"):
            team_cols.append(col)
    return team_cols


def _find_player_columns(columns: list[str]) -> list[str]:
    """
    선수 이름 컬럼 후보를 찾는다.

    Args:
        columns: 데이터프레임 컬럼 목록.

    Returns:
        선수 컬럼 리스트.
    """

    player_cols: list[str] = []
    candidates = {
        "player_name",
        "display_first_last",
        "full_name",
        "first_name",
        "last_name",
    }
    for col in columns:
        lowered = col.lower()
        if col in candidates or "player" in lowered:
            player_cols.append(col)
    return player_cols


def _extract_explicit_profile(user_message: str) -> dict[str, str]:
    """
    사용자가 명시적으로 "기본값"을 지정한 신호를 추출한다.

    Args:
        user_message: 사용자 입력.

    Returns:
        프로필 키/값 딕셔너리.
    """

    lowered = user_message.lower()
    result: dict[str, str] = {}

    # 1) 시즌 기본값: "앞으로 2023-24로", "기본 시즌은 2023-24" 같은 케이스
    if any(keyword in user_message for keyword in ["기본", "앞으로", "항상", "디폴트", "default"]):
        season = _extract_season(user_message)
        if season:
            result[LongTermMemory.PROFILE_DEFAULT_SEASON] = season

    # 2) 응원 팀: "내 최애팀은 LAL" 같은 케이스
    if any(keyword in user_message for keyword in ["응원팀", "최애팀", "좋아하는 팀", "내 팀"]):
        team = _extract_team_abbreviation(user_message)
        if team:
            result[LongTermMemory.PROFILE_PREFERRED_TEAM] = team

    # 3) 영어로도 기본값 표현이 들어오는 경우가 있어 최소한만 보강한다.
    if "my team" in lowered or "favorite team" in lowered:
        team = _extract_team_abbreviation(user_message)
        if team:
            result[LongTermMemory.PROFILE_PREFERRED_TEAM] = team

    return result


def _extract_season(text: str) -> str | None:
    """
    2023-24 형태의 시즌 표현을 추출한다.

    Args:
        text: 입력 텍스트.

    Returns:
        시즌 문자열(없으면 None).
    """

    match = re.search(r"(20\d{2})\s*[-/\.]\s*(\d{2})", text)
    if not match:
        return None
    start_year = match.group(1)
    end_year = match.group(2)
    return f"{start_year}-{end_year}"


def _extract_team_abbreviation(text: str) -> str | None:
    """
    팀 약어(2~3자 영문 대문자)를 추출한다.

    Args:
        text: 입력 텍스트.

    Returns:
        팀 약어(없으면 None).
    """

    match = re.search(r"\b([A-Za-z]{2,3})\b", text)
    if not match:
        return None
    return match.group(1).upper()


if __name__ == "__main__":
    # 1) 장기 메모리(인메모리) 포함 메모리 생성
    memory = ConversationMemory()

    # 2) 턴 기록/학습 흐름 확인
    memory.start_turn("앞으로 기본 시즌은 2023-24로 해줘")
    memory.finish_turn(
        assistant_message="좋아요. 기본 시즌을 2023-24로 기억해둘게요.",
        route="general",
        sql=None,
        planned_slots={"season": "2023-24", "metric": "win_pct", "filters": {}},
    )

    # 3) 장기 메모리 힌트 확인
    print(memory.long_term.build_hint())
