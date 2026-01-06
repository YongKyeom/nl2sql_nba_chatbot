from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

@dataclass(frozen=True)
class TeamEntity:
    """
    팀 엔티티 정보.

    Args:
        team_name: 팀 풀네임.
        abbreviation: 팀 약어.
        nickname: 팀 별칭.
        city: 연고지.
    """

    team_name: str
    abbreviation: str
    nickname: str | None
    city: str | None


@dataclass(frozen=True)
class PlayerEntity:
    """
    선수 엔티티 정보.

    Args:
        person_id: 선수 ID.
        player_name: 표시 이름.
        team_abbreviation: 소속 팀 약어(없을 수 있음).
    """

    person_id: str
    player_name: str
    team_abbreviation: str | None


class EntityResolver:
    """
    팀/선수 엔티티를 표준 이름으로 보강하는 도구.

    DB를 한 번만 읽어 캐시에 올리고, 요청마다 캐시에서 매칭한다.
    """

    _TEAM_ALIAS_OVERRIDES = {
        "골스": "GSW",
        "워리어스": "GSW",
        "레이커스": "LAL",
        "셀틱스": "BOS",
        "불스": "CHI",
        "닉스": "NYK",
        "네츠": "BKN",
        "히트": "MIA",
        "스퍼스": "SAS",
        "썬즈": "PHX",
        "매버릭스": "DAL",
        "클리퍼스": "LAC",
        "캐벌리어스": "CLE",
        "피스톤스": "DET",
        "로케츠": "HOU",
        "재즈": "UTA",
        "호크스": "ATL",
        "벅스": "MIL",
        "랩터스": "TOR",
        "썬더": "OKC",
        "블레이저스": "POR",
        "킹스": "SAC",
        "펠리컨스": "NOP",
        "그리즐리스": "MEM",
        "울브스": "MIN",
        "매직": "ORL",
        "위저즈": "WAS",
        "페이서스": "IND",
        "식서스": "PHI",
        "너겟츠": "DEN",
    }

    _PLAYER_ALIAS_OVERRIDES = {
        "커리": "Stephen Curry",
        "스테픈커리": "Stephen Curry",
        "르브론": "LeBron James",
        "코비": "Kobe Bryant",
        "듀란트": "Kevin Durant",
        "야니스": "Giannis Antetokounmpo",
        "루카": "Luka Doncic",
        "요키치": "Nikola Jokic",
        "엠비드": "Joel Embiid",
        "하든": "James Harden",
        "릴라드": "Damian Lillard",
        "카와이": "Kawhi Leonard",
    }

    def __init__(self, db_path: Path) -> None:
        """
        EntityResolver 초기화.

        Args:
            db_path: SQLite DB 경로.
        """

        self._db_path = db_path
        self._loaded = False
        self._teams: list[TeamEntity] = []
        self._players: list[PlayerEntity] = []
        self._team_alias_index: dict[str, list[TeamEntity]] = {}
        self._player_alias_index: dict[str, list[PlayerEntity]] = {}

    def tool_schema(self) -> dict[str, Any]:
        """
        LLM tool schema를 반환한다.

        Returns:
            tool schema 딕셔너리.
        """

        return {
            "type": "function",
            "function": {
                "name": "entity_resolver",
                "description": "질의에서 팀/선수 엔티티를 표준 이름으로 보강한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "사용자 질문 원문"},
                        "previous_entities": {
                            "type": "object",
                            "description": "직전 결과에서 추출된 엔티티",
                        },
                        "top_k": {"type": "integer", "description": "후보 개수", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                },
            },
        }

    def resolve(
        self,
        *,
        query: str,
        previous_entities: dict[str, list[str]] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        질의에서 팀/선수 엔티티를 추출해 표준화한다.

        Args:
            query: 사용자 질문.
            previous_entities: 직전 결과에서 추출된 엔티티.
            top_k: 후보 개수.

        Returns:
            엔티티 보강 결과 딕셔너리.
        """

        self._ensure_loaded()

        normalized_query = _normalize_token(query)
        team_hits = _match_teams(normalized_query, query, self._team_alias_index, limit=top_k)
        player_hits = _match_players(normalized_query, self._player_alias_index, limit=top_k)

        if (not team_hits and not player_hits) and _looks_like_reference(query) and previous_entities:
            team_hits = _expand_previous_teams(previous_entities, self._teams)
            player_hits = _expand_previous_players(previous_entities, self._players)

        override_team = _resolve_team_override(query, self._teams)
        if override_team:
            team_hits = [override_team]

        override_player = _resolve_player_override(query, self._players)
        if override_player:
            player_hits = [override_player]

        return _build_resolution_payload(team_hits, player_hits)

    def _ensure_loaded(self) -> None:
        """
        엔티티 인덱스를 로드한다.

        Returns:
            None
        """

        if self._loaded:
            return

        self._teams = _load_teams(self._db_path)
        self._players = _load_players(self._db_path)
        self._team_alias_index = _build_team_alias_index(self._teams)
        self._player_alias_index = _build_player_alias_index(self._players)
        self._loaded = True


def _build_readonly_uri(db_path: Path) -> str:
    """
    SQLite 읽기 전용 URI를 만든다.

    Args:
        db_path: DB 경로.

    Returns:
        URI 문자열.
    """

    resolved = db_path.resolve()
    return f"file:{quote(str(resolved))}?mode=ro"


def _load_teams(db_path: Path) -> list[TeamEntity]:
    """
    팀 엔티티를 DB에서 로드한다.

    Args:
        db_path: SQLite DB 경로.

    Returns:
        TeamEntity 리스트.
    """

    query = "SELECT full_name, abbreviation, nickname, city FROM team"
    rows: list[TeamEntity] = []
    try:
        conn = sqlite3.connect(_build_readonly_uri(db_path), uri=True, timeout=5.0)
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(query)
        for full_name, abbreviation, nickname, city in cursor.fetchall():
            if not full_name or not abbreviation:
                continue
            rows.append(
                TeamEntity(
                    team_name=str(full_name),
                    abbreviation=str(abbreviation),
                    nickname=str(nickname) if nickname else None,
                    city=str(city) if city else None,
                )
            )
    except Exception:
        return []
    finally:
        if "conn" in locals():
            conn.close()
    return rows


def _load_players(db_path: Path) -> list[PlayerEntity]:
    """
    선수 엔티티를 DB에서 로드한다.

    Args:
        db_path: SQLite DB 경로.

    Returns:
        PlayerEntity 리스트.
    """

    query = """
    SELECT person_id, display_first_last, team_abbreviation
    FROM common_player_info
    WHERE display_first_last IS NOT NULL
    """
    rows: list[PlayerEntity] = []
    try:
        conn = sqlite3.connect(_build_readonly_uri(db_path), uri=True, timeout=5.0)
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(query)
        for person_id, display_name, team_abbreviation in cursor.fetchall():
            if not display_name:
                continue
            rows.append(
                PlayerEntity(
                    person_id=str(person_id),
                    player_name=str(display_name),
                    team_abbreviation=str(team_abbreviation) if team_abbreviation else None,
                )
            )
    except Exception:
        return []
    finally:
        if "conn" in locals():
            conn.close()
    return rows


def _build_team_alias_index(teams: list[TeamEntity]) -> dict[str, list[TeamEntity]]:
    """
    팀 별칭 인덱스를 만든다.

    Args:
        teams: 팀 목록.

    Returns:
        별칭 인덱스.
    """

    index: dict[str, list[TeamEntity]] = {}
    for team in teams:
        aliases = {
            team.team_name,
            team.abbreviation,
            team.nickname or "",
            team.city or "",
        }
        for alias in aliases:
            cleaned = _normalize_token(alias)
            if not cleaned:
                continue
            index.setdefault(cleaned, []).append(team)
    return index


def _build_player_alias_index(players: list[PlayerEntity]) -> dict[str, list[PlayerEntity]]:
    """
    선수 별칭 인덱스를 만든다.

    Args:
        players: 선수 목록.

    Returns:
        별칭 인덱스.
    """

    index: dict[str, list[PlayerEntity]] = {}
    for player in players:
        cleaned = _normalize_token(player.player_name)
        if cleaned:
            index.setdefault(cleaned, []).append(player)

        last_name = player.player_name.split()[-1] if player.player_name else ""
        cleaned_last = _normalize_token(last_name)
        if cleaned_last and len(cleaned_last) >= 4:
            index.setdefault(cleaned_last, []).append(player)
    return index


def _normalize_token(text: str) -> str:
    """
    문자열을 매칭 가능한 형태로 정규화한다.

    Args:
        text: 원본 문자열.

    Returns:
        정규화된 문자열.
    """

    cleaned = re.sub(r"[^0-9a-zA-Z가-힣]+", "", text.lower())
    return cleaned.strip()


def _match_teams(
    normalized_query: str,
    raw_query: str,
    index: dict[str, list[TeamEntity]],
    *,
    limit: int,
) -> list[TeamEntity]:
    """
    팀 엔티티를 매칭한다.

    Args:
        normalized_query: 정규화된 질의.
        raw_query: 원본 질의.
        index: 팀 별칭 인덱스.
        limit: 최대 반환 수.

    Returns:
        팀 엔티티 리스트.
    """

    hits: list[TeamEntity] = []
    token_hits = _find_abbreviation_tokens(raw_query)
    for token in token_hits:
        entry = _find_team_by_abbreviation(index, token)
        if entry:
            hits.append(entry)

    for alias, teams in index.items():
        if len(alias) < 2:
            continue
        if alias in normalized_query:
            hits.extend(teams)

    return _dedupe_teams(hits)[:limit]


def _match_players(
    normalized_query: str,
    index: dict[str, list[PlayerEntity]],
    *,
    limit: int,
) -> list[PlayerEntity]:
    """
    선수 엔티티를 매칭한다.

    Args:
        normalized_query: 정규화된 질의.
        index: 선수 별칭 인덱스.
        limit: 최대 반환 수.

    Returns:
        선수 엔티티 리스트.
    """

    hits: list[PlayerEntity] = []
    for alias, players in index.items():
        if len(alias) < 4:
            continue
        if alias in normalized_query:
            hits.extend(players)

    return _dedupe_players(hits)[:limit]


def _find_abbreviation_tokens(text: str) -> list[str]:
    """
    약어 형태의 팀 토큰을 찾는다.

    Args:
        text: 원본 텍스트.

    Returns:
        팀 약어 후보 리스트.
    """

    tokens = re.findall(r"\\b[A-Za-z]{2,3}\\b", text)
    return [token.upper() for token in tokens]


def _find_team_by_abbreviation(index: dict[str, list[TeamEntity]], token: str) -> TeamEntity | None:
    """
    약어로 팀 엔티티를 찾는다.

    Args:
        index: 팀 인덱스.
        token: 팀 약어.

    Returns:
        팀 엔티티 또는 None.
    """

    normalized = _normalize_token(token)
    matches = index.get(normalized)
    if matches:
        return matches[0]
    return None


def _resolve_team_override(query: str, teams: list[TeamEntity]) -> TeamEntity | None:
    """
    한글 별칭으로 팀을 보정한다.

    Args:
        query: 사용자 질문.
        teams: 팀 목록.

    Returns:
        TeamEntity 또는 None.
    """

    for alias, abbreviation in EntityResolver._TEAM_ALIAS_OVERRIDES.items():
        if alias in query:
            for team in teams:
                if team.abbreviation == abbreviation:
                    return team
    return None


def _resolve_player_override(query: str, players: list[PlayerEntity]) -> PlayerEntity | None:
    """
    한글 별칭으로 선수를 보정한다.

    Args:
        query: 사용자 질문.
        players: 선수 목록.

    Returns:
        PlayerEntity 또는 None.
    """

    for alias, player_name in EntityResolver._PLAYER_ALIAS_OVERRIDES.items():
        if alias in query:
            for player in players:
                if player.player_name == player_name:
                    return player
    return None


def _build_resolution_payload(teams: list[TeamEntity], players: list[PlayerEntity]) -> dict[str, Any]:
    """
    엔티티 보강 결과를 직렬화한다.

    Args:
        teams: 팀 목록.
        players: 선수 목록.

    Returns:
        직렬화된 결과.
    """

    team_items = [
        {"team_abbreviation": team.abbreviation, "team_name": team.team_name, "city": team.city, "nickname": team.nickname}
        for team in teams
    ]
    player_items = [
        {"player_name": player.player_name, "person_id": player.person_id, "team_abbreviation": player.team_abbreviation}
        for player in players
    ]

    filters: dict[str, Any] = {}
    if len(team_items) == 1:
        filters["team_abbreviation"] = team_items[0]["team_abbreviation"]
    elif len(team_items) >= 2:
        filters["team_abbreviation_in"] = [item["team_abbreviation"] for item in team_items[:3]]
        filters["team_a"] = team_items[0]["team_abbreviation"]
        filters["team_b"] = team_items[1]["team_abbreviation"]

    if len(player_items) == 1:
        filters["player_name"] = player_items[0]["player_name"]
    elif len(player_items) >= 2:
        filters["player_name_in"] = [item["player_name"] for item in player_items[:3]]

    return {
        "teams": team_items,
        "players": player_items,
        "filters": filters,
        "has_match": bool(team_items or player_items),
    }


def _looks_like_reference(query: str) -> bool:
    """
    이전 결과 참조 표현이 있는지 확인한다.

    Args:
        query: 사용자 질문.

    Returns:
        참조 표현 여부.
    """

    keywords = ["그 팀", "해당 팀", "그 선수", "이 선수", "저 팀", "방금", "이전 결과"]
    return any(keyword in query for keyword in keywords)


def _expand_previous_teams(
    previous_entities: dict[str, list[str]],
    teams: list[TeamEntity],
) -> list[TeamEntity]:
    """
    이전 엔티티의 팀 약어를 TeamEntity로 확장한다.

    Args:
        previous_entities: 직전 엔티티 딕셔너리.
        teams: 팀 목록.

    Returns:
        팀 엔티티 리스트.
    """

    abbreviations = set(previous_entities.get("teams", []))
    return [team for team in teams if team.abbreviation in abbreviations]


def _expand_previous_players(
    previous_entities: dict[str, list[str]],
    players: list[PlayerEntity],
) -> list[PlayerEntity]:
    """
    이전 엔티티의 선수 이름을 PlayerEntity로 확장한다.

    Args:
        previous_entities: 직전 엔티티 딕셔너리.
        players: 선수 목록.

    Returns:
        선수 엔티티 리스트.
    """

    names = set(previous_entities.get("players", []))
    return [player for player in players if player.player_name in names]


def _dedupe_teams(teams: list[TeamEntity]) -> list[TeamEntity]:
    """
    팀 엔티티 중복을 제거한다.

    Args:
        teams: 팀 목록.

    Returns:
        중복 제거된 팀 목록.
    """

    seen: set[str] = set()
    unique: list[TeamEntity] = []
    for team in teams:
        if team.abbreviation in seen:
            continue
        seen.add(team.abbreviation)
        unique.append(team)
    return unique


def _dedupe_players(players: list[PlayerEntity]) -> list[PlayerEntity]:
    """
    선수 엔티티 중복을 제거한다.

    Args:
        players: 선수 목록.

    Returns:
        중복 제거된 선수 목록.
    """

    seen: set[str] = set()
    unique: list[PlayerEntity] = []
    for player in players:
        if player.player_name in seen:
            continue
        seen.add(player.player_name)
        unique.append(player)
    return unique


if __name__ == "__main__":
    # 1) 환경 설정과 DB 경로를 확인한다.
    config = load_config()
    if not config.db_path.exists():
        print(f"DB 파일을 찾지 못했습니다: {config.db_path}")
    else:
        # 2) 샘플 질의에 대한 엔티티 보강 결과를 확인한다.
        resolver = EntityResolver(config.db_path)
        sample = resolver.resolve(query="골스 최근 5경기", previous_entities=None, top_k=3)
        print(sample)
