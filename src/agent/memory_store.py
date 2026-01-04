from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PreferenceCount:
    """
    선호도(빈도) 집계 레코드.

    Args:
        category: 카테고리(예: season, team, metric).
        value: 값(예: 2023-24, LAL, win_pct).
        count: 누적 횟수.
    """

    category: str
    value: str
    count: int


class LongTermMemoryStore(Protocol):
    """
    장기 메모리 저장소 인터페이스.

    구현체는 파일/DB 등 저장 방식이 달라도 동일한 API로 동작해야 한다.
    """

    def increment(self, category: str, value: str, *, amount: int = 1) -> None:
        """
        (category, value) 카운트를 증가시킨다.

        Args:
            category: 카테고리.
            value: 값.
            amount: 증가량.

        Returns:
            None
        """

    def top(self, category: str, *, limit: int = 3) -> list[PreferenceCount]:
        """
        특정 카테고리의 상위 N개를 반환한다.

        Args:
            category: 카테고리.
            limit: 반환 개수.

        Returns:
            PreferenceCount 리스트.
        """

    def get_profile(self, key: str) -> str | None:
        """
        사용자 프로필 값을 조회한다.

        Args:
            key: 프로필 키.

        Returns:
            값(없으면 None).
        """

    def set_profile(self, key: str, value: str) -> None:
        """
        사용자 프로필 값을 저장한다.

        Args:
            key: 프로필 키.
            value: 프로필 값.

        Returns:
            None
        """

    def clear(self) -> None:
        """
        저장소 내용을 초기화한다.

        Returns:
            None
        """


class InMemoryLongTermMemoryStore:
    """
    테스트/로컬 실행용 인메모리 저장소.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}
        self._profile: dict[str, str] = {}

    def increment(self, category: str, value: str, *, amount: int = 1) -> None:
        key = (category, value)
        self._counts[key] = self._counts.get(key, 0) + amount

    def top(self, category: str, *, limit: int = 3) -> list[PreferenceCount]:
        pairs = [(value, count) for (cat, value), count in self._counts.items() if cat == category]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [PreferenceCount(category=category, value=value, count=count) for value, count in pairs[:limit]]

    def get_profile(self, key: str) -> str | None:
        return self._profile.get(key)

    def set_profile(self, key: str, value: str) -> None:
        self._profile[key] = value

    def clear(self) -> None:
        self._counts = {}
        self._profile = {}


class SQLiteLongTermMemoryStore:
    """
    SQLite 기반 장기 메모리 저장소.

    파일 하나로 관리되며, 앱을 재시작해도 선호도/프로필이 유지된다.
    """

    def __init__(self, path: Path) -> None:
        """
        SQLiteLongTermMemoryStore 초기화.

        Args:
            path: SQLite 파일 경로.
        """

        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def increment(self, category: str, value: str, *, amount: int = 1) -> None:
        now = _now_iso()
        sql = """
        INSERT INTO preference_counts(category, value, count, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(category, value)
        DO UPDATE SET count = count + excluded.count, updated_at = excluded.updated_at
        """
        with self._connect() as conn:
            conn.execute(sql, (category, value, amount, now))

    def top(self, category: str, *, limit: int = 3) -> list[PreferenceCount]:
        sql = """
        SELECT category, value, count
        FROM preference_counts
        WHERE category = ?
        ORDER BY count DESC, value ASC
        LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (category, limit)).fetchall()
        return [PreferenceCount(category=row[0], value=row[1], count=int(row[2])) for row in rows]

    def get_profile(self, key: str) -> str | None:
        sql = "SELECT value FROM user_profile WHERE key = ? LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, (key,)).fetchone()
        if row is None:
            return None
        return str(row[0])

    def set_profile(self, key: str, value: str) -> None:
        now = _now_iso()
        sql = """
        INSERT INTO user_profile(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key)
        DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """
        with self._connect() as conn:
            conn.execute(sql, (key, value, now))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM preference_counts")
            conn.execute("DELETE FROM user_profile")

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_counts (
                    category TEXT NOT NULL,
                    value TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(category, value)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=3.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn


def _now_iso() -> str:
    """
    UTC ISO 타임스탬프를 반환한다.

    Returns:
        ISO 문자열.
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    # 1) 임시 파일 기반으로 저장/조회가 되는지 빠르게 확인한다.
    store = SQLiteLongTermMemoryStore(Path("result/_memory_dev.sqlite"))
    store.clear()

    # 2) 집계/프로필 저장
    store.increment("season", "2023-24")
    store.increment("season", "2023-24")
    store.increment("team", "LAL")
    store.set_profile("default_season", "2023-24")

    # 3) 조회 결과 확인
    print(store.top("season", limit=3))
    print(store.get_profile("default_season"))
