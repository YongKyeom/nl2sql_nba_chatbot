from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ChatSession:
    """
    채팅 세션 메타 정보.

    Args:
        chat_id: 채팅 식별자.
        title: 채팅 제목.
        created_at: 생성 시각(ISO).
        updated_at: 갱신 시각(ISO).
    """

    chat_id: str
    title: str
    created_at: str
    updated_at: str


class ChatStore:
    """
    채팅 세션/메시지를 SQLite에 저장한다.
    """

    def __init__(self, path: Path) -> None:
        """
        ChatStore 초기화.

        Args:
            path: SQLite 파일 경로.
        """

        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def list_sessions(self, user_id: str) -> list[ChatSession]:
        """
        사용자 채팅 세션 목록을 반환한다.

        Args:
            user_id: 사용자 ID.

        Returns:
            ChatSession 리스트.
        """

        sql = """
        SELECT chat_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [ChatSession(chat_id=row[0], title=row[1], created_at=row[2], updated_at=row[3]) for row in rows]

    def get_session(self, user_id: str, chat_id: str) -> ChatSession | None:
        """
        채팅 세션을 조회한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            ChatSession 또는 None.
        """

        sql = """
        SELECT chat_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE user_id = ? AND chat_id = ?
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (user_id, chat_id)).fetchone()
        if row is None:
            return None
        return ChatSession(chat_id=row[0], title=row[1], created_at=row[2], updated_at=row[3])

    def create_session(self, user_id: str, *, title: str = "새 대화") -> str:
        """
        새로운 채팅 세션을 만든다.

        Args:
            user_id: 사용자 ID.
            title: 초기 제목.

        Returns:
            생성된 chat_id.
        """

        chat_id = uuid4().hex
        now = _now_iso()
        sql = """
        INSERT INTO chat_sessions(user_id, chat_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (user_id, chat_id, title, now, now))
        return chat_id

    def update_title(self, user_id: str, chat_id: str, title: str) -> None:
        """
        채팅 제목을 업데이트한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            title: 새 제목.

        Returns:
            None
        """

        sql = """
        UPDATE chat_sessions
        SET title = ?, updated_at = ?
        WHERE user_id = ? AND chat_id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (title, _now_iso(), user_id, chat_id))

    def delete_session(self, user_id: str, chat_id: str) -> None:
        """
        채팅 세션과 메시지를 삭제한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            None
        """

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            )
            conn.execute(
                "DELETE FROM chat_sessions WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            )

    def add_message(
        self,
        *,
        user_id: str,
        chat_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        메시지를 저장한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            role: 역할(user/assistant).
            content: 본문.
            meta: 부가 정보(없으면 None).

        Returns:
            None
        """

        now = _now_iso()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        sql = """
        INSERT INTO chat_messages(user_id, chat_id, role, content, meta_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (user_id, chat_id, role, content, meta_json, now))
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE user_id = ? AND chat_id = ?",
                (now, user_id, chat_id),
            )

    def list_messages(self, user_id: str, chat_id: str) -> list[dict[str, Any]]:
        """
        채팅 메시지를 시간순으로 조회한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            메시지 딕셔너리 리스트.
        """

        sql = """
        SELECT role, content, meta_json, created_at
        FROM chat_messages
        WHERE user_id = ? AND chat_id = ?
        ORDER BY message_id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (user_id, chat_id)).fetchall()

        messages: list[dict[str, Any]] = []
        for role, content, meta_json, created_at in rows:
            meta = json.loads(meta_json) if meta_json else {}
            message = {
                "role": role,
                "content": content,
                "created_at": created_at,
                **meta,
            }
            messages.append(message)
        return messages

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, chat_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages ON chat_messages(user_id, chat_id, message_id)")

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
    store = ChatStore(Path("result/_chat_dev.sqlite"))
    chat_id = store.create_session("developer", title="테스트 채팅")
    store.add_message(user_id="developer", chat_id=chat_id, role="user", content="안녕", meta={})
    print(store.list_sessions("developer"))
    print(store.list_messages("developer", chat_id))
