from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass(frozen=True)
class HistoryMessage:
    """
    히스토리 메시지 레코드.

    Args:
        role: 메시지 역할(user/assistant).
        content: 메시지 본문.
        meta: 부가 정보.
        created_at: 생성 시각(ISO).
    """

    role: str
    content: str
    meta: dict[str, Any]
    created_at: str


class ChatHistoryStore:
    """
    대화 히스토리를 별도 SQLite에 저장한다.

    UI 로그(chat.sqlite)와 분리하여 대화 요약/복원에 사용한다.
    """

    def __init__(self, path: Path) -> None:
        """
        ChatHistoryStore 초기화.

        Args:
            path: SQLite 파일 경로.
        """

        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

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
            content: 메시지 본문.
            meta: 부가 정보.
        """

        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        now = _now_iso()
        sql = """
        INSERT INTO chat_history_messages(user_id, chat_id, role, content, meta_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (user_id, chat_id, role, content, meta_json, now))

    def list_messages(self, user_id: str, chat_id: str) -> list[HistoryMessage]:
        """
        채팅 히스토리를 시간순으로 조회한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            HistoryMessage 리스트.
        """

        sql = """
        SELECT role, content, meta_json, created_at
        FROM chat_history_messages
        WHERE user_id = ? AND chat_id = ?
        ORDER BY message_id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (user_id, chat_id)).fetchall()

        messages: list[HistoryMessage] = []
        for role, content, meta_json, created_at in rows:
            meta = json.loads(meta_json) if meta_json else {}
            messages.append(
                HistoryMessage(
                    role=str(role),
                    content=str(content),
                    meta=meta,
                    created_at=str(created_at),
                )
            )
        return messages

    def set_summary(self, user_id: str, chat_id: str, summary_text: str) -> None:
        """
        요약 텍스트를 저장한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            summary_text: 요약 문자열.
        """

        now = _now_iso()
        sql = """
        INSERT INTO chat_history_state(user_id, chat_id, summary_text, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, chat_id)
        DO UPDATE SET summary_text = excluded.summary_text, updated_at = excluded.updated_at
        """
        with self._connect() as conn:
            conn.execute(sql, (user_id, chat_id, summary_text, now))

    def get_summary(self, user_id: str, chat_id: str) -> str | None:
        """
        저장된 요약 텍스트를 조회한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            요약 문자열(없으면 None).
        """

        sql = """
        SELECT summary_text
        FROM chat_history_state
        WHERE user_id = ? AND chat_id = ?
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (user_id, chat_id)).fetchone()
        if row is None:
            return None
        return str(row[0])

    def delete_session(self, user_id: str, chat_id: str) -> None:
        """
        채팅 히스토리를 삭제한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
        """

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chat_history_messages WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            )
            conn.execute(
                "DELETE FROM chat_history_state WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            )

    def to_langchain_messages(self, user_id: str, chat_id: str) -> list[BaseMessage]:
        """
        저장된 히스토리를 LangChain 메시지로 변환한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            BaseMessage 리스트.
        """

        messages: list[BaseMessage] = []
        for item in self.list_messages(user_id, chat_id):
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=item.content))
        return messages

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history_messages (
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history_state (
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, chat_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_history_messages ON chat_history_messages(user_id, chat_id, message_id)"
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

    return datetime.now(timezone.utc).isoformat()
