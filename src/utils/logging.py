from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LogEvent:
    """
    로그 이벤트 구조.

    Args:
        timestamp: ISO8601 타임스탬프.
        user_text: 사용자 입력.
        route: 라우팅 결과.
        sql: 실행 SQL(없으면 None).
        rows: 결과 행 수(없으면 None).
        latency_ms: 처리 지연 시간(ms).
        error: 오류 메시지(없으면 None).
        user_id: 사용자 식별자(없으면 None).
    """

    timestamp: str
    user_text: str
    route: str
    sql: str | None
    rows: int | None
    latency_ms: float | None
    error: str | None
    user_id: str | None = None


class JsonlLogger:
    """
    JSONL 로그 기록 유틸.

    민감 정보는 입력으로 받지 않으며, 파일은 append 방식으로 기록한다.
    """

    def __init__(self, log_path: Path) -> None:
        """
        JsonlLogger 초기화.

        Args:
            log_path: 로그 파일 경로.
        """

        self._log_path = log_path

    def log_event(
        self,
        *,
        user_text: str,
        route: str,
        sql: str | None,
        rows: int | None,
        latency_ms: float | None,
        error: str | None,
        user_id: str | None = None,
    ) -> None:
        """
        이벤트를 JSONL로 기록.

        Args:
            user_text: 사용자 입력.
            route: 라우팅 결과.
            sql: 실행 SQL.
            rows: 결과 행 수.
            latency_ms: 처리 지연 시간.
            error: 오류 메시지.

        Returns:
            None
        """

        event = LogEvent(
            timestamp=datetime.utcnow().isoformat(),
            user_text=user_text,
            route=route,
            sql=sql,
            rows=rows,
            latency_ms=latency_ms,
            error=error,
            user_id=user_id,
        )
        payload = dict(event.__dict__)
        if payload.get("user_id") is None:
            payload.pop("user_id", None)
        self._append_json(payload)

    def _append_json(self, payload: dict[str, object]) -> None:
        """
        JSON 객체를 한 줄로 기록.

        Args:
            payload: 기록할 JSON 객체.

        Returns:
            None
        """

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
