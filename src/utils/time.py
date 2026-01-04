from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    """
    처리 시간 측정용 타이머.

    with 문으로 감싸면 시작/종료 시간을 자동으로 기록한다.
    """

    start: float | None = None
    end: float | None = None

    def __enter__(self) -> "Timer":
        """
        타이머 시작.

        Returns:
            Timer 인스턴스.
        """

        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
        """
        타이머 종료.

        Args:
            exc_type: 예외 타입.
            exc: 예외 객체.
            traceback: 트레이스백.

        Returns:
            None
        """

        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float | None:
        """
        경과 시간을 밀리초로 반환.

        Returns:
            경과 시간(ms) 또는 None.
        """

        if self.start is None:
            return None
        if self.end is None:
            return (time.perf_counter() - self.start) * 1000
        return (self.end - self.start) * 1000
