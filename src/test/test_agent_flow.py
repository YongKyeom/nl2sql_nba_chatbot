"""
개발 환경에서 오케스트레이터 기반 응답 흐름을 빠르게 검증하는 러너.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlglot import parse_one


# 로컬 실행 시 src 패키지를 인식하도록 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.memory import ConversationMemory
from src.agent.scene import build_scene, ChatbotScene
from src.config import AppConfig, load_config


class _Tee:
    """
    두 개의 스트림에 동시에 출력하는 간단한 Tee.

    Args:
        streams: 출력 대상 스트림 목록.
    """

    def __init__(self, *streams: Any) -> None:
        """
        Tee 출력기를 초기화한다.

        Args:
            streams: 출력 대상 스트림 목록.
        """

        self._streams = streams

    def write(self, data: str) -> None:
        """
        데이터를 모든 스트림에 기록한다.

        Args:
            data: 출력 문자열.
        """

        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        """
        모든 스트림을 플러시한다.
        """

        for stream in self._streams:
            stream.flush()


def run_live_query(query: str, *, scene: ChatbotScene, stream_output: bool = False) -> dict[str, object]:
    """
    실제 LLM 구성으로 질의를 실행한다.

    Args:
        query: 사용자 질문.
        scene: Scene 인스턴스.
        stream_output: True면 최종 응답 스트림을 즉시 출력한다.

    Returns:
        에이전트 결과 딕셔너리.
    """

    result = scene.ask(query)
    stream = result.get("final_answer_stream")
    if stream and result.get("final_answer") is None:
        parts: list[str] = []
        if stream_output:
            for chunk in stream:
                chunk_text = str(chunk)
                parts.append(chunk_text)
                print(chunk_text, end="", flush=True)
            print()
            result["streamed_output"] = True
        else:
            for chunk in stream:
                parts.append(str(chunk))
            result["streamed_output"] = False
        result["final_answer"] = "".join(parts).strip()
    else:
        result["streamed_output"] = False
    return result


def _build_scene(config: AppConfig, *, memory: ConversationMemory | None = None) -> ChatbotScene:
    """
    Streamlit과 동일한 경로로 Scene을 생성한다.

    Args:
        config: 앱 설정.
        memory: 대화 메모리(없으면 새로 생성).

    Returns:
        ChatbotScene.
    """

    return build_scene(
        config,
        model=config.model,
        temperature=config.temperature,
        memory=memory,
    )


def _print_debug_and_sql(result: dict[str, object]) -> None:
    """
    디버그 정보와 SQL을 콘솔에 출력한다.

    Args:
        result: 에이전트 결과.

    Returns:
        None
    """

    route = result.get("route")
    route_reason = result.get("route_reason")
    planned_slots = result.get("planned_slots")
    if route or route_reason or planned_slots:
        print("\n=== Debug ===")
        if route:
            print(f"route: {route}")
        if route_reason:
            print(f"route_reason: {route_reason}")
        if planned_slots:
            print(f"planned_slots: {planned_slots}")

    sql = result.get("sql")
    if isinstance(sql, str) and sql:
        print("\n=== SQL ===")
        formatted_sql = _format_sql(sql)
        print("```sql")
        print(formatted_sql)
        print("```")


def _format_sql(sql: str) -> str:
    """
    SQL 문자열을 보기 좋게 포맷한다.

    Args:
        sql: 원본 SQL 문자열.

    Returns:
        포맷된 SQL 문자열.
    """

    try:
        return parse_one(sql, read="sqlite").sql(pretty=True)
    except Exception:
        return sql


if __name__ == "__main__":
    # 1) 로그 파일을 준비한다.
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = log_dir / f"test_agent_flow_{timestamp}.log"

    # 2) 테스트 질의 정의
    querys: list[str] = [
        "안녕? 넌 어떤 일을 할 수 있어?",
        "무슨 데이터를 알려줄 수 있어?",
        "TS% 정의 알려줘",
        "최근 리그에서 승률 상위 5개 팀이 어디인 지 알려줘",
        "상위 3개만",
        "Golden State Warriors의 시즌별 성적을 비교분석해줘",
        "2023-24 시즌 팀 득점 상위 10개 보여줘",
        "2018 드래프트 전체 픽 리스트 보여줘",
        "드래프트 1픽 평균 커리어 길이 알려줘",
        "Stephen Curry 프로필 알려줘",
        "Victor Wembanyama 컴바인 스탯 보여줘",
        "LAL과 BOS 맞대결 최근 기록 알려줘",
        "LeBron James vs Stephen Curry 플레이바이플레이 이벤트 빈도 알려줘",
    ]

    # 3) 대화 메모리/Scene 초기화(멀티턴 유지)
    config = load_config()
    scene = _build_scene(config)

    # 4) 콘솔과 로그 파일에 동시에 출력한다.
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _Tee(sys.stdout, log_file)
        tee_stderr = _Tee(sys.stderr, log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            for query in querys:
                print("\n=== 질문 ===")
                print(query)
                print("\n=== 답변 ===")
                result = run_live_query(query, scene=scene, stream_output=True)
                if not result.get("streamed_output"):
                    print(result.get("final_answer"))
                _print_debug_and_sql(result)
            print()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
