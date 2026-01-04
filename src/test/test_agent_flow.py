"""
개발 환경에서 오케스트레이터 기반 응답 흐름을 빠르게 검증하는 러너.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

from sqlglot import parse_one


# 로컬 실행 시 src 패키지를 인식하도록 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.memory import ConversationMemory
from src.agent.scene import build_scene, ChatbotScene
from src.config import AppConfig, load_config


def run_live_query(query: str, *, scene: ChatbotScene) -> dict[str, object]:
    """
    실제 LLM 구성으로 질의를 실행한다.

    Args:
        query: 사용자 질문.
        scene: Scene 인스턴스.

    Returns:
        에이전트 결과 딕셔너리.
    """

    return scene.ask(query)


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


def _print_result(query: str, result: dict[str, object]) -> None:
    """
    결과를 콘솔에 출력한다.

    Args:
        query: 사용자 질문.
        result: 에이전트 결과.

    Returns:
        None
    """

    print("\n=== 질문 ===")
    print(query)
    print("\n=== 답변 ===")
    print(result.get("final_answer"))

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
    # 1) 테스트 질의 정의
    querys: list[str] = [
        "안녕? 넌 어떤 일을 할 수 있어?",
        "무슨 데이터를 알려줄 수 있어?",
        "TS% 정의 알려줘",
        "최근 리그에서 승률 상위 5개 팀이 어디인 지 알려줘",
        "상위 3개만",
        "2023-24 시즌 팀 득점 상위 10개 보여줘",
        "2018 드래프트 전체 픽 리스트 보여줘",
        "드래프트 1픽 평균 커리어 길이 알려줘",
        "Stephen Curry 프로필 알려줘",
        "Victor Wembanyama 컴바인 스탯 보여줘",
        "LAL과 BOS 맞대결 최근 기록 알려줘",
        "LeBron James vs Stephen Curry 플레이바이플레이 이벤트 빈도 알려줘",
    ]

    # 2) 대화 메모리/Scene 초기화(멀티턴 유지)
    config = load_config()
    scene = _build_scene(config)

    # 3) 반복 실행
    for query in querys:
        result = run_live_query(query, scene=scene)
        _print_result(query, result)
    print()
