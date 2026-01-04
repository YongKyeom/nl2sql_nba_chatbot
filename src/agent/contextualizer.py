from __future__ import annotations

from src.agent.memory import ConversationMemory


def build_context_hint(memory: ConversationMemory) -> str:
    """
    이전 대화 맥락을 요약 문자열로 반환한다.

    Args:
        memory: 대화 메모리.

    Returns:
        요약 문자열(없으면 "없음").
    """

    parts: list[str] = []

    if memory.last_slots:
        parts.append(f"last_slots={memory.last_slots}")

    if memory.last_result_schema:
        parts.append(f"last_result_schema={memory.last_result_schema}")

    if memory.last_sql:
        parts.append(f"last_sql={memory.last_sql}")

    recent_dialogue = memory.short_term.build_recent_dialogue(limit=3)
    if recent_dialogue != "없음":
        parts.append(f"recent_dialogue:\n{recent_dialogue}")

    long_term_hint = memory.long_term.build_hint()
    if long_term_hint != "없음":
        parts.append(f"long_term:\n{long_term_hint}")

    return "\n".join(parts) if parts else "없음"


if __name__ == "__main__":
    # 1) 샘플 메모리 준비
    memory = ConversationMemory()
    memory.last_sql = "SELECT team_name FROM team LIMIT 5"
    memory.last_slots = {"metric": "win_pct", "season": "2022-23"}
    memory.last_result_schema = ["team_name", "pct"]

    # 2) 맥락 힌트 출력
    print(build_context_hint(memory))
