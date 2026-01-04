from __future__ import annotations

from textwrap import dedent


TITLE_PROMPT = dedent(
    """
사용자 첫 질문을 기반으로 채팅 목록에 표시할 제목을 만들어라.

규칙:
- 10~20자 이내의 짧은 제목
- 따옴표/이모지/마침표 금지
- 한 줄만 출력

[질문]
{user_question}
"""
).strip()
