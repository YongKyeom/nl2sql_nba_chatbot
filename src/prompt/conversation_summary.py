"""
대화 요약 프롬프트 정의.

변수:
    summary_source: 요약 대상 대화 문자열.
"""

from textwrap import dedent


CONVERSATION_SUMMARY_PROMPT = dedent(
    """
너는 대화 요약 담당자다. 아래 대화 기록을 간결하게 요약하라.

요약 기준:
- 사용자 질문과 최종 답변만 반영한다.
- SQL/툴/디버그 정보는 포함하지 않는다.
- 핵심 의도, 중요한 조건(시즌/팀/지표/범위), 결론, 미해결 질문을 포함한다.
- 3~6줄의 한국어 bullet로 출력한다.

출력 형식(필수):
- ...
- ...

[대화 기록]
{summary_source}
"""
).strip()
