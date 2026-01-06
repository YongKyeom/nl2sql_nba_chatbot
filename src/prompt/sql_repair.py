"""
SQL 수정을 위한 프롬프트 정의.

변수:
    user_question: 사용자 질문 원문.
    failed_sql: 실패한 SQL.
    failure_reason: 실패 원인(가드레일/밸리데이터 결과).
    metric_context: 레지스트리 기반 메트릭 컨텍스트.
    schema_context: schema_store에서 생성한 스키마 요약.
"""

from textwrap import dedent


SQL_REPAIR_PROMPT = dedent(
    """
다음 실패 정보를 참고해 SQL을 한 번만 수정하라.

목표:
- 실패 원인을 제거하고 실행 가능한 SQL을 반환한다.

규칙:
- 테이블/컬럼은 schema.json에 있는 것만 사용.
- SELECT 문만 허용.
- 항상 LIMIT 포함.
- 출력은 SQL만, 코드블록/설명 금지.
- column_parser 도구가 제공되면 수정 SQL을 검증해 오류가 없도록 보정한다.

출력:
- SQL 단일 문자열만 반환한다.

[사용자 질문]
{user_question}

[실패 SQL]
{failed_sql}

[실패 원인]
{failure_reason}

[메트릭 컨텍스트]
{metric_context}

[스키마 요약]
{schema_context}
"""
).strip()
