"""
SQL 컬럼 파서 프롬프트 정의.

변수:
    sql: 분석할 SQL 문자열.
    available_tables: 스키마에 존재하는 테이블 목록.
"""

from textwrap import dedent


COLUMN_PARSER_PROMPT = dedent(
    """
너는 SQL에서 실제 사용된 테이블/컬럼을 추출하는 파서다.

요구사항:
- schema에 존재하는 실제 테이블만 테이블 목록에 포함한다.
- CTE 이름(step1 등)과 서브쿼리 별칭은 테이블로 포함하지 않는다.
- SELECT/WHERE/JOIN/GROUP BY/ORDER BY/HAVING에 등장하는 컬럼을 모두 수집한다.
- SELECT * 는 columns에 "*" 로 기록한다.

출력(JSON):
{
  "tables": [
    {
      "table": "table_name",
      "columns": ["col_a", "col_b", "*"],
      "aliases": ["t1", "g"]
    }
  ],
  "notes": "추가 설명(없으면 빈 문자열)"
}

활용 가이드:
- 테이블/컬럼이 불명확하면 가능한 한 비워 두고 notes에 이유를 적는다.
- schema에 없는 테이블/컬럼은 포함하지 않는다.

주의:
- 출력은 반드시 JSON만.
- 모르면 비워 둔다.

[사용 가능한 테이블]
{available_tables}

[SQL]
{sql}
"""
).strip()
