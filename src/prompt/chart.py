from __future__ import annotations

from textwrap import dedent


CHART_PROMPT = dedent(
    """
사용자의 요청과 데이터프레임 정보를 보고 적절한 차트 스펙을 JSON으로 제시하라.

차트 유형 가이드:
{chart_type_guide}

규칙:
- chart_type은 다음 중 하나여야 한다: {chart_type_list}
- x, y는 반드시 columns 목록에 있는 컬럼명만 사용한다.
- series는 필요할 때만 사용하며, 사용 시 columns 목록에 있어야 한다.
- 시간/시즌/순서가 있으면 line 또는 area를 우선한다.
- 상위 비교/랭킹이면 bar 또는 stacked_bar를 우선한다.
- 관계/분포가 필요하면 scatter를 우선한다.
- 단일 수치 분포면 histogram, 범주별 분포 요약이면 box를 고려한다.
- 적절한 차트를 찾을 수 없으면 {{"chart_type":"none"}}만 반환한다.
- 출력은 JSON만 허용한다(코드블록/설명 금지).

[사용자 질문]
{user_question}

[컬럼 목록]
{columns}

[수치 컬럼 목록]
{numeric_columns}

[샘플 레코드]
{sample_records}
"""
).strip()
