from __future__ import annotations

from typing import Literal, TypedDict


ChartType = Literal["line", "area", "bar", "stacked_bar", "scatter", "histogram", "box"]


class ChartSpec(TypedDict, total=False):
    """
    차트 생성에 필요한 스펙.

    Args:
        chart_type: 차트 유형.
        x: X축 컬럼명.
        y: Y축 컬럼명.
        series: 시리즈(범주) 컬럼명.

    Side Effects:
        None

    Raises:
        예외 없음.
    """

    chart_type: ChartType
    x: str
    y: str
    series: str


SUPPORTED_CHART_TYPES: list[str] = [
    "line",
    "area",
    "bar",
    "stacked_bar",
    "scatter",
    "histogram",
    "box",
]

CHART_TYPE_GUIDE: dict[str, str] = {
    "line": "시간/시즌/순서 기반 추세에 적합",
    "area": "누적 흐름 또는 변화량 강조에 적합",
    "bar": "카테고리 간 비교/랭킹에 적합",
    "stacked_bar": "카테고리 합계와 구성비를 동시에 보여줄 때 적합",
    "scatter": "두 수치 지표의 관계/분포에 적합",
    "histogram": "단일 수치 컬럼의 분포 확인에 적합",
    "box": "분포와 이상치를 요약할 때 적합",
}


def format_chart_type_guide() -> str:
    """
    차트 유형 가이드를 문자열로 구성한다.

    Returns:
        줄바꿈으로 연결된 가이드 문자열.

    Side Effects:
        None

    Raises:
        예외 없음.
    """

    lines = [f"- {chart_type}: {description}" for chart_type, description in CHART_TYPE_GUIDE.items()]
    return "\n".join(lines)
