from __future__ import annotations

from typing import Any


def records_to_markdown(records: list[dict[str, Any]]) -> str:
    """
    레코드 리스트를 마크다운 테이블로 변환한다.

    Args:
        records: 레코드 리스트.

    Returns:
        마크다운 테이블 문자열.
    """

    if not records:
        return "| 결과 없음 |\n| --- |"

    headers = list(records[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    data_rows: list[str] = []
    for record in records:
        row = [str(record.get(header, "")) for header in headers]
        data_rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, separator_row, *data_rows])
