from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class ColumnInfo(TypedDict):
    """
    컬럼 메타데이터 구조.
    """

    cid: int
    name: str
    type: str
    notnull: int
    dflt_value: Any
    pk: int


# "from" 키가 예약어라 함수형 TypedDict로 정의한다.
ForeignKeyInfo = TypedDict(
    "ForeignKeyInfo",
    {
        "id": int,
        "seq": int,
        "table": str,
        "from": str,
        "to": str,
        "on_update": str,
        "on_delete": str,
        "match": str,
    },
)


class TableSchema(TypedDict):
    """
    테이블 스키마 구조.
    """

    row_count: int
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKeyInfo]


@dataclass(frozen=True)
class SearchHit:
    """
    스키마 검색 결과.

    Args:
        table: 테이블 이름.
        column: 컬럼 이름(컬럼 매칭이 없으면 None).
        score: 매칭 점수(정확도/우선순위 판단용).
        reason: 매칭 근거 설명.
    """

    table: str
    column: str | None
    score: int
    reason: str


class SchemaStore:
    """
    schema.json을 로드하고 검색/요약 기능을 제공.

    스키마가 변경될 수 있으므로, SQL 생성 단계는 반드시 이 스토어를 통해
    테이블/컬럼을 확인하도록 설계한다.
    """

    def __init__(self, schema_path: Path) -> None:
        """
        SchemaStore 초기화.

        Args:
            schema_path: schema.json 경로.
        """

        self._schema_path = schema_path
        self._schema: dict[str, TableSchema] = {}

    def load(self) -> None:
        """
        schema.json을 로드.

        Raises:
            FileNotFoundError: schema.json이 없을 때.
            json.JSONDecodeError: JSON 파싱 실패 시.
        """

        if not self._schema_path.exists():
            raise FileNotFoundError(f"schema.json을 찾을 수 없습니다: {self._schema_path}")

        raw = json.loads(self._schema_path.read_text(encoding="utf-8"))
        self._schema = raw

    def ensure_loaded(self) -> None:
        """
        스키마 로드 여부를 확인하고 필요 시 로드.

        Returns:
            None
        """

        if not self._schema:
            self.load()

    def list_tables(self) -> list[str]:
        """
        테이블 목록을 반환.

        Returns:
            테이블 이름 리스트.
        """

        self.ensure_loaded()
        return sorted(self._schema.keys())

    def get_table(self, table_name: str) -> TableSchema | None:
        """
        특정 테이블의 스키마를 반환.

        Args:
            table_name: 테이블 이름.

        Returns:
            테이블 스키마(없으면 None).
        """

        self.ensure_loaded()
        return self._schema.get(table_name)

    def search(self, keyword: str, limit: int = 10) -> list[SearchHit]:
        """
        테이블/컬럼 이름으로 스키마를 검색.

        Args:
            keyword: 검색어.
            limit: 최대 반환 개수.

        Returns:
            검색 결과 리스트.
        """

        self.ensure_loaded()
        query = keyword.strip().lower()
        if not query:
            return []

        hits: list[SearchHit] = []

        for table_name, table_info in self._schema.items():
            table_score = _match_score(table_name.lower(), query)
            if table_score > 0:
                hits.append(SearchHit(table=table_name, column=None, score=table_score, reason="테이블명 매칭"))

            for column in table_info["columns"]:
                column_name = column["name"].lower()
                column_score = _match_score(column_name, query)
                if column_score > 0:
                    hits.append(
                        SearchHit(
                            table=table_name,
                            column=column["name"],
                            score=column_score,
                            reason="컬럼명 매칭",
                        )
                    )

        hits.sort(key=lambda hit: (-hit.score, hit.table, hit.column or ""))
        return hits[:limit]

    def build_context(self, max_tables: int = 8, max_columns: int = 12) -> str:
        """
        LLM 입력에 넣기 좋은 스키마 요약 문자열을 생성.

        Args:
            max_tables: 포함할 최대 테이블 수.
            max_columns: 테이블당 포함할 최대 컬럼 수.

        Returns:
            스키마 요약 문자열.
        """

        self.ensure_loaded()
        lines: list[str] = []

        for table_name in self.list_tables()[:max_tables]:
            table_info = self._schema[table_name]
            columns = [col["name"] for col in table_info["columns"]][:max_columns]
            lines.append(f"- {table_name}: {', '.join(columns)}")

        return "\n".join(lines)

    def build_context_for_tables(
        self,
        table_names: list[str],
        columns_by_table: dict[str, list[str]] | None = None,
        *,
        max_columns: int = 12,
    ) -> str:
        """
        특정 테이블만 포함한 스키마 요약 문자열을 생성.

        Args:
            table_names: 포함할 테이블 목록.
            columns_by_table: 테이블별로 제한할 컬럼 목록(없으면 전체).
            max_columns: 테이블당 포함할 최대 컬럼 수.

        Returns:
            스키마 요약 문자열.
        """

        self.ensure_loaded()
        lines: list[str] = []

        for table_name in table_names:
            table_info = self._schema.get(table_name)
            if not table_info:
                continue

            available = [col["name"] for col in table_info["columns"]]
            if columns_by_table and columns_by_table.get(table_name):
                filtered = [col for col in available if col in columns_by_table[table_name]]
                columns = filtered[:max_columns] if filtered else available[:max_columns]
            else:
                columns = available[:max_columns]

            lines.append(f"- {table_name}: {', '.join(columns)}")

        return "\n".join(lines)

    def build_full_context(self, *, max_columns: int = 12) -> str:
        """
        모든 테이블을 포함한 스키마 요약 문자열을 생성.

        Args:
            max_columns: 테이블당 포함할 최대 컬럼 수.

        Returns:
            스키마 요약 문자열.
        """

        self.ensure_loaded()
        return self.build_context_for_tables(self.list_tables(), max_columns=max_columns)


def _match_score(target: str, query: str) -> int:
    """
    문자열 매칭 점수를 계산.

    Args:
        target: 비교 대상 문자열.
        query: 검색 문자열.

    Returns:
        매칭 점수(0이면 불일치).
    """

    if target == query:
        return 3
    if target.startswith(query):
        return 2
    if query in target:
        return 1
    return 0
