from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.model.llm_operator import LLMOperator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.db.schema_store import SchemaStore
from src.prompt.column_parser import COLUMN_PARSER_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class ColumnParser:
    """
    SQL에서 테이블/컬럼을 추출하고 스키마 정보를 주입하는 도구.

    Args:
        model: 사용할 LLM 모델명.
        temperature: 생성 다양성 파라미터.
        schema_store: 스키마 스토어.
    """

    model: str
    temperature: float
    schema_store: SchemaStore

    def __post_init__(self) -> None:
        """
        LLM 클라이언트를 준비한다.

        Side Effects:
            LLM 클라이언트를 초기화한다.
        """

        object.__setattr__(self, "_client", LLMOperator())

    def tool_schema(self) -> dict[str, Any]:
        """
        LLM tool schema를 반환한다.

        Returns:
            tool schema 딕셔너리.
        """

        return {
            "type": "function",
            "function": {
                "name": "column_parser",
                "description": "SQL에서 테이블/컬럼을 추출하고 스키마 정보를 주입한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "분석할 SQL 문자열"},
                    },
                    "required": ["sql"],
                },
            },
        }

    async def parse(self, *, sql: str) -> dict[str, Any]:
        """
        SQL을 파싱하고 스키마 정보를 보강한다.

        Args:
            sql: SQL 문자열.

        Returns:
            파싱/검증 결과 딕셔너리.
        """

        try:
            parsed = await self._parse_with_llm(sql)
        except Exception:
            parsed = {}
        return self._enrich_with_schema(sql, parsed)

    async def _parse_with_llm(self, sql: str) -> dict[str, Any]:
        """
        LLM으로 테이블/컬럼 파싱을 수행한다.

        Args:
            sql: SQL 문자열.

        Returns:
            파싱 결과 딕셔너리.
        """

        prompt = COLUMN_PARSER_PROMPT.format(
            sql=sql.strip(),
            available_tables=", ".join(self.schema_store.list_tables()),
        )
        response = await self._client.invoke(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        return _parse_json(content)

    def _enrich_with_schema(self, sql: str, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        파싱 결과에 스키마 정보를 주입한다.

        Args:
            sql: 원본 SQL 문자열.
            parsed: 파싱 결과.

        Returns:
            스키마가 주입된 결과.
        """

        tables = parsed.get("tables", [])
        if not isinstance(tables, list):
            tables = []

        if not tables:
            tables = _heuristic_extract_tables(sql)

        results: list[dict[str, Any]] = []
        unknown_tables: list[str] = []
        unknown_columns: dict[str, list[str]] = {}

        for item in tables:
            table_name = str(item.get("table") or "").strip()
            if not table_name:
                continue

            table_schema = self.schema_store.get_table(table_name)
            if not table_schema:
                unknown_tables.append(table_name)
                continue

            columns = item.get("columns", [])
            if not isinstance(columns, list):
                columns = []
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []

            schema_columns = [col["name"] for col in table_schema["columns"]]
            has_wildcard = "*" in columns
            used_columns = schema_columns if has_wildcard else columns
            missing = [col for col in used_columns if col not in schema_columns]
            if missing:
                unknown_columns[table_name] = missing

            results.append(
                {
                    "table": table_name,
                    "aliases": aliases,
                    "columns": columns,
                    "wildcard": has_wildcard,
                    "schema_columns": table_schema["columns"],
                }
            )

        has_from_clause = bool(re.search(r"\\bfrom\\b", sql, re.IGNORECASE))
        is_valid = not unknown_tables and not unknown_columns and (not has_from_clause or bool(results))
        notes = str(parsed.get("notes") or "").strip()
        if has_from_clause and not results:
            notes = f"{notes} 테이블 추출에 실패했습니다.".strip()
        return {
            "sql": _compact_sql(sql),
            "tables": results,
            "unknown_tables": sorted(set(unknown_tables)),
            "unknown_columns": unknown_columns,
            "is_valid": is_valid,
            "notes": notes,
        }


def _parse_json(text: str) -> dict[str, Any]:
    """
    JSON 문자열을 파싱한다.

    Args:
        text: JSON 문자열.

    Returns:
        파싱 결과 딕셔너리.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _compact_sql(sql: str) -> str:
    """
    SQL 문자열을 한 줄로 정리한다.

    Args:
        sql: SQL 문자열.

    Returns:
        정리된 SQL 문자열.
    """

    return re.sub(r"\s+", " ", sql.strip())


def _heuristic_extract_tables(sql: str) -> list[dict[str, Any]]:
    """
    LLM 파싱이 비어 있을 때 SQL에서 테이블명을 추출한다.

    Args:
        sql: SQL 문자열.

    Returns:
        tables 항목 리스트.
    """

    cte_names = _extract_cte_names(sql)
    raw_tables: list[str] = []
    try:
        pattern = re.compile(r"\bfrom\s+([a-zA-Z_]\w*)|\bjoin\s+([a-zA-Z_]\w*)", re.IGNORECASE)
    except re.error:
        return []

    for match in pattern.finditer(sql):
        table = match.group(1) or match.group(2)
        if not table:
            continue
        if table in cte_names:
            continue
        raw_tables.append(table)

    deduped: list[str] = []
    for table in raw_tables:
        if table not in deduped:
            deduped.append(table)

    return [{"table": table, "columns": ["*"], "aliases": []} for table in deduped]


def _extract_cte_names(sql: str) -> set[str]:
    """
    WITH 절의 CTE 이름을 추출한다.

    Args:
        sql: SQL 문자열.

    Returns:
        CTE 이름 집합.
    """

    names = set()
    try:
        pattern = re.compile(r"(?:with|,)\s*([a-zA-Z_]\w*)\s+as\s*\(", re.IGNORECASE)
    except re.error:
        return set()

    for match in pattern.finditer(sql):
        names.add(match.group(1))
    return names


if __name__ == "__main__":
    async def _main() -> None:
        # 1) API 키 확인: 없으면 파서 테스트를 건너뛴다.
        config = load_config()
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY가 없어 ColumnParser 테스트를 건너뜁니다.")
            return

        # 2) 스키마 로드 후 샘플 SQL을 파싱한다.
        if not config.schema_json_path.exists():
            print(f"schema.json이 없습니다: {config.schema_json_path}")
            return

        schema_store = SchemaStore(config.schema_json_path)
        schema_store.ensure_loaded()
        parser = ColumnParser(model=config.model, temperature=0.0, schema_store=schema_store)
        sample_sql = (
            "SELECT team_abbreviation, AVG(pts_home) AS avg_pts "
            "FROM game WHERE season_id = '22022' GROUP BY team_abbreviation LIMIT 5"
        )
        print(await parser.parse(sql=sample_sql))

    asyncio.run(_main())
