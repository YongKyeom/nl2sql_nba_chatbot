from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import expressions as exp

from src.db.schema_store import SchemaStore


@dataclass(frozen=True)
class GuardResult:
    """
    SQL 가드 결과.

    Args:
        is_valid: 검증 통과 여부.
        sql: 보정된 SQL.
        errors: 오류 메시지 목록.
        warnings: 경고 메시지 목록.
    """

    is_valid: bool
    sql: str
    errors: list[str]
    warnings: list[str]


class SQLGuard:
    """
    SQL 정적 검증 및 보정.

    SELECT-only, 테이블 화이트리스트, LIMIT 강제, ON 없는 JOIN 차단을 수행한다.
    """

    def __init__(self, schema_path: Path, *, row_limit: int = 300) -> None:
        """
        SQLGuard 초기화.

        Args:
            schema_path: schema.json 경로.
            row_limit: 최대 행 제한.
        """

        self._schema_store = SchemaStore(schema_path)
        self._row_limit = row_limit

    def validate(self, sql: str) -> GuardResult:
        """
        SQL을 검증하고 필요 시 LIMIT을 보정.

        Args:
            sql: 원본 SQL.

        Returns:
            GuardResult.
        """

        errors: list[str] = []
        warnings: list[str] = []

        try:
            statements = sqlglot.parse(sql, read="sqlite")
        except sqlglot.errors.ParseError as exc:
            return GuardResult(False, sql, [f"SQL 파싱 실패: {exc}"], warnings)

        if len(statements) != 1:
            return GuardResult(False, sql, ["단일 SQL 문만 허용됩니다."], warnings)

        expression = statements[0]
        if not _contains_select(expression):
            errors.append("SELECT 문이 포함되어야 합니다.")

        forbidden = _find_forbidden(expression)
        if forbidden:
            errors.append(f"금지된 구문 포함: {', '.join(forbidden)}")

        if errors:
            return GuardResult(False, sql, errors, warnings)

        table_errors = _validate_tables(expression, self._schema_store)
        if table_errors:
            return GuardResult(False, sql, table_errors, warnings)

        join_errors = _validate_joins(expression)
        if join_errors:
            return GuardResult(False, sql, join_errors, warnings)

        safe_sql = _ensure_limit(expression, self._row_limit, sql)
        return GuardResult(True, safe_sql, errors, warnings)


def _contains_select(expression: exp.Expression) -> bool:
    """
    SELECT 포함 여부를 확인.

    Args:
        expression: SQL AST.

    Returns:
        SELECT 포함 여부.
    """

    return any(isinstance(node, exp.Select) for node in expression.walk())


def _find_forbidden(expression: exp.Expression) -> list[str]:
    """
    금지된 구문을 탐지.

    Args:
        expression: SQL AST.

    Returns:
        금지 구문 목록.
    """

    forbidden_nodes = []
    for name in ["Insert", "Update", "Delete", "Drop", "Alter", "Pragma", "Attach", "Command"]:
        node = getattr(exp, name, None)
        if node is not None:
            forbidden_nodes.append(node)
    forbidden_nodes = tuple(forbidden_nodes)
    found: list[str] = []

    for node in expression.walk():
        if isinstance(node, forbidden_nodes):
            found.append(node.key)

    return sorted(set(found))


def _validate_tables(expression: exp.Expression, schema_store: SchemaStore) -> list[str]:
    """
    테이블 화이트리스트를 검증.

    Args:
        expression: SQL AST.
        schema_store: 스키마 스토어.

    Returns:
        오류 메시지 목록.
    """

    schema_store.ensure_loaded()
    allowed_tables = set(schema_store.list_tables())
    cte_tables = _extract_cte_names(expression)
    used_tables = {table.name for table in expression.find_all(exp.Table)} - cte_tables

    invalid_tables = sorted(table for table in used_tables if table not in allowed_tables)
    if invalid_tables:
        return [f"허용되지 않은 테이블 사용: {', '.join(invalid_tables)}"]
    return []


def _extract_cte_names(expression: exp.Expression) -> set[str]:
    """
    WITH 구문에서 정의된 CTE 이름을 추출.

    Args:
        expression: SQL AST.

    Returns:
        CTE 이름 집합.
    """

    names: set[str] = set()
    for cte in expression.find_all(exp.CTE):
        if cte.alias_or_name:
            names.add(cte.alias_or_name)
    return names


def _validate_joins(expression: exp.Expression) -> list[str]:
    """
    위험한 조인을 검증.

    Args:
        expression: SQL AST.

    Returns:
        오류 메시지 목록.
    """

    errors: list[str] = []
    for join in expression.find_all(exp.Join):
        if join.args.get("on") is None and join.args.get("using") is None and join.kind is None:
            errors.append("ON 조건 없는 JOIN은 카티션 가능성이 있어 금지됩니다.")
    return errors


def _ensure_limit(expression: exp.Expression, row_limit: int, raw_sql: str) -> str:
    """
    LIMIT을 강제한다.

    Args:
        expression: SQL AST.
        row_limit: 최대 행 제한.
        raw_sql: 원본 SQL 문자열.

    Returns:
        LIMIT이 반영된 SQL 문자열.
    """

    target = expression
    if isinstance(expression, exp.With) and expression.this is not None:
        target = expression.this

    normalized = _normalize_sql(raw_sql)
    if target.args.get("limit") is not None:
        return normalized
    if _has_limit_keyword(normalized):
        return normalized
    return f"{normalized} LIMIT {row_limit}"


def _normalize_sql(sql: str) -> str:
    """
    SQL 문자열의 기본 정규화를 수행한다.

    Args:
        sql: 원본 SQL 문자열.

    Returns:
        정규화된 SQL 문자열.
    """

    return sql.strip().rstrip(";")


def _has_limit_keyword(sql: str) -> bool:
    """
    LIMIT 키워드 포함 여부를 확인한다.

    Args:
        sql: SQL 문자열.

    Returns:
        LIMIT 키워드가 있으면 True.
    """

    return bool(re.search(r"\\blimit\\b", sql, re.IGNORECASE))


if __name__ == "__main__":
    # 1) 스키마 경로 확인
    schema_path = Path("result/schema.json")
    if not schema_path.exists():
        print("schema.json이 없어 SQLGuard 테스트를 건너뜁니다.")
    else:
        # 2) SQLGuard 실행
        guard = SQLGuard(schema_path)
        sql = "SELECT team_name, team_abbreviation FROM team LIMIT 5"
        result = guard.validate(sql)
        print(result.is_valid)
        print(result.errors)
        print(result.sql)
