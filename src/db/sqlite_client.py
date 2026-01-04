from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import sqlglot
from sqlglot import expressions as exp


@dataclass(frozen=True)
class QueryResult:
    """
    SQL 실행 결과 요약.

    Args:
        dataframe: 결과 데이터프레임.
        row_count: 결과 행 수.
        columns: 컬럼 목록.
        executed_sql: 실행된 SQL(보정 후).
    """

    dataframe: pd.DataFrame
    row_count: int
    columns: list[str]
    executed_sql: str


class SQLiteClient:
    """
    SQLite 읽기 전용 실행 클라이언트.

    Guard에서 LIMIT을 강제하지만, 안전을 위해 클라이언트에서도
    LIMIT을 보정하여 과도한 행 반환을 방지한다.
    """

    def __init__(self, db_path: Path, *, timeout_sec: float = 5.0, row_limit: int = 200) -> None:
        """
        SQLiteClient 초기화.

        Args:
            db_path: SQLite DB 경로.
            timeout_sec: 연결/잠금 대기 시간(초).
            row_limit: 최대 행 반환 수.
        """

        self._db_path = db_path
        self._timeout_sec = timeout_sec
        self._row_limit = row_limit

    def execute(self, sql: str) -> QueryResult:
        """
        SQL을 실행하고 결과를 반환.

        Args:
            sql: 실행할 SQL 문자열.

        Returns:
            QueryResult 객체.

        Raises:
            FileNotFoundError: DB 파일이 없을 때.
            sqlite3.Error: SQL 실행 중 오류가 발생했을 때.
        """

        if not self._db_path.exists():
            raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {self._db_path}")

        # 1) LIMIT 보정
        safe_sql = enforce_limit(sql, self._row_limit)

        # 2) 읽기 전용 연결 생성
        conn = sqlite3.connect(_build_readonly_uri(self._db_path), uri=True, timeout=self._timeout_sec)

        try:
            # 3) 읽기 전용 모드 강제
            conn.execute("PRAGMA query_only = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(self._timeout_sec * 1000)}")

            # 4) 쿼리 실행
            dataframe = pd.read_sql_query(safe_sql, conn)
            columns = list(dataframe.columns)
            return QueryResult(
                dataframe=dataframe,
                row_count=int(len(dataframe)),
                columns=columns,
                executed_sql=safe_sql,
            )
        finally:
            conn.close()


def _build_readonly_uri(db_path: Path) -> str:
    """
    SQLite 읽기 전용 URI를 생성.

    Args:
        db_path: SQLite DB 경로.

    Returns:
        읽기 전용 SQLite URI 문자열.
    """

    resolved = db_path.resolve()
    return f"file:{quote(str(resolved))}?mode=ro"


def enforce_limit(sql: str, row_limit: int) -> str:
    """
    SQL에 LIMIT을 강제한다.

    Args:
        sql: 원본 SQL 문자열.
        row_limit: 최대 행 제한.

    Returns:
        LIMIT이 포함된 SQL 문자열.

    Notes:
        sqlglot 파싱 실패 시 단순 문자열 검사로 보정한다.
    """

    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
        if expression is None:
            return sql
        if expression.args.get("limit") is None:
            expression.set("limit", exp.Limit(this=exp.Literal.number(row_limit)))
        return expression.sql(dialect="sqlite")
    except sqlglot.ParseError:
        normalized = sql.strip().rstrip(";")
        if " limit " in normalized.lower():
            return normalized
        return f"{normalized} LIMIT {row_limit}"


def preview_dataframe(dataframe: pd.DataFrame, max_rows: int = 200) -> list[dict[str, Any]]:
    """
    데이터프레임을 미리보기 가능한 레코드로 변환.

    Args:
        dataframe: 원본 데이터프레임.
        max_rows: 최대 행 수.

    Returns:
        레코드 리스트.
    """

    return dataframe.head(max_rows).to_dict(orient="records")
