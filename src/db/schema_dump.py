from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class SchemaDumpPaths:
    """
    스키마 덤프에 필요한 경로 모음.

    Args:
        db_path: SQLite DB 경로.
        json_path: schema.json 출력 경로.
        md_path: schema.md 출력 경로.
    """

    db_path: Path
    json_path: Path
    md_path: Path


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


def _fetch_tables(cursor: sqlite3.Cursor) -> list[str]:
    """
    사용자 테이블 목록을 조회.

    Args:
        cursor: SQLite 커서.

    Returns:
        테이블 이름 리스트.
    """

    rows = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _fetch_columns(cursor: sqlite3.Cursor, table_name: str) -> list[dict[str, Any]]:
    """
    테이블의 컬럼 정보를 조회.

    Args:
        cursor: SQLite 커서.
        table_name: 테이블 이름.

    Returns:
        컬럼 메타데이터 리스트.
    """

    rows = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "dflt_value": row[4],
            "pk": row[5],
        }
        for row in rows
    ]


def _fetch_foreign_keys(cursor: sqlite3.Cursor, table_name: str) -> list[dict[str, Any]]:
    """
    테이블의 외래키 정보를 조회.

    Args:
        cursor: SQLite 커서.
        table_name: 테이블 이름.

    Returns:
        외래키 메타데이터 리스트.
    """

    rows = cursor.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
    return [
        {
            "id": row[0],
            "seq": row[1],
            "table": row[2],
            "from": row[3],
            "to": row[4],
            "on_update": row[5],
            "on_delete": row[6],
            "match": row[7],
        }
        for row in rows
    ]


def _fetch_row_count(cursor: sqlite3.Cursor, table_name: str) -> int:
    """
    테이블의 전체 행 수를 조회.

    Args:
        cursor: SQLite 커서.
        table_name: 테이블 이름.

    Returns:
        행 수.
    """

    row = cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()
    return int(row[0])


def build_schema_dict(db_path: Path) -> dict[str, Any]:
    """
    SQLite DB에서 스키마 정보를 추출.

    Args:
        db_path: SQLite DB 경로.

    Returns:
        테이블별 스키마 정보 딕셔너리.

    Raises:
        FileNotFoundError: DB 파일이 없을 때.
        sqlite3.Error: DB 읽기 중 오류가 발생했을 때.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {db_path}")

    # 1) 읽기 전용 연결 생성
    uri = _build_readonly_uri(db_path)
    conn = sqlite3.connect(uri, uri=True)

    try:
        cursor = conn.cursor()

        # 2) 테이블 목록 로딩
        tables = _fetch_tables(cursor)

        # 3) 테이블별 컬럼/외래키/행 수 집계
        schema: dict[str, Any] = {}
        for table in tables:
            schema[table] = {
                "row_count": _fetch_row_count(cursor, table),
                "columns": _fetch_columns(cursor, table),
                "foreign_keys": _fetch_foreign_keys(cursor, table),
            }
        return schema
    finally:
        conn.close()


def render_schema_markdown(schema: dict[str, Any], db_name: str) -> str:
    """
    스키마 딕셔너리를 Markdown 문서로 변환.

    Args:
        schema: 테이블별 스키마 정보.
        db_name: DB 파일명(표시용).

    Returns:
        Markdown 문자열.
    """

    lines: list[str] = [f"# SQLite Schema: {db_name}\n"]

    for table_name, table_info in schema.items():
        # 테이블 헤더
        lines.append(f"## {table_name} (rows={table_info['row_count']})\n")
        lines.append("| cid | name | type | notnull | dflt_value | pk |\n")
        lines.append("|---:|---|---|---:|---|---:|\n")

        # 컬럼 목록
        for column in table_info["columns"]:
            lines.append(
                "| {cid} | {name} | {type} | {notnull} | {dflt_value} | {pk} |\n".format(
                    cid=column["cid"],
                    name=column["name"],
                    type=column["type"],
                    notnull=column["notnull"],
                    dflt_value=column["dflt_value"],
                    pk=column["pk"],
                )
            )

        # 외래키 표
        foreign_keys = table_info.get("foreign_keys", [])
        if foreign_keys:
            lines.append("\n**Foreign Keys**\n\n")
            lines.append("| from | to | table |\n|---|---|---|\n")
            for fk in foreign_keys:
                lines.append(f"| {fk['from']} | {fk['to']} | {fk['table']} |\n")
        lines.append("\n")

    return "".join(lines)


def write_schema_files(schema: dict[str, Any], paths: SchemaDumpPaths) -> None:
    """
    schema.json과 schema.md를 저장.

    Args:
        schema: 테이블별 스키마 정보.
        paths: 스키마 덤프 경로 모음.

    Returns:
        None
    """

    # 1) 결과 디렉터리 준비
    paths.json_path.parent.mkdir(parents=True, exist_ok=True)
    paths.md_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) JSON/Markdown 저장
    paths.json_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.md_path.write_text(render_schema_markdown(schema, paths.db_path.name), encoding="utf-8")


def dump_schema(paths: SchemaDumpPaths) -> dict[str, Any]:
    """
    스키마를 덤프하고 파일로 저장.

    Args:
        paths: 스키마 덤프 경로 모음.

    Returns:
        테이블별 스키마 정보.
    """

    # 1) 스키마 딕셔너리 생성
    schema = build_schema_dict(paths.db_path)

    # 2) 파일 저장
    write_schema_files(schema, paths)
    return schema


def main() -> None:
    """
    CLI 진입점.

    Returns:
        None
    """

    # 1) 기본 경로 구성
    paths = SchemaDumpPaths(
        db_path=Path("data/nba.sqlite"),
        json_path=Path("result/schema.json"),
        md_path=Path("result/schema.md"),
    )

    # 2) 스키마 덤프 실행
    dump_schema(paths)

    # 3) 완료 메시지 출력
    print(f"wrote: {paths.json_path}")
    print(f"wrote: {paths.md_path}")


if __name__ == "__main__":
    main()
