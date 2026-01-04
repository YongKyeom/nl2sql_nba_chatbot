import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/nba.sqlite")
OUT_MD = Path("result/schema.md")
OUT_JSON = Path("result/schema.json")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    tables = [
        r["name"]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]

    schema = {}
    md_lines = [f"# SQLite Schema: {DB_PATH.name}\n"]

    for t in tables:
        cols = [dict(r) for r in cur.execute(f"PRAGMA table_info('{t}')")]
        fk = [dict(r) for r in cur.execute(f"PRAGMA foreign_key_list('{t}')")]
        count = cur.execute(f"SELECT COUNT(*) AS n FROM '{t}'").fetchone()["n"]
        schema[t] = {"row_count": count, "columns": cols, "foreign_keys": fk}

        md_lines.append(f"## {t} (rows={count})\n")
        md_lines.append("| cid | name | type | notnull | dflt_value | pk |\n")
        md_lines.append("|---:|---|---|---:|---|---:|\n")
        for c in cols:
            md_lines.append(
                f"| {c['cid']} | {c['name']} | {c['type']} | {c['notnull']} | {c['dflt_value']} | {c['pk']} |\n"
            )
        if fk:
            md_lines.append("\n**Foreign Keys**\n\n")
            md_lines.append("| from | to | table |\n|---|---|---|\n")
            for f in fk:
                md_lines.append(f"| {f['from']} | {f['to']} | {f['table']} |\n")
        md_lines.append("\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md_lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    conn.close()
    print(f"wrote: {OUT_MD}")
    print(f"wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
