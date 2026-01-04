from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """
    앱 실행에 필요한 설정.

    Args:
        db_path: SQLite DB 경로.
        model: 사용할 LLM 모델명.
        temperature: LLM temperature.
        schema_json_path: 스키마 JSON 경로.
        schema_md_path: 스키마 Markdown 경로.
        log_path: 로그 파일 경로.
        memory_db_path: 장기 메모리 DB 경로.
    """

    db_path: Path
    model: str
    temperature: float
    schema_json_path: Path
    schema_md_path: Path
    log_path: Path
    memory_db_path: Path


def load_config() -> AppConfig:
    """
    .env를 로드하고 설정을 생성.

    Returns:
        AppConfig 객체.
    """

    load_dotenv()

    db_path = Path(os.getenv("DB_PATH", "data/nba.sqlite"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    memory_db_path = Path(os.getenv("MEMORY_DB_PATH", "result/memory.sqlite"))

    return AppConfig(
        db_path=db_path,
        model=model,
        temperature=temperature,
        schema_json_path=Path("result/schema.json"),
        schema_md_path=Path("result/schema.md"),
        log_path=Path("log") / f"log_{datetime.now().strftime('%Y%m%d')}.json",
        memory_db_path=memory_db_path,
    )
