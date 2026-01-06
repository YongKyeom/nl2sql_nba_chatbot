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
        model: 사용할 LLM 모델명(라우팅/SQL 생성용).
        final_answer_model: 최종 답변 생성에 사용할 모델명.
        temperature: LLM temperature.
        schema_json_path: 스키마 JSON 경로.
        schema_md_path: 스키마 Markdown 경로.
        log_path: 로그 파일 경로.
        memory_db_path: 장기 메모리 DB 경로.
        chat_db_path: 채팅 로그 DB 경로.
        history_db_path: 대화 히스토리 DB 경로.
        fewshot_candidate_limit: few-shot 후보 sql_template 최대 개수.
        conversation_summary_char_limit: 대화 요약 트리거 문자 수.
        conversation_summary_keep_turns: 요약 시 유지할 최근 턴 수.
    """

    db_path: Path
    model: str
    final_answer_model: str
    temperature: float
    schema_json_path: Path
    schema_md_path: Path
    log_path: Path
    memory_db_path: Path
    chat_db_path: Path
    history_db_path: Path
    fewshot_candidate_limit: int
    conversation_summary_char_limit: int
    conversation_summary_keep_turns: int


def load_config() -> AppConfig:
    """
    .env를 로드하고 설정을 생성.

    Returns:
        AppConfig 객체.
    """

    load_dotenv()

    db_path = Path(os.getenv("DB_PATH", "data/nba.sqlite"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    final_answer_model = os.getenv("FINAL_ANSWER_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    memory_db_path = Path(os.getenv("MEMORY_DB_PATH", "result/memory.sqlite"))
    chat_db_path = Path(os.getenv("CHAT_DB_PATH", "result/chat.sqlite"))
    history_db_path = Path(os.getenv("HISTORY_DB_PATH", "result/chat_history.sqlite"))
    fewshot_candidate_limit = int(os.getenv("FEWSHOT_CANDIDATE_LIMIT", "3"))
    conversation_summary_char_limit = int(os.getenv("CONVERSATION_SUMMARY_CHAR_LIMIT", "10000"))
    conversation_summary_keep_turns = int(os.getenv("CONVERSATION_SUMMARY_KEEP_TURNS", "3"))

    return AppConfig(
        db_path=db_path,
        model=model,
        final_answer_model=final_answer_model,
        temperature=temperature,
        schema_json_path=Path("result/schema.json"),
        schema_md_path=Path("result/schema.md"),
        log_path=Path("log") / f"log_{datetime.now().strftime('%Y%m%d')}.json",
        memory_db_path=memory_db_path,
        chat_db_path=chat_db_path,
        history_db_path=history_db_path,
        fewshot_candidate_limit=fewshot_candidate_limit,
        conversation_summary_char_limit=conversation_summary_char_limit,
        conversation_summary_keep_turns=conversation_summary_keep_turns,
    )
