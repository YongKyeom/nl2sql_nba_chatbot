from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.prompt.sql_generation import SQL_FEWSHOT_EXAMPLES, SQL_GENERATION_PROMPT
from src.prompt.sql_repair import SQL_REPAIR_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class SQLGenerationInput:
    """
    SQL 생성 입력 정보.

    Args:
        user_question: 사용자 질문.
        planned_slots: 플래너 슬롯 딕셔너리.
        metric_context: 메트릭 컨텍스트.
        schema_context: 스키마 요약 문자열.
        last_sql: 직전 SQL.
        context_hint: 대화 맥락 요약.
        fewshot_examples: SQL 생성 예시 문자열.
    """

    user_question: str
    planned_slots: dict[str, Any]
    metric_context: dict[str, Any]
    schema_context: str
    last_sql: str | None = None
    context_hint: str | None = None
    fewshot_examples: str | None = None


@dataclass(frozen=True)
class SQLRepairInput:
    """
    SQL 수정 입력 정보.

    Args:
        user_question: 사용자 질문.
        failed_sql: 실패한 SQL.
        failure_reason: 실패 원인.
        metric_context: 메트릭 컨텍스트.
        schema_context: 스키마 요약 문자열.
    """

    user_question: str
    failed_sql: str
    failure_reason: str
    metric_context: dict[str, Any]
    schema_context: str


class SQLGenerator:
    """
    LLM을 사용해 SQL 생성/수정을 수행.

    메트릭/스키마 정의가 바뀔 수 있으므로, 입력으로 전달된 컨텍스트만을
    사용해 SQL을 만든다.
    """

    def __init__(self, *, model: str, temperature: float) -> None:
        """
        SQLGenerator 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
        """

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def generate_sql(self, payload: SQLGenerationInput) -> str:
        """
        SQL을 생성한다.

        Args:
            payload: SQL 생성 입력 정보.

        Returns:
            생성된 SQL 문자열.
        """

        fewshot_examples = payload.fewshot_examples or SQL_FEWSHOT_EXAMPLES
        prompt = SQL_GENERATION_PROMPT.format(
            user_question=payload.user_question,
            planned_slots=json.dumps(payload.planned_slots, ensure_ascii=False),
            metric_context=json.dumps(payload.metric_context, ensure_ascii=False),
            schema_context=payload.schema_context,
            last_sql=payload.last_sql or "없음",
            context_hint=payload.context_hint or "없음",
            fewshot_examples=fewshot_examples,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _strip_code_fence(content)

    def repair_sql(self, payload: SQLRepairInput) -> str:
        """
        실패한 SQL을 한 번만 수정한다.

        Args:
            payload: SQL 수정 입력 정보.

        Returns:
            수정된 SQL 문자열.
        """

        prompt = SQL_REPAIR_PROMPT.format(
            user_question=payload.user_question,
            failed_sql=payload.failed_sql,
            failure_reason=payload.failure_reason,
            metric_context=json.dumps(payload.metric_context, ensure_ascii=False),
            schema_context=payload.schema_context,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _strip_code_fence(content)


def _strip_code_fence(text: str) -> str:
    """
    코드블록이 포함된 응답에서 SQL만 추출.

    Args:
        text: LLM 응답 문자열.

    Returns:
        코드블록 제거된 SQL 문자열.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
    if cleaned.lower().startswith("sql\n"):
        cleaned = cleaned.split("\n", maxsplit=1)[1].strip()
    return cleaned.strip()


if __name__ == "__main__":
    # 1) 코드블록 제거 테스트
    sample = "```sql\nSELECT 1;\n```"
    print(_strip_code_fence(sample))

    # 2) API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없어 SQLGenerator 테스트를 건너뜁니다.")
    else:
        # 3) SQL 생성 테스트
        from pathlib import Path

        from src.config import load_config
        from src.db.schema_store import SchemaStore
        from src.metrics.registry import MetricsRegistry

        config = load_config()
        registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
        registry.ensure_loaded()

        schema_store = SchemaStore(config.schema_json_path)
        if not config.schema_json_path.exists():
            print("schema.json이 없어 SQLGenerator 테스트를 건너뜁니다.")
        else:
            schema_store.ensure_loaded()

        metric = registry.get("win_pct")
        if metric and config.schema_json_path.exists():
            generator = SQLGenerator(model=config.model, temperature=0.0)
            sql = generator.generate_sql(
                SQLGenerationInput(
                    user_question="최근 시즌 승률 상위 5개 팀",
                    planned_slots={"metric": "win_pct", "top_k": 5, "season": "2022-23"},
                    metric_context=registry.build_sql_context(metric),
                    schema_context=schema_store.build_context(),
                    last_sql=None,
                )
            )
            print(sql)
