from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from src.model.llm_operator import LLMOperator

from src.prompt.multi_step_sql import MULTI_STEP_SQL_PROMPT
from src.prompt.sql_generation import SQL_FEWSHOT_EXAMPLES, SQL_GENERATION_PROMPT
from src.prompt.sql_repair import SQL_REPAIR_PROMPT
from src.prompt.system import SYSTEM_PROMPT
from src.agent.tools.column_parser import ColumnParser


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


@dataclass(frozen=True)
class MultiStepSQLInput:
    """
    멀티 스텝 계획을 단일 SQL로 합성하기 위한 입력 정보.

    Args:
        user_question: 사용자 질문.
        planned_slots: 플래너 슬롯 딕셔너리.
        multi_step_plan: 멀티 스텝 계획 딕셔너리.
        metric_contexts: 관련 메트릭 컨텍스트 목록.
        schema_context: 스키마 요약 문자열.
        last_sql: 직전 SQL.
        context_hint: 대화 맥락 요약.
        fewshot_examples: SQL 생성 예시 문자열.
    """

    user_question: str
    planned_slots: dict[str, Any]
    multi_step_plan: dict[str, Any]
    metric_contexts: list[dict[str, Any]]
    schema_context: str
    last_sql: str | None = None
    context_hint: str | None = None
    fewshot_examples: str | None = None


class SQLGenerator:
    """
    LLM을 사용해 SQL 생성/수정을 수행.

    메트릭/스키마 정의가 바뀔 수 있으므로, 입력으로 전달된 컨텍스트만을
    사용해 SQL을 만든다.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        column_parser: ColumnParser | None = None,
        max_tool_attempts: int = 3,
    ) -> None:
        """
        SQLGenerator 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
            column_parser: 컬럼 파서 도구(없으면 비활성).
            max_tool_attempts: 컬럼 파서 재시도 횟수.
        """

        self._client = LLMOperator()
        self._model = model
        self._temperature = temperature
        self._column_parser = column_parser
        self._max_tool_attempts = max(1, max_tool_attempts)
        self._last_column_parser: dict[str, Any] | None = None
        self._last_column_parser_used: bool = False

    def get_last_column_parser(self) -> dict[str, Any] | None:
        """
        직전 컬럼 파서 결과를 반환한다.

        Returns:
            컬럼 파서 결과(없으면 None).
        """

        return self._last_column_parser

    def get_last_column_parser_used(self) -> bool:
        """
        직전 컬럼 파서 사용 여부를 반환한다.

        Returns:
            컬럼 파서 사용 여부.
        """

        return self._last_column_parser_used

    async def generate_sql(self, payload: SQLGenerationInput) -> str:
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
        return await self._generate_with_column_parser(prompt)

    async def repair_sql(self, payload: SQLRepairInput) -> str:
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
        return await self._generate_with_column_parser(prompt)

    async def generate_multi_step_sql(self, payload: MultiStepSQLInput) -> str:
        """
        멀티 스텝 계획을 단일 SQL로 합성한다.

        Args:
            payload: 멀티 스텝 SQL 합성 입력.

        Returns:
            생성된 SQL 문자열.
        """

        fewshot_examples = payload.fewshot_examples or SQL_FEWSHOT_EXAMPLES
        prompt = MULTI_STEP_SQL_PROMPT.format(
            user_question=payload.user_question,
            planned_slots=json.dumps(payload.planned_slots, ensure_ascii=False),
            multi_step_plan=json.dumps(payload.multi_step_plan, ensure_ascii=False),
            metric_contexts=json.dumps(payload.metric_contexts, ensure_ascii=False),
            schema_context=payload.schema_context,
            last_sql=payload.last_sql or "없음",
            context_hint=payload.context_hint or "없음",
            fewshot_examples=fewshot_examples,
        )
        return await self._generate_with_column_parser(prompt)

    async def _generate_with_column_parser(self, prompt: str) -> str:
        """
        컬럼 파서 도구를 포함해 SQL을 생성한다.

        Args:
            prompt: SQL 생성 프롬프트.

        Returns:
            SQL 문자열.
        """

        self._last_column_parser = None
        self._last_column_parser_used = False
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if not self._column_parser:
            response = await self._client.invoke(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            return _strip_code_fence(content)

        tools = [self._column_parser.tool_schema()]
        sql = ""
        for attempt in range(self._max_tool_attempts):
            response = await self._client.invoke(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            message = response.choices[0].message
            if message.tool_calls:
                messages.append(_build_tool_call_message(message))
                for call in message.tool_calls:
                    if call.function.name != "column_parser":
                        continue
                    args = _safe_json(call.function.arguments)
                    sql_text = str(args.get("sql") or "")
                    tool_result = await self._column_parser.parse(sql=sql_text)
                    self._last_column_parser = tool_result
                    self._last_column_parser_used = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                continue

            content = message.content or ""
            sql = _strip_code_fence(content)
            if not sql:
                messages.append({"role": "user", "content": "SQL을 다시 출력해 주세요. SQL만 반환하세요."})
                continue

            validation = await self._column_parser.parse(sql=sql)
            self._last_column_parser = validation
            self._last_column_parser_used = True
            if validation.get("is_valid") or attempt == self._max_tool_attempts - 1:
                return sql

            feedback = _build_column_feedback(validation)
            messages.append({"role": "user", "content": feedback})

        return sql


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


def _safe_json(text: str) -> dict[str, Any]:
    """
    JSON 문자열을 안전하게 파싱한다.

    Args:
        text: JSON 문자열.

    Returns:
        파싱 결과(실패 시 빈 딕셔너리).
    """

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _build_tool_call_message(message: Any) -> dict[str, Any]:
    """
    tool_calls가 포함된 assistant 메시지를 직렬화한다.

    Args:
        message: LLM 응답 메시지.

    Returns:
        직렬화된 메시지.
    """

    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    payload = {"role": "assistant", "tool_calls": tool_calls}
    if message.content:
        payload["content"] = message.content
    return payload


def _build_column_feedback(result: dict[str, Any]) -> str:
    """
    컬럼 파서 결과를 SQL 수정 지시로 요약한다.

    Args:
        result: 컬럼 파서 결과.

    Returns:
        수정 지시 문자열.
    """

    unknown_tables = result.get("unknown_tables") or []
    unknown_columns = result.get("unknown_columns") or {}

    lines = [
        "ColumnParser 결과에 오류가 있습니다.",
        "아래 오류를 수정해 SQL을 다시 작성하세요.",
    ]
    if unknown_tables:
        lines.append(f"- 알 수 없는 테이블: {', '.join(sorted(set(unknown_tables)))}")
    if unknown_columns:
        for table, cols in unknown_columns.items():
            if not isinstance(cols, list):
                continue
            col_list = ", ".join(cols)
            lines.append(f"- 테이블 {table}에 없는 컬럼: {col_list}")
    return "\n".join(lines)


if __name__ == "__main__":
    async def _main() -> None:
        # 1) 코드블록 제거 테스트
        sample = "```sql\nSELECT 1;\n```"
        print(_strip_code_fence(sample))

        # 2) API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY가 없어 SQLGenerator 테스트를 건너뜁니다.")
            return

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
            return

        schema_store.ensure_loaded()

        metric = registry.get("win_pct")
        if metric:
            generator = SQLGenerator(model=config.model, temperature=0.0)
            sql = await generator.generate_sql(
                SQLGenerationInput(
                    user_question="최근 시즌 승률 상위 5개 팀",
                    planned_slots={"metric": "win_pct", "top_k": 5, "season": "2022-23"},
                    metric_context=registry.build_sql_context(metric),
                    schema_context=schema_store.build_context(),
                    last_sql=None,
                )
            )
            print(sql)

    asyncio.run(_main())
