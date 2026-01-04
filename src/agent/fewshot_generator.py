from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import load_config
from src.db.schema_store import SchemaStore
from src.metrics.registry import MetricsRegistry
from src.prompt.fewshot import FEWSHOT_PROMPT
from src.prompt.sql_generation import SQL_FEWSHOT_EXAMPLES
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class FewshotGenerationInput:
    """
    Few-shot 생성 입력 정보.

    Args:
        user_question: 사용자 질문 원문.
        planned_slots: 플래너 슬롯 딕셔너리.
        candidate_metrics: 후보 메트릭 컨텍스트 리스트.
        schema_context: 스키마 요약 문자열.
        context_hint: 대화 맥락 요약.
        target_count: 생성할 예시 개수.
    """

    user_question: str
    planned_slots: dict[str, Any]
    candidate_metrics: list[dict[str, Any]]
    schema_context: str
    context_hint: str | None = None
    target_count: int = 3


class FewshotGenerator:
    """
    LLM 기반 few-shot 예시 생성기.

    SQL 생성 단계 이전에 유사 예시를 구성해 프롬프트 품질을 보강한다.
    """

    def __init__(self, *, model: str, temperature: float = 0.2) -> None:
        """
        FewshotGenerator 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
        """

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def generate_examples(self, payload: FewshotGenerationInput) -> str:
        """
        few-shot 예시 문자열을 생성한다.

        Args:
            payload: few-shot 생성 입력 정보.

        Returns:
            few-shot 예시 문자열.
        """

        if not payload.candidate_metrics:
            return SQL_FEWSHOT_EXAMPLES

        prompt = FEWSHOT_PROMPT.format(
            user_question=payload.user_question,
            planned_slots=json.dumps(payload.planned_slots, ensure_ascii=False),
            candidate_metrics=json.dumps(payload.candidate_metrics, ensure_ascii=False),
            schema_context=payload.schema_context,
            context_hint=payload.context_hint or "없음",
            target_count=payload.target_count,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:  # noqa: BLE001
            return SQL_FEWSHOT_EXAMPLES

        content = response.choices[0].message.content or ""
        examples = _parse_examples(content)
        formatted = _format_examples(examples, limit=payload.target_count)
        return formatted or SQL_FEWSHOT_EXAMPLES


def _parse_examples(text: str) -> list[dict[str, str]]:
    """
    LLM 응답에서 예시 목록을 파싱한다.

    Args:
        text: LLM 응답 문자열.

    Returns:
        예시 목록.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    raw_examples = payload.get("examples", [])
    if not isinstance(raw_examples, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        sql = _strip_code_fence(str(item.get("sql") or "").strip())
        note = str(item.get("note") or "").strip()
        if not question or not sql:
            continue
        normalized.append({"question": question, "sql": sql, "note": note})
    return normalized


def _format_examples(examples: list[dict[str, str]], *, limit: int) -> str:
    """
    예시 목록을 SQL 생성 프롬프트 형식으로 직렬화한다.

    Args:
        examples: 예시 목록.
        limit: 최대 출력 개수.

    Returns:
        직렬화된 문자열.
    """

    if not examples:
        return ""

    selected = examples[: max(limit, 0)]
    lines: list[str] = []
    for idx, example in enumerate(selected, start=1):
        lines.append(f"[예시 {idx}]")
        lines.append(f"질문: {example['question']}")
        lines.append("SQL:")
        lines.append(example["sql"])
        if example.get("note"):
            lines.append(f"설명: {example['note']}")
        lines.append("")
    return "\n".join(lines).strip()


def _strip_code_fence(text: str) -> str:
    """
    코드블록이 포함된 응답에서 SQL만 추출한다.

    Args:
        text: LLM 응답 문자열.

    Returns:
        코드블록 제거된 문자열.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned.split("\n", maxsplit=1)[1].strip()
    return cleaned.strip()


if __name__ == "__main__":
    config = load_config()
    registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
    registry.ensure_loaded()
    schema_store = SchemaStore(config.schema_json_path)
    schema_store.ensure_loaded()

    generator = FewshotGenerator(model=config.model, temperature=0.1)
    metric = registry.search("팀 득점 상위", limit=1)
    examples = generator.generate_examples(
        FewshotGenerationInput(
            user_question="2023-24 시즌 팀 득점 상위 10개 보여줘",
            planned_slots={"metric": "top_scorers", "season": "2023-24", "top_k": 10, "filters": {}},
            candidate_metrics=[registry.build_sql_context(metric[0])] if metric else [],
            schema_context=schema_store.build_context(),
            context_hint="없음",
            target_count=3,
        )
    )
    print(examples)
