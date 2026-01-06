from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.model.llm_operator import LLMOperator

from src.prompt.multi_step_plan import MULTI_STEP_PLAN_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class MultiStepPlanInput:
    """
    멀티 스텝 계획 입력 정보.

    Args:
        user_question: 사용자 질문 원문.
        planned_slots: 플래너 슬롯 딕셔너리.
        schema_context: 스키마 요약 문자열.
        context_hint: 대화 맥락 요약.
    """

    user_question: str
    planned_slots: dict[str, Any]
    schema_context: str
    context_hint: str | None = None


@dataclass(frozen=True)
class MultiStepPlanResult:
    """
    멀티 스텝 계획 결과.

    Args:
        use_multi_step: 멀티 스텝 여부.
        execution_mode: 실행 방식(single_sql 또는 multi_sql).
        steps: 멀티 스텝 단계 목록.
        combine: 결과 결합 규칙.
        reason: 판단 근거.
    """

    use_multi_step: bool
    execution_mode: str | None
    steps: list[dict[str, Any]]
    combine: dict[str, Any] | None
    reason: str | None


class MultiStepPlanner:
    """
    LLM 기반 멀티 스텝 계획 생성기.
    """

    def __init__(self, *, model: str, temperature: float = 0.0) -> None:
        """
        MultiStepPlanner 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
        """

        self._client = LLMOperator()
        self._model = model
        self._temperature = temperature

    def plan(self, payload: MultiStepPlanInput) -> MultiStepPlanResult:
        """
        멀티 스텝 계획을 결정한다.

        Args:
            payload: 멀티 스텝 계획 입력 정보.

        Returns:
            MultiStepPlanResult.
        """

        prompt = MULTI_STEP_PLAN_PROMPT.format(
            user_question=payload.user_question,
            planned_slots=json.dumps(payload.planned_slots, ensure_ascii=False),
            schema_context=payload.schema_context,
            context_hint=payload.context_hint or "없음",
        )
        try:
            response = self._client.invoke(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:  # noqa: BLE001
            return MultiStepPlanResult(use_multi_step=False, execution_mode=None, steps=[], combine=None, reason=None)

        content = response.choices[0].message.content or ""
        return _parse_plan_result(content)


def _parse_plan_result(text: str) -> MultiStepPlanResult:
    """
    멀티 스텝 계획 응답을 파싱한다.

    Args:
        text: LLM 응답 문자열.
    Returns:
        MultiStepPlanResult.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", maxsplit=2)[1].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return MultiStepPlanResult(use_multi_step=False, execution_mode=None, steps=[], combine=None, reason=None)

    use_multi_step = bool(payload.get("use_multi_step"))
    reason = str(payload.get("reason") or "").strip() if isinstance(payload, dict) else None
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    combine = payload.get("combine") if isinstance(payload, dict) else None
    execution_mode = str(payload.get("execution_mode") or "").strip() if isinstance(payload, dict) else ""
    if execution_mode not in {"single_sql", "multi_sql"}:
        execution_mode = ""

    if not use_multi_step:
        # LLM이 "단일 SQL로 가능"이라는 의미로 use_multi_step=false를 반환하더라도,
        # steps/combine이 존재하면 합성 계획으로 간주해 후속 단계에서 활용한다.
        if not (isinstance(steps, list) and steps):
            return MultiStepPlanResult(
                use_multi_step=False,
                execution_mode=None,
                steps=[],
                combine=None,
                reason=reason or None,
            )
        use_multi_step = True

    if not isinstance(steps, list):
        return MultiStepPlanResult(use_multi_step=False, execution_mode=None, steps=[], combine=None, reason=reason or None)

    normalized_steps = [step for step in steps if isinstance(step, dict)]
    if not normalized_steps:
        return MultiStepPlanResult(use_multi_step=False, execution_mode=None, steps=[], combine=None, reason=reason or None)

    return MultiStepPlanResult(
        use_multi_step=True,
        execution_mode=execution_mode or "multi_sql",
        steps=normalized_steps,
        combine=combine,
        reason=reason or None,
    )
