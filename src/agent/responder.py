from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.model.llm_operator import LLMOperator

from src.metrics.registry import MetricDefinition
from src.prompt.clarify import CLARIFY_PROMPT
from src.prompt.direct_answer import DIRECT_ANSWER_PROMPT
from src.prompt.general_answer import GENERAL_ANSWER_PROMPT
from src.prompt.missing_metric import MISSING_METRIC_PROMPT
from src.prompt.reuse_answer import REUSE_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class ResponderConfig:
    """
    응답 생성기 설정.

    Args:
        model: 사용할 LLM 모델명.
        temperature: 생성 다양성 파라미터.
    """

    model: str
    temperature: float = 0.2


class Responder:
    """
    LLM 기반 응답 생성기.

    Direct Answer/Clarify/Reuse 메시지를 프롬프트로 생성한다.
    """

    def __init__(self, config: ResponderConfig) -> None:
        """
        Responder 초기화.

        Args:
            config: 응답 생성기 설정.
        """

        self._client = LLMOperator()
        self._model = config.model
        self._temperature = config.temperature

    async def compose_direct(
        self,
        metric: MetricDefinition,
        *,
        conversation_context: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        Direct Answer 응답을 생성.

        Args:
            metric: 메트릭 정의.
            conversation_context: 최근 대화 컨텍스트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        prompt = DIRECT_ANSWER_PROMPT.format(
            metric_name=metric.name,
            metric_description=metric.description_ko,
            metric_formula=metric.formula_ko or "정의에 계산식이 명시되어 있지 않습니다.",
            metric_cut_rules=str(metric.cut_rules),
            conversation_context=conversation_context or "없음",
        )
        return await self._invoke(prompt, stream=stream)

    async def compose_clarify(
        self,
        user_message: str,
        clarify_question: str,
        *,
        conversation_context: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        확인 질문 응답을 생성.

        Args:
            user_message: 사용자 질문.
            clarify_question: 확인 질문.
            conversation_context: 최근 대화 컨텍스트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        prompt = CLARIFY_PROMPT.format(
            user_message=user_message,
            clarify_question=clarify_question,
            conversation_context=conversation_context or "없음",
        )
        return await self._invoke(prompt, stream=stream)

    async def compose_reuse(
        self,
        user_message: str,
        reuse_summary: str,
        result_markdown: str,
        *,
        conversation_context: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        이전 결과 후처리 응답을 생성.

        Args:
            user_message: 사용자 질문.
            reuse_summary: 후처리 요약.
            result_markdown: 마크다운 테이블.
            conversation_context: 최근 대화 컨텍스트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        prompt = REUSE_PROMPT.format(
            user_message=user_message,
            reuse_summary=reuse_summary,
            result_markdown=result_markdown,
            conversation_context=conversation_context or "없음",
        )
        return await self._invoke(prompt, stream=stream)

    async def compose_missing_metric(
        self,
        user_message: str,
        *,
        conversation_context: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        메트릭 정의 누락 응답을 생성.

        Args:
            user_message: 사용자 질문.
            conversation_context: 최근 대화 컨텍스트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        prompt = MISSING_METRIC_PROMPT.format(
            user_message=user_message,
            conversation_context=conversation_context or "없음",
        )
        return await self._invoke(prompt, stream=stream)

    async def compose_general(
        self,
        user_message: str,
        *,
        conversation_context: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        일반 안내 응답을 생성.

        Args:
            user_message: 사용자 질문.
            conversation_context: 최근 대화 컨텍스트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        prompt = GENERAL_ANSWER_PROMPT.format(
            user_message=user_message,
            conversation_context=conversation_context or "없음",
        )
        return await self._invoke(prompt, stream=stream)

    async def _invoke(self, prompt: str, *, stream: bool = False) -> str | AsyncIterator[str]:
        """
        LLM 호출을 수행.

        Args:
            prompt: 사용자 프롬프트.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            응답 문자열 또는 스트리밍 이터레이터.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if stream:
            return self._client.stream(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
            )

        response = await self._client.invoke(
            model=self._model,
            temperature=self._temperature,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    async def _main() -> None:
        # 1) API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY가 없어 Responder 테스트를 건너뜁니다.")
            return

        # 2) 일반 안내 응답 테스트
        from pathlib import Path

        from src.config import load_config
        from src.metrics.registry import MetricsRegistry

        config = load_config()
        responder = Responder(ResponderConfig(model=config.model, temperature=0.0))
        registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
        registry.ensure_loaded()
        metric = registry.get("win_pct")

        if metric:
            print(await responder.compose_direct(metric))
        print(await responder.compose_general("무슨 데이터를 알려줄 수 있어?"))

    asyncio.run(_main())
