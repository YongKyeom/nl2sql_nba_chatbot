from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.model.llm_operator import LLMOperator

from src.prompt.summarize import SUMMARY_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class SummaryInput:
    """
    요약 생성 입력 구조.

    Args:
        user_question: 사용자 질문.
        sql: 실행 SQL.
        result_preview: 결과 미리보기.
        applied_filters: 적용 조건 요약.
        conversation_context: 최근 대화 컨텍스트.
    """

    user_question: str
    sql: str
    result_preview: list[dict[str, object]]
    applied_filters: str
    conversation_context: str | None = None


class Summarizer:
    """
    결과 요약 생성기.
    """

    def __init__(self, *, model: str, temperature: float = 0.2) -> None:
        """
        Summarizer 초기화.

        Args:
            model: 사용할 LLM 모델명.
            temperature: 생성 다양성 파라미터.
        """

        self._client = LLMOperator()
        self._model = model
        self._temperature = temperature

    async def summarize(self, payload: SummaryInput, *, stream: bool = False) -> str | AsyncIterator[str]:
        """
        요약 텍스트를 생성.

        Args:
            payload: 요약 입력.
            stream: True면 스트리밍 이터레이터를 반환.

        Returns:
            요약 문자열 또는 스트리밍 이터레이터.
        """

        prompt = SUMMARY_PROMPT.format(
            user_question=payload.user_question,
            sql=payload.sql,
            result_preview=json.dumps(payload.result_preview, ensure_ascii=False),
            applied_filters=payload.applied_filters,
            conversation_context=payload.conversation_context or "없음",
        )
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
            print("OPENAI_API_KEY가 없어 Summarizer 테스트를 건너뜁니다.")
            return

        # 2) 요약 생성 테스트
        from src.config import load_config

        config = load_config()
        summarizer = Summarizer(model=config.model, temperature=0.0)
        result = await summarizer.summarize(
            SummaryInput(
                user_question="최근 시즌 승률 상위 3개 팀 알려줘",
                sql="SELECT team_name, pct FROM team LIMIT 3",
                result_preview=[
                    {"team_name": "A", "pct": 0.7},
                    {"team_name": "B", "pct": 0.65},
                ],
                applied_filters="시즌: 2022-23",
            )
        )
        print(result)

    asyncio.run(_main())
