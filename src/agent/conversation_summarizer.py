from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from src.model.llm_operator import LLMOperator
from src.prompt.conversation_summary import CONVERSATION_SUMMARY_PROMPT
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class ConversationSummarizerConfig:
    """
    대화 요약기 설정.

    Args:
        model: 사용할 LLM 모델명.
        temperature: 생성 다양성 파라미터.
    """

    model: str
    temperature: float = 0.0


class ConversationSummarizer:
    """
    대화 요약 생성기.
    """

    def __init__(self, config: ConversationSummarizerConfig) -> None:
        """
        ConversationSummarizer 초기화.

        Args:
            config: 요약기 설정.
        """

        self._client = LLMOperator()
        self._model = config.model
        self._temperature = config.temperature

    async def summarize(self, summary_source: str) -> str:
        """
        대화 기록을 요약한다.

        Args:
            summary_source: 요약 대상 문자열.

        Returns:
            요약 문자열.
        """

        prompt = CONVERSATION_SUMMARY_PROMPT.format(summary_source=summary_source)
        response = await self._client.invoke(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    async def _main() -> None:
        # 1) API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY가 없어 ConversationSummarizer 테스트를 건너뜁니다.")
            return

        # 2) 요약 테스트
        summarizer = ConversationSummarizer(ConversationSummarizerConfig(model="gpt-4.1-mini"))
        sample = """
        - 사용자: 2022-23 시즌 승률 상위 5개 팀 알려줘
          어시스턴트: 승률 상위 5개 팀은 MIL, BOS, PHI, DEN, CLE 입니다.
        - 사용자: 상위 3개만
          어시스턴트: 상위 3개 팀은 MIL, BOS, PHI 입니다.
        """
        print(await summarizer.summarize(sample))

    asyncio.run(_main())
