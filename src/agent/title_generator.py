from __future__ import annotations

from dataclasses import dataclass

from src.model.llm_operator import LLMOperator

from src.prompt.system import SYSTEM_PROMPT
from src.prompt.title import TITLE_PROMPT


@dataclass(frozen=True)
class TitleGenerator:
    """
    첫 질문을 기반으로 채팅 제목을 생성한다.

    Args:
        model: 사용할 LLM 모델명.
        temperature: 생성 다양성 파라미터.
    """

    model: str
    temperature: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "_client", LLMOperator())

    def generate(self, user_question: str) -> str:
        """
        채팅 제목을 생성한다.

        Args:
            user_question: 사용자 첫 질문.

        Returns:
            제목 문자열.
        """

        prompt = TITLE_PROMPT.format(user_question=user_question)
        response = self._client.invoke(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    generator = TitleGenerator(model="gpt-4o-mini", temperature=0.0)
    print(generator.generate("2023-24 시즌 팀 득점 상위 10개 보여줘"))
