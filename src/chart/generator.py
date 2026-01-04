from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.prompt.chart import CHART_PROMPT
from src.chart.types import SUPPORTED_CHART_TYPES, format_chart_type_guide
from src.prompt.system import SYSTEM_PROMPT


@dataclass(frozen=True)
class ChartGenerator:
    """
    사용자 질문과 데이터프레임 정보를 기반으로 차트 스펙을 생성한다.

    Args:
        model: 사용할 LLM 모델명.
        temperature: 생성 다양성 파라미터.
    """

    model: str
    temperature: float = 0.2

    def __post_init__(self) -> None:
        """
        OpenAI 클라이언트를 준비한다.

        Side Effects:
            내부에 OpenAI 클라이언트를 초기화한다.

        Raises:
            예외 없음.
        """

        object.__setattr__(self, "_client", OpenAI())

    def generate(
        self,
        *,
        user_question: str,
        columns: list[str],
        numeric_columns: list[str],
        sample_records: str,
    ) -> dict[str, Any]:
        """
        차트 스펙을 생성한다.

        Args:
            user_question: 사용자 질문.
            columns: 전체 컬럼 목록.
            numeric_columns: 수치형 컬럼 목록.
            sample_records: 샘플 레코드 문자열.

        Returns:
            차트 스펙 딕셔너리.

        Side Effects:
            LLM API 호출을 수행한다.

        Raises:
            ValueError: 응답 JSON이 올바르지 않을 때 발생할 수 있다.
        """

        prompt = CHART_PROMPT.format(
            user_question=user_question,
            columns=columns,
            numeric_columns=numeric_columns,
            sample_records=sample_records,
            chart_type_guide=format_chart_type_guide(),
            chart_type_list=", ".join(SUPPORTED_CHART_TYPES),
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        return json.loads(content)


if __name__ == "__main__":
    # 1) 차트 생성기를 구성한다.
    generator = ChartGenerator(model="gpt-4o-mini", temperature=0.0)
    # 2) 샘플 입력으로 차트 스펙을 확인한다.
    print(
        generator.generate(
            user_question="관중 상위 팀 그래프로 보여줘",
            columns=["team_name", "avg_attendance"],
            numeric_columns=["avg_attendance"],
            sample_records="[{'team_name': 'BOS', 'avg_attendance': 19000}]",
        )
    )
