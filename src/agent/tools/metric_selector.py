from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.registry import MetricDefinition, MetricsRegistry


@dataclass(frozen=True)
class MetricCandidate:
    """
    메트릭 후보 요약 정보.

    Args:
        name: 메트릭 식별자.
        aliases: 별칭 목록.
        description_ko: 한글 설명.
        formula_ko: 계산식/규칙 요약.
        required_tables: 필요한 테이블 목록.
        required_columns: 필요한 컬럼 목록.
    """

    name: str
    aliases: list[str]
    description_ko: str
    formula_ko: str | None
    required_tables: list[str]
    required_columns: list[str]


class MetricSelector:
    """
    metrics.yaml 기반으로 메트릭 후보를 선택하는 도구.
    """

    def __init__(self, registry: MetricsRegistry) -> None:
        """
        MetricSelector 초기화.

        Args:
            registry: 메트릭 레지스트리.
        """

        self._registry = registry

    def tool_schema(self) -> dict[str, Any]:
        """
        OpenAI tool schema를 반환한다.

        Returns:
            tool schema 딕셔너리.
        """

        return {
            "type": "function",
            "function": {
                "name": "metric_selector",
                "description": "사용자 질의에 맞는 메트릭 후보를 Top-K로 반환한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "사용자 질문 원문"},
                        "top_k": {"type": "integer", "description": "후보 개수", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
        }

    def select(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        """
        메트릭 후보를 선택한다.

        Args:
            query: 사용자 질문.
            top_k: 반환할 후보 개수.

        Returns:
            후보 목록을 담은 딕셔너리.
        """

        limit = max(1, min(top_k, 10))
        candidates = self._registry.search(query, limit=limit)
        payload = [_metric_to_candidate(metric).__dict__ for metric in candidates]
        return {"candidates": payload, "count": len(payload)}


def _metric_to_candidate(metric: MetricDefinition) -> MetricCandidate:
    """
    MetricDefinition을 MetricCandidate로 변환한다.

    Args:
        metric: 메트릭 정의.

    Returns:
        MetricCandidate 객체.
    """

    return MetricCandidate(
        name=metric.name,
        aliases=metric.aliases[:6],
        description_ko=metric.description_ko,
        formula_ko=metric.formula_ko,
        required_tables=metric.required_tables,
        required_columns=metric.required_columns,
    )


if __name__ == "__main__":
    # 1) 레지스트리를 로드해 메트릭 후보를 확인한다.
    registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
    registry.ensure_loaded()

    # 2) 샘플 질의를 기반으로 후보를 출력한다.
    selector = MetricSelector(registry)
    sample_result = selector.select(query="2022-23 시즌 팀 득점 상위 5개 보여줘", top_k=5)
    print(sample_result)
