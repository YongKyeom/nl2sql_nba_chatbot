"""
체인(라우팅 → SQL 생성/실행 → 요약) 흐름을 LLM 없이 단위 테스트한다.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.chain import AgentDependencies, build_agent_chain
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.planner import Planner
from src.agent.router import Route, RouterResult
from src.agent.validator import ResultValidator
from src.db.schema_store import SchemaStore
from src.db.sqlite_client import SQLiteClient
from src.metrics.registry import MetricsRegistry


class DummyRouter:
    """
    테스트용 라우터.

    Args:
        route: 라우팅 결과.
        metric_name: 강제로 지정할 메트릭 이름.
    """

    def __init__(self, route: Route, metric_name: str | None = None) -> None:
        """
        DummyRouter 초기화.

        Args:
            route: 라우팅 결과.
            metric_name: 메트릭 이름.
        """

        self._route = route
        self._metric_name = metric_name

    def route(self, context: Any, registry: MetricsRegistry) -> RouterResult:
        """
        라우팅 결과를 반환.

        Args:
            context: 라우팅 컨텍스트.
            registry: 메트릭 레지스트리.

        Returns:
            RouterResult.
        """

        return RouterResult(route=self._route, metric_name=self._metric_name, reason="테스트 라우터")


class DummySQLGenerator:
    """
    테스트용 SQL 생성기.

    Args:
        sql: 반환할 SQL 문자열.
    """

    def __init__(self, sql: str) -> None:
        """
        DummySQLGenerator 초기화.

        Args:
            sql: 반환할 SQL 문자열.
        """

        self._sql = sql

    def generate_sql(self, payload: Any) -> str:
        """
        SQL을 반환.

        Args:
            payload: SQL 생성 입력.

        Returns:
            SQL 문자열.
        """

        return self._sql

    def repair_sql(self, payload: Any) -> str:
        """
        실패 SQL 수정을 대신한다.

        Args:
            payload: SQL 수정 입력.

        Returns:
            SQL 문자열.
        """

        return self._sql


class DummySummarizer:
    """
    테스트용 요약기.

    Args:
        answer: 반환할 요약 문자열.
    """

    def __init__(self, answer: str) -> None:
        """
        DummySummarizer 초기화.

        Args:
            answer: 반환할 요약 문자열.
        """

        self._answer = answer

    def summarize(self, payload: Any) -> str:
        """
        요약 문자열을 반환.

        Args:
            payload: 요약 입력.

        Returns:
            요약 문자열.
        """

        return self._answer


class DummyResponder:
    """
    테스트용 응답 생성기.

    Args:
        answer: 반환할 응답 문자열.
    """

    def __init__(self, answer: str) -> None:
        """
        DummyResponder 초기화.

        Args:
            answer: 반환할 응답 문자열.
        """

        self._answer = answer

    def compose_direct(self, metric: Any) -> str:
        """
        Direct Answer 응답을 반환.

        Args:
            metric: 메트릭 정의.

        Returns:
            응답 문자열.
        """

        return self._answer

    def compose_clarify(self, user_message: str, clarify_question: str) -> str:
        """
        확인 질문 응답을 반환.

        Args:
            user_message: 사용자 질문.
            clarify_question: 확인 질문.

        Returns:
            응답 문자열.
        """

        return f"{self._answer} - {clarify_question}"

    def compose_reuse(self, user_message: str, reuse_summary: str, result_markdown: str) -> str:
        """
        후처리 응답을 반환.

        Args:
            user_message: 사용자 질문.
            reuse_summary: 후처리 요약.
            result_markdown: 마크다운 테이블.

        Returns:
            응답 문자열.
        """

        return f"{self._answer} - {reuse_summary}"

    def compose_missing_metric(self, user_message: str) -> str:
        """
        메트릭 누락 응답을 반환.

        Args:
            user_message: 사용자 질문.

        Returns:
            응답 문자열.
        """

        return self._answer


class TestAgentAnswerFlow(unittest.TestCase):
    """
    에이전트 응답 흐름 테스트.
    """

    def test_direct_answer_response(self) -> None:
        """
        Direct Answer 응답 형식을 검증한다.
        """

        deps = _build_dependencies(
            router=DummyRouter(Route.DIRECT, metric_name="ts_pct"),
            sql="SELECT 1",
        )
        chain = build_agent_chain(deps)
        result = chain.invoke({"user_message": "TS% 정의 알려줘"})

        self.assertIn("요약 테스트", result["final_answer"])

    def test_reuse_response(self) -> None:
        """
        이전 결과 후처리 응답을 검증한다.
        """

        memory = ConversationMemory()
        memory.last_sql = "SELECT team, value FROM sample LIMIT 10"
        memory.last_result = pd.DataFrame({"team": ["A", "B", "C"], "value": [1, 2, 3]})
        memory.last_result_schema = ["team", "value"]

        deps = _build_dependencies(
            router=DummyRouter(Route.REUSE),
            sql="SELECT 1",
            memory=memory,
        )
        chain = build_agent_chain(deps)
        result = chain.invoke({"user_message": "상위 1개만"})

        self.assertIsNotNone(result.get("result_df"))
        self.assertEqual(len(result["result_df"]), 1)
        self.assertIn("상위 1개", result["final_answer"])

    def test_sql_response(self) -> None:
        """
        실제 DB에서 SQL 실행 흐름을 검증한다.
        """

        deps = _build_dependencies(
            router=DummyRouter(Route.SQL_REQUIRED),
            sql=(
                "SELECT season, overall_pick, player_name "
                "FROM draft_history "
                "WHERE season = '2018' "
                "ORDER BY overall_pick "
                "LIMIT 5"
            ),
        )
        chain = build_agent_chain(deps)
        result = chain.invoke({"user_message": "2018 드래프트 상위 5픽 보여줘"})

        self.assertIsNotNone(result.get("result_df"))
        self.assertGreater(len(result["result_df"]), 0)
        self.assertEqual(result["final_answer"], "요약 테스트")


def _build_dependencies(
    *,
    router: DummyRouter,
    sql: str,
    memory: ConversationMemory | None = None,
) -> AgentDependencies:
    """
    테스트용 의존성을 구성한다.

    Args:
        router: 테스트 라우터.
        sql: 테스트 SQL.
        memory: 대화 메모리.

    Returns:
        AgentDependencies.
    """

    schema_path = Path("result/schema.json")
    db_path = Path("data/nba.sqlite")

    _skip_if_missing(schema_path, "result/schema.json이 없습니다. Dump Schema를 실행하세요.")
    _skip_if_missing(db_path, "data/nba.sqlite이 없습니다. DB 경로를 확인하세요.")

    registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
    registry.ensure_loaded()

    schema_store = SchemaStore(schema_path)
    schema_store.ensure_loaded()

    return AgentDependencies(
        registry=registry,
        schema_store=schema_store,
        router=router,
        responder=DummyResponder("요약 테스트"),
        planner=Planner(registry),
        sql_generator=DummySQLGenerator(sql),
        guard=SQLGuard(schema_path),
        validator=ResultValidator(),
        sqlite_client=SQLiteClient(db_path),
        summarizer=DummySummarizer("요약 테스트"),
        memory=memory or ConversationMemory(),
    )


def _skip_if_missing(path: Path, message: str) -> None:
    """
    파일이 없으면 테스트를 건너뛴다.

    Args:
        path: 검사할 경로.
        message: 스킵 사유 메시지.

    Returns:
        None
    """

    if not path.exists():
        raise unittest.SkipTest(message)


def run_query(
    query: str,
    *,
    route: Route,
    metric_name: str | None = None,
    sql: str | None = None,
    memory: ConversationMemory | None = None,
) -> dict[str, object]:
    """
    단일 질의를 실행하고 결과를 반환.

    Args:
        query: 사용자 질문.
        route: 강제 라우팅 값.
        metric_name: Direct Answer용 메트릭 이름.
        sql: SQL 경로에서 사용할 SQL 문자열.
        memory: 이전 결과 재사용 테스트용 메모리.

    Returns:
        에이전트 결과 딕셔너리.
    """

    if route == Route.DIRECT and not metric_name:
        raise ValueError("Direct Answer 테스트는 metric_name이 필요합니다.")
    if route == Route.REUSE and memory is None:
        raise ValueError("Reuse 테스트는 memory가 필요합니다.")

    deps = _build_dependencies(
        router=DummyRouter(route, metric_name=metric_name),
        sql=sql or "SELECT 1",
        memory=memory,
    )
    chain = build_agent_chain(deps)
    return chain.invoke({"user_message": query})
