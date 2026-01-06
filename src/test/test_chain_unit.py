"""
체인(라우팅 → SQL 생성/실행 → 요약) 흐름을 LLM 없이 단위 테스트한다.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# 로컬 실행 시 src 패키지를 인식하도록 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.chain import AgentDependencies, build_agent_chain
from src.agent.fewshot_generator import SchemaSelectionResult
from src.agent.multi_step_planner import MultiStepPlanResult
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.planner import Planner
from src.agent.router import Route, RouterResult
from src.agent.validator import ResultValidator
from src.db.schema_store import SchemaStore
from src.db.sqlite_client import SQLiteClient
from src.metrics.registry import MetricsRegistry
from src.prompt.sql_generation import SQL_FEWSHOT_EXAMPLES


class _Tee:
    """
    두 개의 스트림에 동시에 출력하는 간단한 Tee.

    Args:
        streams: 출력 대상 스트림 목록.
    """

    def __init__(self, *streams: Any) -> None:
        """
        Tee 출력기를 초기화한다.

        Args:
            streams: 출력 대상 스트림 목록.
        """

        self._streams = streams

    def write(self, data: str) -> None:
        """
        데이터를 모든 스트림에 기록한다.

        Args:
            data: 출력 문자열.
        """

        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        """
        모든 스트림을 플러시한다.
        """

        for stream in self._streams:
            stream.flush()


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

    def get_last_column_parser(self) -> None:
        """
        컬럼 파서 결과를 반환한다.

        Returns:
            None
        """

        return None

    def get_last_column_parser_used(self) -> bool:
        """
        컬럼 파서 사용 여부를 반환한다.

        Returns:
            False.
        """

        return False


class DummyFewshotGenerator:
    """
    테스트용 Few-shot 생성기.
    """

    def generate_examples(self, payload: Any) -> str:
        """
        기본 few-shot 예시를 반환한다.

        Args:
            payload: few-shot 입력.

        Returns:
            few-shot 예시 문자열.
        """

        return SQL_FEWSHOT_EXAMPLES

    def select_schema(self, payload: Any) -> Any:
        """
        스키마 선별을 비워 반환한다.

        Args:
            payload: 스키마 선별 입력.

        Returns:
            빈 선별 결과.
        """

        return SchemaSelectionResult(tables=[], columns_by_table={}, reason=None)


class DummyMultiStepPlanner:
    """
    테스트용 멀티 스텝 플래너.
    """

    def plan(self, payload: Any) -> Any:
        """
        멀티 스텝 미사용으로 고정한다.

        Args:
            payload: 멀티 스텝 입력.

        Returns:
            멀티 스텝 계획 결과.
        """

        return MultiStepPlanResult(use_multi_step=False, execution_mode=None, steps=[], combine=None, reason=None)


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

    def summarize(self, payload: Any, *, stream: bool = False) -> str:
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

    def compose_direct(self, metric: Any, *, stream: bool = False) -> str:
        """
        Direct Answer 응답을 반환.

        Args:
            metric: 메트릭 정의.

        Returns:
            응답 문자열.
        """

        return self._answer

    def compose_general(self, user_message: str, *, stream: bool = False) -> str:
        """
        일반 안내 응답을 반환.

        Args:
            user_message: 사용자 질문.

        Returns:
            응답 문자열.
        """

        return self._answer

    def compose_clarify(self, user_message: str, clarify_question: str, *, stream: bool = False) -> str:
        """
        확인 질문 응답을 반환.

        Args:
            user_message: 사용자 질문.
            clarify_question: 확인 질문.

        Returns:
            응답 문자열.
        """

        return f"{self._answer} - {clarify_question}"

    def compose_reuse(
        self,
        user_message: str,
        reuse_summary: str,
        result_markdown: str,
        *,
        stream: bool = False,
    ) -> str:
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

    def compose_missing_metric(self, user_message: str, *, stream: bool = False) -> str:
        """
        메트릭 누락 응답을 반환.

        Args:
            user_message: 사용자 질문.
            stream: 스트리밍 여부.

        Returns:
            응답 문자열.
        """

        return self._answer

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
        fewshot_generator=DummyFewshotGenerator(),
        fewshot_candidate_limit=5,
        multi_step_planner=DummyMultiStepPlanner(),
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
    debug: bool = False,
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
    result = chain.invoke({"user_message": query})
    if debug:
        print("\n=== Debug ===")
        print({"route": result.get("route"), "planned_slots": result.get("planned_slots")})
    return result


if __name__ == "__main__":
    # 1) 로그 파일을 준비한다.
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = log_dir / f"test_chain_unit_{timestamp}.log"

    # 2) 콘솔과 로그 파일에 동시에 출력한다.
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _Tee(sys.stdout, log_file)
        tee_stderr = _Tee(sys.stderr, log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            unittest.main(verbosity=2)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
