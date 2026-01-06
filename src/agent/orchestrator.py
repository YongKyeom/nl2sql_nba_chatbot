from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from src.agent.chain import AgentDependencies, build_agent_chain
from src.agent.fewshot_generator import FewshotGenerator
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.multi_step_planner import MultiStepPlanner
from src.agent.planner import Planner
from src.agent.responder import Responder, ResponderConfig
from src.agent.router import RouterLLM
from src.agent.sql_generator import SQLGenerator
from src.agent.summarizer import Summarizer
from src.agent.tools import ColumnParser, EntityResolver, MetricSelector
from src.agent.validator import ResultValidator
from src.config import AppConfig, load_config
from src.db.schema_dump import dump_schema, SchemaDumpPaths
from src.db.schema_store import SchemaStore
from src.db.sqlite_client import SQLiteClient
from src.metrics.registry import MetricsRegistry


@dataclass(frozen=True)
class OrchestratorOptions:
    """
    오케스트레이터 동작 옵션.

    Args:
        model: 사용할 LLM 모델명(라우팅/SQL 생성용).
        final_answer_model: 최종 답변 생성 모델명.
        temperature: SQL 생성 temperature.
        router_temperature: 라우터 temperature.
        responder_temperature: 답변 생성 temperature.
        summarizer_temperature: 요약 생성 temperature.
    """

    model: str
    temperature: float
    final_answer_model: str | None = None
    router_temperature: float = 0.0
    responder_temperature: float = 0.2
    summarizer_temperature: float = 0.2


class AgentOrchestrator:
    """
    에이전트 의존성을 캡슐화하는 오케스트레이터.

    앱/테스트에서 단일 인스턴스로 사용하며, 내부에서 체인을 구성한다.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        options: OrchestratorOptions | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        """
        AgentOrchestrator 초기화.

        Args:
            config: 앱 설정.
            options: 오케스트레이터 옵션.
            memory: 대화 메모리(없으면 새로 생성).
        """

        self._config: AppConfig = config
        resolved_options = options or OrchestratorOptions(
            model=config.model,
            temperature=config.temperature,
        )

        self._options: OrchestratorOptions = resolved_options
        self.memory: ConversationMemory = memory or ConversationMemory.persistent(config.memory_db_path)

        final_answer_model = resolved_options.final_answer_model or config.final_answer_model

        # 1) 레지스트리/스키마 준비
        self.registry: MetricsRegistry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
        self.registry.ensure_loaded()

        self.schema_store: SchemaStore = SchemaStore(config.schema_json_path)
        if not config.schema_json_path.exists():
            dump_schema(
                SchemaDumpPaths(
                    db_path=config.db_path,
                    json_path=config.schema_json_path,
                    md_path=config.schema_md_path,
                )
            )
        self.schema_store.ensure_loaded()

        # 2) 의존성 구성
        metric_selector = MetricSelector(self.registry)
        entity_resolver = EntityResolver(config.db_path)
        column_parser = ColumnParser(
            model=resolved_options.model,
            temperature=0.0,
            schema_store=self.schema_store,
        )
        self._dependencies: AgentDependencies = AgentDependencies(
            registry=self.registry,
            schema_store=self.schema_store,
            router=RouterLLM(model=resolved_options.model, temperature=resolved_options.router_temperature),
            responder=Responder(
                ResponderConfig(model=final_answer_model, temperature=resolved_options.responder_temperature)
            ),
            planner=Planner(
                self.registry,
                model=resolved_options.model,
                temperature=0.0,
                metric_selector=metric_selector,
                entity_resolver=entity_resolver,
                enable_tools=True,
            ),
            fewshot_generator=FewshotGenerator(model=resolved_options.model, temperature=0.0),
            fewshot_candidate_limit=config.fewshot_candidate_limit,
            multi_step_planner=MultiStepPlanner(model=resolved_options.model, temperature=0.0),
            sql_generator=SQLGenerator(
                model=resolved_options.model,
                temperature=resolved_options.temperature,
                column_parser=column_parser,
                max_tool_attempts=3,
            ),
            guard=SQLGuard(config.schema_json_path),
            validator=ResultValidator(),
            sqlite_client=SQLiteClient(config.db_path),
            summarizer=Summarizer(model=final_answer_model, temperature=resolved_options.summarizer_temperature),
            memory=self.memory,
        )

        # 3) 체인 컴파일
        self._chain: Any = build_agent_chain(self._dependencies)

    @staticmethod
    def create_memory() -> ConversationMemory:
        """
        대화 메모리를 생성한다.

        Returns:
            ConversationMemory 객체.
        """

        return ConversationMemory()

    async def invoke(self, user_message: str) -> dict[str, object]:
        """
        에이전트를 실행한다.

        Args:
            user_message: 사용자 질문.

        Returns:
            실행 결과 딕셔너리.
        """

        # 1) 턴 시작: 질문 원문을 단기 메모리에 먼저 기록한다.
        self.memory.start_turn(user_message)

        # 2) 체인 실행: 내부에서 SQL/결과/슬롯 같은 구조화 데이터가 메모리에 업데이트된다.
        result = await self._invoke_chain(user_message)
        if result.get("route") == "error":
            await self.memory.finish_turn_async(
                assistant_message=str(result.get("final_answer")),
                route=str(result.get("route")),
                sql=None,
                planned_slots=None,
            )
            return result

        # 3) 턴 마무리: 최종 응답과 함께 장기 메모리 학습(선호/기본값)을 갱신한다.
        final_answer = result.get("final_answer")
        planned_slots = result.get("planned_slots")
        await self.memory.finish_turn_async(
            assistant_message=str(final_answer) if isinstance(final_answer, str) else None,
            route=str(result.get("route")) if result.get("route") else None,
            sql=str(result.get("sql")) if isinstance(result.get("sql"), str) else None,
            planned_slots=planned_slots if isinstance(planned_slots, dict) else None,
        )
        return result

    async def stream(self, user_message: str) -> AsyncIterator[dict[str, object]]:
        """
        LangGraph 스트리밍 API를 사용해 단계별 상태를 방출한다.

        Args:
            user_message: 사용자 질문.

        Yields:
            부분 상태 딕셔너리.
        """

        if not hasattr(self._chain, "astream"):
            # 스트리밍을 지원하지 않는 경우 폴백
            yield {"event_type": "final", "node": None, "state": await self.invoke(user_message)}
            return

        # 스트리밍에서는 메모리 턴 마킹만 하고, finish_turn은 최종 상태에서 처리한다.
        self.memory.start_turn(user_message)
        merged_state: dict[str, object] = {"user_message": user_message}
        last_state: dict[str, object] | None = None

        stream_mode = "debug"
        try:
            stream_iter = self._chain.astream({"user_message": user_message}, stream_mode=stream_mode)
        except TypeError:
            stream_mode = "updates"
            stream_iter = self._chain.astream({"user_message": user_message}, stream_mode=stream_mode)
        except Exception:
            fallback = await self._invoke_chain(user_message)
            await self._finish_stream_turn(fallback)
            yield {"event_type": "final", "node": None, "state": fallback}
            return

        if stream_mode == "debug":
            async for event in stream_iter:
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                node_name = payload.get("name")

                if event_type == "task_result":
                    result = payload.get("result")
                    if isinstance(result, dict):
                        merged_state.update(result)
                        last_state = dict(merged_state)

                if event_type in {"task", "task_result"}:
                    yield {
                        "event_type": event_type,
                        "node": node_name,
                        "state": dict(merged_state),
                    }
        else:
            async for state in stream_iter:
                if not isinstance(state, dict):
                    continue
                merged_state.update(state)
                last_state = dict(merged_state)
                yield {"event_type": "state", "node": None, "state": dict(merged_state)}

        if last_state is None:
            fallback = await self._invoke_chain(user_message)
            await self._finish_stream_turn(fallback)
            yield {"event_type": "final", "node": None, "state": fallback}
            return

        await self._finish_stream_turn(last_state)
        yield {"event_type": "final", "node": None, "state": last_state}

    async def _invoke_chain(self, user_message: str) -> dict[str, object]:
        """
        체인을 실행하고 결과를 반환한다.

        Args:
            user_message: 사용자 질문.

        Returns:
            실행 결과 딕셔너리.
        """

        try:
            return await self._chain.ainvoke({"user_message": user_message})
        except Exception as exc:  # noqa: BLE001
            return {
                "route": "error",
                "error": f"오케스트레이터 실행 실패: {exc}",
                "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
                "final_answer": "요청을 처리하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            }

    async def _finish_stream_turn(self, state: dict[str, object]) -> None:
        """
        스트리밍 종료 시 메모리 턴을 마무리한다.

        Args:
            state: 최종 상태.

        Returns:
            None
        """

        final_answer = state.get("final_answer")
        final_answer_stream = state.get("final_answer_stream")
        if not isinstance(final_answer, str) and final_answer_stream is not None:
            return
        planned_slots = state.get("planned_slots")
        await self.memory.finish_turn_async(
            assistant_message=str(final_answer) if isinstance(final_answer, str) else None,
            route=str(state.get("route")) if state.get("route") else None,
            sql=str(state.get("sql")) if isinstance(state.get("sql"), str) else None,
            planned_slots=planned_slots if isinstance(planned_slots, dict) else None,
        )

    def reset_memory(self) -> None:
        """
        대화 메모리를 초기화한다.

        Returns:
            None
        """

        self.memory.reset()

    def clear_long_term_memory(self) -> None:
        """
        장기 메모리를 초기화한다.

        Returns:
            None
        """

        self.memory.clear_long_term()

    def reset_all_memory(self) -> None:
        """
        단기/장기 메모리를 모두 초기화한다.

        Returns:
            None
        """

        self.memory.reset_all()

    def dump_schema(self) -> None:
        """
        스키마를 덤프하고 스토어를 갱신한다.

        Returns:
            None
        """

        dump_schema(
            SchemaDumpPaths(
                db_path=self._config.db_path,
                json_path=self._config.schema_json_path,
                md_path=self._config.schema_md_path,
            )
        )
        self.schema_store.load()


if __name__ == "__main__":
    async def _main() -> None:
        # 1) API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY가 없어 오케스트레이터 테스트를 건너뜁니다.")
            return

        # 2) 오케스트레이터 구성
        config = load_config()
        orchestrator = AgentOrchestrator(config=config)

        # 3) 샘플 질의 실행
        result = await orchestrator.invoke("최근 리그에서 승률 상위 5개 팀 알려줘")
        print(result.get("final_answer"))

    asyncio.run(_main())
