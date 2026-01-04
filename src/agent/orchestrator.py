from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.chain import AgentDependencies, build_agent_chain
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.planner import Planner
from src.agent.responder import Responder, ResponderConfig
from src.agent.router import RouterLLM
from src.agent.sql_generator import SQLGenerator
from src.agent.summarizer import Summarizer
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
        model: 사용할 LLM 모델명.
        temperature: SQL 생성 temperature.
        router_temperature: 라우터 temperature.
        responder_temperature: 답변 생성 temperature.
        summarizer_temperature: 요약 생성 temperature.
    """

    model: str
    temperature: float
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
        self._dependencies: AgentDependencies = AgentDependencies(
            registry=self.registry,
            schema_store=self.schema_store,
            router=RouterLLM(model=resolved_options.model, temperature=resolved_options.router_temperature),
            responder=Responder(
                ResponderConfig(model=resolved_options.model, temperature=resolved_options.responder_temperature)
            ),
            planner=Planner(self.registry),
            sql_generator=SQLGenerator(model=resolved_options.model, temperature=resolved_options.temperature),
            guard=SQLGuard(config.schema_json_path),
            validator=ResultValidator(),
            sqlite_client=SQLiteClient(config.db_path),
            summarizer=Summarizer(model=resolved_options.model, temperature=resolved_options.summarizer_temperature),
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

    def invoke(self, user_message: str) -> dict[str, object]:
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
        try:
            result: dict[str, object] = self._chain.invoke({"user_message": user_message})
        except Exception as exc:  # noqa: BLE001
            failure = {
                "route": "error",
                "error": f"오케스트레이터 실행 실패: {exc}",
                "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
                "final_answer": "요청을 처리하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            }
            self.memory.finish_turn(
                assistant_message=str(failure.get("final_answer")),
                route=str(failure.get("route")),
                sql=None,
                planned_slots=None,
            )
            return failure

        # 3) 턴 마무리: 최종 응답과 함께 장기 메모리 학습(선호/기본값)을 갱신한다.
        final_answer = result.get("final_answer")
        planned_slots = result.get("planned_slots")
        self.memory.finish_turn(
            assistant_message=str(final_answer) if isinstance(final_answer, str) else None,
            route=str(result.get("route")) if result.get("route") else None,
            sql=str(result.get("sql")) if isinstance(result.get("sql"), str) else None,
            planned_slots=planned_slots if isinstance(planned_slots, dict) else None,
        )
        return result

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
    # 1) API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없어 오케스트레이터 테스트를 건너뜁니다.")
    else:
        # 2) 오케스트레이터 구성
        config = load_config()
        orchestrator = AgentOrchestrator(config=config)

        # 3) 샘플 질의 실행
        result = orchestrator.invoke("최근 리그에서 승률 상위 5개 팀 알려줘")
        print(result.get("final_answer"))
