"""
Streamlit에서 에이전트를 실행하기 위한 Chatbot Scene 레이어.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.memory import ConversationMemory
from src.agent.orchestrator import AgentOrchestrator, OrchestratorOptions
from src.config import AppConfig


@dataclass(slots=True)
class ChatbotScene:
    """
    실행 단위를 캡슐화한 Scene.

    Args:
        config: 앱 설정.
        options: 오케스트레이터 옵션.
        orchestrator: 오케스트레이터 인스턴스.
    """

    config: AppConfig
    options: OrchestratorOptions
    orchestrator: AgentOrchestrator

    @property
    def memory(self) -> ConversationMemory:
        """
        Scene이 사용하는 대화 메모리를 반환한다.

        Returns:
            ConversationMemory.
        """

        return self.orchestrator.memory

    def ask(self, user_message: str) -> dict[str, object]:
        """
        사용자 질문을 실행하고 결과를 반환한다.

        Args:
            user_message: 사용자 질문.

        Returns:
            실행 결과 딕셔너리.
        """

        return self.orchestrator.invoke(user_message)

    def reset(self) -> None:
        """
        대화 메모리를 초기화한다.

        Returns:
            None
        """

        self.orchestrator.reset_memory()


def build_scene(
    config: AppConfig,
    *,
    model: str,
    temperature: float,
    memory: ConversationMemory | None = None,
) -> ChatbotScene:
    """
    Streamlit/CLI에서 공통으로 사용할 Scene을 생성한다.

    Args:
        config: 앱 설정.
        model: 사용할 모델명.
        temperature: SQL 생성 temperature.
        memory: 대화 메모리(없으면 새로 생성).

    Returns:
        ChatbotScene.
    """

    resolved_memory = memory or ConversationMemory.persistent(config.memory_db_path)
    options = OrchestratorOptions(model=model, temperature=temperature)
    orchestrator = AgentOrchestrator(config=config, options=options, memory=resolved_memory)
    return ChatbotScene(config=config, options=options, orchestrator=orchestrator)
