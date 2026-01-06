from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMToolFunction:
    """
    도구 호출 함수 정보.

    Args:
        name: 함수 이름.
        arguments: 함수 인자(JSON 문자열).
    """

    name: str
    arguments: str


@dataclass(frozen=True)
class LLMToolCall:
    """
    LLM 도구 호출 정보.

    Args:
        id: 호출 ID.
        function: 호출 함수 정보.
    """

    id: str
    function: LLMToolFunction


@dataclass(frozen=True)
class LLMMessage:
    """
    LLM 메시지 정보.

    Args:
        role: 역할(system/user/assistant).
        content: 메시지 본문.
        tool_calls: 도구 호출 목록.
    """

    role: str
    content: str | None
    tool_calls: list[LLMToolCall] | None = None


@dataclass(frozen=True)
class LLMChoice:
    """
    LLM 응답 choice.

    Args:
        message: 응답 메시지.
    """

    message: LLMMessage


@dataclass(frozen=True)
class LLMResponse:
    """
    LLM 응답 래퍼.

    Args:
        choices: 응답 choices.
        raw: 원본 응답 객체.
    """

    choices: list[LLMChoice]
    raw: Any


class LLMOperator:
    """
    LLM 호출을 모듈화한 래퍼.

    OpenAI 호환 호출을 내부에서 감싼다.
    """

    def __init__(self, *, client: OpenAI | None = None) -> None:
        """
        LLMOperator 초기화.

        Args:
            client: OpenAI 클라이언트(없으면 기본 생성).
        """

        self._client = client or OpenAI()

    def invoke(self, **kwargs: Any) -> LLMResponse:
        """
        LLM 요청을 수행하고 응답을 반환한다.

        Args:
            **kwargs: OpenAI 호환 파라미터.

        Returns:
            LLMResponse.
        """

        response = self._client.chat.completions.create(**kwargs)
        return _convert_response(response)

    def stream(self, **kwargs: Any) -> Iterator[str]:
        """
        LLM 응답을 스트리밍으로 반환한다.

        Args:
            **kwargs: OpenAI 호환 파라미터.

        Yields:
            텍스트 델타 문자열.
        """

        request = {**kwargs, "stream": True}
        stream = self._client.chat.completions.create(**request)
        for event in stream:
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield str(content)


def _convert_response(response: Any) -> LLMResponse:
    """
    OpenAI 응답을 LLMResponse로 변환한다.

    Args:
        response: OpenAI 응답 객체.

    Returns:
        LLMResponse.
    """

    choices: list[LLMChoice] = []
    for choice in response.choices:
        message = choice.message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                LLMToolCall(
                    id=call.id,
                    function=LLMToolFunction(
                        name=call.function.name,
                        arguments=call.function.arguments,
                    ),
                )
                for call in message.tool_calls
            ]
        choices.append(
            LLMChoice(
                message=LLMMessage(
                    role=str(message.role),
                    content=message.content,
                    tool_calls=tool_calls,
                )
            )
        )
    return LLMResponse(choices=choices, raw=response)
