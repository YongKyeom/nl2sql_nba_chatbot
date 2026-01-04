# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit 실행 시 src 패키지를 인식하도록 경로를 보정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.scene import build_scene, ChatbotScene
from src.config import AppConfig, load_config
from src.utils.logging import JsonlLogger
from src.utils.time import Timer


def main() -> None:
    """
    Streamlit 엔트리 포인트.

    Returns:
        None
    """

    # 1) 설정/세션 초기화
    config = load_config()
    _init_session_state(config)

    # 2) 사이드바 설정
    st.sidebar.header("설정")
    st.sidebar.text(f"DB 경로: {config.db_path}")

    model = st.sidebar.selectbox("모델", ["gpt-4o-mini"], index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    # 3) Scene/오케스트레이터 구성(세션 내 동일 인스턴스 재사용)
    scene = _ensure_scene(config, model=model, temperature=temperature)
    orchestrator = scene.orchestrator
    logger = JsonlLogger(config.log_path)

    if st.sidebar.button("Dataset Info"):
        st.session_state.show_dataset_info = not st.session_state.show_dataset_info

    if st.session_state.show_dataset_info:
        _render_dataset_info(scene)

    if st.sidebar.button("Dump Schema"):
        orchestrator.dump_schema()
        st.success("schema.json / schema.md 생성 완료")

    if st.sidebar.button("Reset Conversation"):
        scene.reset()
        st.session_state.messages = []
        st.success("대화가 초기화되었습니다.")

    if st.sidebar.button("Reset Long-term Memory"):
        orchestrator.clear_long_term_memory()
        st.success("장기 메모리가 초기화되었습니다.")

    # 4) 대화 로그 렌더링
    st.title("NBA NL2SQL 챗봇")
    _render_messages(st.session_state.messages)

    # 5) 사용자 입력 처리
    if user_text := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with Timer() as timer:
                result = _run_agent_with_thinking(scene, user_text)
            _handle_agent_result(result, timer.elapsed_ms, logger, user_text, show_thinking=False)


def _init_session_state(config: AppConfig) -> None:
    """
    Streamlit 세션 상태를 초기화.

    Args:
        config: 앱 설정.

    Returns:
        None
    """

    if "scene" not in st.session_state:
        st.session_state.scene = None
    if st.session_state.scene is None:
        st.session_state.scene = build_scene(config, model="gpt-4o-mini", temperature=0.2)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_dataset_info" not in st.session_state:
        st.session_state.show_dataset_info = False


def _ensure_scene(config: AppConfig, *, model: str, temperature: float) -> ChatbotScene:
    """
    세션 상태의 Scene을 옵션에 맞게 준비한다.

    Args:
        config: 앱 설정.
        model: 선택된 모델.
        temperature: 선택된 temperature.

    Returns:
        ChatbotScene.
    """

    scene: ChatbotScene = st.session_state.scene
    if scene.options.model == model and scene.options.temperature == temperature:
        return scene

    st.session_state.scene = build_scene(config, model=model, temperature=temperature, memory=scene.memory)
    return st.session_state.scene


def _render_dataset_info(scene: ChatbotScene) -> None:
    """
    데이터셋 상세 정보를 표시.

    Args:
        scene: Scene.

    Returns:
        None
    """

    tables = scene.orchestrator.schema_store.list_tables()
    metric_names = [metric.name for metric in scene.orchestrator.registry.list_metrics()]

    with st.sidebar.expander("데이터셋 상세 설명", expanded=True):
        st.markdown(
            """
- 데이터 출처: Kaggle (wyattowalsh/basketball)
- 원본 CSV를 SQLite로 적재해 읽기 전용으로 조회합니다.
- 팀 경기 스탯이 중심이며, 선수 단위는 프로필/드래프트/컴바인 정보 위주입니다.
- 정규시즌/플레이오프는 season_type 기준으로 구분됩니다.
- 선수 맞대결은 play_by_play 이벤트 동시 관여 빈도로 근사합니다.
- 지표 정의는 metrics.yaml에서 주입되며, 변경 즉시 반영됩니다.
"""
        )
        st.markdown(f"**테이블 수**: {len(tables)}")
        st.markdown(f"**테이블 목록**: {', '.join(tables)}")
        st.markdown(f"**사용 가능한 지표**: {', '.join(metric_names)}")


def _render_messages(messages: list[dict[str, object]]) -> None:
    """
    저장된 메시지를 렌더링.

    Args:
        messages: 메시지 리스트.

    Returns:
        None
    """

    for idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            sql = message.get("sql")
            dataframe = message.get("dataframe")
            error = message.get("error")
            error_detail = message.get("error_detail")
            route = message.get("route")
            route_reason = message.get("route_reason")
            planned_slots = message.get("planned_slots")
            last_result_schema = message.get("last_result_schema")

            if message["role"] == "assistant" and (route or route_reason or planned_slots or last_result_schema):
                _render_thinking_panel(
                    route=route,
                    route_reason=route_reason,
                    planned_slots=planned_slots,
                    last_result_schema=last_result_schema,
                    sql=sql,
                    message_index=idx,
                )

            st.write(message["content"])

            if sql:
                st.code(sql, language="sql")
            if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                st.dataframe(dataframe, use_container_width=True)
                st.download_button(
                    "CSV 다운로드",
                    dataframe.to_csv(index=False).encode("utf-8"),
                    file_name="result.csv",
                    mime="text/csv",
                    key=f"download_{idx}",
                )
            if error:
                with st.expander("디버그 정보"):
                    st.write(error)
                    if error_detail:
                        st.write(error_detail)


def _handle_agent_result(
    result: dict[str, object],
    latency_ms: float | None,
    logger: JsonlLogger,
    user_text: str,
    show_thinking: bool = True,
) -> None:
    """
    에이전트 결과를 화면과 로그에 반영.

    Args:
        result: 에이전트 실행 결과.
        latency_ms: 처리 시간.
        logger: 로그 기록기.
        user_text: 사용자 입력.

    Returns:
        None
    """

    final_answer = result.get("final_answer", "응답을 생성하지 못했습니다.")
    sql = result.get("sql")
    dataframe = result.get("result_df")
    error = result.get("error")
    error_detail = result.get("error_detail")
    route = result.get("route", "unknown")
    route_reason = result.get("route_reason")
    planned_slots = result.get("planned_slots")

    assistant_message = {
        "role": "assistant",
        "content": final_answer,
        "sql": sql,
        "dataframe": dataframe,
        "error": error,
        "error_detail": error_detail,
        "route": route,
        "route_reason": route_reason,
        "planned_slots": planned_slots,
        "last_result_schema": result.get("last_result_schema"),
    }
    st.session_state.messages.append(assistant_message)
    message_index = len(st.session_state.messages) - 1

    if show_thinking:
        _render_thinking_panel(
            route=route,
            route_reason=route_reason,
            planned_slots=planned_slots,
            last_result_schema=result.get("last_result_schema"),
            sql=sql,
            message_index=message_index,
        )

    st.write_stream(_stream_text(final_answer))

    if sql:
        st.code(sql, language="sql")
    if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
        st.dataframe(dataframe, use_container_width=True)
        st.download_button(
            "CSV 다운로드",
            dataframe.to_csv(index=False).encode("utf-8"),
            file_name="result.csv",
            mime="text/csv",
            key=f"download_{message_index}",
        )

    if error:
        with st.expander("디버그 정보"):
            st.write(error)
            st.write({"route": route, "planned_slots": result.get("planned_slots")})
            if error_detail:
                st.write(error_detail)

    logger.log_event(
        user_text=user_text,
        route=route,
        sql=sql if isinstance(sql, str) else None,
        rows=int(len(dataframe)) if isinstance(dataframe, pd.DataFrame) else None,
        latency_ms=latency_ms,
        error=error if isinstance(error, str) else None,
        user_id=_get_user_id(),
    )


def _run_agent_with_thinking(scene: ChatbotScene, user_message: str) -> dict[str, object]:
    """
    스트리밍으로 Thinking 패널을 업데이트하며 에이전트를 실행한다.

    Args:
        scene: Scene 인스턴스.
        user_message: 사용자 입력.

    Returns:
        최종 결과 딕셔너리.
    """

    placeholder = st.empty()
    final_state: dict[str, object] | None = None

    try:
        for state in scene.orchestrator.stream(user_message):
            if not isinstance(state, dict):
                continue
            final_state = state
            with placeholder.container():
                _render_thinking_panel(
                    route=state.get("route"),
                    route_reason=state.get("route_reason"),
                    planned_slots=state.get("planned_slots"),
                    last_result_schema=state.get("last_result_schema"),
                    sql=state.get("sql"),
                    message_index=len(st.session_state.messages),
                )
    except Exception:
        placeholder.empty()
        return scene.ask(user_message)

    if final_state is None or final_state.get("final_answer") is None:
        placeholder.empty()
        return scene.ask(user_message)

    with placeholder.container():
        _render_thinking_panel(
            route=final_state.get("route"),
            route_reason=final_state.get("route_reason"),
            planned_slots=final_state.get("planned_slots"),
            last_result_schema=final_state.get("last_result_schema"),
            sql=final_state.get("sql"),
            message_index=len(st.session_state.messages),
        )

    return final_state


def _render_thinking_panel(
    *,
    route: str | None,
    route_reason: str | None,
    planned_slots: dict[str, object] | None,
    last_result_schema: list[str] | None,
    sql: str | None,
    message_index: int,
) -> None:
    """
    Thinking(단계별 상태)을 렌더링한다.

    Args:
        route: 라우팅 결과.
        route_reason: 라우팅 근거.
        planned_slots: 플래너 슬롯.
        last_result_schema: 직전 결과 스키마.
        sql: 생성/실행 SQL.
        message_index: 메시지 인덱스(고유 키용).

    Returns:
        None
    """

    with st.expander("Thinking", expanded=False):
        steps = [
            ("Routing", bool(route)),
            ("Planning", bool(planned_slots)),
            ("SQL", bool(sql)),
        ]
        for label, done in steps:
            icon = "✅" if done else "⏳"
            st.markdown(f"- {icon} {label}")

        if route_reason:
            st.caption(f"route_reason: {route_reason}")

        payload: dict[str, object] = {}
        if route:
            payload["route"] = route
        if planned_slots:
            payload["planned_slots"] = planned_slots
        if last_result_schema:
            payload["last_result_schema"] = last_result_schema
        if sql:
            payload["sql"] = sql

        if payload:
            st.json(payload, expanded=True)


def _get_user_id() -> str | None:
    """
    가능한 범위에서 사용자 식별자를 추출한다.

    Returns:
        사용자 ID(없으면 None).
    """

    user_id = st.session_state.get("user_id")
    if user_id:
        return str(user_id)

    if hasattr(st, "query_params"):
        params = st.query_params
        if "user_id" in params:
            value = params.get("user_id")
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)

    return None


def _stream_text(text: str, chunk_size: int = 24) -> Iterator[str]:
    """
    텍스트를 일정 크기씩 나눠 스트리밍용 청크로 반환한다.

    Args:
        text: 원본 텍스트.
        chunk_size: 한 번에 출력할 문자 수.

    Yields:
        텍스트 청크.
    """

    if chunk_size <= 0:
        yield text
        return

    for offset in range(0, len(text), chunk_size):
        yield text[offset : offset + chunk_size]


if __name__ == "__main__":
    # 1) Streamlit 앱 실행
    main()
