# ruff: noqa: E402, I001
from __future__ import annotations

import re
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
from src.utils.markdown import records_to_markdown
from src.utils.time import Timer


COLUMN_LABELS = {
    "team_name": "팀 이름",
    "team_abbreviation": "팀 약어",
    "team_abbreviation_home": "홈 팀",
    "team_abbreviation_away": "원정 팀",
    "season_year": "시즌",
    "season_id": "시즌 ID",
    "w": "승",
    "l": "패",
    "pct": "승률",
    "avg_attendance": "평균 관중",
    "total_attendance": "총 관중",
    "game_count": "경기 수",
    "player_name": "선수 이름",
    "overall_pick": "전체 픽",
    "game_date": "경기 날짜",
    "pts": "득점",
    "opp_pts": "실점",
}



def main() -> None:
    """
    Streamlit 엔트리 포인트.

    Returns:
        None
    """

    # 1) 설정/세션 초기화
    config = load_config()
    _init_session_state(config)
    _apply_custom_theme()

    # 2) 사이드바 설정
    st.sidebar.header("설정")
    st.sidebar.caption(f"DB: {config.db_path}")

    st.sidebar.markdown("### 모델")
    model = st.sidebar.selectbox("모델", ["gpt-4o-mini"], index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    # 3) Scene/오케스트레이터 구성(세션 내 동일 인스턴스 재사용)
    scene = _ensure_scene(config, model=model, temperature=temperature)
    orchestrator = scene.orchestrator
    logger = JsonlLogger(config.log_path)

    st.sidebar.divider()
    st.sidebar.markdown("### 데이터")
    if st.sidebar.button("Dataset Info"):
        st.session_state.show_dataset_info = not st.session_state.show_dataset_info

    if st.session_state.show_dataset_info:
        _render_dataset_info(scene)

    st.sidebar.divider()
    st.sidebar.markdown("### 유틸")
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
    _render_hero()
    quick_prompt = _render_quick_prompts()
    _render_messages(st.session_state.messages)

    # 5) 사용자 입력 처리
    user_text = st.chat_input("질문을 입력하세요")
    if quick_prompt and not user_text:
        user_text = quick_prompt

    if user_text:
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


def _render_hero() -> None:
    """
    헤더 영역을 렌더링한다.

    Returns:
        None
    """

    st.markdown(
        """
<div class="hero">
  <div class="hero-title">NBA NL2SQL 챗봇</div>
  <div class="hero-subtitle">NBA 데이터 질의를 자연어로 입력하면 SQL 실행 결과와 요약을 제공합니다.</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_quick_prompts() -> str | None:
    """
    빠른 질문 버튼을 렌더링한다.

    Returns:
        선택된 질문(없으면 None).
    """

    prompts = [
        ("2023-24 득점 Top10", "2023-24 시즌 팀 득점 상위 10개 보여줘"),
        ("승률 상위 5팀", "최근 리그에서 승률 상위 5개 팀 알려줘"),
        ("2018 드래프트", "2018 드래프트 전체 픽 리스트 보여줘"),
        ("LAL 최근 5경기", "LAL 최근 5경기 결과 알려줘"),
        ("관중 상위 팀 분석", "관중 수 상위 10개 팀의 시즌 승률/순위를 같이 보여줘"),
        ("관중 Top10 + 순위", "관중 수 상위 10개 팀의 최근 리그 순위를 분석해줘"),
        ("관중 Top5 + 승률", "관중 수 상위 5개 팀의 승률과 순위를 비교해줘"),
        ("관중 Top10 + 성적", "관중 수 상위 10팀의 시즌 성적과 순위를 정리해줘"),
        ("관중 상위 팀 승패", "관중 수 상위 10개 팀의 승패와 리그 순위를 알려줘"),
        ("관중 상위 팀 요약", "관중 상위 10개 팀의 시즌 승률을 표로 보여줘"),
    ]

    st.markdown('<div class="section-label">추천 질의</div>', unsafe_allow_html=True)
    selection: str | None = None
    cols = st.columns(3)
    for idx, (label, question) in enumerate(prompts):
        with cols[idx % 3]:
            if st.button(label, key=f"quick_prompt_{idx}"):
                selection = question
    return selection


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

            if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                _render_result_summary(dataframe, planned_slots)
                display_df = _prepare_dataframe_for_display(dataframe, True)
                if not _has_markdown_table(str(message.get("content", ""))):
                    st.dataframe(display_df, use_container_width=True)
                st.download_button(
                    "CSV 다운로드",
                    display_df.to_csv(index=False).encode("utf-8"),
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

    display_df: pd.DataFrame | None = None
    if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
        display_df = _prepare_dataframe_for_display(dataframe, True)
        table_markdown = _dataframe_to_markdown(display_df)
        final_answer = _merge_markdown_table(final_answer, table_markdown)

    assistant_message = {
        "role": "assistant",
        "content": final_answer,
        "sql": sql,
        "dataframe": display_df if display_df is not None else dataframe,
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

    if isinstance(display_df, pd.DataFrame) and not display_df.empty:
        _render_result_summary(display_df, planned_slots)
        if not _has_markdown_table(str(final_answer)):
            st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "CSV 다운로드",
            display_df.to_csv(index=False).encode("utf-8"),
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

    st.markdown('<span class="thinking-pill">Thinking</span>', unsafe_allow_html=True)
    with st.expander("Details", expanded=False):
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


def _render_result_summary(dataframe: pd.DataFrame, planned_slots: dict[str, object] | None) -> None:
    """
    결과 요약 카드를 렌더링한다.

    Args:
        dataframe: 결과 데이터프레임.
        planned_slots: 플래너 슬롯.

    Returns:
        None
    """

    planned_slots = planned_slots or {}
    season = planned_slots.get("season") or "-"
    metric = planned_slots.get("metric") or "-"
    top_k = planned_slots.get("top_k") or "-"

    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(dataframe):,}")
    cols[1].metric("Season", str(season))
    cols[2].metric("Metric", str(metric))
    cols[3].metric("Top-K", str(top_k))

    filters = planned_slots.get("filters")
    if isinstance(filters, dict) and filters:
        filters_text = ", ".join(f"{key}={value}" for key, value in filters.items())
        st.caption(f"필터: {filters_text}")


def _prepare_dataframe_for_display(dataframe: pd.DataFrame, use_friendly_columns: bool) -> pd.DataFrame:
    """
    화면 표시용 데이터프레임을 준비한다.

    Args:
        dataframe: 원본 데이터프레임.
        use_friendly_columns: 컬럼명 변환 여부.

    Returns:
        표시용 데이터프레임.
    """

    if not use_friendly_columns:
        return dataframe

    rename_map = {col: COLUMN_LABELS.get(col, col) for col in dataframe.columns}
    return dataframe.rename(columns=rename_map)


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """
    데이터프레임을 마크다운 테이블로 변환한다.

    Args:
        dataframe: 결과 데이터프레임.

    Returns:
        마크다운 테이블 문자열.
    """

    records = dataframe.to_dict(orient="records")
    return records_to_markdown(records)


def _merge_markdown_table(text: str, table_markdown: str) -> str:
    """
    기존 답변에 마크다운 테이블을 주입하거나 교체한다.

    Args:
        text: 원본 응답 텍스트.
        table_markdown: 교체할 테이블 문자열.

    Returns:
        테이블이 포함된 응답 텍스트.
    """

    lines = text.splitlines()
    table_range = _find_markdown_table_range(lines)
    if table_range:
        start, end = table_range
        merged = lines[:start] + table_markdown.splitlines() + lines[end:]
        return "\n".join(merged).strip()

    for idx, line in enumerate(lines):
        if "📌 조회 결과" in line:
            merged = lines[: idx + 1] + ["", table_markdown, ""] + lines[idx + 1 :]
            return "\n".join(merged).strip()

    return (text.rstrip() + "\n\n" + table_markdown).strip()


def _find_markdown_table_range(lines: list[str]) -> tuple[int, int] | None:
    """
    마크다운 테이블 영역을 찾는다.

    Args:
        lines: 텍스트 라인 목록.

    Returns:
        (start, end) 범위 또는 None.
    """

    for idx in range(len(lines) - 1):
        if "|" in lines[idx] and _is_markdown_separator(lines[idx + 1]):
            end = idx + 2
            while end < len(lines) and "|" in lines[end]:
                end += 1
            return idx, end
    return None


def _has_markdown_table(text: str) -> bool:
    """
    마크다운 테이블 포함 여부를 확인한다.

    Args:
        text: 응답 텍스트.

    Returns:
        테이블이 있으면 True.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    for idx in range(len(lines) - 1):
        if "|" in lines[idx] and _is_markdown_separator(lines[idx + 1]):
            return True
    return False


def _is_markdown_separator(line: str) -> bool:
    """
    마크다운 테이블 구분선인지 확인한다.

    Args:
        line: 한 줄 문자열.

    Returns:
        구분선이면 True.
    """

    return bool(re.match(r"^\|?\s*[-:|\s]+\s*\|?$", line.strip()))


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


def _apply_custom_theme() -> None:
    """
    커스텀 UI 테마를 적용한다.

    Returns:
        None
    """

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Space Grotesk", sans-serif;
}

.stApp {
  background: #f8fafc;
}

.hero {
  padding: 0.5rem 0 0.2rem;
}

.hero-title {
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0f172a;
}

.hero-subtitle {
  margin-top: 0.35rem;
  font-size: 1.05rem;
  color: #475569;
}

.section-label {
  margin: 1.2rem 0 0.6rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #0f172a;
}

.thinking-pill {
  display: inline-block;
  margin: 0.4rem 0 0.6rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  font-size: 0.8rem;
  font-weight: 600;
  color: #475569;
}

.stButton>button {
  border-radius: 999px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  color: #0f172a;
  font-weight: 600;
  padding: 0.35rem 0.9rem;
}

.stButton>button:hover {
  border-color: #0f172a;
  color: #0f172a;
}
</style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # 1) Streamlit 앱 실행
    main()
