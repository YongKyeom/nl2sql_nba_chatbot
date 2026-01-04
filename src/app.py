# ruff: noqa: E402, I001
from __future__ import annotations

import base64
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit 실행 시 src 패키지를 인식하도록 경로를 보정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.scene import ChatbotScene, build_scene
from src.agent.title_generator import TitleGenerator
from src.chart.generator import ChartGenerator
from src.chart.renderer import ChartRenderer
from src.chart.types import ChartSpec, SUPPORTED_CHART_TYPES
from src.config import AppConfig, load_config
from src.db.chat_store import ChatSession, ChatStore
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

QUICK_PROMPTS = [
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

DEFAULT_USER_ID = "developer"


class StreamlitChatApp:
    """
    Streamlit 기반 NBA NL2SQL 앱을 구성하고 실행한다.

    Args:
        config: 앱 설정.

    Side Effects:
        Streamlit UI 렌더링과 세션 상태를 갱신한다.

    Raises:
        예외 없음.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        앱 인스턴스를 초기화한다.

        Args:
            config: 앱 설정.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        self._config = config

    def main(self) -> None:
        """
        Streamlit 엔트리 포인트를 실행한다.

        Side Effects:
            Streamlit UI 렌더링, 세션 상태 갱신.

        Raises:
            예외 없음.
        """

        # 1) 상태/테마 초기화: 세션 유지와 UI 일관성을 확보한다.
        self._init_session_state()
        self._apply_custom_theme()

        # 2) 사용자/채팅 세션 준비: 채팅 목록과 세션 복원을 선행한다.
        user_id = self._get_user_id()
        chat_store = self._ensure_chat_store()
        active_chat_id = self._ensure_active_chat(chat_store, user_id)
        selected_chat_id = self._render_chat_sidebar(chat_store, user_id, active_chat_id)

        # 3) 설정/모델 UI: 모델과 온도 변경을 먼저 반영한다.
        st.sidebar.divider()
        st.sidebar.header("설정")
        st.sidebar.caption(f"DB: {self._config.db_path}")

        st.sidebar.markdown("### 모델")
        model_options = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
        default_model = self._config.model
        default_index = model_options.index(default_model) if default_model in model_options else 0
        model = st.sidebar.selectbox("모델", model_options, index=default_index)
        temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        scene = self._ensure_scene(model, temperature)
        orchestrator = scene.orchestrator
        logger = JsonlLogger(self._config.log_path)
        title_generator = self._ensure_title_generator(model)
        chart_generator = self._ensure_chart_generator(model)
        chart_renderer = self._ensure_chart_renderer()

        # 4) 채팅 전환 처리: 선택된 채팅의 메시지와 메모리를 복원한다.
        if selected_chat_id != st.session_state.loaded_chat_id:
            st.session_state.active_chat_id = selected_chat_id
            st.session_state.messages = self._load_chat_messages(chat_store, user_id, selected_chat_id)
            self._restore_memory_from_messages(scene, st.session_state.messages)
            st.session_state.loaded_chat_id = selected_chat_id

        # 5) 데이터/유틸 UI: 데이터셋 정보와 스키마 덤프를 제공한다.
        st.sidebar.divider()
        st.sidebar.markdown("### 데이터")
        if st.sidebar.button("Dataset Info"):
            st.session_state.show_dataset_info = not st.session_state.show_dataset_info

        if st.session_state.show_dataset_info:
            self._render_dataset_info(scene)

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

        # 6) 메인 화면: 헤더와 추천 질의를 보여준다.
        self._render_hero()
        quick_prompt = self._render_quick_prompts()
        self._render_messages(st.session_state.messages, chart_renderer=chart_renderer)

        # 7) 입력 처리: 질문 저장 -> 실행 -> 결과 반영 순서로 처리한다.
        user_text = st.chat_input("질문을 입력하세요")
        if quick_prompt and not user_text:
            user_text = quick_prompt

        if user_text:
            chat_store.add_message(user_id=user_id, chat_id=st.session_state.active_chat_id, role="user", content=user_text)
            self._maybe_set_chat_title(chat_store, title_generator, user_id, st.session_state.active_chat_id, user_text)
            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.chat_message("user"):
                st.write(user_text)

            with st.chat_message("assistant"):
                with Timer() as timer:
                    result = self._run_agent_with_thinking(scene, user_text)
                self._handle_agent_result(
                    result,
                    timer.elapsed_ms,
                    logger,
                    user_text,
                    chart_generator=chart_generator,
                    chart_renderer=chart_renderer,
                    chat_store=chat_store,
                    user_id=user_id,
                    chat_id=st.session_state.active_chat_id,
                    show_thinking=False,
                )

            if st.session_state.title_refresh_needed:
                st.session_state.title_refresh_needed = False
                st.rerun()

    def _init_session_state(self) -> None:
        """
        Streamlit 세션 상태를 초기화한다.

        Side Effects:
            st.session_state에 기본 키를 생성한다.

        Raises:
            예외 없음.
        """

        if "scene" not in st.session_state:
            st.session_state.scene = None
        if st.session_state.scene is None:
            st.session_state.scene = build_scene(
                self._config,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "show_dataset_info" not in st.session_state:
            st.session_state.show_dataset_info = False
        if "chat_store" not in st.session_state:
            st.session_state.chat_store = None
        if "title_generator" not in st.session_state:
            st.session_state.title_generator = None
        if "chart_generator" not in st.session_state:
            st.session_state.chart_generator = None
        if "chart_renderer" not in st.session_state:
            st.session_state.chart_renderer = None
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = []
        if "active_chat_id" not in st.session_state:
            st.session_state.active_chat_id = None
        if "loaded_chat_id" not in st.session_state:
            st.session_state.loaded_chat_id = None
        if "title_refresh_needed" not in st.session_state:
            st.session_state.title_refresh_needed = False

    def _ensure_scene(self, model: str, temperature: float) -> ChatbotScene:
        """
        세션 상태의 Scene을 옵션에 맞게 준비한다.

        Args:
            model: 사용할 모델명.
            temperature: temperature 값.

        Returns:
            ChatbotScene.

        Side Effects:
            st.session_state.scene을 갱신할 수 있다.

        Raises:
            예외 없음.
        """

        scene: ChatbotScene = st.session_state.scene
        if scene.options.model == model and scene.options.temperature == temperature:
            return scene

        st.session_state.scene = build_scene(self._config, model=model, temperature=temperature, memory=scene.memory)
        return st.session_state.scene

    def _ensure_chat_store(self) -> ChatStore:
        """
        ChatStore를 세션에 준비한다.

        Returns:
            ChatStore.

        Side Effects:
            st.session_state.chat_store를 갱신할 수 있다.

        Raises:
            예외 없음.
        """

        if st.session_state.chat_store is None:
            st.session_state.chat_store = ChatStore(self._config.chat_db_path)
        return st.session_state.chat_store

    def _ensure_title_generator(self, model: str) -> TitleGenerator:
        """
        제목 생성기를 세션에 준비한다.

        Args:
            model: 사용할 모델명.

        Returns:
            TitleGenerator.

        Side Effects:
            st.session_state.title_generator를 갱신할 수 있다.

        Raises:
            예외 없음.
        """

        generator = st.session_state.title_generator
        if generator is None or generator.model != model:
            st.session_state.title_generator = TitleGenerator(model=model, temperature=0.2)
        return st.session_state.title_generator

    def _ensure_chart_generator(self, model: str) -> ChartGenerator:
        """
        차트 생성기를 세션에 준비한다.

        Args:
            model: 사용할 모델명.

        Returns:
            ChartGenerator.

        Side Effects:
            st.session_state.chart_generator를 갱신할 수 있다.

        Raises:
            예외 없음.
        """

        generator = st.session_state.chart_generator
        if generator is None or generator.model != model:
            st.session_state.chart_generator = ChartGenerator(model=model, temperature=0.2)
        return st.session_state.chart_generator

    def _ensure_chart_renderer(self) -> ChartRenderer:
        """
        차트 렌더러를 세션에 준비한다.

        Returns:
            ChartRenderer.

        Side Effects:
            st.session_state.chart_renderer를 갱신할 수 있다.

        Raises:
            예외 없음.
        """

        renderer = st.session_state.chart_renderer
        if renderer is None:
            st.session_state.chart_renderer = ChartRenderer()
        return st.session_state.chart_renderer

    def _ensure_active_chat(self, chat_store: ChatStore, user_id: str) -> str:
        """
        활성 채팅을 확보한다.

        Args:
            chat_store: 채팅 저장소.
            user_id: 사용자 ID.

        Returns:
            활성 chat_id.

        Side Effects:
            st.session_state.chat_sessions/active_chat_id를 갱신한다.

        Raises:
            예외 없음.
        """

        sessions = chat_store.list_sessions(user_id)
        if not sessions:
            chat_id = chat_store.create_session(user_id, title="새 대화")
            sessions = chat_store.list_sessions(user_id)
            st.session_state.chat_sessions = sessions
            st.session_state.active_chat_id = chat_id
            return chat_id

        st.session_state.chat_sessions = sessions
        valid_ids = [session.chat_id for session in sessions]
        active_chat_id = st.session_state.active_chat_id
        if active_chat_id not in valid_ids:
            active_chat_id = valid_ids[0]
            st.session_state.active_chat_id = active_chat_id
        return active_chat_id

    def _render_chat_sidebar(self, chat_store: ChatStore, user_id: str, active_chat_id: str) -> str:
        """
        사이드바 채팅 목록을 렌더링한다.

        Args:
            chat_store: 채팅 저장소.
            user_id: 사용자 ID.
            active_chat_id: 현재 활성 채팅 ID.

        Returns:
            선택된 chat_id.

        Side Effects:
            Streamlit 사이드바에 채팅 목록을 표시한다.

        Raises:
            예외 없음.
        """

        st.sidebar.header("채팅")
        if st.sidebar.button("새 채팅"):
            new_chat_id = chat_store.create_session(user_id, title="새 대화")
            st.session_state.active_chat_id = new_chat_id
            st.session_state.loaded_chat_id = None
            st.session_state.messages = []
            st.session_state.chat_sessions = chat_store.list_sessions(user_id)
            return new_chat_id

        sessions = chat_store.list_sessions(user_id)
        st.session_state.chat_sessions = sessions
        if not sessions:
            return active_chat_id

        options = [session.chat_id for session in sessions]
        if active_chat_id not in options:
            active_chat_id = options[0]

        selected = active_chat_id
        for session in sessions:
            cols = st.sidebar.columns([10, 1])
            prefix = "●" if session.chat_id == active_chat_id else "○"
            label = f"{prefix} {session.title}"
            if cols[0].button(label, key=f"chat_select_{session.chat_id}", use_container_width=True):
                selected = session.chat_id
            self._render_chat_actions(cols[1], chat_store, user_id, session)

        st.session_state.active_chat_id = selected
        return selected

    def _render_chat_actions(self, column: Any, chat_store: ChatStore, user_id: str, session: ChatSession) -> None:
        """
        채팅 항목의 액션 메뉴를 렌더링한다.

        Args:
            column: 렌더링할 컬럼.
            chat_store: 채팅 저장소.
            user_id: 사용자 ID.
            session: 채팅 세션.

        Side Effects:
            채팅 제목 변경/삭제 UI를 렌더링한다.

        Raises:
            예외 없음.
        """

        chat_id = session.chat_id
        title_value = session.title

        with column:
            menu_factory = getattr(st, "popover", None)
            if menu_factory:
                menu = menu_factory("⋯", use_container_width=False)
            else:
                menu = st.expander("⋯", expanded=False)

        with menu:
            new_title = st.text_input(
                "채팅 제목",
                value=title_value,
                key=f"title_input_{chat_id}",
            )
            if st.button("이름 바꾸기", key=f"update_title_{chat_id}"):
                if new_title.strip():
                    chat_store.update_title(user_id, chat_id, new_title.strip())
                    st.session_state.chat_sessions = chat_store.list_sessions(user_id)
                    st.rerun()

            confirm = st.checkbox("삭제 확인", key=f"confirm_delete_{chat_id}")
            if st.button("삭제", key=f"delete_chat_{chat_id}"):
                if confirm:
                    chat_store.delete_session(user_id, chat_id)
                    st.session_state.active_chat_id = None
                    st.session_state.loaded_chat_id = None
                    st.session_state.messages = []
                    st.session_state.chat_sessions = chat_store.list_sessions(user_id)
                    st.rerun()

    def _render_dataset_info(self, scene: ChatbotScene) -> None:
        """
        데이터셋 상세 정보를 표시한다.

        Args:
            scene: Scene.

        Side Effects:
            사이드바에 데이터셋 설명을 렌더링한다.

        Raises:
            예외 없음.
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

    def _render_hero(self) -> None:
        """
        헤더 영역을 렌더링한다.

        Side Effects:
            메인 화면 상단에 로고와 타이틀을 표시한다.

        Raises:
            예외 없음.
        """

        logo_uri = self._build_logo_data_uri(Path("data/nba_image.png"))
        logo_html = f'<img src="{logo_uri}" class="hero-logo" />' if logo_uri else ""
        st.markdown(
            f"""
<div class="hero-wrap">
  {logo_html}
  <div class="hero">
    <div class="hero-title">NL2SQL 기반 NBA 챗봇</div>
    <div class="hero-subtitle">자연어 질의를 하면 AI 에이전트가 스키마를 읽고 데이터를 종합 분석해 핵심을 알려드립니다.</div>
  </div>
</div>
        """,
            unsafe_allow_html=True,
        )

    def _render_quick_prompts(self) -> str | None:
        """
        추천 질의 버튼을 렌더링한다.

        Returns:
            선택된 질문(없으면 None).

        Side Effects:
            추천 질의 버튼 UI를 렌더링한다.

        Raises:
            예외 없음.
        """

        st.markdown('<div class="section-label">추천 질의</div>', unsafe_allow_html=True)
        selection: str | None = None
        cols = st.columns(3)
        for idx, (label, question) in enumerate(QUICK_PROMPTS):
            with cols[idx % 3]:
                if st.button(label, key=f"quick_prompt_{idx}"):
                    selection = question
        return selection

    def _render_messages(self, messages: list[dict[str, object]], *, chart_renderer: ChartRenderer) -> None:
        """
        저장된 메시지를 렌더링한다.

        Args:
            messages: 메시지 리스트.
            chart_renderer: 차트 렌더러.

        Side Effects:
            대화 로그를 화면에 표시한다.

        Raises:
            예외 없음.
        """

        current_user_id = str(st.session_state.get("user_id") or DEFAULT_USER_ID)
        active_chat_id = str(st.session_state.get("active_chat_id") or "unknown")

        for idx, message in enumerate(messages):
            with st.chat_message(message["role"]):
                dataframe = message.get("dataframe")
                error = message.get("error")
                error_detail = message.get("error_detail")
                route = message.get("route")
                route_reason = message.get("route_reason")
                planned_slots = message.get("planned_slots")
                multi_step_plan = message.get("multi_step_plan")
                last_result_schema = message.get("last_result_schema")
                sql = message.get("sql")
                fewshot_examples = message.get("fewshot_examples")
                schema_context = message.get("schema_context")

                if message["role"] == "assistant" and (
                    route or route_reason or planned_slots or last_result_schema or multi_step_plan
                ):
                    self._render_thinking_panel(
                        route=route,
                        route_reason=route_reason,
                        planned_slots=planned_slots,
                        multi_step_plan=multi_step_plan,
                        last_result_schema=last_result_schema,
                        sql=sql,
                        fewshot_examples=fewshot_examples,
                        schema_context=schema_context,
                        message_index=idx,
                    )

                chart_spec = message.get("chart_spec")
                chart_image_path = message.get("chart_image_path")
                if chart_spec and isinstance(dataframe, pd.DataFrame):
                    rendered_path = self._render_answer_with_chart(
                        str(message["content"]),
                        dataframe,
                        chart_spec,
                        user_id=current_user_id,
                        chat_id=active_chat_id,
                        chart_image_path=chart_image_path if isinstance(chart_image_path, str) else None,
                        chart_renderer=chart_renderer,
                        stream=False,
                    )
                    if rendered_path and not chart_image_path:
                        message["chart_image_path"] = rendered_path
                else:
                    st.markdown(str(message["content"]))

                if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                    self._render_result_summary(dataframe, planned_slots)
                    display_df = self._prepare_dataframe_for_display(dataframe, True)
                    if not self._has_markdown_table(str(message.get("content", ""))):
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
        self,
        result: dict[str, object],
        latency_ms: float | None,
        logger: JsonlLogger,
        user_text: str,
        *,
        chart_generator: ChartGenerator,
        chart_renderer: ChartRenderer,
        chat_store: ChatStore,
        user_id: str,
        chat_id: str,
        show_thinking: bool = True,
    ) -> None:
        """
        에이전트 결과를 화면과 로그에 반영한다.

        Args:
            result: 에이전트 실행 결과.
            latency_ms: 처리 시간.
            logger: 로그 기록기.
            user_text: 사용자 입력.
            chart_generator: 차트 생성기.
            chart_renderer: 차트 렌더러.
            chat_store: 채팅 저장소.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            show_thinking: Thinking 패널 표시 여부.

        Side Effects:
            메시지 저장, UI 렌더링, 로그 기록을 수행한다.

        Raises:
            예외 없음.
        """

        final_answer = result.get("final_answer", "응답을 생성하지 못했습니다.")
        sql = result.get("sql")
        dataframe = result.get("result_df")
        error = result.get("error")
        error_detail = result.get("error_detail")
        route = result.get("route", "unknown")
        route_reason = result.get("route_reason")
        planned_slots = result.get("planned_slots")
        multi_step_plan = result.get("multi_step_plan")
        fewshot_examples = result.get("fewshot_examples")
        schema_context = result.get("schema_context")

        display_df: pd.DataFrame | None = None
        chart_spec: ChartSpec | None = None
        chart_image_path: str | None = None
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            display_df = self._prepare_dataframe_for_display(dataframe, True)
            table_markdown = self._dataframe_to_markdown(display_df)
            final_answer = self._merge_markdown_table(str(final_answer), table_markdown)

            if self._is_chart_request(user_text):
                chart_spec = self._build_chart_spec(user_text, display_df, chart_generator)
                if chart_spec:
                    final_answer = self._ensure_chart_section(final_answer)
                    chart_result = chart_renderer.prepare_chart_image(
                        display_df,
                        chart_spec,
                        user_id=user_id,
                        chat_id=chat_id,
                        existing_path=None,
                    )
                    chart_image_path = chart_result.path

        assistant_message = {
            "role": "assistant",
            "content": final_answer,
            "sql": sql,
            "dataframe": display_df if display_df is not None else dataframe,
            "chart_spec": chart_spec,
            "chart_image_path": chart_image_path,
            "error": error,
            "error_detail": error_detail,
            "route": route,
            "route_reason": route_reason,
            "planned_slots": planned_slots,
            "multi_step_plan": multi_step_plan,
            "last_result_schema": result.get("last_result_schema"),
            "fewshot_examples": fewshot_examples,
            "schema_context": schema_context,
        }
        st.session_state.messages.append(assistant_message)
        message_index = len(st.session_state.messages) - 1

        chat_meta: dict[str, object] = {
            "sql": sql,
            "planned_slots": planned_slots,
            "route": route,
            "route_reason": route_reason,
            "error": error,
            "error_detail": error_detail,
            "last_result_schema": result.get("last_result_schema"),
            "chart_spec": chart_spec,
            "chart_image_path": chart_image_path,
            "fewshot_examples": fewshot_examples,
            "multi_step_plan": multi_step_plan,
            "schema_context": schema_context,
        }
        if isinstance(display_df, pd.DataFrame):
            chat_meta["dataframe_records"] = display_df.to_dict(orient="records")
            chat_meta["dataframe_columns"] = list(display_df.columns)

        chat_store.add_message(
            user_id=user_id,
            chat_id=chat_id,
            role="assistant",
            content=final_answer,
            meta=chat_meta,
        )

        if show_thinking:
            self._render_thinking_panel(
                route=route,
                route_reason=route_reason,
                planned_slots=planned_slots,
                multi_step_plan=multi_step_plan,
                last_result_schema=result.get("last_result_schema"),
                sql=sql,
                fewshot_examples=fewshot_examples,
                schema_context=schema_context,
                message_index=message_index,
            )

        if chart_spec and isinstance(display_df, pd.DataFrame):
            self._render_answer_with_chart(
                str(final_answer),
                display_df,
                chart_spec,
                user_id=user_id,
                chat_id=chat_id,
                chart_image_path=chart_image_path,
                chart_renderer=chart_renderer,
                stream=True,
            )
        else:
            st.write_stream(self._stream_text(str(final_answer)))

        if isinstance(display_df, pd.DataFrame) and not display_df.empty:
            self._render_result_summary(display_df, planned_slots)
            if not self._has_markdown_table(str(final_answer)):
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
            rows=int(len(display_df)) if isinstance(display_df, pd.DataFrame) else None,
            latency_ms=latency_ms,
            error=error if isinstance(error, str) else None,
            user_id=user_id,
        )

    def _run_agent_with_thinking(self, scene: ChatbotScene, user_message: str) -> dict[str, object]:
        """
        스트리밍으로 Thinking 패널을 업데이트하며 에이전트를 실행한다.

        Args:
            scene: Scene 인스턴스.
            user_message: 사용자 입력.

        Returns:
            최종 결과 딕셔너리.

        Side Effects:
            Thinking 패널 렌더링을 갱신한다.

        Raises:
            예외 없음.
        """

        placeholder = st.empty()
        final_state: dict[str, object] | None = None

        try:
            for state in scene.orchestrator.stream(user_message):
                if not isinstance(state, dict):
                    continue
                final_state = state
                with placeholder.container():
                    self._render_thinking_panel(
                        route=state.get("route"),
                        route_reason=state.get("route_reason"),
                        planned_slots=state.get("planned_slots"),
                        multi_step_plan=state.get("multi_step_plan"),
                        last_result_schema=state.get("last_result_schema"),
                        sql=state.get("sql"),
                        fewshot_examples=state.get("fewshot_examples"),
                        schema_context=state.get("schema_context"),
                        message_index=len(st.session_state.messages),
                    )
        except Exception:
            placeholder.empty()
            return scene.ask(user_message)

        if final_state is None or final_state.get("final_answer") is None:
            placeholder.empty()
            return scene.ask(user_message)

        with placeholder.container():
            self._render_thinking_panel(
                route=final_state.get("route"),
                route_reason=final_state.get("route_reason"),
                planned_slots=final_state.get("planned_slots"),
                multi_step_plan=final_state.get("multi_step_plan"),
                last_result_schema=final_state.get("last_result_schema"),
                sql=final_state.get("sql"),
                fewshot_examples=final_state.get("fewshot_examples"),
                schema_context=final_state.get("schema_context"),
                message_index=len(st.session_state.messages),
            )

        return final_state

    def _render_thinking_panel(
        self,
        *,
        route: str | None,
        route_reason: str | None,
        planned_slots: dict[str, object] | None,
        multi_step_plan: dict[str, object] | None,
        last_result_schema: list[str] | None,
        sql: str | None,
        fewshot_examples: str | None,
        schema_context: str | None,
        message_index: int,
    ) -> None:
        """
        Thinking(단계별 상태)을 렌더링한다.

        Args:
            route: 라우팅 결과.
            route_reason: 라우팅 근거.
            planned_slots: 플래너 슬롯.
            multi_step_plan: 멀티 스텝 계획.
            last_result_schema: 직전 결과 스키마.
            sql: 생성/실행 SQL.
            fewshot_examples: few-shot 예시 문자열.
            schema_context: 선별된 스키마 컨텍스트.
            message_index: 메시지 인덱스(고유 키용).

        Side Effects:
            Thinking 패널 UI를 렌더링한다.

        Raises:
            예외 없음.
        """

        st.markdown('<span class="thinking-pill">Thinking</span>', unsafe_allow_html=True)
        with st.expander("Details", expanded=False):
            def _render_step(label: str, data: dict[str, object]) -> None:
                with st.expander(f"✅ {label}", expanded=False):
                    st.json(data, expanded=True)

            if route:
                _render_step("Routing", {"route": route, "route_reason": route_reason})
            else:
                st.markdown("- ⏳ Routing")

            if planned_slots:
                _render_step("Planning", {"planned_slots": planned_slots})
            else:
                st.markdown("- ⏳ Planning")

            if multi_step_plan:
                _render_step("Multi-step", {"multi_step_plan": multi_step_plan})

            if schema_context:
                _render_step("선별된 스키마", {"schema_context": schema_context})
            else:
                st.markdown("- ⏳ 선별된 스키마")

            if fewshot_examples:
                _render_step("Few-shot", {"fewshot_examples": fewshot_examples})
            else:
                st.markdown("- ⏳ Few-shot")

            if sql:
                _render_step("SQL", {"sql": sql})
            else:
                st.markdown("- ⏳ SQL")

    def _render_result_summary(self, dataframe: pd.DataFrame, planned_slots: dict[str, object] | None) -> None:
        """
        결과 요약 카드를 렌더링한다.

        Args:
            dataframe: 결과 데이터프레임.
            planned_slots: 플래너 슬롯.

        Side Effects:
            결과 요약 카드와 필터 정보를 표시한다.

        Raises:
            예외 없음.
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

    def _render_answer_with_chart(
        self,
        text: str,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
        *,
        user_id: str,
        chat_id: str,
        chart_image_path: str | None,
        chart_renderer: ChartRenderer,
        stream: bool,
    ) -> str | None:
        """
        차트를 포함해 응답을 렌더링한다.

        Args:
            text: 응답 텍스트.
            dataframe: 결과 데이터프레임.
            chart_spec: 차트 스펙.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            chart_image_path: 저장된 차트 이미지 경로(있으면 재사용).
            chart_renderer: 차트 렌더러.
            stream: 스트리밍 여부.

        Returns:
            사용한 차트 이미지 경로(없으면 None).

        Side Effects:
            차트와 텍스트를 순서대로 출력한다. 필요 시 이미지를 저장한다.

        Raises:
            예외 없음.
        """

        pre, post = self._split_chart_text(text)
        if pre.strip():
            if stream:
                st.write_stream(self._stream_text(pre))
            else:
                st.markdown(pre)

        st.markdown("📊 차트")
        image_path = self._render_chart(
            dataframe,
            chart_spec,
            user_id=user_id,
            chat_id=chat_id,
            chart_image_path=chart_image_path,
            chart_renderer=chart_renderer,
        )

        if post.strip():
            if stream:
                st.write_stream(self._stream_text(post))
            else:
                st.markdown(post)

        return image_path

    def _render_chart(
        self,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
        *,
        user_id: str,
        chat_id: str,
        chart_image_path: str | None,
        chart_renderer: ChartRenderer,
    ) -> str | None:
        """
        차트 스펙에 따라 그래프를 렌더링한다.

        Args:
            dataframe: 결과 데이터프레임.
            chart_spec: 차트 스펙.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            chart_image_path: 저장된 차트 이미지 경로(있으면 재사용).
            chart_renderer: 차트 렌더러.

        Returns:
            사용한 차트 이미지 경로(없으면 None).

        Side Effects:
            차트를 Streamlit에 출력하고, 이미지 파일을 저장할 수 있다.

        Raises:
            예외 없음.
        """

        result = chart_renderer.prepare_chart_image(
            dataframe,
            chart_spec,
            user_id=user_id,
            chat_id=chat_id,
            existing_path=chart_image_path,
        )
        if result.error == "path":
            st.caption("차트 저장 경로를 만들 수 없습니다.")
            return None
        if result.path is None:
            st.caption("차트를 그릴 수 있는 수치형 데이터가 없습니다.")
            return None

        st.image(result.path, use_container_width=True)
        return result.path

    def _is_chart_request(self, text: str) -> bool:
        """
        차트/그래프 요청인지 판별한다.

        Args:
            text: 사용자 질문.

        Returns:
            차트 요청이면 True.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        lowered = text.lower()
        keywords = [
            "그래프",
            "차트",
            "시각화",
            "히스토그램",
            "박스플롯",
            "박스 플롯",
            "plot",
            "chart",
            "histogram",
            "boxplot",
            "box plot",
            "visualize",
            "visualization",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _build_chart_spec(
        self,
        user_text: str,
        dataframe: pd.DataFrame,
        chart_generator: ChartGenerator,
    ) -> ChartSpec | None:
        """
        LLM과 규칙 기반으로 차트 스펙을 생성한다.

        Args:
            user_text: 사용자 질문.
            dataframe: 결과 데이터프레임.
            chart_generator: 차트 생성기.

        Returns:
            차트 스펙 딕셔너리(불가 시 None).

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            return None

        columns = list(dataframe.columns)
        sample_records = dataframe.head(5).to_dict(orient="records")
        try:
            spec = chart_generator.generate(
                user_question=user_text,
                columns=columns,
                numeric_columns=numeric_columns,
                sample_records=str(sample_records),
            )
        except Exception:
            spec = {}

        normalized = self._normalize_chart_spec(spec or {}, dataframe)
        if normalized:
            return normalized
        return self._fallback_chart_spec(dataframe)

    def _normalize_chart_spec(self, spec: dict[str, Any], dataframe: pd.DataFrame) -> ChartSpec | None:
        """
        차트 스펙을 정규화한다.

        Args:
            spec: 원본 스펙.
            dataframe: 결과 데이터프레임.

        Returns:
            정규화된 스펙 또는 None.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        chart_type = str(spec.get("chart_type", "")).lower()
        if chart_type == "none":
            return None
        if chart_type not in SUPPORTED_CHART_TYPES:
            return None

        x = spec.get("x")
        y = spec.get("y")

        if chart_type == "histogram":
            candidate = x if x in dataframe.columns else y
            if candidate not in dataframe.columns:
                return None
            if not pd.api.types.is_numeric_dtype(dataframe[candidate]):
                return None
            return {"chart_type": chart_type, "x": candidate, "y": candidate}

        if chart_type == "box":
            candidate = y if y in dataframe.columns else x
            if candidate not in dataframe.columns:
                return None
            if not pd.api.types.is_numeric_dtype(dataframe[candidate]):
                return None
            normalized: dict[str, object] = {"chart_type": chart_type, "y": candidate}
            if x in dataframe.columns:
                normalized["x"] = x
            return normalized

        if x not in dataframe.columns or y not in dataframe.columns:
            return None
        if not pd.api.types.is_numeric_dtype(dataframe[y]):
            return None

        normalized = {"chart_type": chart_type, "x": x, "y": y}
        series = spec.get("series")
        if series in dataframe.columns:
            normalized["series"] = series
        return normalized

    def _fallback_chart_spec(self, dataframe: pd.DataFrame) -> ChartSpec | None:
        """
        안전한 기본 차트 스펙을 생성한다.

        Args:
            dataframe: 결과 데이터프레임.

        Returns:
            차트 스펙 또는 None.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            return None

        y = numeric_columns[0]
        candidate_x = [col for col in dataframe.columns if col not in numeric_columns]
        x = candidate_x[0] if candidate_x else dataframe.columns[0]

        chart_type = "bar"
        if any(token in str(x).lower() for token in ["date", "day", "season", "year"]):
            chart_type = "line"

        return {"chart_type": chart_type, "x": x, "y": y}

    def _ensure_chart_section(self, text: str) -> str:
        """
        응답 텍스트에 차트 섹션을 주입한다.

        Args:
            text: 원본 응답 텍스트.

        Returns:
            차트 섹션이 포함된 텍스트.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        if "📊 차트" in text:
            return text

        lines = text.splitlines()
        table_range = self._find_markdown_table_range(lines)
        if table_range:
            start, end = table_range
            merged = lines[:end] + ["", "📊 차트", ""] + lines[end:]
            return "\n".join(merged).strip()

        for idx, line in enumerate(lines):
            if "📌 조회 결과" in line:
                merged = lines[: idx + 1] + ["", "📊 차트", ""] + lines[idx + 1 :]
                return "\n".join(merged).strip()

        return (text.rstrip() + "\n\n📊 차트\n").strip()

    def _split_chart_text(self, text: str) -> tuple[str, str]:
        """
        차트 섹션 기준으로 텍스트를 분할한다.

        Args:
            text: 원본 텍스트.

        Returns:
            (차트 이전, 차트 이후) 튜플.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        if "📊 차트" not in text:
            return text, ""

        before, after = text.split("📊 차트", maxsplit=1)
        return before.rstrip(), after.lstrip()

    def _prepare_dataframe_for_display(self, dataframe: pd.DataFrame, use_friendly_columns: bool) -> pd.DataFrame:
        """
        화면 표시용 데이터프레임을 준비한다.

        Args:
            dataframe: 원본 데이터프레임.
            use_friendly_columns: 컬럼명 변환 여부.

        Returns:
            표시용 데이터프레임.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        if not use_friendly_columns:
            return dataframe

        rename_map = {col: COLUMN_LABELS.get(col, col) for col in dataframe.columns}
        return dataframe.rename(columns=rename_map)

    def _dataframe_to_markdown(self, dataframe: pd.DataFrame) -> str:
        """
        데이터프레임을 마크다운 테이블로 변환한다.

        Args:
            dataframe: 결과 데이터프레임.

        Returns:
            마크다운 테이블 문자열.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        records = dataframe.to_dict(orient="records")
        return records_to_markdown(records)

    def _merge_markdown_table(self, text: str, table_markdown: str) -> str:
        """
        기존 답변에 마크다운 테이블을 주입하거나 교체한다.

        Args:
            text: 원본 응답 텍스트.
            table_markdown: 교체할 테이블 문자열.

        Returns:
            테이블이 포함된 응답 텍스트.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        lines = text.splitlines()
        table_range = self._find_markdown_table_range(lines)
        if table_range:
            start, end = table_range
            merged = lines[:start] + table_markdown.splitlines() + lines[end:]
            return "\n".join(merged).strip()

        for idx, line in enumerate(lines):
            if "📌 조회 결과" in line:
                merged = lines[: idx + 1] + ["", table_markdown, ""] + lines[idx + 1 :]
                return "\n".join(merged).strip()

        return (text.rstrip() + "\n\n" + table_markdown).strip()

    def _find_markdown_table_range(self, lines: list[str]) -> tuple[int, int] | None:
        """
        마크다운 테이블 영역을 찾는다.

        Args:
            lines: 텍스트 라인 목록.

        Returns:
            (start, end) 범위 또는 None.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        for idx in range(len(lines) - 1):
            if "|" in lines[idx] and self._is_markdown_separator(lines[idx + 1]):
                end = idx + 2
                while end < len(lines) and "|" in lines[end]:
                    end += 1
                return idx, end
        return None

    def _has_markdown_table(self, text: str) -> bool:
        """
        마크다운 테이블 포함 여부를 확인한다.

        Args:
            text: 응답 텍스트.

        Returns:
            테이블이 있으면 True.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        for idx in range(len(lines) - 1):
            if "|" in lines[idx] and self._is_markdown_separator(lines[idx + 1]):
                return True
        return False

    def _is_markdown_separator(self, line: str) -> bool:
        """
        마크다운 테이블 구분선인지 확인한다.

        Args:
            line: 한 줄 문자열.

        Returns:
            구분선이면 True.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        return bool(re.match(r"^\|?\s*[-:|\s]+\s*\|?$", line.strip()))

    def _stream_text(self, text: str, chunk_size: int = 24, delay_sec: float = 0.01) -> Iterator[str]:
        """
        텍스트를 일정 크기씩 나눠 스트리밍용 청크로 반환한다.

        Args:
            text: 원본 텍스트.
            chunk_size: 한 번에 출력할 문자 수.
            delay_sec: 청크 사이 대기 시간.

        Yields:
            텍스트 청크.

        Side Effects:
            delay_sec이 0보다 크면 지연을 발생시킨다.

        Raises:
            예외 없음.
        """

        if chunk_size <= 0:
            yield text
            return

        for offset in range(0, len(text), chunk_size):
            yield text[offset : offset + chunk_size]
            if delay_sec > 0:
                time.sleep(delay_sec)

    def _apply_custom_theme(self) -> None:
        """
        커스텀 UI 테마를 적용한다.

        Side Effects:
            CSS를 주입한다.

        Raises:
            예외 없음.
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
  padding: 0.1rem 0 0.2rem;
}

.hero-wrap {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 0.6rem 0 0.2rem;
}

.hero-logo {
  width: 52px;
  height: 52px;
  object-fit: contain;
  border-radius: 12px;
  background: #ffffff;
  padding: 6px;
  border: 1px solid #e2e8f0;
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

section[data-testid="stSidebar"] {
  background: #f1f5f9;
  border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .stButton>button {
  justify-content: flex-start;
  text-align: left;
  padding-left: 0.9rem;
  padding-right: 0.9rem;
}

section[data-testid="stSidebar"] .stButton>button span {
  width: 100%;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

section[data-testid="stSidebar"] button {
  border-radius: 16px;
}
</style>
            """,
            unsafe_allow_html=True,
        )

    def _build_logo_data_uri(self, path: Path) -> str | None:
        """
        로고 이미지의 data URI를 만든다.

        Args:
            path: 이미지 경로.

        Returns:
            data URI 문자열(없으면 None).

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        if not path.exists():
            return None

        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _load_chat_messages(self, chat_store: ChatStore, user_id: str, chat_id: str) -> list[dict[str, object]]:
        """
        저장된 메시지를 불러와 렌더링용 형태로 변환한다.

        Args:
            chat_store: 채팅 저장소.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            메시지 리스트.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        messages = chat_store.list_messages(user_id, chat_id)
        hydrated: list[dict[str, object]] = []
        for message in messages:
            dataframe_records = message.pop("dataframe_records", None)
            dataframe_columns = message.pop("dataframe_columns", None)
            if dataframe_records is not None:
                dataframe = pd.DataFrame.from_records(dataframe_records)
                if isinstance(dataframe_columns, list):
                    dataframe = dataframe.reindex(columns=dataframe_columns)
                message["dataframe"] = dataframe
            hydrated.append(message)
        return hydrated

    def _restore_memory_from_messages(self, scene: ChatbotScene, messages: list[dict[str, object]]) -> None:
        """
        저장된 메시지로 대화 메모리를 복원한다.

        Args:
            scene: ChatbotScene.
            messages: 메시지 리스트.

        Side Effects:
            단기 메모리를 초기화하고 재구성한다.

        Raises:
            예외 없음.
        """

        memory = scene.memory
        memory.short_term.reset()

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                memory.short_term.start_turn(str(content))
                continue

            if role != "assistant":
                continue

            memory.short_term.finish_turn(
                assistant_message=str(content),
                route=message.get("route"),
                sql=message.get("sql"),
                planned_slots=message.get("planned_slots"),
            )

            dataframe = message.get("dataframe")
            sql = message.get("sql")
            planned_slots = message.get("planned_slots") or {}
            if isinstance(dataframe, pd.DataFrame) and isinstance(sql, str):
                memory.short_term.update_sql_result(sql, dataframe, planned_slots)
            elif isinstance(dataframe, pd.DataFrame):
                memory.short_term.last_result = dataframe
                memory.short_term.last_result_schema = list(dataframe.columns)
                memory.short_term.last_slots = planned_slots

    def _maybe_set_chat_title(
        self,
        chat_store: ChatStore,
        title_generator: TitleGenerator,
        user_id: str,
        chat_id: str,
        user_message: str,
    ) -> None:
        """
        첫 질문 기반으로 채팅 제목을 설정한다.

        Args:
            chat_store: 채팅 저장소.
            title_generator: 제목 생성기.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            user_message: 첫 질문.

        Side Effects:
            채팅 제목을 업데이트하고 목록을 갱신한다.

        Raises:
            예외 없음.
        """

        session = chat_store.get_session(user_id, chat_id)
        if session is None or session.title.strip() not in {"", "새 대화"}:
            return

        try:
            title = title_generator.generate(user_message)
        except Exception:
            return

        if not title:
            return
        chat_store.update_title(user_id, chat_id, title)
        st.session_state.chat_sessions = chat_store.list_sessions(user_id)
        st.session_state.title_refresh_needed = True

    def _get_user_id(self) -> str:
        """
        가능한 범위에서 사용자 식별자를 추출한다.

        Returns:
            사용자 ID.

        Side Effects:
            기본 사용자 ID를 세션 상태에 저장한다.

        Raises:
            예외 없음.
        """

        user_id = st.session_state.get("user_id")
        if user_id:
            return str(user_id)

        if hasattr(st, "query_params"):
            params = st.query_params
            if "user_id" in params:
                value = params.get("user_id")
                if isinstance(value, list):
                    resolved = str(value[0]) if value else DEFAULT_USER_ID
                else:
                    resolved = str(value)
                st.session_state.user_id = resolved
                return resolved

        st.session_state.user_id = DEFAULT_USER_ID
        return DEFAULT_USER_ID


def main() -> None:
    """
    Streamlit 실행을 위한 엔트리 포인트.

    Side Effects:
        Streamlit 앱을 실행한다.

    Raises:
        예외 없음.
    """

    app = StreamlitChatApp(load_config())
    app.main()


if __name__ == "__main__":
    # 1) Streamlit 앱 실행
    main()
