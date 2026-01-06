from __future__ import annotations

import asyncio
import copy
import re
import traceback
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.agent.fewshot_generator import (
    FewshotGenerationInput,
    FewshotGenerator,
    SchemaSelectionInput,
    SchemaSelectionResult,
)
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.multi_step_planner import MultiStepPlanInput, MultiStepPlanner
from src.agent.planner import Planner, PlannerOutput, PlannerSlots
from src.agent.responder import Responder
from src.agent.router import apply_reuse_rules, Route, RouterLLM, RoutingContext
from src.agent.sql_generator import MultiStepSQLInput, SQLGenerationInput, SQLGenerator, SQLRepairInput
from src.agent.summarizer import Summarizer, SummaryInput
from src.agent.validator import ResultValidator
from src.db.schema_store import SchemaStore
from src.db.sqlite_client import preview_dataframe, SQLiteClient
from src.metrics.registry import MetricDefinition, MetricsRegistry
from src.utils.markdown import records_to_markdown


MAX_SQL_RETRIES = 5

class AgentState(TypedDict, total=False):
    """
    LangGraph 상태 구조.
    """

    user_message: str
    route: str
    route_reason: str | None
    metric_name: str | None
    planned_slots: dict[str, Any]
    metric_candidates: list[dict[str, Any]] | None
    entity_resolution: dict[str, Any] | None
    metric_tool_used: bool | None
    entity_tool_used: bool | None
    clarify_question: str | None
    multi_step_plan: MultiStepPlan | None
    fewshot_examples: str | None
    schema_context: str | None
    sql: str | None
    column_parser: dict[str, Any] | None
    column_parser_used: bool | None
    result_df: pd.DataFrame | None
    last_result_schema: list[str] | None
    final_answer: str | None
    final_answer_stream: AsyncIterator[str] | None
    error: str | None
    error_detail: dict[str, Any] | None


class MultiStepPlan(TypedDict, total=False):
    """
    멀티 스텝 실행 계획.

    steps:
        실행할 단계 목록.
    """

    steps: list[dict[str, Any]]
    combine: dict[str, Any] | None
    reason: str | None
    execution_mode: str | None


@dataclass(frozen=True)
class AgentDependencies:
    """
    에이전트 의존성 모음.

    Args:
        registry: 메트릭 레지스트리.
        schema_store: 스키마 스토어.
        router: 라우터.
        responder: 응답 생성기.
        planner: 플래너.
        fewshot_generator: Few-shot 생성기.
        fewshot_candidate_limit: few-shot 후보 sql_template 최대 개수.
        multi_step_planner: 멀티 스텝 플래너.
        sql_generator: SQL 생성기.
        guard: SQL 가드.
        validator: 결과 검증기.
        sqlite_client: SQLite 클라이언트.
        summarizer: 요약 생성기.
        memory: 대화 메모리.
    """

    registry: MetricsRegistry
    schema_store: SchemaStore
    router: RouterLLM
    responder: Responder
    planner: Planner
    fewshot_generator: FewshotGenerator
    fewshot_candidate_limit: int
    multi_step_planner: MultiStepPlanner
    sql_generator: SQLGenerator
    guard: SQLGuard
    validator: ResultValidator
    sqlite_client: SQLiteClient
    summarizer: Summarizer
    memory: ConversationMemory


def build_agent_chain(deps: AgentDependencies):
    """
    LangGraph 상태 머신을 구성한다.

    Args:
        deps: 에이전트 의존성.

    Returns:
        LangGraph 컴파일 결과.
    """

    graph = StateGraph(AgentState)

    async def _router(state: AgentState) -> AgentState:
        return await _router_node(state, deps)

    async def _general_answer(state: AgentState) -> AgentState:
        return await _general_answer_node(state, deps)

    async def _direct_answer(state: AgentState) -> AgentState:
        return await _direct_answer_node(state, deps)

    async def _reuse(state: AgentState) -> AgentState:
        return await _reuse_node(state, deps)

    async def _plan(state: AgentState) -> AgentState:
        return await _plan_node(state, deps)

    async def _clarify(state: AgentState) -> AgentState:
        return await _clarify_node(state, deps)

    async def _multi_step(state: AgentState) -> AgentState:
        return await _multi_step_node(state, deps)

    async def _multi_step_single_sql(state: AgentState) -> AgentState:
        return await _multi_step_single_sql_node(state, deps)

    async def _fewshot(state: AgentState) -> AgentState:
        return await _fewshot_node(state, deps)

    async def _generate_sql(state: AgentState) -> AgentState:
        return await _generate_sql_node(state, deps)

    async def _guard(state: AgentState) -> AgentState:
        return await _guard_node(state, deps)

    async def _execute(state: AgentState) -> AgentState:
        return await _execute_node(state, deps)

    async def _validate(state: AgentState) -> AgentState:
        return await _validate_node(state, deps)

    async def _summarize(state: AgentState) -> AgentState:
        return await _summarize_node(state, deps)

    graph.add_node("router", _router)
    graph.add_node("general_answer", _general_answer)
    graph.add_node("direct_answer", _direct_answer)
    graph.add_node("reuse", _reuse)
    graph.add_node("plan", _plan)
    graph.add_node("clarify", _clarify)
    graph.add_node("multi_step", _multi_step)
    graph.add_node("multi_step_single_sql", _multi_step_single_sql)
    graph.add_node("fewshot", _fewshot)
    graph.add_node("generate_sql", _generate_sql)
    graph.add_node("guard", _guard)
    graph.add_node("execute", _execute)
    graph.add_node("validate", _validate)
    graph.add_node("summarize", _summarize)
    graph.add_node("finalize_error", _finalize_error_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_selector,
        {
            Route.GENERAL.value: "general_answer",
            Route.DIRECT.value: "direct_answer",
            Route.REUSE.value: "reuse",
            Route.SQL_REQUIRED.value: "plan",
        },
    )

    graph.add_conditional_edges(
        "plan",
        _plan_selector,
        {
            "clarify": "clarify",
            "continue": "fewshot",
            "multi_step": "fewshot",
        },
    )

    graph.add_conditional_edges(
        "fewshot",
        _fewshot_selector,
        {
            "multi_step": "multi_step",
            "multi_step_single_sql": "multi_step_single_sql",
            "generate_sql": "generate_sql",
        },
    )
    graph.add_edge("generate_sql", "guard")

    graph.add_conditional_edges(
        "multi_step_single_sql",
        _single_sql_selector,
        {
            "fallback": "multi_step",
            "continue": "guard",
        },
    )
    graph.add_conditional_edges(
        "guard",
        _guard_selector,
        {
            "fallback": "multi_step",
            "error": "finalize_error",
            "continue": "execute",
        },
    )

    graph.add_edge("execute", "validate")
    graph.add_conditional_edges(
        "validate",
        _validate_selector,
        {
            "fallback": "multi_step",
            "error": "finalize_error",
            "continue": "summarize",
        },
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("general_answer", END)
    graph.add_edge("reuse", END)
    graph.add_edge("clarify", END)
    graph.add_conditional_edges(
        "multi_step",
        _error_selector,
        {
            "error": "finalize_error",
            "continue": "summarize",
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("finalize_error", END)

    return graph.compile()


async def _router_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    라우터 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    context = RoutingContext(
        user_message=state["user_message"],
        conversation_context=deps.memory.build_conversation_context(),
        has_previous=deps.memory.last_result is not None,
        last_result_schema=deps.memory.last_result_schema,
        last_sql=deps.memory.last_sql,
        last_slots=deps.memory.last_slots,
        available_metrics=_build_metric_aliases(deps.registry),
    )

    result = await deps.router.route(context, deps.registry)

    deps.memory.update_route(result.route.value)
    return {"route": result.route.value, "metric_name": result.metric_name, "route_reason": result.reason}


async def _direct_answer_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    Direct Answer 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    metric_name = state.get("metric_name")
    if metric_name:
        metric = deps.registry.get(metric_name)
    else:
        metric = None

    try:
        if metric is None:
            response = await deps.responder.compose_missing_metric(
                state["user_message"],
                conversation_context=deps.memory.build_conversation_context(),
                stream=True,
            )
            return _build_answer_payload(response)

        response = await deps.responder.compose_direct(
            metric,
            conversation_context=deps.memory.build_conversation_context(),
            stream=True,
        )
        return _build_answer_payload(response)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Direct Answer 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "지표 설명을 생성하는 중 오류가 발생했습니다.",
        }


async def _general_answer_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    일반 안내 응답 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    try:
        response = await deps.responder.compose_general(
            state["user_message"],
            conversation_context=deps.memory.build_conversation_context(),
            stream=True,
        )
        return _build_answer_payload(response)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"일반 응답 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "안내 답변을 생성하는 중 오류가 발생했습니다.",
        }


async def _reuse_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    이전 결과 재사용 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    if deps.memory.last_result is None:
        return {
            "final_answer": "이전 결과가 없어 후처리를 적용할 수 없습니다. 먼저 조회를 실행해 주세요.",
        }

    reuse = apply_reuse_rules(state["user_message"], deps.memory.last_result)
    deps.memory.last_result = reuse.dataframe
    deps.memory.last_result_schema = list(reuse.dataframe.columns)
    deps.memory.update_entities_from_dataframe(reuse.dataframe)

    preview_records = preview_dataframe(reuse.dataframe, max_rows=10)
    preview_markdown = records_to_markdown(preview_records)

    try:
        response_text = await deps.responder.compose_reuse(
            state["user_message"],
            reuse.summary,
            preview_markdown,
            conversation_context=deps.memory.build_conversation_context(),
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "result_df": reuse.dataframe,
            "sql": deps.memory.last_sql,
            "last_result_schema": deps.memory.last_result_schema,
            "error": f"후처리 응답 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "후처리 결과를 생성하는 중 오류가 발생했습니다.",
        }

    answer_payload = _build_answer_payload(response_text)
    return {
        "result_df": reuse.dataframe,
        "sql": deps.memory.last_sql,
        "last_result_schema": deps.memory.last_result_schema,
        **answer_payload,
    }


async def _plan_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    플래너 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    previous = None
    if deps.memory.last_slots:
        previous = PlannerSlots(**deps.memory.last_slots)

    plan_output: PlannerOutput = await deps.planner.plan_with_retry(
        state["user_message"],
        previous,
        previous_entities=deps.memory.last_entities,
        conversation_context=deps.memory.build_conversation_context(),
        max_retries=5,
    )
    planned_slots = plan_output.slots.model_dump()
    _apply_entity_resolution(planned_slots, plan_output.entity_resolution)
    await _fill_recent_season(planned_slots, state["user_message"], deps.sqlite_client)
    await _normalize_season(planned_slots, deps.sqlite_client)
    _apply_reference_filters(planned_slots, state["user_message"], deps.memory)
    await _apply_team_name_filter(planned_slots, state["user_message"], deps.sqlite_client)
    _override_season_comparison(planned_slots, state["user_message"])
    await _apply_default_season(planned_slots, state["user_message"], deps)
    _override_attendance_metric(planned_slots, state["user_message"])
    multi_step_plan = await _build_multi_step_plan_llm(
        state["user_message"],
        planned_slots,
        deps.schema_store,
        deps.registry,
        deps.multi_step_planner,
        deps.memory,
    )

    return {
        "planned_slots": planned_slots,
        "clarify_question": plan_output.clarify_question,
        "multi_step_plan": multi_step_plan,
        "metric_candidates": plan_output.metric_candidates or None,
        "entity_resolution": plan_output.entity_resolution,
        "metric_tool_used": plan_output.metric_tool_used,
        "entity_tool_used": plan_output.entity_tool_used,
    }


async def _clarify_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    확인 질문 노드.

    Args:
        state: 현재 상태.

    Returns:
        업데이트된 상태.
    """

    clarify_question = state.get("clarify_question", "추가 확인이 필요합니다.")
    try:
        response_text = await deps.responder.compose_clarify(
            state["user_message"],
            clarify_question,
            conversation_context=deps.memory.build_conversation_context(),
            stream=True,
        )
        return _build_answer_payload(response_text)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"확인 질문 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": clarify_question,
        }


async def _fewshot_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    Few-shot 생성 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    planned_slots = state.get("planned_slots", {})
    user_message = state["user_message"]
    multi_step_plan = state.get("multi_step_plan")
    target_count = _decide_fewshot_count(user_message, multi_step_plan)
    candidate_metrics = _collect_candidate_metrics(
        user_message,
        planned_slots,
        multi_step_plan,
        deps.registry,
        limit=deps.fewshot_candidate_limit,
    )
    full_schema_context = deps.schema_store.build_full_context(max_columns=12)
    selection = await deps.fewshot_generator.select_schema(
        SchemaSelectionInput(
            user_question=user_message,
            planned_slots=planned_slots,
            candidate_metrics=candidate_metrics,
            schema_context=full_schema_context,
            context_hint="없음",
        )
    )
    schema_context = _build_schema_context_from_selection(selection, deps.schema_store)
    if not schema_context:
        schema_context = _build_schema_context_for_metrics(candidate_metrics, deps.schema_store)

    try:
        fewshot_examples = await deps.fewshot_generator.generate_examples(
            FewshotGenerationInput(
                user_question=user_message,
                planned_slots=planned_slots,
                candidate_metrics=candidate_metrics,
                schema_context=schema_context,
                context_hint="없음",
                target_count=target_count,
            )
        )
    except Exception:  # noqa: BLE001
        return {}

    return {"fewshot_examples": fewshot_examples, "schema_context": schema_context}


async def _generate_sql_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    SQL 생성 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    slots = state.get("planned_slots", {})
    metric_name = slots.get("metric")
    metric = deps.registry.get(metric_name) if metric_name else None

    if metric is None:
        return {
            "error": "메트릭 정의를 찾을 수 없습니다.",
            "final_answer": "요청하신 지표 정의를 찾을 수 없습니다. 다른 지표로 질문해 주세요.",
        }

    template_sql = _render_metric_template(metric, slots)
    if template_sql:
        return {"sql": template_sql, "column_parser": None, "column_parser_used": False}

    last_error: Exception | None = None
    last_traceback = ""
    for attempt in range(MAX_SQL_RETRIES):
        try:
            sql = await deps.sql_generator.generate_sql(
                SQLGenerationInput(
                    user_question=state["user_message"],
                    planned_slots=slots,
                    metric_context=deps.registry.build_sql_context(metric),
                    schema_context=state.get("schema_context") or deps.schema_store.build_context(),
                    last_sql=deps.memory.last_sql,
                    context_hint="없음",
                    fewshot_examples=state.get("fewshot_examples"),
                )
            )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            last_traceback = traceback.format_exc()
            sql = ""

    if last_error or not sql:
        return {
            "error": f"SQL 생성 실패: {last_error}" if last_error else "SQL 생성 실패: 빈 SQL",
            "error_detail": {
                "exception": str(last_error) if last_error else "empty_sql",
                "traceback": last_traceback,
                "retry_count": MAX_SQL_RETRIES,
            },
            "final_answer": f"SQL 생성에 실패했습니다. {MAX_SQL_RETRIES}회 재시도했지만 오류가 발생했습니다.",
        }
    return {
        "sql": sql,
        "column_parser": deps.sql_generator.get_last_column_parser(),
        "column_parser_used": deps.sql_generator.get_last_column_parser_used(),
    }


async def _multi_step_single_sql_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    멀티 스텝 계획을 단일 SQL로 합성한다.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    plan = state.get("multi_step_plan")
    if not isinstance(plan, dict):
        return {
            "error": "멀티 스텝 계획이 없습니다.",
            "final_answer": "요청을 처리하는 중 오류가 발생했습니다.",
        }

    metric_contexts: list[dict[str, Any]] = []
    for metric_name in _metrics_for_multi_step(plan):
        metric = deps.registry.get(metric_name)
        if metric is None:
            continue
        metric_contexts.append(deps.registry.build_sql_context(metric))

    if not metric_contexts:
        return {
            "error": "메트릭 정의를 찾을 수 없습니다.",
            "final_answer": "요청을 처리할 수 없습니다. 다른 지표로 질문해 주세요.",
        }

    base_slots = state.get("planned_slots", {})
    planned_slots_for_sql = copy.deepcopy(base_slots) if isinstance(base_slots, dict) else {}
    default_season_id = _season_year_to_id(str(planned_slots_for_sql.get("season") or ""))
    if default_season_id:
        planned_slots_for_sql.setdefault("filters", {})
        if isinstance(planned_slots_for_sql["filters"], dict):
            planned_slots_for_sql["filters"].setdefault("season_id", default_season_id)

    plan_for_sql: dict[str, Any] = copy.deepcopy(plan)
    for step in plan_for_sql.get("steps", []):
        if not isinstance(step, dict):
            continue
        overrides = step.get("overrides")
        filters = step.get("filters")

        season_value: str | None = None
        if isinstance(overrides, dict) and isinstance(overrides.get("season"), str):
            season_value = overrides.get("season")
        elif isinstance(filters, dict) and isinstance(filters.get("season"), str):
            season_value = filters.get("season")

        if not season_value:
            continue

        season_id = _season_year_to_id(season_value)
        if not season_id:
            continue

        step.setdefault("resolved", {})
        if isinstance(step["resolved"], dict):
            step["resolved"].setdefault("season_id", season_id)

        if isinstance(overrides, dict):
            overrides.setdefault("season_id", season_id)
        if isinstance(filters, dict):
            filters.setdefault("season_id", season_id)

    schema_context = state.get("schema_context") or deps.schema_store.build_context()
    try:
        sql = await deps.sql_generator.generate_multi_step_sql(
            MultiStepSQLInput(
                user_question=state["user_message"],
                planned_slots=planned_slots_for_sql,
                multi_step_plan=plan_for_sql,
                metric_contexts=metric_contexts,
                schema_context=schema_context,
                last_sql=deps.memory.last_sql,
                context_hint="없음",
                fewshot_examples=state.get("fewshot_examples"),
            )
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"단일 SQL 합성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
        }

    if not sql or not str(sql).strip():
        return {
            "error": "단일 SQL 합성 결과가 비어 있습니다.",
        }

    return {
        "sql": sql,
        "column_parser": deps.sql_generator.get_last_column_parser(),
        "column_parser_used": deps.sql_generator.get_last_column_parser_used(),
        "error": None,
        "error_detail": None,
    }


async def _multi_step_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    멀티 스텝 실행 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    plan = state.get("multi_step_plan")
    if not plan:
        return {"error": "멀티 스텝 계획이 없습니다.", "final_answer": "요청을 처리하는 중 오류가 발생했습니다."}

    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return {"error": "멀티 스텝 단계가 비어 있습니다.", "final_answer": "요청을 처리할 수 없습니다."}

    base_slots = state.get("planned_slots", {})
    previous_slots = PlannerSlots(**base_slots) if base_slots else PlannerSlots()
    results: list[pd.DataFrame] = []
    sql_chunks: list[str] = []
    last_dataframe: pd.DataFrame | None = None
    column_parser_used = False

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return {"error": "멀티 스텝 단계 형식이 올바르지 않습니다.", "final_answer": "요청을 처리할 수 없습니다."}

        step_question = str(step.get("question") or state["user_message"]).strip()
        step_output = await deps.planner.plan_with_retry(
            step_question,
            previous_slots,
            previous_entities=deps.memory.last_entities,
            max_retries=5,
        )
        step_slots = step_output.slots.model_dump()
        _apply_step_overrides(step_slots, step, last_dataframe)

        metric_name = step_slots.get("metric")
        metric = deps.registry.get(metric_name) if metric_name else None
        if metric is None:
            return {"error": "메트릭 정의를 찾을 수 없습니다.", "final_answer": "요청을 처리할 수 없습니다."}

        template_sql = _render_metric_template(metric, step_slots)
        if template_sql:
            sql = template_sql
        else:
            try:
                sql = await deps.sql_generator.generate_sql(
                    SQLGenerationInput(
                        user_question=step_question,
                        planned_slots=step_slots,
                        metric_context=deps.registry.build_sql_context(metric),
                        schema_context=state.get("schema_context") or deps.schema_store.build_context(),
                        last_sql=deps.memory.last_sql,
                        context_hint="없음",
                        fewshot_examples=state.get("fewshot_examples"),
                    )
                )
                column_parser_used = column_parser_used or deps.sql_generator.get_last_column_parser_used()
            except Exception as exc:  # noqa: BLE001
                return {
                    "error": f"SQL 생성 실패: {exc}",
                    "error_detail": {"exception": str(exc), "traceback": traceback.format_exc(), "step": idx},
                    "final_answer": "요청하신 내용을 조회하지 못했습니다. 질문을 조금 더 구체화해 주세요.",
                }

        if not sql or not str(sql).strip():
            return {
                "error": "SQL 생성 결과가 비어 있습니다.",
                "error_detail": {"step": idx},
                "final_answer": "요청하신 내용을 조회하지 못했습니다. 질문을 조금 더 구체화해 주세요.",
            }

        guard_result = await _guard_sql_with_retry(sql, step_question, step_slots, deps)
        if not guard_result:
            return {"error": "SQL 가드레일을 통과하지 못했습니다.", "final_answer": "요청을 처리할 수 없습니다."}

        try:
            result = await asyncio.to_thread(deps.sqlite_client.execute, guard_result)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"SQL 실행 실패: {exc}", "final_answer": "SQL 실행 중 오류가 발생했습니다."}

        deps.memory.update_sql_result(result.executed_sql, result.dataframe, step_slots)
        results.append(result.dataframe)
        sql_chunks.append(f"-- step{idx}\n{result.executed_sql}")
        last_dataframe = result.dataframe
        previous_slots = PlannerSlots(**step_slots)

    final_df = _combine_multi_step_results(results, plan.get("combine"))
    final_sql = "\n".join(sql_chunks)
    deps.memory.update_sql_result(final_sql, final_df, state.get("planned_slots", {}))

    return {
        "sql": final_sql,
        "result_df": final_df,
        "last_result_schema": list(final_df.columns),
        "column_parser_used": column_parser_used,
        "error": None,
        "error_detail": None,
    }


def _apply_step_overrides(
    step_slots: dict[str, Any],
    step: dict[str, Any],
    previous_df: pd.DataFrame | None,
) -> None:
    """
    멀티 스텝 단계별 override를 적용한다.

    Args:
        step_slots: 단계별 슬롯.
        step: 단계 정의.
        previous_df: 이전 단계 결과.

    Returns:
        None
    """

    overrides = step.get("overrides", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            step_slots[key] = value

    if isinstance(step.get("top_k"), int):
        step_slots["top_k"] = step["top_k"]

    filters = step_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    step_filters = step.get("filters", {})
    if isinstance(step_filters, dict):
        for key, value in step_filters.items():
            if value == "__from_previous__" and isinstance(previous_df, pd.DataFrame):
                if key in {"team_abbreviation", "team_abbreviation_in"}:
                    filters["team_abbreviation_in"] = _extract_team_list(previous_df)
                    continue
                if key in {"player_name", "player_name_in"}:
                    filters["player_name_in"] = _extract_player_list(previous_df)
                    continue
            filters[key] = value

    step_slots["filters"] = filters


async def _guard_sql_with_retry(
    sql: str,
    user_question: str,
    planned_slots: dict[str, Any],
    deps: AgentDependencies,
) -> str | None:
    """
    SQL 가드와 수정을 반복해 유효 SQL을 확보한다.

    Args:
        sql: 생성된 SQL.
        user_question: 사용자 질문.
        planned_slots: 플래너 슬롯.
        deps: 에이전트 의존성.

    Returns:
        유효 SQL(없으면 None).
    """

    guard_result = deps.guard.validate(sql)
    if guard_result.is_valid:
        return guard_result.sql

    metric_context = _get_metric_context(planned_slots, deps.registry)
    failure_reason = "; ".join(guard_result.errors)
    failed_sql = sql

    for _ in range(MAX_SQL_RETRIES):
        repair_sql = await deps.sql_generator.repair_sql(
            SQLRepairInput(
                user_question=user_question,
                failed_sql=failed_sql,
                failure_reason=failure_reason,
                metric_context=metric_context,
                schema_context=deps.schema_store.build_context(),
            )
        )
        repaired = deps.guard.validate(repair_sql)
        if repaired.is_valid:
            return repaired.sql
        failure_reason = "; ".join(repaired.errors)
        failed_sql = repair_sql

    return None


def _combine_multi_step_results(results: list[pd.DataFrame], combine: dict[str, Any] | None) -> pd.DataFrame:
    """
    멀티 스텝 결과를 결합한다.

    Args:
        results: 단계별 결과.
        combine: 결합 규칙.

    Returns:
        결합된 데이터프레임.
    """

    if not results:
        return pd.DataFrame()

    if not isinstance(combine, dict):
        return results[-1]

    method = str(combine.get("method") or "").lower()
    left_step = int(combine.get("left_step") or 1) - 1
    right_step = int(combine.get("right_step") or 2) - 1
    on_keys = combine.get("on") or []

    if left_step < 0 or right_step < 0:
        return results[-1]
    if left_step >= len(results) or right_step >= len(results):
        return results[-1]
    if not isinstance(on_keys, list) or not on_keys:
        return results[-1]

    how_map = {
        "left_join": "left",
        "inner_join": "inner",
        "right_join": "right",
    }
    how = how_map.get(method)
    if not how:
        return results[-1]

    left_df = results[left_step]
    right_df = results[right_step]

    try:
        return left_df.merge(right_df, on=on_keys, how=how, suffixes=("_left", "_right"))
    except Exception:
        return results[-1]


def _sanitize_multi_step_steps(
    steps: list[dict[str, Any]],
    registry: MetricsRegistry,
) -> list[dict[str, Any]]:
    """
    멀티 스텝 계획의 메트릭/필터를 레지스트리에 맞게 보정한다.

    Args:
        steps: 멀티 스텝 단계 목록.
        registry: 메트릭 레지스트리.

    Returns:
        보정된 단계 목록.
    """

    sanitized: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue

        overrides = step.get("overrides")
        if isinstance(overrides, dict) and "metric" in overrides:
            metric_value = overrides.get("metric")
            resolved = _resolve_metric_name(metric_value, registry, str(step.get("question") or ""))
            if resolved:
                overrides["metric"] = resolved
            else:
                overrides.pop("metric", None)

        sanitized.append(step)

    return sanitized


def _resolve_metric_name(
    metric_value: Any,
    registry: MetricsRegistry,
    fallback_query: str,
) -> str | None:
    """
    메트릭 이름을 레지스트리 기준으로 보정한다.

    Args:
        metric_value: 후보 메트릭 값.
        registry: 메트릭 레지스트리.
        fallback_query: 대체 검색용 문장.

    Returns:
        보정된 메트릭 이름 또는 None.
    """

    if isinstance(metric_value, str):
        raw = metric_value.strip()
        if raw:
            metric = registry.get(raw)
            if metric is not None:
                return metric.name
            normalized = raw.replace("_", " ").strip()
            if normalized:
                metric = registry.get(normalized)
                if metric is not None:
                    return metric.name

            search_seed = f"{raw} {fallback_query}".strip()
            candidates = registry.search(search_seed, limit=1)
            if candidates:
                return candidates[0].name

    if fallback_query:
        candidates = registry.search(fallback_query, limit=1)
        if candidates:
            return candidates[0].name

    return None


def _infer_execution_mode(
    steps: list[dict[str, Any]],
    combine: dict[str, Any] | None,
) -> str:
    """
    멀티 스텝 계획이 단일 SQL로 합성 가능한지 간단히 추정한다.

    LLM이 execution_mode를 잘못 판단하거나 누락하는 경우를 대비해,
    보수적으로 single_sql 후보를 탐지한다.

    Args:
        steps: 멀티 스텝 단계 목록.
        combine: 결합 규칙.

    Returns:
        "single_sql" 또는 "multi_sql".
    """

    if not isinstance(combine, dict):
        return "multi_sql"

    method = str(combine.get("method") or "").lower()
    if method not in {"left_join", "inner_join", "right_join"}:
        return "multi_sql"

    on_keys = combine.get("on")
    if not isinstance(on_keys, list) or not on_keys:
        return "multi_sql"

    if len(steps) < 2:
        return "multi_sql"

    for step in steps:
        filters = step.get("filters")
        if not isinstance(filters, dict):
            continue
        if "__from_previous__" in filters.values():
            return "single_sql"

    return "multi_sql"


async def _guard_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    SQL 가드 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    sql = state.get("sql")
    if not sql:
        return {"error": "SQL이 생성되지 않았습니다.", "final_answer": "SQL 생성에 실패했습니다."}

    guard_result = deps.guard.validate(sql)
    if guard_result.is_valid:
        return {"sql": guard_result.sql}

    metric_context = _get_metric_context(state.get("planned_slots", {}), deps.registry)

    failure_reason = "; ".join(guard_result.errors)
    failed_sql = sql

    for _ in range(MAX_SQL_RETRIES):
        repair_sql = await deps.sql_generator.repair_sql(
            SQLRepairInput(
                user_question=state["user_message"],
                failed_sql=failed_sql,
                failure_reason=failure_reason,
                metric_context=metric_context,
                schema_context=deps.schema_store.build_context(),
            )
        )
        repaired_result = deps.guard.validate(repair_sql)
        if repaired_result.is_valid:
            return {"sql": repaired_result.sql}

        failure_reason = "; ".join(repaired_result.errors)
        failed_sql = repair_sql

    return {
        "error": failure_reason,
        "error_detail": {"guard_errors": guard_result.errors, "sql": sql},
        "final_answer": "SQL 가드레일을 통과하지 못했습니다. 질문을 구체화해 주세요.",
    }


async def _execute_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    SQL 실행 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    sql = state["sql"]
    metric_context = _get_metric_context(state.get("planned_slots", {}), deps.registry)
    failure_reason = ""
    last_traceback = ""

    for attempt in range(MAX_SQL_RETRIES):
        try:
            result = await asyncio.to_thread(deps.sqlite_client.execute, sql)
            deps.memory.update_sql_result(result.executed_sql, result.dataframe, state.get("planned_slots", {}))
            return {
                "sql": result.executed_sql,
                "result_df": result.dataframe,
                "last_result_schema": result.columns,
            }
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"SQL 실행 실패: {exc}"
            last_traceback = traceback.format_exc()
            repair_sql = await deps.sql_generator.repair_sql(
                SQLRepairInput(
                    user_question=state["user_message"],
                    failed_sql=sql,
                    failure_reason=str(exc),
                    metric_context=metric_context,
                    schema_context=deps.schema_store.build_context(),
                )
            )
            guard_result = deps.guard.validate(repair_sql)
            if not guard_result.is_valid:
                failure_reason = "; ".join(guard_result.errors)
                sql = repair_sql
                continue
            sql = guard_result.sql

    return {
        "error": failure_reason,
        "error_detail": {
            "exception": failure_reason,
            "traceback": last_traceback,
            "retry_count": MAX_SQL_RETRIES,
        },
        "final_answer": f"쿼리 실행에 실패했습니다. {MAX_SQL_RETRIES}회 재시도했지만 오류가 발생했습니다.",
    }


async def _validate_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    결과 검증 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    dataframe = state.get("result_df")
    if dataframe is None:
        return {
            "error": "실행 결과가 없습니다.",
            "error_detail": {"reason": "result_df가 None"},
            "final_answer": "결과를 얻지 못했습니다.",
        }

    validation = deps.validator.validate(dataframe)
    if validation.is_valid:
        return {}

    if not validation.should_retry:
        return {
            "error": validation.reason,
            "error_detail": {"validation_reason": validation.reason},
            "final_answer": "결과 품질이 낮아 응답을 생성할 수 없습니다.",
        }

    metric_context = _get_metric_context(state.get("planned_slots", {}), deps.registry)
    failed_sql = state.get("sql", "")
    failure_reason = validation.reason

    for _ in range(MAX_SQL_RETRIES):
        repair_sql = await deps.sql_generator.repair_sql(
            SQLRepairInput(
                user_question=state["user_message"],
                failed_sql=failed_sql,
                failure_reason=failure_reason,
                metric_context=metric_context,
                schema_context=deps.schema_store.build_context(),
            )
        )
        guard_result = deps.guard.validate(repair_sql)
        if not guard_result.is_valid:
            failure_reason = "; ".join(guard_result.errors)
            failed_sql = repair_sql
            continue

        try:
            retry_result = await asyncio.to_thread(deps.sqlite_client.execute, guard_result.sql)
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"재시도 SQL 실행 실패: {exc}"
            failed_sql = guard_result.sql
            continue

        validation_retry = deps.validator.validate(retry_result.dataframe)
        if validation_retry.is_valid:
            deps.memory.update_sql_result(
                retry_result.executed_sql,
                retry_result.dataframe,
                state.get("planned_slots", {}),
            )
            return {
                "sql": retry_result.executed_sql,
                "result_df": retry_result.dataframe,
                "last_result_schema": retry_result.columns,
            }

        if not validation_retry.should_retry:
            return {
                "error": validation_retry.reason,
                "error_detail": {"validation_reason": validation_retry.reason},
                "final_answer": "재시도 결과 품질이 낮아 응답을 생성할 수 없습니다.",
            }

        failure_reason = validation_retry.reason
        failed_sql = guard_result.sql

    return {
        "error": failure_reason,
        "error_detail": {"validation_reason": failure_reason},
        "final_answer": "재시도 결과 품질이 낮아 응답을 생성할 수 없습니다.",
    }


async def _summarize_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    요약 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    dataframe = state.get("result_df")
    if dataframe is None:
        return {"final_answer": "요약할 결과가 없습니다."}

    applied_filters = _build_filter_summary(state.get("planned_slots", {}))
    try:
        summary = await deps.summarizer.summarize(
            SummaryInput(
                user_question=state["user_message"],
                sql=state.get("sql", ""),
                result_preview=preview_dataframe(dataframe, max_rows=10),
                applied_filters=applied_filters,
                conversation_context=deps.memory.build_conversation_context(),
            ),
            stream=True,
        )
        return _build_answer_payload(summary)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"요약 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "요약 생성 중 오류가 발생했습니다.",
        }


def _build_answer_payload(response: str | AsyncIterator[str]) -> AgentState:
    """
    LLM 응답 유형에 맞춰 최종 답변 필드를 구성한다.

    Args:
        response: 문자열 또는 스트리밍 이터레이터.

    Returns:
        AgentState에 병합할 payload.
    """

    if isinstance(response, AsyncIterator):
        return {"final_answer_stream": response}
    return {"final_answer": response}


def _finalize_error_node(state: AgentState) -> AgentState:
    """
    에러 처리 노드.

    Args:
        state: 현재 상태.

    Returns:
        업데이트된 상태.
    """

    if state.get("final_answer") or state.get("final_answer_stream"):
        return {}
    error = state.get("error")
    if isinstance(error, str) and error.strip():
        return {"final_answer": f"오류가 발생했습니다: {error}"}
    return {"final_answer": "요청을 처리하는 중 오류가 발생했습니다. 질문을 조금 더 구체화해 주세요."}


def _route_selector(state: AgentState) -> str:
    """
    라우팅 분기 선택.

    Args:
        state: 현재 상태.

    Returns:
        라우팅 문자열.
    """

    return state.get("route", Route.SQL_REQUIRED.value)


def _plan_selector(state: AgentState) -> str:
    """
    플래너 결과에 따른 분기 선택.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    if state.get("clarify_question"):
        return "clarify"
    if state.get("multi_step_plan"):
        return "multi_step"
    return "continue"


def _fewshot_selector(state: AgentState) -> str:
    """
    few-shot 이후 분기를 결정한다.

    Args:
        state: 현재 상태.

    Returns:
        다음 노드 키.
    """

    multi_step_plan = state.get("multi_step_plan")
    if isinstance(multi_step_plan, dict):
        if multi_step_plan.get("execution_mode") == "single_sql":
            return "multi_step_single_sql"
        return "multi_step"
    return "generate_sql"


def _single_sql_selector(state: AgentState) -> str:
    """
    단일 SQL 합성 실패 시 멀티스텝으로 폴백한다.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    return "fallback" if state.get("error") else "continue"


def _guard_selector(state: AgentState) -> str:
    """
    SQL 가드 실패 시 멀티스텝으로 폴백할지 결정한다.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    if not state.get("error"):
        return "continue"

    multi_step_plan = state.get("multi_step_plan")
    if isinstance(multi_step_plan, dict) and multi_step_plan.get("execution_mode") == "single_sql":
        return "fallback"

    return "error"


def _validate_selector(state: AgentState) -> str:
    """
    결과 검증 실패 시 멀티스텝으로 폴백할지 결정한다.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    if not state.get("error"):
        return "continue"

    multi_step_plan = state.get("multi_step_plan")
    if isinstance(multi_step_plan, dict) and multi_step_plan.get("execution_mode") == "single_sql":
        return "fallback"

    return "error"


def _error_selector(state: AgentState) -> str:
    """
    에러 여부에 따른 분기 선택.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    return "error" if state.get("error") else "continue"


def _build_filter_summary(planned_slots: dict[str, Any]) -> str:
    """
    적용 필터 요약 문자열을 생성.

    Args:
        planned_slots: 플래너 슬롯.

    Returns:
        필터 요약 문자열.
    """

    parts: list[str] = []
    season = planned_slots.get("season")
    date_range = planned_slots.get("date_range")
    filters = planned_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    if season:
        parts.append(f"시즌: {season}")
    if date_range:
        parts.append(f"기간: {date_range}")
    if filters:
        filters_text = ", ".join(f"{key}={value}" for key, value in filters.items())
        parts.append(f"필터: {filters_text}")

    return " / ".join(parts) if parts else "적용 조건 없음"


def _apply_reference_filters(
    planned_slots: dict[str, Any],
    user_message: str,
    memory: ConversationMemory,
) -> None:
    """
    이전 결과를 지칭하는 표현을 감지해 필터를 보강한다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.
        memory: 대화 메모리.

    Returns:
        None
    """

    if not _has_team_reference(user_message):
        return

    entities = memory.short_term.last_entities
    teams = entities.get("teams") if entities else None
    if not teams:
        return

    filters = planned_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    filters["team_abbreviation_in"] = teams
    planned_slots["filters"] = filters
    planned_slots["entity_type"] = "team"

    if planned_slots.get("metric") is None and _looks_like_team_performance_query(user_message):
        planned_slots["metric"] = "win_pct"


async def _apply_team_name_filter(
    planned_slots: dict[str, Any],
    user_message: str,
    sqlite_client: SQLiteClient,
) -> None:
    """
    팀 이름 기반 필터를 보강한다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.
        sqlite_client: SQLite 클라이언트.

    Returns:
        None
    """

    filters = planned_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    if filters.get("team_abbreviation") or filters.get("team_abbreviation_in"):
        planned_slots["filters"] = filters
        return

    lowered = user_message.lower()
    team_catalog = await _fetch_team_catalog(sqlite_client)
    if not team_catalog:
        return

    abbreviations = {row["abbreviation"] for row in team_catalog if row.get("abbreviation")}
    tokens = re.findall(r"\\b[A-Za-z]{2,3}\\b", user_message)
    for token in tokens:
        token_upper = token.upper()
        if token_upper in abbreviations:
            filters["team_abbreviation"] = token_upper
            filters["team_abbreviation_in"] = [token_upper]
            planned_slots["filters"] = filters
            planned_slots["entity_type"] = "team"
            return

    matched_abbreviation = _match_team_name(lowered, team_catalog)
    if matched_abbreviation:
        filters["team_abbreviation"] = matched_abbreviation
        filters["team_abbreviation_in"] = [matched_abbreviation]
        planned_slots["filters"] = filters
        planned_slots["entity_type"] = "team"


def _apply_entity_resolution(
    planned_slots: dict[str, Any],
    entity_resolution: dict[str, Any] | None,
) -> None:
    """
    EntityResolver 결과를 슬롯에 반영한다.

    Args:
        planned_slots: 플래너 슬롯.
        entity_resolution: 엔티티 보강 결과.

    Returns:
        None
    """

    if not entity_resolution:
        return

    filters = planned_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    tool_filters = entity_resolution.get("filters")
    if isinstance(tool_filters, dict):
        for key, value in tool_filters.items():
            if key not in filters:
                filters[key] = value

    teams = entity_resolution.get("teams")
    players = entity_resolution.get("players")
    if teams and planned_slots.get("entity_type") in (None, "", "team"):
        planned_slots["entity_type"] = "team"
    if players and planned_slots.get("entity_type") in (None, "", "player"):
        planned_slots["entity_type"] = "player"

    planned_slots["filters"] = filters


async def _fetch_team_catalog(sqlite_client: SQLiteClient) -> list[dict[str, str]]:
    """
    팀 이름/약어 목록을 조회한다.

    Args:
        sqlite_client: SQLite 클라이언트.

    Returns:
        팀 메타 리스트.
    """

    sql = "SELECT full_name, abbreviation, nickname, city FROM team"
    try:
        result = await asyncio.to_thread(sqlite_client.execute, sql)
    except Exception:
        return []

    records: list[dict[str, str]] = []
    for record in result.dataframe.to_dict(orient="records"):
        records.append(
            {
                "full_name": str(record.get("full_name") or ""),
                "abbreviation": str(record.get("abbreviation") or "").upper(),
                "nickname": str(record.get("nickname") or ""),
                "city": str(record.get("city") or ""),
            }
        )
    return records


def _match_team_name(message_lowered: str, catalog: list[dict[str, str]]) -> str | None:
    """
    사용자 문장에서 팀 이름을 찾아 약어로 반환한다.

    Args:
        message_lowered: 사용자 질문(lower).
        catalog: 팀 메타 목록.

    Returns:
        팀 약어 또는 None.
    """

    best_match: tuple[int, str] | None = None
    for row in catalog:
        abbreviation = row.get("abbreviation")
        if not abbreviation:
            continue

        candidates = []
        full_name = row.get("full_name")
        nickname = row.get("nickname")
        city = row.get("city")
        if full_name:
            candidates.append(full_name)
        if nickname:
            candidates.append(nickname)
        if city and nickname:
            candidates.append(f"{city} {nickname}")

        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower and candidate_lower in message_lowered:
                length = len(candidate_lower)
                if best_match is None or length > best_match[0]:
                    best_match = (length, abbreviation)

    return best_match[1] if best_match else None


def _override_season_comparison(planned_slots: dict[str, Any], user_message: str) -> None:
    """
    시즌별 성적 비교 질의를 보정한다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.

    Returns:
        None
    """

    if not _looks_like_season_comparison(user_message):
        return

    filters = planned_slots.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    team_abbreviation = filters.get("team_abbreviation")
    if not team_abbreviation:
        candidates = filters.get("team_abbreviation_in")
        if isinstance(candidates, list) and len(candidates) == 1:
            team_abbreviation = str(candidates[0])
            filters["team_abbreviation"] = team_abbreviation

    if not team_abbreviation:
        return

    planned_slots["metric"] = "team_win_pct_by_season"
    planned_slots.pop("season", None)
    for key in ("season_default", "season_override"):
        filters.pop(key, None)
    planned_slots["filters"] = filters
    if not _has_explicit_top_k(user_message):
        planned_slots["top_k"] = 200


def _looks_like_season_comparison(user_message: str) -> bool:
    """
    시즌별 비교 요청인지 판단한다.

    Args:
        user_message: 사용자 질문.

    Returns:
        시즌 비교 요청이면 True.
    """

    lowered = user_message.lower()
    keywords = [
        "시즌별",
        "시즌 별",
        "연도별",
        "년도별",
        "시즌 비교",
        "season by season",
        "season-by-season",
        "seasonal",
        "season comparison",
    ]
    return any(keyword in user_message or keyword in lowered for keyword in keywords)


def _has_explicit_top_k(user_message: str) -> bool:
    """
    상위 N 등 명시적 제한이 있는지 확인한다.

    Args:
        user_message: 사용자 질문.

    Returns:
        명시적 제한이 있으면 True.
    """

    return bool(re.search(r"(?:상위|top)\\s*\\d+|\\d+\\s*개", user_message, re.IGNORECASE))


def _override_attendance_metric(planned_slots: dict[str, Any], user_message: str) -> None:
    """
    관중 관련 질의에서 팀/경기 구분을 보정한다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.

    Returns:
        None
    """

    lowered = user_message.lower()
    has_attendance = "관중" in user_message or "attendance" in lowered
    if not has_attendance:
        return

    has_game = "경기" in user_message or "game" in lowered
    has_team = "팀" in user_message or planned_slots.get("entity_type") == "team"

    if has_game:
        planned_slots["metric"] = "attendance_top_games"
        return

    if has_team and planned_slots.get("metric") in {None, "attendance_top_games"}:
        planned_slots["metric"] = "attendance_top_teams"


def _has_team_reference(user_message: str) -> bool:
    """
    이전 팀 목록을 가리키는 표현이 있는지 확인한다.

    Args:
        user_message: 사용자 질문.

    Returns:
        지칭 표현이 있으면 True.
    """

    keywords = [
        "해당 팀",
        "그 팀",
        "이 팀",
        "위 팀",
        "해당 팀들",
        "그 팀들",
        "이 팀들",
        "위의 팀",
        "앞의 팀",
        "상위 팀",
        "해당 팀의",
        "그 팀의",
        "이 팀의",
    ]
    return any(keyword in user_message for keyword in keywords)


def _looks_like_team_performance_query(user_message: str) -> bool:
    """
    팀 성적/순위성 질문인지 간단히 판별한다.

    Args:
        user_message: 사용자 질문.

    Returns:
        성적성 질문이면 True.
    """

    keywords = ["성적", "순위", "승률", "랭킹", "리그 순위"]
    return any(keyword in user_message for keyword in keywords)


async def _build_multi_step_plan_llm(
    user_message: str,
    planned_slots: dict[str, Any],
    schema_store: SchemaStore,
    registry: MetricsRegistry,
    planner: MultiStepPlanner,
    memory: ConversationMemory,
) -> MultiStepPlan | None:
    """
    LLM 기반 멀티 스텝 계획을 생성한다.

    Args:
        user_message: 사용자 질문.
        planned_slots: 플래너 슬롯.
        schema_store: 스키마 스토어.
        planner: 멀티 스텝 플래너.
        memory: 대화 메모리.

    Returns:
        멀티 스텝 계획(없으면 None).
    """

    schema_context = schema_store.build_full_context(max_columns=8)
    result = await planner.plan(
        MultiStepPlanInput(
            user_question=user_message,
            planned_slots=planned_slots,
            schema_context=schema_context,
            context_hint="없음",
        )
    )

    if not result.use_multi_step or not result.steps:
        return None

    sanitized_steps = _sanitize_multi_step_steps(result.steps, registry)
    if not sanitized_steps:
        return None

    execution_mode = result.execution_mode or "multi_sql"
    inferred_mode = _infer_execution_mode(sanitized_steps, result.combine)
    if inferred_mode == "single_sql":
        execution_mode = "single_sql"

    return {
        "steps": sanitized_steps,
        "combine": result.combine,
        "reason": result.reason,
        "execution_mode": execution_mode,
    }


def _decide_fewshot_count(user_message: str, multi_step_plan: MultiStepPlan | None) -> int:
    """
    few-shot 예시 개수를 결정한다.

    Args:
        user_message: 사용자 질문.
        multi_step_plan: 멀티 스텝 계획.

    Returns:
        예시 개수(3 또는 5).
    """

    if multi_step_plan:
        return 5

    lowered = user_message.lower()
    keywords = ["그리고", "및", "비교", "추세", "상관", "분석", "vs", "대결", "맞대결"]
    if any(keyword in user_message or keyword in lowered for keyword in keywords):
        return 5
    return 3


def _collect_candidate_metrics(
    user_message: str,
    planned_slots: dict[str, Any],
    multi_step_plan: MultiStepPlan | None,
    registry: MetricsRegistry,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """
    few-shot 후보 메트릭 컨텍스트를 수집한다.

    Args:
        user_message: 사용자 질문.
        planned_slots: 플래너 슬롯.
        multi_step_plan: 멀티 스텝 계획.
        registry: 메트릭 레지스트리.

    Returns:
        메트릭 컨텍스트 리스트.
    """

    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_metric(metric_name: str | None) -> None:
        if not metric_name or metric_name in seen:
            return
        metric = registry.get(metric_name)
        if metric is None:
            return
        collected.append(registry.build_sql_context(metric))
        seen.add(metric.name)

    add_metric(planned_slots.get("metric"))

    if multi_step_plan:
        for metric_name in _metrics_for_multi_step(multi_step_plan):
            add_metric(metric_name)

    for metric in registry.search(user_message, limit=5):
        add_metric(metric.name)

    return collected[: max(limit, 0)]


def _build_schema_context_for_metrics(
    candidate_metrics: list[dict[str, Any]],
    schema_store: SchemaStore,
) -> str:
    """
    메트릭 후보에 필요한 테이블 스키마만 요약한다.

    Args:
        candidate_metrics: 후보 메트릭 컨텍스트 리스트.
        schema_store: 스키마 스토어.

    Returns:
        스키마 요약 문자열.
    """

    tables: list[str] = []
    required_columns: set[str] = set()

    for metric in candidate_metrics:
        for table in metric.get("required_tables", []):
            if table not in tables:
                tables.append(table)
        for column in metric.get("required_columns", []):
            if isinstance(column, str) and column:
                required_columns.add(column)

    columns_by_table: dict[str, list[str]] = {}
    for table in tables:
        table_info = schema_store.get_table(table)
        if not table_info:
            continue
        available = [col["name"] for col in table_info["columns"]]
        filtered = [col for col in available if col in required_columns]
        if filtered:
            columns_by_table[table] = filtered

    context = schema_store.build_context_for_tables(tables, columns_by_table, max_columns=12)
    if not context:
        context = schema_store.build_context(max_tables=6, max_columns=12)

    return "\n".join(["[필요 테이블 스키마]", context, "---"])


def _build_schema_context_from_selection(
    selection: SchemaSelectionResult,
    schema_store: SchemaStore,
) -> str:
    """
    LLM 선별 결과로 스키마 요약을 구성한다.

    Args:
        selection: 스키마 선별 결과.
        schema_store: 스키마 스토어.

    Returns:
        스키마 요약 문자열.
    """

    if not selection.tables:
        return ""

    context = schema_store.build_context_for_tables(
        selection.tables,
        selection.columns_by_table,
        max_columns=12,
    )
    if not context:
        return ""

    return "\n".join(["[필요 테이블 스키마]", context, "---"])


def _metrics_for_multi_step(plan: MultiStepPlan) -> list[str]:
    """
    멀티 스텝 계획에서 관련 메트릭을 반환한다.

    Args:
        plan: 멀티 스텝 계획.

    Returns:
        메트릭 이름 리스트.
    """

    steps = plan.get("steps", [])
    metrics: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        overrides = step.get("overrides", {})
        if isinstance(overrides, dict):
            metric = overrides.get("metric")
            if isinstance(metric, str) and metric not in metrics:
                metrics.append(metric)
    return metrics



def _extract_team_list(dataframe: pd.DataFrame) -> list[str]:
    """
    데이터프레임에서 팀 약어 목록을 추출한다.

    Args:
        dataframe: 결과 데이터프레임.

    Returns:
        팀 약어 리스트.
    """

    candidates = [
        "team_abbreviation",
        "team_abbreviation_home",
        "team_abbreviation_away",
    ]
    values: list[str] = []
    for col in candidates:
        if col in dataframe.columns:
            values.extend(dataframe[col].dropna().astype(str).tolist())
            break
    return _normalize_team_list(values)


def _extract_player_list(dataframe: pd.DataFrame) -> list[str]:
    """
    데이터프레임에서 선수 이름 목록을 추출한다.

    Args:
        dataframe: 결과 데이터프레임.

    Returns:
        선수 이름 리스트.
    """

    candidates = ["player_name", "display_first_last", "player"]
    values: list[str] = []
    for col in candidates:
        if col in dataframe.columns:
            values.extend(dataframe[col].dropna().astype(str).tolist())
            break
    return sorted({value for value in values if value})



def _normalize_team_list(values: list[str]) -> list[str]:
    """
    팀 약어 목록을 정규화한다.

    Args:
        values: 원본 값 목록.

    Returns:
        정규화된 팀 약어 리스트.
    """

    normalized: set[str] = set()
    for value in values:
        token = str(value).strip().upper()
        if re.fullmatch(r"[A-Z]{2,3}", token):
            normalized.add(token)
    return sorted(normalized)


def _build_team_filter(filters: dict[str, Any], placeholder: str) -> str:
    """
    팀 목록 필터 문자열을 생성한다.

    Args:
        filters: 플래너 필터.
        placeholder: 템플릿 플레이스홀더 이름.

    Returns:
        SQL 필터 문자열.
    """

    teams = filters.get("team_abbreviation_in")
    if not teams:
        return ""
    if isinstance(teams, str):
        team_list = _normalize_team_list([teams])
    elif isinstance(teams, list):
        team_list = _normalize_team_list([str(team) for team in teams])
    else:
        return ""

    if not team_list:
        return ""

    column = "team_abbreviation"
    if placeholder.endswith("_home"):
        column = "team_abbreviation_home"
    elif placeholder.endswith("_away"):
        column = "team_abbreviation_away"

    joined = ", ".join(f"'{team}'" for team in team_list)
    return f"AND {column} IN ({joined})"


def _render_metric_template(metric: MetricDefinition, planned_slots: dict[str, Any]) -> str | None:
    """
    메트릭 템플릿을 슬롯 값으로 채운다.

    Args:
        metric: 메트릭 정의.
        planned_slots: 플래너 슬롯.

    Returns:
        채워진 SQL 문자열(불가 시 None).
    """

    template = metric.sql_template
    if not template:
        return None

    placeholders = set(re.findall(r"{([a-zA-Z_]+)}", template))
    if not placeholders:
        return template

    values: dict[str, Any] = {}
    season = planned_slots.get("season")
    top_k = planned_slots.get("top_k") or 200
    filters = planned_slots.get("filters", {})

    for placeholder in placeholders:
        if placeholder in {"limit", "n"}:
            values[placeholder] = top_k
            continue
        if placeholder in {"season_year", "season"}:
            if not season:
                return None
            values[placeholder] = season
            continue
        if placeholder == "season_id":
            if not season:
                return None
            season_id = _season_year_to_id(season)
            if not season_id:
                return None
            values[placeholder] = season_id
            continue
        if placeholder in {"team_abbreviation", "team_a", "team_b"}:
            value = filters.get(placeholder)
            if not value:
                return None
            values[placeholder] = value
            continue
        if placeholder.startswith("team_filter"):
            values[placeholder] = _build_team_filter(filters, placeholder)
            continue
        if placeholder in filters:
            values[placeholder] = filters.get(placeholder)
            continue
        return None

    try:
        return template.format(**values)
    except KeyError:
        return None


async def _fill_recent_season(
    planned_slots: dict[str, Any],
    user_message: str,
    sqlite_client: SQLiteClient,
) -> None:
    """
    최근/최신 표현이 있을 때 시즌 값을 보강한다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.
        sqlite_client: SQLite 클라이언트.

    Returns:
        None
    """

    if planned_slots.get("season"):
        return

    date_range = planned_slots.get("date_range")
    lowered = user_message.lower()
    if date_range != "최근" and "최근" not in user_message and "latest" not in lowered:
        return

    latest_season = await _fetch_latest_season(sqlite_client)
    if latest_season:
        planned_slots["season"] = latest_season


async def _fetch_latest_season(sqlite_client: SQLiteClient) -> str | None:
    """
    사용 가능한 최신 시즌을 조회한다.

    Args:
        sqlite_client: SQLite 클라이언트.

    Returns:
        시즌 문자열(없으면 None).
    """

    sql = (
        "SELECT season_id FROM game "
        "WHERE season_type = 'Regular Season' "
        "ORDER BY season_id DESC LIMIT 1"
    )
    try:
        result = await asyncio.to_thread(sqlite_client.execute, sql)
    except Exception:
        return None

    if result.dataframe.empty or "season_id" not in result.dataframe.columns:
        return None

    value = result.dataframe.iloc[0]["season_id"]
    if value is None:
        return None
    return _season_id_to_year(str(value))


async def _normalize_season(planned_slots: dict[str, Any], sqlite_client: SQLiteClient) -> None:
    """
    시즌 값이 실제 데이터에 없으면 최신 시즌으로 보정한다.

    Args:
        planned_slots: 플래너 슬롯.
        sqlite_client: SQLite 클라이언트.

    Returns:
        None
    """

    season = planned_slots.get("season")
    if not season:
        return

    season_id = _season_year_to_id(season)
    if not season_id:
        return

    if await _season_id_exists(season_id, sqlite_client):
        return

    latest = await _fetch_latest_season(sqlite_client)
    if not latest:
        return

    planned_slots["season"] = latest
    filters = planned_slots.get("filters", {})
    filters["season_override"] = f"{season} -> {latest}"
    planned_slots["filters"] = filters


async def _apply_default_season(
    planned_slots: dict[str, Any],
    user_message: str,
    deps: AgentDependencies,
) -> None:
    """
    시즌이 비어있는 경우, 장기 메모리/최신 시즌으로 보강한다.

    핵심은 "아무 때나 채우지 않는다"는 점이다.
    - 프로필/드래프트처럼 시즌이 의미 없는 질의에 시즌을 억지로 붙이면 오히려 품질이 떨어진다.
    - 반대로 팀/리그 현황성 지표는 시즌이 없으면 SQL이 모호해져서 실패 확률이 높다.

    Args:
        planned_slots: 플래너 슬롯.
        user_message: 사용자 질문.
        deps: 에이전트 의존성.

    Returns:
        None
    """

    if planned_slots.get("season"):
        return

    if planned_slots.get("metric") == "team_win_pct_by_season" and _looks_like_season_comparison(user_message):
        return

    # "역대/커리어"처럼 시즌을 고정하면 뉘앙스가 망가지는 질문은 제외한다.
    lowered = user_message.lower()
    if any(keyword in user_message for keyword in ["역대", "커리어"]) or any(
        keyword in lowered for keyword in ["all-time", "career"]
    ):
        return

    metric_name = planned_slots.get("metric")
    if not isinstance(metric_name, str) or not metric_name:
        return

    metric = deps.registry.get(metric_name)
    if metric is None:
        return

    if not _metric_requires_season(metric):
        return

    preferred = deps.memory.long_term.get_default_season()
    season = preferred or await _fetch_latest_season(deps.sqlite_client)
    if not season:
        return

    planned_slots["season"] = season
    filters = planned_slots.get("filters", {})
    if isinstance(filters, dict):
        filters["season_default"] = season
        planned_slots["filters"] = filters


def _metric_requires_season(metric: MetricDefinition) -> bool:
    """
    메트릭이 시즌 조건을 필요로 하는지 판단한다.

    Args:
        metric: 메트릭 정의.

    Returns:
        시즌이 필요하면 True.
    """

    if metric.sql_template:
        placeholders = set(re.findall(r"{([a-zA-Z_]+)}", metric.sql_template))
        if placeholders.intersection({"season", "season_year", "season_id"}):
            return True

    # 템플릿이 없는 경우는 required_tables로 보수적으로 추정한다.
    seasonal_tables = {"game", "team_info_common"}
    return any(table in seasonal_tables for table in metric.required_tables)


async def _season_id_exists(season_id: str, sqlite_client: SQLiteClient) -> bool:
    """
    season_id가 데이터에 존재하는지 확인한다.

    Args:
        season_id: season_id 문자열.
        sqlite_client: SQLite 클라이언트.

    Returns:
        존재 여부.
    """

    sql = (
        "SELECT 1 FROM game "
        "WHERE season_type = 'Regular Season' "
        "AND season_id = '{season_id}' "
        "LIMIT 1"
    ).format(season_id=season_id)
    try:
        result = await asyncio.to_thread(sqlite_client.execute, sql)
    except Exception:
        return False
    return not result.dataframe.empty


def _season_year_to_id(season_year: str) -> str | None:
    """
    시즌 문자열(예: 2023-24)을 season_id(예: 22023)로 변환.

    Args:
        season_year: 시즌 문자열.

    Returns:
        season_id 문자열(없으면 None).
    """

    match = re.match(r"(20\d{2})\s*[-/\.]\s*\d{2}", season_year)
    if not match:
        return None
    start_year = match.group(1)
    return f"2{start_year}"


def _season_id_to_year(season_id: str) -> str | None:
    """
    season_id(예: 22023)를 시즌 문자열(예: 2023-24)로 변환.

    Args:
        season_id: season_id 문자열.

    Returns:
        시즌 문자열(없으면 None).
    """

    if not season_id.isdigit() or len(season_id) < 5:
        return None
    start_year = int(season_id[1:])
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def _get_metric_context(planned_slots: dict[str, Any], registry: MetricsRegistry) -> dict[str, Any]:
    """
    플래너 슬롯에서 메트릭 컨텍스트를 추출.

    Args:
        planned_slots: 플래너 슬롯.
        registry: 메트릭 레지스트리.

    Returns:
        메트릭 컨텍스트(없으면 빈 딕셔너리).
    """

    metric_name = planned_slots.get("metric")
    metric = registry.get(metric_name) if metric_name else None
    if metric is None:
        return {}
    return registry.build_sql_context(metric)


def _build_metric_aliases(registry: MetricsRegistry) -> list[str]:
    """
    레지스트리의 메트릭 이름/별칭을 평탄화.

    Args:
        registry: 메트릭 레지스트리.

    Returns:
        메트릭 이름/별칭 리스트.
    """

    aliases: list[str] = []
    for metric in registry.list_metrics():
        aliases.extend(metric.all_names())
    return sorted(set(aliases))


if __name__ == "__main__":
    # 1) 레지스트리 로드
    from pathlib import Path

    registry = MetricsRegistry(Path("src/metrics/metrics.yaml"))
    registry.ensure_loaded()

    # 2) 메트릭 템플릿 렌더링 확인
    metric = registry.get("win_pct")
    if metric:
        sql = _render_metric_template(metric, {"season": "2022-23", "top_k": 5, "filters": {}})
        print(sql)

    # 3) 필터 요약 확인
    summary = _build_filter_summary({"season": "2022-23", "filters": {"team": "BOS"}})
    print(summary)
