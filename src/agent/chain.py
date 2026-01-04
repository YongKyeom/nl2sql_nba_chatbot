from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.agent.contextualizer import build_context_hint
from src.agent.guard import SQLGuard
from src.agent.memory import ConversationMemory
from src.agent.planner import Planner, PlannerOutput, PlannerSlots
from src.agent.responder import Responder
from src.agent.router import apply_reuse_rules, Route, RouterLLM, RoutingContext
from src.agent.sql_generator import SQLGenerationInput, SQLGenerator, SQLRepairInput
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
    metric_name: str | None
    planned_slots: dict[str, Any]
    clarify_question: str | None
    sql: str | None
    result_df: pd.DataFrame | None
    last_result_schema: list[str] | None
    final_answer: str | None
    error: str | None
    error_detail: dict[str, Any] | None


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

    graph.add_node("router", lambda state: _router_node(state, deps))
    graph.add_node("general_answer", lambda state: _general_answer_node(state, deps))
    graph.add_node("direct_answer", lambda state: _direct_answer_node(state, deps))
    graph.add_node("reuse", lambda state: _reuse_node(state, deps))
    graph.add_node("plan", lambda state: _plan_node(state, deps))
    graph.add_node("clarify", lambda state: _clarify_node(state, deps))
    graph.add_node("generate_sql", lambda state: _generate_sql_node(state, deps))
    graph.add_node("guard", lambda state: _guard_node(state, deps))
    graph.add_node("execute", lambda state: _execute_node(state, deps))
    graph.add_node("validate", lambda state: _validate_node(state, deps))
    graph.add_node("summarize", lambda state: _summarize_node(state, deps))
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
        _clarify_selector,
        {
            "clarify": "clarify",
            "continue": "generate_sql",
        },
    )

    graph.add_edge("generate_sql", "guard")
    graph.add_conditional_edges(
        "guard",
        _error_selector,
        {
            "error": "finalize_error",
            "continue": "execute",
        },
    )

    graph.add_edge("execute", "validate")
    graph.add_conditional_edges(
        "validate",
        _error_selector,
        {
            "error": "finalize_error",
            "continue": "summarize",
        },
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("general_answer", END)
    graph.add_edge("reuse", END)
    graph.add_edge("clarify", END)
    graph.add_edge("summarize", END)
    graph.add_edge("finalize_error", END)

    return graph.compile()


def _router_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
        has_previous=deps.memory.last_result is not None,
        last_result_schema=deps.memory.last_result_schema,
        last_sql=deps.memory.last_sql,
        last_slots=deps.memory.last_slots,
        available_metrics=_build_metric_aliases(deps.registry),
    )

    result = deps.router.route(context, deps.registry)

    deps.memory.update_route(result.route.value)
    return {"route": result.route.value, "metric_name": result.metric_name}


def _direct_answer_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
            return {"final_answer": deps.responder.compose_missing_metric(state["user_message"])}

        return {"final_answer": deps.responder.compose_direct(metric)}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Direct Answer 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "지표 설명을 생성하는 중 오류가 발생했습니다.",
        }


def _general_answer_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    일반 안내 응답 노드.

    Args:
        state: 현재 상태.
        deps: 의존성.

    Returns:
        업데이트된 상태.
    """

    try:
        return {"final_answer": deps.responder.compose_general(state["user_message"])}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"일반 응답 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "안내 답변을 생성하는 중 오류가 발생했습니다.",
        }


def _reuse_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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

    preview_records = preview_dataframe(reuse.dataframe, max_rows=10)
    preview_markdown = records_to_markdown(preview_records)

    try:
        response_text = deps.responder.compose_reuse(state["user_message"], reuse.summary, preview_markdown)
    except Exception as exc:  # noqa: BLE001
        return {
            "result_df": reuse.dataframe,
            "sql": deps.memory.last_sql,
            "last_result_schema": deps.memory.last_result_schema,
            "error": f"후처리 응답 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "후처리 결과를 생성하는 중 오류가 발생했습니다.",
        }

    return {
        "result_df": reuse.dataframe,
        "sql": deps.memory.last_sql,
        "last_result_schema": deps.memory.last_result_schema,
        "final_answer": response_text,
    }


def _plan_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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

    plan_output: PlannerOutput = deps.planner.plan(state["user_message"], previous)
    planned_slots = plan_output.slots.model_dump()
    _fill_recent_season(planned_slots, state["user_message"], deps.sqlite_client)
    _normalize_season(planned_slots, deps.sqlite_client)
    _apply_default_season(planned_slots, state["user_message"], deps)

    return {
        "planned_slots": planned_slots,
        "clarify_question": plan_output.clarify_question,
    }


def _clarify_node(state: AgentState, deps: AgentDependencies) -> AgentState:
    """
    확인 질문 노드.

    Args:
        state: 현재 상태.

    Returns:
        업데이트된 상태.
    """

    clarify_question = state.get("clarify_question", "추가 확인이 필요합니다.")
    try:
        response_text = deps.responder.compose_clarify(state["user_message"], clarify_question)
        return {"final_answer": response_text}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"확인 질문 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": clarify_question,
        }


def _generate_sql_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
        return {"sql": template_sql}

    try:
        sql = deps.sql_generator.generate_sql(
            SQLGenerationInput(
                user_question=state["user_message"],
                planned_slots=slots,
                metric_context=deps.registry.build_sql_context(metric),
                schema_context=deps.schema_store.build_context(),
                last_sql=deps.memory.last_sql,
                context_hint=build_context_hint(deps.memory),
            )
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"SQL 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "SQL 생성 중 오류가 발생했습니다.",
        }
    return {"sql": sql}


def _guard_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
        repair_sql = deps.sql_generator.repair_sql(
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


def _execute_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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

    for _ in range(MAX_SQL_RETRIES + 1):
        try:
            result = deps.sqlite_client.execute(sql)
            deps.memory.update_sql_result(result.executed_sql, result.dataframe, state.get("planned_slots", {}))
            return {
                "sql": result.executed_sql,
                "result_df": result.dataframe,
                "last_result_schema": result.columns,
            }
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"SQL 실행 실패: {exc}"
            last_traceback = traceback.format_exc()
            repair_sql = deps.sql_generator.repair_sql(
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
        "error_detail": {"exception": failure_reason, "traceback": last_traceback},
        "final_answer": "쿼리 실행 중 오류가 발생했습니다.",
    }


def _validate_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
        repair_sql = deps.sql_generator.repair_sql(
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
            retry_result = deps.sqlite_client.execute(guard_result.sql)
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


def _summarize_node(state: AgentState, deps: AgentDependencies) -> AgentState:
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
        summary = deps.summarizer.summarize(
            SummaryInput(
                user_question=state["user_message"],
                sql=state.get("sql", ""),
                result_preview=preview_dataframe(dataframe, max_rows=10),
                applied_filters=applied_filters,
            )
        )
        return {"final_answer": summary}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"요약 생성 실패: {exc}",
            "error_detail": {"exception": str(exc), "traceback": traceback.format_exc()},
            "final_answer": "요약 생성 중 오류가 발생했습니다.",
        }


def _finalize_error_node(state: AgentState) -> AgentState:
    """
    에러 처리 노드.

    Args:
        state: 현재 상태.

    Returns:
        업데이트된 상태.
    """

    if state.get("final_answer"):
        return {}
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


def _clarify_selector(state: AgentState) -> str:
    """
    확인 질문 여부에 따른 분기 선택.

    Args:
        state: 현재 상태.

    Returns:
        분기 키.
    """

    return "clarify" if state.get("clarify_question") else "continue"


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

    if season:
        parts.append(f"시즌: {season}")
    if date_range:
        parts.append(f"기간: {date_range}")
    if filters:
        filters_text = ", ".join(f"{key}={value}" for key, value in filters.items())
        parts.append(f"필터: {filters_text}")

    return " / ".join(parts) if parts else "적용 조건 없음"


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
        return None

    try:
        return template.format(**values)
    except KeyError:
        return None


def _fill_recent_season(
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

    latest_season = _fetch_latest_season(sqlite_client)
    if latest_season:
        planned_slots["season"] = latest_season


def _fetch_latest_season(sqlite_client: SQLiteClient) -> str | None:
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
        result = sqlite_client.execute(sql)
    except Exception:
        return None

    if result.dataframe.empty or "season_id" not in result.dataframe.columns:
        return None

    value = result.dataframe.iloc[0]["season_id"]
    if value is None:
        return None
    return _season_id_to_year(str(value))


def _normalize_season(planned_slots: dict[str, Any], sqlite_client: SQLiteClient) -> None:
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

    if _season_id_exists(season_id, sqlite_client):
        return

    latest = _fetch_latest_season(sqlite_client)
    if not latest:
        return

    planned_slots["season"] = latest
    filters = planned_slots.get("filters", {})
    filters["season_override"] = f"{season} -> {latest}"
    planned_slots["filters"] = filters


def _apply_default_season(
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
    season = preferred or _fetch_latest_season(deps.sqlite_client)
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


def _season_id_exists(season_id: str, sqlite_client: SQLiteClient) -> bool:
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
        result = sqlite_client.execute(sql)
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
