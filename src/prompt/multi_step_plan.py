"""
멀티 스텝 계획 생성 프롬프트 정의.

변수:
    user_question: 사용자 질문 원문.
    planned_slots: 플래너 슬롯.
    schema_context: 스키마 요약.
    context_hint: 대화 맥락 요약.
"""

from textwrap import dedent


MULTI_STEP_PLAN_PROMPT = dedent(
    """
너는 멀티 스텝 SQL 계획을 결정하는 에이전트다.

목표:
- 질문이 단일 SQL로 해결 가능한지, 멀티 스텝이 필요한지 판단한다.
- 멀티 스텝이 필요하면 단계별 계획(steps)을 제시한다.

규칙:
- 멀티 스텝이 필요 없으면 use_multi_step=false로 반환한다.
- steps는 2~4개로 제한한다.
- 이전 결과를 사용하는 경우 filters에 "__from_previous__"를 넣는다.
- metric은 반드시 metrics.yaml의 name 또는 alias 중 하나로만 지정한다.
- 이전 단계 결과 목록을 참조할 때는 team_abbreviation_in / player_name_in 키를 사용한다.
- 출력은 JSON 한 줄로만 응답한다.

출력 형식(필수):
{{"use_multi_step":true,"reason":"...","steps":[{{"question":"...","overrides":{{...}},"filters":{{...}}}},...],"combine":{{"method":"left_join","left_step":1,"right_step":2,"on":["team_abbreviation"]}}}}
또는
{{"use_multi_step":false,"reason":"..."}}

[사용자 질문]
{user_question}

[플래너 슬롯]
{planned_slots}

[스키마 요약]
{schema_context}

[대화 맥락]
{context_hint}
"""
).strip()
