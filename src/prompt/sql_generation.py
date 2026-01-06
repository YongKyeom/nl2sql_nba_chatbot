"""
SQL 생성 프롬프트 정의.

변수:
    user_question: 사용자 질문 원문.
    planned_slots: 플래너가 추출한 슬롯 정보.
    metric_context: 레지스트리 기반 메트릭 컨텍스트.
    schema_context: schema_store에서 생성한 스키마 요약.
    last_sql: 직전 SQL(있을 때만).
    context_hint: 대화 맥락 요약.
    fewshot_examples: SQL 생성 예시.
"""

from textwrap import dedent


SQL_FEWSHOT_EXAMPLES = dedent(
    """
    [예시 1] 시즌별 팀 득점 상위
    질문: 2023-24 시즌 팀 득점 상위 10개 보여줘
    SQL:
    SELECT team_name, team_abbreviation, season_year, pts_pg
    FROM team_info_common
    WHERE season_year = '2023-24'
    ORDER BY pts_pg DESC
    LIMIT 10;

    [예시 2] 드래프트 1픽 평균 커리어
    질문: 드래프트 1픽 평균 커리어 길이 알려줘
    SQL:
    SELECT AVG(CAST(cpi.season_exp AS REAL)) AS avg_season_exp
    FROM draft_history AS dh
    JOIN common_player_info AS cpi ON cpi.person_id = dh.person_id
    WHERE dh.overall_pick = 1
    LIMIT 200;

    [예시 3] 특정 연도 드래프트 전체 픽
    질문: 2018 드래프트 전체 픽 리스트 보여줘
    SQL:
    SELECT season, overall_pick, player_name, team_name, team_abbreviation
    FROM draft_history
    WHERE season = '2018'
    ORDER BY overall_pick
    LIMIT 200;

    [예시 4] 선수 프로필 조회
    질문: Stephen Curry 프로필 알려줘
    SQL:
    SELECT display_first_last, position, height, weight, country, school, season_exp, from_year, to_year, team_name
    FROM common_player_info
    WHERE display_first_last LIKE '%Stephen Curry%'
    LIMIT 50;

    [예시 5] 컴바인 스탯 조회
    질문: Victor Wembanyama 컴바인 스탯 보여줘
    SQL:
    SELECT player_name, position, height_wo_shoes, height_w_shoes, weight, wingspan, standing_reach, body_fat_pct
    FROM draft_combine_stats
    WHERE player_name LIKE '%Victor Wembanyama%'
    ORDER BY season DESC
    LIMIT 50;

    [예시 6] 팀 맞대결
    질문: LAL과 BOS 맞대결 최근 기록 알려줘
    SQL:
    SELECT game_id, game_date, team_abbreviation_home, team_abbreviation_away, pts_home, pts_away, wl_home, wl_away
    FROM game
    WHERE (team_abbreviation_home = 'LAL' AND team_abbreviation_away = 'BOS')
       OR (team_abbreviation_home = 'BOS' AND team_abbreviation_away = 'LAL')
    ORDER BY game_date DESC
    LIMIT 50;

    [예시 7] 선수 이벤트 맞대결
    질문: LeBron James vs Stephen Curry 플레이바이플레이 이벤트 빈도 알려줘
    SQL:
    SELECT player1_name, player2_name, COUNT(*) AS event_count
    FROM play_by_play
    WHERE (player1_name LIKE '%LeBron James%' AND player2_name LIKE '%Stephen Curry%')
       OR (player1_name LIKE '%Stephen Curry%' AND player2_name LIKE '%LeBron James%')
    GROUP BY player1_name, player2_name
    ORDER BY event_count DESC
    LIMIT 50;
    """
).strip()


SQL_GENERATION_PROMPT = dedent(
    """
SQLite용 SQL을 생성하라.

목표:
- 사용자 질문을 schema/metrics.yaml 컨텍스트에 맞춰 SQL로 변환한다.

필수 규칙:
- 지표 정의는 metrics.yaml 레지스트리를 따른다.
- 테이블/컬럼은 schema.json에 존재하는 것만 사용한다.
- 모르면 추측하지 말고 확인 질문을 요청하거나 schema_store.search 결과를 요구하라.
- 출력은 SQL만, 코드블록/설명/주석 금지.
- 항상 LIMIT을 포함한다(기본 200).
- 이전 대화 맥락이 있으면 생략된 조건을 보완하되, 충돌하면 사용자 질문을 우선한다.
- planned_slots에 season 값이 있으면 season_year/season 조건에 그대로 사용한다.
- column_parser 도구가 제공되면 SQL 생성 후 호출해 테이블/컬럼 정합성을 검증한다.
  - 오류가 있으면 수정한 SQL로 다시 시도한 뒤 최종 SQL만 출력한다.

출력:
- SQL 단일 문자열만 반환한다.

[사용자 질문]
{user_question}

[플래너 슬롯]
{planned_slots}

[메트릭 컨텍스트]
{metric_context}

[스키마 요약]
{schema_context}

[직전 SQL]
{last_sql}

[대화 맥락]
{context_hint}

[Few-shot 예시]
{fewshot_examples}
"""
).strip()
