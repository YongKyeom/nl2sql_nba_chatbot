# NBA NL2SQL 챗봇 (SQLite + Streamlit)

NBA SQLite DB를 기반으로 NL2SQL 에이전트를 제공하는 프로젝트입니다. 지표 정의는 `src/metrics/metrics.yaml`에 한글로 상세히 주입되며, 변경 시 코드 수정 없이 반영되도록 설계했습니다.

## 주요 기능

- Scene/Orchestrator 기반으로 Streamlit과 로컬 러너가 동일한 경로로 실행됩니다.
- LLM 라우터 + 플래너 + SQL 생성/가드 + 요약까지 end-to-end 파이프라인을 제공합니다.
- 단기/장기 메모리를 분리해 멀티턴 문맥과 선호(기본 시즌 등)를 관리합니다.
- SQL Guard로 SELECT-only를 강제하고, 품질 검증/재시도를 통해 안정성을 높입니다.
- 이전 결과 재사용(정렬/필터/Top-K)과 일반 안내 답변도 지원합니다.
- 멀티 스텝 플래닝으로 복합 질의를 단계별 SQL로 처리합니다(예: 관중 상위 팀 + 시즌 성적 결합).
- Thinking 패널로 단계별 상태를 확인하며, 최종 답변은 UI 스트리밍(청크 출력)으로 제공합니다.
- 결과 표는 앱에서 생성한 마크다운 테이블로 통일되어 CSV 다운로드와 동일한 데이터가 보이도록 합니다.
- 채팅 목록/제목 생성/수정/삭제를 지원하며, 첫 질문을 기반으로 채팅 제목을 한 번 자동 생성합니다.
- 그래프/차트/시각화 요청 시 📊 차트 섹션에 자동 시각화를 추가합니다(차트는 Matplotlib로 렌더링).
- 지원 차트 유형: line, area, bar, stacked_bar, scatter, histogram, box

## 데이터 출처

- Kaggle: https://www.kaggle.com/datasets/wyattowalsh/basketball

## 데이터셋 상세

### 포함 테이블

- game, game_info, game_summary, line_score, other_stats
- team, team_info_common, team_details, team_history
- common_player_info, player, inactive_players
- draft_history, draft_combine_stats
- play_by_play
- officials

### 제공 가능한 범위

- 팀 경기 스탯 기반 지표(득점/리바운드/어시스트/슈팅 효율 등)
- 팀 비교/순위/트렌드(시즌/기간 기준)
- 관중/흥행 지표(경기/팀 기준 집계)
- 드래프트/컴바인/선수 프로필 정보
- 플레이바이플레이 이벤트 기반 간접 지표(동시 관여 빈도 등)

### 제한 사항

- 선수 경기 로그(월별/경기별 득점·출전시간·슛 성공률) 테이블이 없어 해당 질의는 정확히 지원하지 않습니다.
- 질문이 데이터 범위를 벗어나면 확인 질문 또는 대체 제안을 제공합니다.

## 환경 설정

`.env` 파일에 API 키를 설정합니다(실제 키는 노출하지 마세요).

```
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
DB_PATH=data/nba.sqlite
MEMORY_DB_PATH=result/memory.sqlite
CHAT_DB_PATH=result/chat.sqlite
```

## 스키마 덤프

Streamlit 사이드바의 **Dump Schema** 버튼을 누르거나 아래 명령을 실행합니다.

```
python -m src.db.schema_dump
```

생성 파일:
- `result/schema.json`
- `result/schema.md`

## 실행 방법

```
streamlit run src/app.py
```

## Streamlit UI

- 모델: `gpt-4o-mini` 고정(선택형 UI 유지)
- Temperature: 0.1 ~ 2.0
- Dataset Info: 데이터셋 요약/출처/테이블/지표 목록 표시
- 추천 질의: 클릭으로 빠른 질문 실행
- 결과 요약 카드: Rows/Season/Metric/Top-K를 카드로 표시
- 표 출력 정책: 응답 본문에 포함된 마크다운 테이블을 기준으로 출력하며, 동일 데이터를 CSV로 제공합니다
- 차트 출력: 사용자가 그래프/차트/시각화를 요청하면 📊 차트 섹션에 자동 렌더링
- 차트 이미지 저장: `result/plot/{user_id}/{chat_id}/` 경로에 PNG로 저장(깃 추적 제외)
- 지원 차트 유형: line, area, bar, stacked_bar, scatter, histogram, box
- 채팅 목록: 좌측 사이드바에서 채팅 선택/제목 수정/삭제 가능
- Dump Schema: `schema.json` / `schema.md` 갱신
- Reset Conversation: 단기 메모리 초기화
- Reset Long-term Memory: 장기 메모리 초기화
- Thinking: 라우팅/플래닝/SQL 상태를 단계별로 표시(기본 접힘, Details 토글)

## 로컬 테스트 (Streamlit 없이)

Streamlit과 동일한 Scene/오케스트레이터 구성으로 로컬 실행합니다.

```
python src/test/test_agent_flow.py
```

## 단위 테스트

더미 구성으로 체인 로직만 빠르게 검증합니다.

```
python -m unittest -q src.test.test_chain_unit
```

## 예시 질문

### 지표 정의 질문 (Direct Answer)
1. "TS% 뭐야?"
2. "트리플더블 정의 알려줘"
3. "eFG% 계산식은?"
4. "백투백 기준이 뭐야?"
5. "승률 계산 공식 설명해줘"

### 데이터 질의 질문 (SQL)
1. "2023-24 시즌 팀 득점 상위 10개 보여줘"
2. "LAL 최근 5경기 결과 알려줘"
3. "접전 경기(5점 이하) 최근 10개 보여줘"
4. "팀 평균 리바운드 상위 5팀 알려줘"
5. "관중 수 상위 10경기 알려줘"

### 멀티턴 예시
1. "최근 리그에서 승률 상위 5개 팀 알려줘" → "상위 3개만"
2. "앞으로 기본 시즌은 2023-24로 해줘" → "승률 상위 5팀 보여줘"

## 메모리 구조

- 단기 메모리: 마지막 SQL/결과/슬롯 + 최근 대화 일부를 유지합니다(채팅 세션 단위).
- 단기 메모리에는 직전 결과에서 추출한 팀/선수 엔티티도 포함되어 참조 질의를 보강합니다.
- 장기 메모리: 자주 묻는 시즌/팀/지표 등 선호를 사용자 단위로 SQLite에 누적합니다(앱 재시작 이후에도 유지).
- 장기 메모리는 `MEMORY_DB_PATH`로 경로를 변경할 수 있으며, Streamlit에서 별도 초기화 버튼을 제공합니다.

## 채팅 세션

- 채팅 목록은 `CHAT_DB_PATH` SQLite에 저장됩니다.
- 첫 질문 입력 후 제목을 한 번 자동 생성하고 이후 고정합니다.
- 채팅별로 메시지/표/메타가 저장되어 새로고침 없이 복원됩니다.

## 로컬 SQLite 가이드 (Agent 메모리 / Chat 히스토리)

### 생성 시점

- `result/chat.sqlite` (Chat 히스토리): Streamlit 실행 중 최초로 채팅을 생성할 때 자동으로 만들어집니다.
- `result/memory.sqlite` (Agent 메모리): 장기 메모리에 선호/프로필을 기록하는 순간 자동으로 만들어집니다.
- 별도의 수동 생성 작업은 필요하지 않습니다. 로컬 실행 또는 테스트 실행만으로 생성됩니다.

### 스키마 요약

- `chat.sqlite` (Chat 히스토리)
  - `chat_sessions`: 채팅 목록 메타(사용자 ID, 제목, 생성/갱신 시각).
  - `chat_messages`: 채팅별 메시지/메타 JSON 기록.
- `memory.sqlite` (Agent 메모리)
  - `preference_counts`: 선호 카운트(카테고리, 값, 횟수).
  - `user_profile`: 사용자 기본값(예: 기본 시즌).

## 비즈니스 로직 주입

- 지표 정의/컷 규칙/SQL 템플릿은 `src/metrics/metrics.yaml`에 한글로 상세히 정의됩니다.
- 레지스트리(`src/metrics/registry.py`)가 해당 정의를 읽어 Direct Answer 및 SQL 생성 컨텍스트에 주입합니다.

## Fewshot 추가 방법

1. 기본 예시는 `src/prompt/sql_generation.py`의 `SQL_FEWSHOT_EXAMPLES`에서 관리합니다.
2. 런타임에는 `FewshotGenerator`가 사용자 질문과 후보 메트릭을 기반으로 few-shot을 동적으로 생성합니다.
3. `FEWSHOT_CANDIDATE_LIMIT` 환경 변수로 후보 sql_template 개수를 조절할 수 있습니다(기본값 5).
4. 예시에 쓰는 테이블/컬럼은 `result/schema.json`에 존재해야 합니다.
5. 특정 지표의 정확도를 높이려면 `src/metrics/metrics.yaml`의 `aliases`와 `sql_template`도 함께 보강합니다.

예시 추가 패턴:
```
[예시 N] 설명
질문: ...
SQL:
SELECT ...
LIMIT 50;
```

## Agent 아키텍처
```mermaid
flowchart TD
  U["User"] --> S["Scene"]
  S --> O["Orchestrator"]
  O --> M["Memory<br/>단기(최근 결과/엔티티)<br/>장기(선호/기본값)"]

  O --> R["Router"]
  R --> GA["General Answer"]
  R --> DA["Direct Answer"]
  R --> RE["Reuse Answer"]
  R --> P["Planner"]

  P -->|확인 필요| C["Clarifier"]
  C --> P

  P -->|멀티 스텝| MS["Multi-step SQL Runner"]
  P -->|단일 SQL| FS["Fewshot Generator"]
  FS --> SG["SQL Generator"]

  O --> TS["Title Generator"]
  O --> CS["Chat Store"]

  MS --> G["SQL Guard"]
  SG --> G
  G --> EX["DB Execute"]
  EX --> V["Result Validator"]
  V --> SUM["Summarizer"]
  SUM --> RESP["Responder"]

  GA --> U
  DA --> U
  RE --> U
  RESP --> U

  GA --> M
  DA --> M
  RE --> M
  RESP --> M
```

### 에이전트 상세 설명
- **Scene (`src/agent/scene.py`)**: Streamlit/로컬 테스트에서 동일한 실행 경로를 제공하는 래퍼입니다. `ask()`로 오케스트레이터를 호출하고, `reset()`으로 대화 메모리를 초기화합니다.
- **Orchestrator (`src/agent/orchestrator.py`)**: 레지스트리/스키마 로딩, 체인 구성, 스트리밍 실행을 책임집니다. 예외 발생 시 안전한 폴백 응답을 만들고, 턴 시작/종료 시 메모리를 업데이트합니다.
- **Router (`src/agent/router.py`)**: LLM 라우터가 기본이며, JSON 파싱 실패 시 키워드 폴백을 수행합니다. 메트릭 별칭을 통해 Direct/Reuse/SQL_REQUIRED를 분기하고, `route_reason`을 남깁니다.
- **Planner (`src/agent/planner.py`)**: 엔티티/시즌/기간/지표/Top-K 슬롯을 구성하고, 부족하면 `clarify_question`을 제공합니다. 직전 슬롯을 참고해 멀티턴 추론을 보강합니다.
- **Clarifier (`src/prompt/clarify.py`)**: 플래너가 부족한 정보를 발견했을 때 확인 질문을 생성합니다.
- **Reuse Answer (`src/agent/router.py` 내 `apply_reuse_rules`)**: 직전 결과에 대해 정렬/필터/Top-K 후처리를 적용합니다. SQL 재실행 없이 DataFrame을 변형하고, 요약 문자열을 남깁니다.
- **Direct Answer (`src/agent/responder.py`)**: 메트릭 레지스트리 정의를 기반으로 지표 설명 응답을 생성합니다.
- **General Answer (`src/agent/responder.py`)**: 가능한 데이터 범위와 사용법을 일반 안내로 응답합니다.
- **Multi-step SQL Runner (`src/agent/chain.py`)**: 복합 질의를 단계별 SQL로 분해합니다. 예: 관중 상위 팀을 먼저 추출한 뒤 해당 팀 성적을 2단계로 결합합니다.
- **SQL Generator (`src/agent/sql_generator.py`)**: 스키마/메트릭 정의/퓨샷을 바탕으로 SQL을 생성하고, 필요 시 수정을 재시도합니다.
- **SQL Guard (`src/agent/guard.py`)**: SELECT-only/조건 없는 JOIN 금지/LIMIT 강제 등 안전 규칙을 적용합니다.
- **DB Execute (`src/db/sqlite_client.py`)**: SQL을 실행하고 DataFrame을 반환합니다.
- **Result Validator (`src/agent/validator.py`)**: 0행/컬럼 없음/NULL 과다를 감지해 재시도 여부를 결정합니다.
- **Summarizer (`src/agent/summarizer.py`)**: 결과 미리보기와 적용 필터를 바탕으로 요약 응답을 생성합니다.
- **Responder (`src/agent/responder.py`)**: Direct/Clarify/Reuse/General/Missing Metric 응답을 프롬프트 기반으로 생성합니다.
- **Title Generator (`src/agent/title_generator.py`)**: 첫 질문을 기반으로 채팅 제목을 1회 생성합니다.
- **Chat Store (`src/db/chat_store.py`)**: 채팅 세션/메시지를 SQLite에 저장해 새로고침 후에도 복원합니다.

### 메모리 동작 방식
1. **턴 시작**: 오케스트레이터가 `memory.start_turn()`으로 사용자 입력 원문을 단기 메모리에 기록합니다.
2. **라우팅 보강**: 라우터는 직전 SQL/결과/슬롯을 참고해 Reuse 여부를 판단하고, 사용 가능한 메트릭 별칭을 함께 고려합니다.
3. **플래닝 보강**: 플래너는 직전 슬롯을 참고해 생략된 시즌/엔티티를 추정합니다. 체인은 참조 필터(직전 결과 팀/선수)를 반영합니다.
4. **결과 저장**: SQL 실행 후 `update_sql_result()`로 마지막 SQL, 결과 DataFrame, 결과 스키마, 슬롯을 저장하고 엔티티를 추출합니다.
5. **턴 종료 + 학습**: `finish_turn()`에서 응답/라우트/SQL/슬롯을 기록하고, 장기 메모리에 선호 신호를 누적합니다.
6. **장기 메모리 구성**: `preference_counts`에는 시즌/팀/지표의 빈도, `user_profile`에는 기본 시즌/선호 팀 같은 명시적 기본값이 저장됩니다.
7. **기본값 적용**: 다음 플래닝 시 장기 메모리의 기본 시즌/선호 팀이 슬롯과 필터에 자동 반영됩니다.
8. **리셋 정책**: 새 채팅 전환 시 단기 메모리만 초기화하고, 장기 메모리는 Reset 버튼에서만 삭제됩니다.
