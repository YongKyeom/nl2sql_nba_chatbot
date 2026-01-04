# NBA NL2SQL 챗봇 (SQLite + Streamlit)

NBA SQLite DB를 기반으로 NL2SQL 에이전트를 제공하는 프로젝트입니다. 지표 정의는 `src/metrics/metrics.yaml`에 한글로 상세히 주입되며, 변경 시 코드 수정 없이 반영되도록 설계했습니다.

## 주요 기능

- Scene/Orchestrator 기반으로 Streamlit과 로컬 러너가 동일한 경로로 실행됩니다.
- LLM 라우터 + 플래너 + SQL 생성/가드 + 요약까지 end-to-end 파이프라인을 제공합니다.
- 단기/장기 메모리를 분리해 멀티턴 문맥과 선호(기본 시즌 등)를 관리합니다.
- SQL Guard로 SELECT-only를 강제하고, 품질 검증/재시도를 통해 안정성을 높입니다.
- 이전 결과 재사용(정렬/필터/Top-K)과 일반 안내 답변도 지원합니다.

## 환경 설정

`.env` 파일에 API 키를 설정합니다(실제 키는 노출하지 마세요).

```
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
DB_PATH=data/nba.sqlite
MEMORY_DB_PATH=result/memory.sqlite
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
- Temperature: 0.0 ~ 0.3
- Dataset Info: 데이터셋 요약/테이블/지표 목록 표시
- Dump Schema: `schema.json` / `schema.md` 갱신
- Reset Conversation: 단기 메모리 초기화
- Reset Long-term Memory: 장기 메모리 초기화

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

- 단기 메모리: 마지막 SQL/결과/슬롯 + 최근 대화 일부를 유지합니다(세션 단위).
- 장기 메모리: 자주 묻는 시즌/팀/지표 등 선호를 SQLite로 누적합니다(앱 재시작 이후에도 유지).
- 장기 메모리는 `MEMORY_DB_PATH`로 경로를 변경할 수 있으며, Streamlit에서 별도 초기화 버튼을 제공합니다.

## 비즈니스 로직 주입

- 지표 정의/컷 규칙/SQL 템플릿은 `src/metrics/metrics.yaml`에 한글로 상세히 정의됩니다.
- 레지스트리(`src/metrics/registry.py`)가 해당 정의를 읽어 Direct Answer 및 SQL 생성 컨텍스트에 주입합니다.

## Agent 아키텍처
```mermaid
flowchart TD
  U["User"] --> S["Scene"]
  S --> A["Orchestrator"]
  A --> CM["Memory<br/>단기(최근 결과/슬롯)<br/>장기(선호/기본값)"]

  CM --> R["Router<br/>무엇을 조회해야 답이 나오나?"]

  %% 1) Knowledge Retrieval only (Direct Answer but grounded)
  R -->|지식/정의/가이드| KR["Knowledge Retriever<br/>Metric Registry + Docs/KB"]
  KR --> DA["Direct Answer Composer<br/>정의 + 예시 + 적용룰"]
  DA --> U
  DA --> CM

  %% 2) Previous result reuse
  R -->|이전 결과로 해결| PR["Post-Process<br/>필터/정렬/TopK/포맷변환"]
  PR --> U
  PR --> CM

  %% 3) Need DB query (SQL)
  R -->|DB 조회 필요| P["Planner<br/>의도/슬롯 확정"]
  P -->|부족한 슬롯| Q["Clarifier<br/>확인질문 0~1개"]
  Q --> P

  P --> ER["Entity Resolver<br/>선수/팀 표준화"]
  P --> KR2["Knowledge Retriever<br/>Metric Registry + Schema"]
  ER --> SG["SQL Generator"]
  KR2 --> SG

  SG --> GV["SQL Guard<br/>SELECT-only / whitelist / LIMIT"]
  GV --> RUN["DB Execute"]
  RUN --> SV["Sanity Validator"]
  SV --> OUT["Answer Composer<br/>결과 + 요약 + 적용룰"]
  OUT --> U
  OUT --> CM
```
