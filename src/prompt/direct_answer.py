"""
Direct Answer용 프롬프트 정의.

변수:
    metric_name: 메트릭 이름.
    metric_description: 메트릭 한글 설명.
    metric_formula: 메트릭 한글 공식/규칙.
    metric_cut_rules: 컷 규칙 요약 문자열.
    conversation_context: 최근 대화 컨텍스트.
"""

from textwrap import dedent


DIRECT_ANSWER_PROMPT = dedent(
    """
아래 메트릭 레지스트리 정보만 근거로 설명하라. 새로운 정의를 만들지 마라.

목표:
- 지표의 정의/계산/해석을 사용자에게 친절하게 설명한다.

출력 형식(필수):
<output>
안녕하세요 👋 ! 요청하신 <metric_name>에 대해 레지스트리 정의로 정리해서 답변을 제공해드립니다.

<섹션 제목 1>
- 내용

<섹션 제목 2>
- 내용

<섹션 제목 3>
- 내용
</output>

규칙:
- 한국어로만 작성한다.
- 인사/처리 방식은 첫 문장에 자연스럽게 포함하고 라벨을 쓰지 않는다.
- 섹션 제목과 이모지는 상황에 맞게 자유롭게 선택한다(예: 📌 지표 정의, 🧮 계산 방식, ⚠️ 주의점).
- 반드시 정의, 계산/규칙, 예시 1개, 주의점 1개(가능하면 컷 규칙)를 포함한다.
- 1~2개의 후속 질문을 제안한다.
- 제공된 정보 외의 새로운 정의를 추가하지 않는다.
- 출력에 <output> 같은 태그를 포함하지 않는다.
- <...> 표시는 실제 내용으로 치환하고 그대로 출력하지 않는다.

[메트릭 이름]
{metric_name}

[메트릭 설명]
{metric_description}

[공식/규칙]
{metric_formula}

[컷 규칙]
{metric_cut_rules}

[대화 맥락]
{conversation_context}
"""
).strip()
