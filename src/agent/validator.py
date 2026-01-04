from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """
    실행 결과 검증 결과.

    Args:
        is_valid: 유효 여부.
        should_retry: 재시도 권장 여부.
        reason: 실패 사유(유효하면 빈 문자열).
    """

    is_valid: bool
    should_retry: bool
    reason: str


class ResultValidator:
    """
    SQL 실행 결과를 검증한다.

    0행/컬럼 없음/NULL 과다 등 품질 이슈를 감지해 재시도 여부를 판단한다.
    - 0행: 조건이 지나치게 엄격하거나 조인 조건이 잘못된 경우가 많아 재시도 대상.
    - 컬럼 없음: SELECT 절 구성 실패 가능성이 높아 재시도 대상.
    - NULL 과다: 의미 있는 값이 거의 없어 요약이 왜곡될 수 있으므로 재시도 대상.
    """

    def __init__(self, *, null_ratio_threshold: float = 0.7) -> None:
        """
        ResultValidator 초기화.

        Args:
            null_ratio_threshold: NULL 비율 임계값.
        """

        self._null_ratio_threshold = null_ratio_threshold

    def validate(self, dataframe: pd.DataFrame) -> ValidationResult:
        """
        결과 데이터프레임을 검증.

        Args:
            dataframe: SQL 실행 결과.

        Returns:
            ValidationResult.

        Notes:
            NULL 비율 임계값은 기본 0.7로 설정하며, 전체 컬럼이 임계값을
            초과하는 경우에만 재시도를 권고한다. 일부 컬럼만 NULL이 많은
            경우는 요약에 활용 가능한 값이 남아있을 수 있어 허용한다.
        """

        if dataframe.empty:
            return ValidationResult(False, True, "결과가 0행입니다. 조건을 완화해야 합니다.")

        if dataframe.columns.empty:
            return ValidationResult(False, True, "결과 컬럼이 없습니다. SELECT 항목을 확인해야 합니다.")

        null_ratio = dataframe.isna().mean()
        if (null_ratio > self._null_ratio_threshold).all():
            return ValidationResult(False, True, "NULL 비율이 과도하게 높습니다. 컬럼 선택을 재검토해야 합니다.")

        return ValidationResult(True, False, "")


if __name__ == "__main__":
    # 1) 샘플 데이터 준비
    df_empty = pd.DataFrame()
    df_null = pd.DataFrame({"a": [None, None], "b": [None, None]})
    df_valid = pd.DataFrame({"a": [1, 2], "b": [3, None]})

    # 2) 검증 결과 확인
    validator = ResultValidator()
    print("empty:", validator.validate(df_empty))
    print("null:", validator.validate(df_null))
    print("valid:", validator.validate(df_valid))
