from __future__ import annotations

from pathlib import Path
import datetime as dt


def ensure_dir(path: str | Path) -> None:
    """폴더가 없으면 생성한다.

    Args:
        path: 생성할(보장할) 폴더 경로.

    Notes:
        - 부모 폴더까지 함께 생성한다(parents=True).
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def create_run_dir(base_dir: str | Path) -> Path:
    """타임스탬프 기반 실행 폴더를 생성하고 경로를 반환한다.

    예) base_dir="result/heptathlon" → "result/heptathlon/run_YYYYmmdd_HHMMSS"

    Args:
        base_dir: 결과 상위 폴더.

    Returns:
        Path: 생성된 실행 폴더 경로.
    """
    base = Path(base_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out
