from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from src.chart.types import ChartSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ChartImageResult:
    """
    차트 이미지 생성 결과.

    Args:
        path: 저장된 이미지 경로.
        error: 실패 사유(path/data).

    Side Effects:
        None

    Raises:
        예외 없음.
    """

    path: str | None
    error: str | None


@dataclass(frozen=True)
class ChartRenderer:
    """
    Matplotlib 기반 차트 이미지를 생성/저장한다.

    Args:
        output_root: 이미지 저장 루트 경로.

    Side Effects:
        None

    Raises:
        예외 없음.
    """

    output_root: Path = Path("result/plot")

    def __post_init__(self) -> None:
        """
        Matplotlib 기본 폰트를 설정한다.

        Side Effects:
            Matplotlib rcParams를 갱신한다.

        Raises:
            예외 없음.
        """

        self._configure_matplotlib_fonts()

    def prepare_chart_image(
        self,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
        *,
        user_id: str,
        chat_id: str,
        existing_path: str | None = None,
    ) -> ChartImageResult:
        """
        차트 이미지를 준비하고 저장 경로를 반환한다.

        Args:
            dataframe: 결과 데이터프레임.
            chart_spec: 차트 스펙.
            user_id: 사용자 ID.
            chat_id: 채팅 ID.
            existing_path: 기존 경로가 있으면 재사용한다.

        Returns:
            ChartImageResult.

        Side Effects:
            이미지 파일을 디스크에 저장할 수 있다.

        Raises:
            예외 없음.
        """

        if existing_path:
            path = Path(existing_path)
            if path.exists():
                return ChartImageResult(path=str(path), error=None)

        self._configure_matplotlib_fonts()

        image_path = self._build_image_path(user_id, chat_id)
        if image_path is None:
            return ChartImageResult(path=None, error="path")

        saved = self._save_chart_image(dataframe, chart_spec, image_path)
        if not saved:
            return ChartImageResult(path=None, error="data")

        return ChartImageResult(path=image_path, error=None)

    def _save_chart_image(self, dataframe: pd.DataFrame, chart_spec: ChartSpec, image_path: str) -> bool:
        """
        차트 이미지를 저장한다.

        Args:
            dataframe: 결과 데이터프레임.
            chart_spec: 차트 스펙.
            image_path: 저장할 이미지 경로.

        Returns:
            저장 성공 여부.

        Side Effects:
            이미지 파일을 디스크에 저장한다.

        Raises:
            예외 없음.
        """

        figure = self._build_matplotlib_figure(dataframe, chart_spec)
        if figure is None:
            return False

        path = Path(image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return True

    def _build_matplotlib_figure(self, dataframe: pd.DataFrame, chart_spec: ChartSpec) -> "Figure | None":
        """
        Matplotlib Figure를 생성한다.

        Args:
            dataframe: 결과 데이터프레임.
            chart_spec: 차트 스펙.

        Returns:
            Matplotlib Figure 또는 None.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        chart_type = str(chart_spec.get("chart_type", "bar")).lower()
        x = chart_spec.get("x")
        y = chart_spec.get("y")
        series = chart_spec.get("series")

        if chart_type == "histogram":
            x = self._select_numeric_column(dataframe, x, y)
            if x is None:
                return None
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.hist(dataframe[x].dropna(), bins=12, alpha=0.85, color="#4C78A8")
            ax.set_xlabel(str(x))
            ax.set_ylabel("count")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            return fig

        if chart_type == "box":
            target = self._select_numeric_column(dataframe, y, x)
            if target is None:
                return None
            fig, ax = plt.subplots(figsize=(8, 4.6))
            if x and x in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[x]):
                grouped: list[pd.Series] = []
                labels: list[str] = []
                for label, group in dataframe.groupby(x):
                    values = group[target].dropna()
                    if values.empty:
                        continue
                    grouped.append(values)
                    labels.append(str(label))
                if not grouped:
                    return None
                ax.boxplot(grouped, labels=labels, showfliers=True)
                ax.set_xlabel(str(x))
            else:
                ax.boxplot(dataframe[target].dropna(), showfliers=True)
            ax.set_ylabel(str(target))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            return fig

        if x not in dataframe.columns or y not in dataframe.columns:
            return None
        if not pd.api.types.is_numeric_dtype(dataframe[y]):
            return None

        columns = [x, y]
        if series and series in dataframe.columns:
            columns.append(series)
        chart_df = dataframe[columns].copy()

        fig, ax = plt.subplots(figsize=(8, 4.6))
        try:
            if chart_type in {"line", "area"}:
                if series and series in chart_df.columns:
                    for label, group in chart_df.groupby(series):
                        group = group.sort_values(by=x)
                        ax.plot(group[x], group[y], marker="o", label=str(label))
                        if chart_type == "area":
                            ax.fill_between(group[x], group[y], alpha=0.2)
                    ax.legend()
                else:
                    chart_df = chart_df.sort_values(by=x)
                    ax.plot(chart_df[x], chart_df[y], marker="o")
                    if chart_type == "area":
                        ax.fill_between(chart_df[x], chart_df[y], alpha=0.2)
            elif chart_type == "scatter":
                if series and series in chart_df.columns:
                    for label, group in chart_df.groupby(series):
                        ax.scatter(group[x], group[y], label=str(label), alpha=0.8)
                    ax.legend()
                else:
                    ax.scatter(chart_df[x], chart_df[y], alpha=0.8)
            else:
                if series and series in chart_df.columns:
                    pivot = chart_df.pivot_table(index=x, columns=series, values=y, aggfunc="mean")
                    stacked = chart_type == "stacked_bar"
                    pivot.plot(kind="bar", stacked=stacked, ax=ax)
                else:
                    ax.bar(chart_df[x], chart_df[y], color="#4C78A8")

            ax.set_xlabel(str(x))
            ax.set_ylabel(str(y))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if len(chart_df[x].astype(str).unique()) > 8:
                ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            return fig
        except Exception:
            plt.close(fig)
            return None

    def _select_numeric_column(self, dataframe: pd.DataFrame, primary: str | None, secondary: str | None) -> str | None:
        """
        수치형 컬럼을 선택한다.

        Args:
            dataframe: 결과 데이터프레임.
            primary: 1차 후보 컬럼.
            secondary: 2차 후보 컬럼.

        Returns:
            수치형 컬럼명 또는 None.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        for candidate in [primary, secondary]:
            if candidate and candidate in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[candidate]):
                return candidate
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        return numeric_columns[0] if numeric_columns else None

    def _configure_matplotlib_fonts(self) -> None:
        """
        한글 폰트를 우선으로 Matplotlib 폰트를 설정한다.

        Side Effects:
            Matplotlib rcParams를 갱신한다.

        Raises:
            예외 없음.
        """

        available = {font.name for font in fm.fontManager.ttflist}
        preferred = [
            "Apple SD Gothic Neo",
            "AppleGothic",
            "Malgun Gothic",
            "NanumGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "NanumSquare",
            "NanumSquareRound",
            "Arial Unicode MS",
        ]
        for name in preferred:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return

    def _build_image_path(self, user_id: str, chat_id: str) -> str | None:
        """
        차트 이미지 저장 경로를 생성한다.

        Args:
            user_id: 사용자 ID.
            chat_id: 채팅 ID.

        Returns:
            저장 경로 문자열(없으면 None).

        Side Effects:
            저장 폴더를 생성한다.

        Raises:
            예외 없음.
        """

        safe_user = self._sanitize_identifier(user_id or "unknown")
        safe_chat = self._sanitize_identifier(chat_id or "unknown")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = self.output_root / safe_user / safe_chat
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_user}__{safe_chat}__{timestamp}.png"
        return str(path / filename)

    def _sanitize_identifier(self, value: str) -> str:
        """
        파일 경로에 사용할 문자열을 안전하게 정리한다.

        Args:
            value: 원본 문자열.

        Returns:
            안전한 식별자 문자열.

        Side Effects:
            None

        Raises:
            예외 없음.
        """

        cleaned = re.sub(r"[^0-9a-zA-Z_-]", "_", value.strip())
        return cleaned or "unknown"


if __name__ == "__main__":
    renderer = ChartRenderer()
    dummy = pd.DataFrame({"team": ["A", "B", "C"], "score": [10, 12, 9]})
    result = renderer.prepare_chart_image(
        dummy,
        {"chart_type": "bar", "x": "team", "y": "score"},
        user_id="developer",
        chat_id="demo",
    )
    print(result)
