
"""실습 04: Matplotlib을 활용한 데이터 시각화

각 분석 목적에 맞는 그래프를 독립된 Figure로 생성합니다.
생성한 그래프는 PNG 이미지로 저장하고, 실제 사용한 수치도 함께 반환합니다.

- 막대그래프: 에이전트별 평균 실행 시간 비교
- 히스토그램: 실행 시간의 구간별 분포 확인
- 산점도: 토큰 수와 실행 시간의 관계 관찰
- 선그래프: 날짜별 평균 실행 시간의 변화 확인
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# GUI 창을 띄우지 않고 PNG 이미지를 생성하는 백엔드를 사용합니다.
# 서버 환경에서도 그래프를 파일로 저장할 수 있습니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from analyze import daily_summary


def _save(fig, chart_dir, filename):
    """그래프를 지정한 폴더에 PNG 파일로 저장하는 공통 함수입니다."""

    # 저장 경로를 Path 객체로 변환하고 폴더가 없으면 생성합니다.
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 그래프 제목, 축 이름 등이 잘리지 않도록 여백을 자동 조정합니다.
        fig.tight_layout()

        # dpi=160 해상도로 PNG 파일을 저장합니다.
        fig.savefig(chart_dir / filename, dpi=160)

    finally:
        # 저장 후 Figure를 닫아 메모리 사용을 줄입니다.
        # finally는 저장 중 오류가 발생해도 실행됩니다.
        plt.close(fig)


def plot_bar(summary, chart_dir):
    """에이전트별 평균 실행 시간을 막대그래프로 표현합니다.

    summary의 agent, mean_ms, measured, runs 열을 사용합니다.
    각 막대에 평균 실행 시간과 유효 측정 수/전체 실행 수를 표시합니다.
    측정값이 없는 경우 0ms로 오해하지 않도록 별도 안내합니다.
    """

    # Figure(전체 그림판)와 Axes(실제 그래프 영역)를 생성합니다.
    # figsize=(8, 4.5)는 그래프의 가로·세로 크기(인치)입니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [1단계] 에이전트별 평균 실행 시간을 막대그래프로 생성합니다.
    # x축: agent / y축: mean_ms
    # ax.bar()가 반환하는 bars는 각 막대의 위치와 크기 정보를 포함합니다.
    bars = ax.bar(summary["agent"], summary["mean_ms"])

    # [2단계] NaN을 제외한 유효 평균값만 선택합니다.
    # 그래프의 y축 범위를 결정할 때 사용합니다.
    finite = summary["mean_ms"].dropna()

    # y축 상한을 계산합니다.
    # 최소 650ms까지 표시하되, 데이터 최댓값이 크면 22%의 여유를 둡니다.
    # 유효 평균값이 전혀 없다면 기본 상한값 650을 사용합니다.
    top = max(650.0, float(finite.max()) * 1.22) if not finite.empty else 650.0

    # y축을 0부터 시작하도록 설정하여 막대 높이의 비교가 왜곡되지 않게 합니다.
    ax.set_ylim(0, top)

    # [3단계] 각 막대 위에 표시할 설명 문구를 생성합니다.
    #
    # zip(): 여러 열을 같은 순서로 묶어 처리합니다.
    # np.isfinite(): 평균값이 NaN이나 무한대가 아닌지 확인합니다.
    #
    # 정상 측정: "150.00 ms\nn=8/10"
    # 측정 없음: "No measurement\nn=0/10"
    #
    # :.2f는 소수점 둘째 자리까지 표시한다는 의미입니다.
    labels = [
        f"{v:.2f} ms\nn={n}/{r}"
        if np.isfinite(v)
        else f"No measurement\nn=0/{r}"
        for v, n, r in zip(
            summary["mean_ms"],
            summary["measured"],
            summary["runs"]
        )
    ]

    # [4단계] 각 막대 위에 평균값과 측정 횟수를 표시합니다.
    # enumerate(): 반복하면서 인덱스 번호도 함께 가져옵니다.
    for index, (bar, label) in enumerate(zip(bars, labels)):

        # 해당 막대의 실제 평균 실행 시간을 가져옵니다.
        value = summary.iloc[index]["mean_ms"]

        # 정상 평균이면 해당 높이에 표시하고,
        # NaN이면 안내 문구를 그래프 하단에 표시합니다.
        y = float(value) if np.isfinite(value) else 0.0

        # ax.annotate(): 그래프의 특정 위치에 문자나 주석을 표시합니다.
        #
        # bar.get_x() + bar.get_width()/2 : 막대의 가로 중앙 위치
        # xytext=(0, 5) : 기준 위치에서 위쪽으로 5포인트 이동
        # ha="center" : 가로 중앙 정렬
        # va="bottom" : 문자의 아래쪽을 기준으로 정렬
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width()/2, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # [5단계] 그래프 제목과 x축·y축 이름을 지정합니다.
    # y축 단위는 밀리초(ms)입니다.
    ax.set(
        title="Mean latency by agent (valid / all runs)",
        xlabel="Agent",
        ylabel="Latency (ms)"
    )

    # [6단계] 생성된 막대그래프를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "latency_by_agent.png")

    # 그래프에 사용한 에이전트, 측정 개수, 평균값을 딕셔너리로 반환합니다.
    # tolist()는 Pandas Series를 일반 파이썬 리스트로 변환합니다.
    return {
        "agents": summary["agent"].tolist(),
        "measured": summary["measured"].tolist(),
        "runs": summary["runs"].tolist(),
        "means": summary["mean_ms"].tolist()
    }


def plot_hist(clean, chart_dir):
    """실행 시간(latency_ms)의 분포를 히스토그램으로 표현합니다.

    실행 시간을 200ms 간격의 구간으로 나누어 빈도를 계산합니다.
    기본 구간은 0~1000ms이며, 더 큰 값이 있으면 범위를 확장합니다.
    결측값은 제외하지만 900ms처럼 큰 유효 측정값은 보존합니다.
    """

    # 그래프를 그릴 Figure와 Axes를 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [1단계] 실행 시간에서 NaN만 제외합니다.
    # 유효 측정값을 대상으로 히스토그램을 생성합니다.
    values = clean["latency_ms"].dropna()

    # [2단계] 히스토그램의 구간 범위를 결정합니다.
    # 데이터가 없다면 최댓값을 0으로 설정합니다.
    maximum = float(values.max()) if len(values) else 0.0

    # 200ms 단위로 올림하여 구간의 상한값을 결정합니다.
    # np.ceil(): 올림 연산
    # max(1000, ...): 최소 1000ms 범위까지는 표시합니다.
    #
    # 예: 최대 1250ms → 상한값 1400ms
    upper = max(1000, int(np.ceil(maximum / 200)) * 200)

    # [3단계] 0부터 상한값까지 200ms 간격의 구간 경계를 생성합니다.
    # 예: [0, 200, 400, 600, 800, 1000]
    edges = np.arange(0, upper + 200, 200)

    # [4단계] 히스토그램을 생성하고 결과를 반환받습니다.
    #
    # ax.hist() 반환값:
    # counts : 각 구간에 포함된 데이터 개수
    # edges  : 구간의 경계값
    # _      : 생성된 막대 객체(여기서는 사용하지 않음)
    counts, edges, _ = ax.hist(values, bins=edges)

    # [5단계] 제목에 유효 측정 개수와 결측 개수를 함께 표시합니다.
    # x축은 실행 시간(ms), y축은 해당 구간의 실행 횟수입니다.
    ax.set(
        title=f"Latency distribution (n={len(values)}, missing={len(clean)-len(values)})",
        xlabel="Latency (ms)",
        ylabel="Runs"
    )

    # x축 눈금을 히스토그램의 구간 경계와 일치시킵니다.
    ax.set_xticks(edges)

    # [6단계] 유효한 측정값이 하나도 없다면 안내 문구를 표시합니다.
    # transAxes를 사용하면 (0.5, 0.5)가 그래프 영역의 중앙을 의미합니다.
    if values.empty:
        ax.text(
            0.5, 0.5,
            "No valid measurement",
            transform=ax.transAxes,
            ha="center"
        )

    # [7단계] 히스토그램을 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "latency_hist.png")

    # 그래프에 실제 사용한 데이터 수, 결측 수, 구간별 빈도를 반환합니다.
    # counts.astype(int)는 구간별 데이터 개수를 정수형으로 변환합니다.
    return {
        "n": len(values),
        "missing": len(clean) - len(values),
        "counts": counts.astype(int).tolist(),
        "edges": edges.tolist()
    }


def plot_scatter(clean, chart_dir):
    """토큰 사용량(tokens)과 실행 시간(latency_ms)의 관계를 산점도로 표현합니다.

    두 열의 값이 모두 존재하는 실행 기록만 사용합니다.
    산점도는 두 변수의 관계를 관찰하기 위한 것으로,
    상관관계가 있다고 해서 인과관계가 있다고 단정하지 않습니다.
    """

    # Figure와 Axes를 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [1단계] tokens와 latency_ms 열을 선택합니다.
    # dropna()로 두 값 중 하나라도 결측인 행은 제외합니다.
    # 따라서 pair에는 두 측정값이 모두 존재하는 기록만 남습니다.
    pair = clean[["tokens", "latency_ms"]].dropna()

    # [2단계] 산점도를 생성합니다.
    # x축: 토큰 수 / y축: 실행 시간(ms)
    # 각각의 점은 하나의 실행 기록을 의미합니다.
    ax.scatter(pair["tokens"], pair["latency_ms"])

    # [3단계] 제목에 실제 분석에 사용한 데이터 개수(n)를 표시합니다.
    ax.set(
        title=f"Tokens vs latency (n={len(pair)})",
        xlabel="Tokens",
        ylabel="Latency (ms)"
    )

    # 두 값이 모두 유효한 행이 없다면 그래프 중앙에 안내 문구를 표시합니다.
    if pair.empty:
        ax.text(
            0.5, 0.5,
            "No paired measurement",
            transform=ax.transAxes,
            ha="center"
        )

    # [4단계] 산점도를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "tokens_latency.png")

    # 사용한 데이터 개수와 결측으로 제외된 기록 수를 반환합니다.
    return {
        "n": len(pair),
        "excluded": len(clean) - len(pair),
        "selection": "두 측정값이 모두 유효한 행"
    }


def plot_daily(clean, chart_dir):
    """날짜별 평균 실행 시간의 변화를 선그래프로 표현합니다.

    daily_summary()를 이용해 날짜별 평균과 측정 횟수를 계산합니다.
    날짜가 없는 기록(NaT)은 별도로 집계하며,
    중간에 기록이 없는 날짜는 NaN으로 유지하여 선을 연결하지 않습니다.
    """

    # Figure와 Axes를 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [1단계] 실습 03의 날짜별 집계 함수를 호출합니다.
    # 결과에는 날짜별 runs, measured, mean_ms가 포함됩니다.
    daily = daily_summary(clean)

    # [2단계] 날짜가 없는 실행 기록의 전체 개수를 계산합니다.
    # index.isna(): 날짜 인덱스가 NaT인지 확인합니다.
    unknown = int(daily.loc[daily.index.isna(), "runs"].sum())

    # 날짜가 존재하는 데이터만 선택하고 날짜순으로 정렬합니다.
    known = daily.loc[daily.index.notna()].sort_index()

    # [3단계] 날짜가 정상적으로 기록된 데이터가 있는 경우 그래프를 생성합니다.
    if not known.empty:

        # 가장 이른 날짜부터 마지막 날짜까지 하루 간격의 날짜를 생성합니다.
        # freq="D"는 하루(Day) 간격을 의미합니다.
        dates = pd.date_range(
            known.index.min(),
            known.index.max(),
            freq="D"
        )

        # [4단계] 전체 날짜 목록을 기준으로 데이터를 재배치합니다.
        # 기록이 없는 날짜는 NaN이 되어 그래프의 선이 끊어집니다.
        # 없는 날짜에 임의의 평균값을 만들어 넣지 않습니다.
        expanded = known.reindex(dates)

        # [5단계] 날짜별 평균 실행 시간을 선그래프로 표현합니다.
        # marker="o"는 실제 측정 위치를 원형 점으로 표시합니다.
        ax.plot(
            expanded.index,
            expanded["mean_ms"],
            marker="o"
        )

        # [6단계] 각 날짜의 평균값 위에 유효 측정 수/전체 실행 수를 표시합니다.
        # iterrows()는 날짜별 데이터를 한 행씩 가져옵니다.
        for day, row in known.iterrows():

            # 평균값이 NaN이 아닌 경우에만 주석을 표시합니다.
            if pd.notna(row["mean_ms"]):
                ax.annotate(
                    f'n={int(row["measured"])}/{int(row["runs"])}',
                    (day, row["mean_ms"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center"
                )

        # [7단계] x축의 날짜 눈금을 지정합니다.
        ax.set_xticks(dates)

        # 날짜를 월-일(MM-DD) 형식으로 표시하고 30도 회전시킵니다.
        # 날짜가 겹쳐 보이는 현상을 줄이기 위한 설정입니다.
        ax.set_xticklabels(
            [d.strftime("%m-%d") for d in dates],
            rotation=30
        )

        # [8단계] y축 상한을 계산합니다.
        # 최댓값보다 25% 높은 위치까지 표시하여 주석 공간을 확보합니다.
        ymax = known["mean_ms"].max()

        # 평균값이 전부 NaN이면 y축 상한을 기본값 1.0으로 설정합니다.
        ax.set_ylim(
            0,
            max(1.0, float(ymax) * 1.25) if pd.notna(ymax) else 1.0
        )

    else:
        # 정상적인 날짜가 하나도 없으면 그래프 중앙에 안내 문구를 표시합니다.
        ax.text(
            0.5, 0.5,
            "No known observation date",
            transform=ax.transAxes,
            ha="center"
        )

    # [9단계] 제목과 축 이름을 설정합니다.
    # 제목에 날짜가 없는 실행 기록 수(unknown)를 함께 표시합니다.
    ax.set(
        title=f"Daily mean latency (unknown-date runs={unknown})",
        xlabel="Observation date",
        ylabel="Latency (ms)"
    )

    # [10단계] 날짜별 선그래프를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "daily_latency.png")

    # 날짜가 없는 실행 수와 정상 날짜의 개수, 결측 처리 정책을 반환합니다.
    return {
        "unknown_date_runs": unknown,
        "known_dates": len(known),
        "policy": "날짜 미상 별도 보고, 없는 날짜는 보간하지 않음"
    }


# ============================================================
# 실행부: 실습 데이터 로드 → 그래프 생성 → 결과 저장
# ============================================================

from pathlib import Path
from support import write_json, json_safe
import json

# 현재 실행 중인 파이썬 파일이 위치한 폴더의 절대 경로입니다.
BASE_DIR = Path(__file__).resolve().parent

from support import load_inputs


def main():
    # 결과 저장용 out 폴더를 생성합니다. 이미 존재하면 그대로 사용합니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # 실습용 데이터를 불러옵니다.
    # clean: 개별 실행 기록 / summary: 에이전트별 집계 데이터
    clean, summary = load_inputs(BASE_DIR)

    # 생성된 그래프 이미지가 저장될 폴더를 지정합니다.
    charts = BASE_DIR / "out/charts"

    # 막대그래프와 히스토그램을 생성하고 통계 결과를 딕셔너리에 저장합니다.
    # 현재 main()에서는 산점도와 선그래프를 호출하지 않습니다.
    result = {
        "bar": plot_bar(summary, charts),
        "hist": plot_hist(clean, charts)
    }

    # 그래프에 사용한 통계 데이터를 JSON 파일로 저장합니다.
    write_json(BASE_DIR / "out/result.json", result)

    # 분석 결과를 한글이 유지되도록 들여쓰기 2칸의 JSON 형태로 출력합니다.
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


# 현재 파일을 직접 실행했을 때만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()
