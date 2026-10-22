
"""실습 04: Matplotlib을 이용한 데이터 시각화

[실습 목표]
앞선 실습에서 정제하고 집계한 데이터를 그래프로 표현합니다.

1. 막대그래프(Bar Chart)
   → 에이전트별 평균 실행 시간을 비교합니다.

2. 히스토그램(Histogram)
   → 실행 시간이 어떤 구간에 많이 분포하는지 확인합니다.

3. 산점도(Scatter Plot)
   → 토큰 사용량과 실행 시간의 관계를 관찰합니다.

4. 선그래프(Line Chart)
   → 날짜별 평균 실행 시간의 변화를 확인합니다.

각 그래프는 독립된 이미지(PNG) 파일로 저장합니다.
그래프를 생성하는 데 사용한 수치도 함께 반환합니다.

[주요 용어]
- Figure : 그래프 전체를 담는 그림판
- Axes   : Figure 안에서 실제 그래프가 그려지는 영역
- axis   : 그래프의 x축 또는 y축
- figsize: 그래프의 크기 (가로, 세로 / 인치)
- dpi    : 이미지의 해상도를 결정하는 값
- NaN    : 측정되지 않았거나 유효하지 않은 숫자 데이터
- NaT    : 측정 날짜가 없거나 유효하지 않은 날짜 데이터

[중요한 원칙]
측정되지 않은 값을 임의로 0으로 바꾸지 않습니다.
실제 측정값과 결측값을 구분하여 시각화합니다.
"""

# 파일과 폴더의 경로를 관리하기 위한 라이브러리입니다.
from pathlib import Path

# NumPy: 배열, 숫자 계산, 구간 설정 등에 사용합니다.
import numpy as np

# Pandas: 표 형태의 데이터와 날짜를 처리합니다.
import pandas as pd

# Matplotlib: 데이터를 그래프로 시각화하는 라이브러리입니다.
import matplotlib

# Agg는 화면에 그래프 창을 띄우지 않고 이미지를 생성하는 방식입니다.
# 서버나 GUI가 없는 환경에서도 PNG 파일을 저장할 수 있습니다.
# matplotlib.pyplot을 import하기 전에 설정합니다.
matplotlib.use("Agg")

# 그래프 생성과 꾸미기 기능을 사용합니다.
import matplotlib.pyplot as plt

# 이전 실습(실습 03)에서 작성한 날짜별 집계 함수를 가져옵니다.
from analyze import daily_summary


def _save(fig, chart_dir, filename):
    """생성된 그래프를 PNG 파일로 저장하는 공통 함수입니다.

    [매개변수]
    fig       : 저장할 Figure(그림판) 객체
    chart_dir : 이미지를 저장할 폴더 경로
    filename  : 저장할 이미지 파일 이름

    [예시]
    _save(fig, "out/charts", "example.png")

    → out/charts/example.png 파일이 생성됩니다.

    여러 그래프 함수에서 동일한 저장 코드를
    반복하지 않기 위해 별도 함수로 작성했습니다.
    """

    # [1단계] 저장할 폴더 경로를 Path 객체로 변환합니다.
    # Path를 사용하면 운영체제에 맞는 경로를 쉽게 처리할 수 있습니다.
    chart_dir = Path(chart_dir)

    # [2단계] 저장 폴더를 생성합니다.
    #
    # parents=True
    #   → 상위 폴더가 없어도 함께 생성합니다.
    #
    # exist_ok=True
    #   → 폴더가 이미 존재해도 오류가 발생하지 않습니다.
    chart_dir.mkdir(parents=True, exist_ok=True)

    try:
        # [3단계] 그래프의 여백과 요소 배치를 자동 조정합니다.
        # 제목이나 축 이름이 이미지 밖으로 잘리는 것을 줄여줍니다.
        fig.tight_layout()

        # [4단계] 그래프를 PNG 이미지 파일로 저장합니다.
        #
        # chart_dir / filename
        #   → 폴더 경로와 파일 이름을 결합합니다.
        #
        # dpi=160
        #   → 이미지 해상도를 설정합니다.
        #   → 일반적으로 값이 클수록 이미지가 선명해집니다.
        fig.savefig(chart_dir / filename, dpi=160)

    finally:
        # [5단계] 사용한 Figure를 닫습니다.
        #
        # 여러 그래프를 계속 생성하면 메모리를 사용하므로
        # 저장 후 그래프 객체를 닫아 자원을 정리합니다.
        #
        # finally는 저장 과정에서 오류가 발생하더라도 실행됩니다.
        plt.close(fig)


def plot_bar(summary, chart_dir):
    """에이전트별 평균 실행 시간을 막대그래프로 표현합니다.

    [막대그래프란?]
    항목별 값을 막대의 높이 또는 길이로 표현하는 그래프입니다.
    서로 다른 집단의 크기를 비교할 때 주로 사용합니다.

    [입력]
    summary   : 실습 03에서 집계한 에이전트별 통계 DataFrame
    chart_dir : 그래프 이미지를 저장할 폴더

    [주요 데이터]
    agent    : 에이전트 이름
    runs     : 전체 실행 횟수
    measured : 실행 시간이 정상적으로 측정된 횟수
    mean_ms  : 평균 실행 시간(ms)

    [예시]
    agent     runs   measured   mean_ms
    planner   10     8          150
    coder     12     10         250
    reviewer  8      0          NaN

    planner와 coder는 평균 실행 시간을 막대로 표시합니다.

    reviewer는 유효 측정값이 0건이므로 평균을 계산할 수 없습니다.
    따라서 0ms라고 표시하지 않고 No measurement로 안내합니다.

    [저장 파일]
    latency_by_agent.png
    """

    # [1단계] 그래프를 그릴 Figure와 Axes를 생성합니다.
    #
    # plt.subplots()
    #   → 그래프 전체 영역과 실제 그래프를 그릴 영역을 만듭니다.
    #
    # fig
    #   → 전체 그림판(Figure)
    #
    # ax
    #   → 그래프를 그릴 좌표 영역(Axes)
    #
    # figsize=(8, 4.5)
    #   → 그래프의 가로 8인치, 세로 4.5인치를 의미합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # =========================================================
    # TODO P_BAR: 수강생이 직접 구현할 부분
    # =========================================================
    #
    # [목표]
    # 에이전트별 평균 실행 시간을 막대그래프로 표현합니다.
    #
    # [구현 순서]
    #
    # ① summary에서 agent와 mean_ms 데이터를 가져옵니다.
    #    → agent는 x축, mean_ms는 y축에 사용합니다.
    bars = ax.bar(summary["agent"], summary["mean_ms"])

    # ② ax.bar()를 이용해 막대그래프를 생성합니다.
    #    → 항목별 평균 실행 시간이 막대 높이로 표현됩니다.

    # [2단계] NaN을 제외한 유효 평균값만 선택합니다.
        # 그래프의 y축 범위를 결정할 때 사용합니다.
    finite = summary["mean_ms"].dropna()
    
        # y축 상한을 계산합니다.
        # 최소 650ms까지 표시하되, 데이터 최댓값이 크면 22%의 여유를 둡니다.
        # 유효 평균값이 전혀 없다면 기본 상한값 650을 사용합니다.
    top = max(650.0, float(finite.max()) * 1.22) if not finite.empty else 650.0
    
        # y축을 0부터 시작하도록 설정하여 막대 높이의 비교가 왜곡되지 않게 합니다.
    ax.set_ylim(0, top)
    
    # ④ 그래프 제목과 축 이름을 지정합니다.
    #    → x축: Agent
    #    → y축: Mean latency (ms)
    ax.set_title("Mean Execution Time by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean latency (ms)")

    # ⑤ 각 에이전트의 측정 횟수를 표시합니다.
    #    → measured / runs
    #    → 예: n=8/10
    #    → 전체 10건 중 8건만 측정되었다는 의미입니다.
    for i, row in summary.iterrows():
        measured = row["measured"]
        runs = row["runs"]
        mean = row["mean_ms"]

    # ⑥ measured가 0인 에이전트는 별도로 표시합니다.
    #    → 평균이 NaN이므로 0ms 막대로 오해하지 않도록
    #      "No measurement"라는 문구를 표시합니다.
        if measured==0 or not np.isfinite(mean):
            ax.text(
                i,0, "No measurement",
                ha="center", va="bottom", rotation=0
            )
        else:
            ax.text(
                i, mean, f"n={int(measured)}/{int(runs)}",
                ha="center", va="bottom"
            )
    # [참고할 함수]
    # ax.bar()       : 막대그래프 생성
    # ax.set_title() : 그래프 제목 설정
    # ax.set_xlabel(): x축 이름 설정
    # ax.set_ylabel(): y축 이름 설정
    # ax.set_ylim()  : y축 표시 범위 설정
    # ax.text()      : 그래프에 문자 표시
    #
    # 아래 오류는 아직 구현되지 않았음을 나타냅니다.


    # [2단계] 완성된 막대그래프를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "latency_by_agent.png")

    # [3단계] 그래프에 사용한 데이터를 딕셔너리로 반환합니다.
    #
    # .tolist()
    #   → Pandas Series를 파이썬 리스트로 변환합니다.
    #
    # 이미지뿐 아니라 실제 수치도 저장하여
    # 그래프와 데이터가 일치하는지 검증할 수 있습니다.
    return {
        "agents": summary["agent"].tolist(),
        "measured": summary["measured"].tolist(),
        "runs": summary["runs"].tolist(),
        "means": summary["mean_ms"].tolist()
    }


def plot_hist(clean, chart_dir):
    """실행 시간의 분포를 히스토그램으로 표현합니다.

    [히스토그램이란?]
    연속적인 숫자 데이터를 일정한 구간으로 나누어
    각 구간에 데이터가 몇 개 포함되는지 보여주는 그래프입니다.

    막대그래프는 에이전트와 같은 항목을 비교하지만,
    히스토그램은 실행 시간처럼 연속적인 숫자의 분포를 확인합니다.

    [예시]
    실행 시간: [100, 150, 220, 400, 450, 900]

    구간(ms)        데이터 개수
    0~200           2
    200~400         1
    400~600         2
    600~800         0
    800~1000        1

    ※ 구간 경계의 포함 방식은 NumPy의 histogram 규칙을 따릅니다.
    ※ 900ms처럼 상대적으로 큰 측정값도 삭제하지 않습니다.

    [저장 파일]
    latency_hist.png
    """

    # [1단계] 그래프를 그릴 Figure와 Axes를 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # =========================================================
    # TODO P_HIST: 수강생이 직접 구현할 부분
    # =========================================================
    #
    # [목표]
    # 실행 시간을 구간별로 나누어 측정값의 분포를 확인합니다.
    #
    # [구현 순서]
    #
    # ① clean의 latency_ms 열에서 NaN을 제외합니다.
    #    → 결측값은 실행 시간이 아니므로 분포 계산에서 제외합니다.
    #    → 정제 과정에서 유효하다고 판단된 측정값은 유지합니다.
    #
    # ② 유효 측정값을 values 변수에 저장합니다.
    #    → 실제 히스토그램을 그릴 때 사용하는 데이터입니다.
    #
    # ③ 히스토그램의 구간(bins)을 설정합니다.
    #    → 기본적으로 0~1000ms를 200ms 간격으로 구분합니다.
    #    → 예: 0, 200, 400, 600, 800, 1000
    #
    # ④ 측정값이 1000ms보다 크다면 구간을 확장합니다.
    #    → 큰 측정값을 임의로 제외하면 분포가 왜곡됩니다.
    #
    # ⑤ ax.hist()로 히스토그램을 생성합니다.
    #
    #    ax.hist()는 주요 결과로 다음 두 값을 반환합니다.
    #
    #    counts : 각 구간에 포함된 데이터 개수
    #    edges  : 각 구간의 경계값
    #
    # ⑥ 그래프 제목에 유효 측정 개수(n)를 표시합니다.
    #    → n은 분석에 실제로 사용한 데이터 수입니다.
    #
    # ⑦ x축과 y축 이름을 설정합니다.
    #    → x축: Latency (ms)
    #    → y축: Count
    #
    # [참고할 함수]
    # .dropna()       : NaN 제거
    # np.arange()     : 일정 간격의 숫자 배열 생성
    # ax.hist()       : 히스토그램 생성
    # ax.set()        : 제목 및 축 이름 지정
    #
    # 아래 오류는 실습 코드를 완성하면 제거합니다.

    # [2단계] 히스토그램을 이미지로 저장합니다.
    _save(fig, chart_dir, "latency_hist.png")

    # [3단계] 실제 분석에 사용한 수치를 반환합니다.
    #
    # n       : 유효 측정값 개수
    # missing : 제외된 결측값 개수
    # counts  : 구간별 데이터 개수
    # edges   : 구간 경계값
    #
    # astype(int)
    #   → 구간별 데이터 개수를 정수형으로 변환합니다.
    return {
        "n": len(values),
        "missing": len(clean) - len(values),
        "counts": counts.astype(int).tolist(),
        "edges": edges.tolist()
    }


def plot_scatter(clean, chart_dir):
    """토큰 사용량과 실행 시간의 관계를 산점도로 표현합니다.

    [산점도란?]
    두 수치형 데이터의 관계를 점으로 나타내는 그래프입니다.

    예를 들어
    x축: 사용 토큰 수(tokens)
    y축: 실행 시간(latency_ms)

    각 점은 하나의 실행 기록을 의미합니다.

    [예시]
    tokens    latency_ms
    100       200
    200       350
    300       500

    토큰이 증가할수록 실행 시간이 증가하는 경향이 있는지
    시각적으로 확인할 수 있습니다.

    주의:
    두 값에 관계가 보인다고 해서 토큰 증가가
    실행 시간 증가의 직접적인 원인이라고 단정할 수는 없습니다.

    [저장 파일]
    tokens_latency.png
    """

    # [1단계] 그래프 영역을 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [2단계] 산점도에 사용할 두 열을 선택합니다.
    #
    # clean[["tokens", "latency_ms"]]
    #   → tokens와 latency_ms 열만 선택합니다.
    #
    # .dropna()
    #   → 두 열 중 하나라도 NaN이면 해당 행을 제외합니다.
    #
    # 산점도는 x값과 y값이 모두 있어야 점을 그릴 수 있습니다.
    #
    # 예시:
    # tokens   latency_ms
    # 100      200        → 사용
    # NaN      300        → 제외
    # 200      NaN        → 제외
    # 300      400        → 사용
    #
    # 따라서 위 데이터의 유효한 쌍(pair)은 2개입니다.
    pair = clean[["tokens", "latency_ms"]].dropna()

    # [3단계] 산점도를 생성합니다.
    #
    # ax.scatter(x, y)
    #   → 두 수치 데이터를 x축과 y축에 배치합니다.
    #
    # 각 점은 하나의 유효한 실행 기록을 의미합니다.
    ax.scatter(pair["tokens"], pair["latency_ms"])

    # [4단계] 그래프 제목과 축 이름을 설정합니다.
    #
    # len(pair)
    #   → 두 측정값이 모두 존재하는 기록의 개수입니다.
    #
    # n=숫자 형태로 실제 사용한 데이터 수를 표시합니다.
    ax.set(
        title=f"Tokens vs latency (n={len(pair)})",
        xlabel="Tokens",
        ylabel="Latency (ms)"
    )

    # [5단계] 두 수치가 모두 존재하는 기록이 없을 경우
    # 빈 그래프에 안내 문구를 표시합니다.
    #
    # pair.empty
    #   → 데이터가 비어 있는지 확인합니다.
    #
    # ax.text()
    #   → 그래프 영역에 문자를 표시합니다.
    #
    # transform=ax.transAxes
    #   → 그래프 내부의 상대 좌표(0~1)를 사용합니다.
    #
    # (0.5, 0.5)
    #   → 그래프 영역의 중앙 위치입니다.
    #
    # ha="center"
    #   → 문자를 가로 방향 가운데 정렬합니다.
    if pair.empty:
        ax.text(
            0.5, 0.5,
            "No paired measurement",
            transform=ax.transAxes,
            ha="center"
        )

    # [6단계] 산점도를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "tokens_latency.png")

    # [7단계] 산점도에 사용한 데이터 개수와
    # 결측으로 제외된 데이터 개수를 반환합니다.
    return {
        "n": len(pair),
        "excluded": len(clean) - len(pair),
        "selection": "두 측정값이 모두 유효한 행"
    }


def plot_daily(clean, chart_dir):
    """날짜별 평균 실행 시간의 변화를 선그래프로 표현합니다.

    [선그래프란?]
    시간에 따른 수치의 변화 추이를 보여주는 그래프입니다.

    [예시]
    날짜          평균 실행 시간
    10월 1일      100ms
    10월 2일      200ms
    10월 3일      데이터 없음
    10월 4일      150ms

    날짜별 평균을 점으로 표시하고 선으로 연결합니다.

    단, 10월 3일처럼 데이터가 없는 날짜에
    임의의 평균값을 만들어 넣지 않습니다.

    [중요]
    날짜가 없는 기록(NaT)은 별도로 개수를 계산합니다.
    없는 날짜의 평균을 0이나 이전 날짜의 값으로 채우지 않습니다.

    [저장 파일]
    daily_latency.png
    """

    # [1단계] 그래프 영역을 생성합니다.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # [2단계] 앞선 실습의 daily_summary()를 호출합니다.
    #
    # 날짜별로 다음 세 가지 통계가 집계됩니다.
    #
    # runs     : 전체 실행 횟수
    # measured : 유효 측정 횟수
    # mean_ms  : 평균 실행 시간
    #
    # 날짜가 없는 기록도 별도로 집계되어 있습니다.
    daily = daily_summary(clean)

    # [3단계] 날짜가 없는 실행 기록의 개수를 계산합니다.
    #
    # daily.index.isna()
    #   → 날짜 인덱스가 NaT인지 검사합니다.
    #
    # .loc[조건, "runs"]
    #   → 날짜가 없는 그룹의 전체 실행 횟수를 선택합니다.
    #
    # .sum()
    #   → 날짜가 없는 실행 횟수를 합산합니다.
    unknown = int(daily.loc[daily.index.isna(), "runs"].sum())

    # [4단계] 날짜가 정상적으로 존재하는 데이터만 선택합니다.
    #
    # index.notna()
    #   → 날짜가 NaT가 아닌 그룹을 선택합니다.
    #
    # sort_index()
    #   → 날짜를 시간순으로 정렬합니다.
    known = daily.loc[daily.index.notna()].sort_index()

    # [5단계] 유효한 날짜가 하나 이상 있으면 선그래프를 그립니다.
    if not known.empty:

        # 가장 이른 날짜부터 가장 늦은 날짜까지
        # 하루 간격으로 연속된 날짜 목록을 생성합니다.
        #
        # pd.date_range()
        #   → 일정한 간격의 날짜 배열을 만듭니다.
        #
        # freq="D"
        #   → Day(일) 단위, 즉 하루 간격을 의미합니다.
        dates = pd.date_range(
            known.index.min(),
            known.index.max(),
            freq="D"
        )

        # [6단계] 누락된 날짜도 날짜 목록에 포함합니다.
        #
        # reindex(dates)
        #   → 생성한 날짜 목록을 기준으로 데이터를 다시 배치합니다.
        #
        # 실제 기록이 없는 날짜에는 NaN이 들어갑니다.
        #
        # 예시:
        # 10월 1일: 100
        # 10월 2일: 200
        # 10월 3일: NaN
        # 10월 4일: 150
        #
        # NaN 구간은 선그래프에서 끊어져 표현됩니다.
        # 이를 통해 측정되지 않은 날짜를 구분할 수 있습니다.
        expanded = known.reindex(dates)

        # [7단계] 날짜별 평균 실행 시간을 선그래프로 표현합니다.
        #
        # ax.plot(x, y)
        #   → x축과 y축 데이터를 선으로 연결합니다.
        #
        # marker="o"
        #   → 실제 데이터 위치를 원형 점으로 표시합니다.
        ax.plot(
            expanded.index,
            expanded["mean_ms"],
            marker="o"
        )

        # [8단계] 실제 측정 기록이 있는 날짜에
        # 유효 측정 수와 전체 실행 수를 표시합니다.
        #
        # iterrows()
        #   → DataFrame의 각 행을 하나씩 가져옵니다.
        #
        # day
        #   → 해당 행의 날짜
        #
        # row
        #   → 해당 날짜의 집계 데이터
        for day, row in known.iterrows():

            # 평균 실행 시간이 NaN이 아니어야
            # 그래프에 수치를 표시할 수 있습니다.
            if pd.notna(row["mean_ms"]):

                # ax.annotate()
                #   → 그래프의 특정 위치에 설명 문구를 추가합니다.
                #
                # n=측정수/전체수
                #   → 예: n=8/10
                #
                # xytext=(0, 10)
                #   → 데이터 점으로부터 위쪽으로 10포인트 이동합니다.
                #
                # textcoords="offset points"
                #   → 문자 위치를 점 기준 상대 거리로 지정합니다.
                ax.annotate(
                    f'n={int(row["measured"])}/{int(row["runs"])}',
                    (day, row["mean_ms"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center"
                )

        # [9단계] x축에 표시할 날짜를 설정합니다.
        #
        # set_xticks(dates)
        #   → x축 눈금 위치를 날짜별로 지정합니다.
        ax.set_xticks(dates)

        # 날짜를 MM-DD 형식으로 표시합니다.
        #
        # strftime("%m-%d")
        #   → 날짜를 월-일 형태의 문자열로 변환합니다.
        #
        # rotation=30
        #   → 날짜 글자를 30도 기울여 겹침을 줄입니다.
        ax.set_xticklabels(
            [d.strftime("%m-%d") for d in dates],
            rotation=30
        )

        # [10단계] y축 표시 범위를 설정합니다.
        #
        # ymax
        #   → 실제 측정된 평균 실행 시간 중 최댓값입니다.
        #
        # 최댓값의 1.25배까지 공간을 확보하여
        # 그래프 위쪽의 n=측정수/전체수 문구가
        # 이미지 밖으로 잘리지 않도록 합니다.
        ymax = known["mean_ms"].max()

        # ax.set_ylim(최솟값, 최댓값)
        #   → y축의 표시 범위를 지정합니다.
        #
        # 모든 평균이 NaN이면 기본 상한값 1.0을 사용합니다.
        ax.set_ylim(
            0,
            max(1.0, float(ymax) * 1.25)
            if pd.notna(ymax) else 1.0
        )

    else:
        # [11단계] 날짜가 정상적으로 기록된 데이터가 없다면
        # 빈 그래프에 안내 문구를 표시합니다.
        #
        # 이는 데이터가 없다는 사실을 보여주기 위한 처리입니다.
        ax.text(
            0.5, 0.5,
            "No known observation date",
            transform=ax.transAxes,
            ha="center"
        )

    # [12단계] 그래프 제목과 축 이름을 설정합니다.
    #
    # 제목에 unknown-date runs를 표시하여
    # 날짜가 없는 실행 기록의 개수를 함께 안내합니다.
    ax.set(
        title=f"Daily mean latency (unknown-date runs={unknown})",
        xlabel="Observation date",
        ylabel="Latency (ms)"
    )

    # [13단계] 날짜별 선그래프를 PNG 파일로 저장합니다.
    _save(fig, chart_dir, "daily_latency.png")

    # [14단계] 그래프에 사용한 날짜 정보와
    # 결측 날짜 처리 정책을 반환합니다.
    return {
        "unknown_date_runs": unknown,
        "known_dates": len(known),
        "policy": "날짜 미상 별도 보고, 없는 날짜는 보간하지 않음"
    }


# ============================================================
# 프로그램 실행부
# ============================================================
#
# 앞에서 정의한 함수들을 호출하여
# 그래프를 생성하고 분석 결과를 저장합니다.
#
# 이번 실습의 main()에서는 막대그래프와 히스토그램만 실행합니다.
# 산점도와 선그래프는 별도로 함수를 호출하여 확인할 수 있습니다.
# ============================================================

# 파일 경로 관리와 JSON 저장을 위한 라이브러리입니다.
from pathlib import Path
from support import write_json, json_safe
import json

# 현재 파이썬 파일이 위치한 폴더의 절대 경로를 가져옵니다.
#
# __file__ : 현재 실행 중인 파이썬 파일
# resolve(): 절대 경로로 변환
# parent   : 해당 파일이 포함된 폴더
BASE_DIR = Path(__file__).resolve().parent

# 실습 입력 데이터를 불러오는 제공 함수입니다.
from support import load_inputs


def main():
    """실습 데이터를 불러와 그래프와 JSON 결과를 저장합니다."""

    # [1단계] 결과를 저장할 out 폴더를 생성합니다.
    # 폴더가 이미 존재해도 오류가 발생하지 않습니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # [2단계] 실습 데이터를 불러옵니다.
    #
    # clean
    #   → 정제된 개별 실행 기록 데이터
    #
    # summary
    #   → 에이전트별 통계가 계산된 집계 데이터
    #
    # 막대그래프는 집계 데이터(summary)를 사용하고,
    # 히스토그램은 개별 실행 데이터(clean)를 사용합니다.
    clean, summary = load_inputs(BASE_DIR)

    # [3단계] 그래프 이미지를 저장할 폴더 경로를 지정합니다.
    #
    # 저장 위치: 현재 실습 폴더 / out / charts
    charts = BASE_DIR / "out/charts"

    # [4단계] 두 종류의 그래프를 생성합니다.
    #
    # plot_bar()
    #   → 에이전트별 평균 실행 시간 막대그래프
    #
    # plot_hist()
    #   → 전체 실행 시간 분포 히스토그램
    #
    # 각 함수가 반환한 통계값을 result에 저장합니다.
    #
    # 주의:
    # TODO P_BAR가 미완성 상태이면 NotImplementedError가
    # 발생하여 다음 함수인 plot_hist()는 실행되지 않습니다.
    result = {
        "bar": plot_bar(summary, charts),
        "hist": plot_hist(clean, charts)
    }

    # [5단계] 그래프 생성에 사용한 통계 데이터를 JSON 파일로 저장합니다.
    #
    # 이미지 파일: out/charts/ 폴더
    # 통계 파일: out/result.json
    write_json(BASE_DIR / "out/result.json", result)

    # [6단계] JSON 결과를 콘솔에 보기 좋게 출력합니다.
    #
    # json_safe(): JSON으로 표현 가능한 데이터로 변환
    # ensure_ascii=False: 한글을 그대로 표시
    # indent=2: 들여쓰기 2칸 적용
    print(
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=2
        )
    )


# 현재 파이썬 파일을 직접 실행했을 때만 main()을 호출합니다.
# 다른 파일에서 import할 때는 자동 실행하지 않습니다.
if __name__ == "__main__":
    main()
