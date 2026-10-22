
"""실습 03: EDA(탐색적 데이터 분석)

[실습 목표]
AI 에이전트의 실행 기록을 분석하여 다음 내용을 확인합니다.

1. 에이전트별로 몇 번 실행되었는가?
2. 실행 시간(latency_ms)의 평균과 중앙값은 얼마인가?
3. 전체 실행 중 성공한 비율은 얼마인가?
4. 서로 다른 표를 정확하게 연결할 수 있는가?
5. 데이터가 없을 때 잘못된 통계를 계산하지 않는가?

[주요 용어]
- agent       : 작업을 수행한 AI 에이전트의 종류 또는 역할
- run_id      : 각각의 실행 기록을 구분하는 고유 번호
- status      : 실행 결과. ok는 성공, error는 실패
- latency_ms  : 실행에 걸린 시간 (밀리초, ms)
- tokens      : AI가 사용한 토큰의 개수
- day         : 실행 날짜
- NaN         : 숫자 데이터가 없거나 유효하지 않은 상태
- NaT         : 날짜 데이터가 없거나 유효하지 않은 상태

※ 중요한 원칙
측정값이 없다는 것과 측정값이 0이라는 것은 다릅니다.
예를 들어 실행 시간이 NaN이면 시간을 측정하지 못한 것이고,
0이면 실제 측정값이 0이라는 의미입니다.
"""

# NumPy: 배열 계산, 평균, 결측값 처리 등에 사용합니다.
import numpy as np

# Pandas: 엑셀 표와 비슷한 DataFrame 데이터를 처리합니다.
import pandas as pd

# support.py에 미리 정의된 함수입니다.
# 데이터에 반드시 필요한 열이 있는지 검사합니다.
from support import require_columns


def aggregate_runs(clean):
    """에이전트별 실행 통계를 계산하는 함수입니다.

    [입력]
    clean: 정제된 실행 기록 DataFrame

    [출력]
    에이전트별로 아래 6가지 통계를 계산한 DataFrame

    1. runs         : 전체 실행 횟수
    2. measured     : 실행 시간이 정상적으로 측정된 횟수
    3. mean_ms      : 실행 시간의 평균
    4. median_ms    : 실행 시간의 중앙값
    5. success_rate : 전체 실행 중 성공 비율 (0~1)
    6. success_pct  : 성공 비율을 백분율로 표현 (0~100)

    [예시]
    어떤 에이전트가 4번 실행했다고 가정합니다.

    실행 결과 : ok, ok, error, ok
    실행 시간 : 100, 200, NaN, NaN

    runs         = 4
    measured     = 2
    mean_ms      = (100 + 200) / 2 = 150
    median_ms    = 150
    success_rate = 3 / 4 = 0.75
    success_pct  = 75.0

    주의할 점은 평균과 성공률의 분모가 다르다는 것입니다.

    - 평균: 시간이 정상 측정된 2건으로 나눕니다.
    - 성공률: 전체 실행 4건으로 나눕니다.

    측정하지 못한 시간을 임의로 0으로 바꾸지 않습니다.
    """

    # [1단계] 필요한 열이 존재하는지 검사합니다.
    #
    # 예를 들어 데이터에 agent, status 등의 필수 열이
    # 누락되었다면 이후 계산에서 오류가 발생할 수 있습니다.
    #
    # require_columns()는 이러한 문제를 미리 확인합니다.
    require_columns(clean)

    # [2단계] 실행 상태가 올바른지 확인합니다.
    #
    # clean["status"]
    #   → status 열의 모든 값을 가져옵니다.
    #
    # .isin(["ok", "error"])
    #   → 각 값이 ok 또는 error인지 확인합니다.
    #   → 결과는 True 또는 False입니다.
    #
    # .all()
    #   → 모든 값이 True인지 확인합니다.
    #
    # 예시:
    # ["ok", "error", "ok"]      → True
    # ["ok", "unknown", "error"] → False
    #
    # 알 수 없는 상태를 임의로 실패라고 처리하면
    # 성공률이 잘못 계산될 수 있으므로 오류를 발생시킵니다.
    if not clean["status"].isin(["ok", "error"]).all():
        raise ValueError("알 수 없는 상태를 실패로 간주할 수 없습니다.")

    # [3단계] 에이전트별 통계 계산 (수강생 실습)
    #
    # TODO A_GROUP
    #
    # 아래 과정을 직접 구현해 보세요.
    #
    # ① 원본 clean을 복사합니다.
    #    → 원본 데이터가 변경되지 않도록 하기 위함입니다.
    
    df = clean.copy()
    df["ok"] = df["status"].eq("ok")

    # ② status가 "ok"인지 확인하는 새 열을 만듭니다.
    #    → 성공이면 True, 실패이면 False
    #
    # ③ groupby("agent")를 사용합니다.
    #    → 같은 에이전트의 실행 기록끼리 묶습니다.
    #
    # ④ agg()를 사용해 통계를 계산합니다.
    #    → size()   : 전체 행 개수 (NaN 포함)
    #    → count()  : 결측값을 제외한 개수
    #    → mean()   : 평균
    #    → median() : 중앙값
    
    result = df.groupby("agent", as_index=False, sort=True, observed=True).agg(
            runs=("run_id", "size"),
            measured=("latency_ms", "count"),
            mean_ms=("latency_ms", "mean"),
            median_ms=("latency_ms", "median"),
            success_rate=("ok", "mean"),
    )
    
    result["success_pct"] = result["success_rate"] * 100

    # ⑤ 성공률을 계산합니다.
    #    → 성공 횟수 / 전체 실행 횟수
    #
    # ⑥ 성공률에 100을 곱해 백분율을 계산합니다.
    #    → 0.75 * 100 = 75%
    #
    # [참고]
    # NotImplementedError는 아직 구현하지 않았다는 뜻입니다.
    # 수강생이 코드를 완성하면 아래 줄을 제거해야 합니다.
    return result


def make_pivot(clean):
    """에이전트와 실행 상태별 평균 실행 시간을 표로 정리합니다.

    [Pivot Table이란?]
    엑셀의 피벗 테이블처럼 데이터를 원하는 기준으로
    묶어서 요약한 표입니다.

    [예시: 원본 데이터]

    agent   status   latency_ms
    A       ok       100
    A       ok       200
    A       error    300
    B       ok       400

    [피벗 테이블 결과]

    agent   error     ok
    A       300       150
    B       NaN       400

    A의 ok 평균 = (100 + 200) / 2 = 150

    B의 error가 NaN인 이유는 해당 상태의
    측정 기록이 없기 때문입니다.
    """

    # [1단계] 피벗 테이블을 생성합니다.
    #
    # index="agent"
    #   → 행을 에이전트 이름으로 구분합니다.
    #
    # columns="status"
    #   → 열을 실행 상태(ok, error)로 구분합니다.
    #
    # values="latency_ms"
    #   → 계산할 대상은 실행 시간입니다.
    #
    # aggfunc="mean"
    #   → 같은 그룹의 실행 시간 평균을 계산합니다.
    #
    # observed=True
    #   → 범주형 데이터인 경우 실제 관측된 조합을
    #     기준으로 집계합니다.
    pivot = clean.pivot_table(
        index="agent",
        columns="status",
        values="latency_ms",
        aggfunc="mean",
        observed=True
    )

    # [2단계] 출력 표의 행과 열을 정리합니다.
    #
    # clean["agent"].unique()
    #   → 중복되지 않은 에이전트 이름을 가져옵니다.
    #
    # sorted(...)
    #   → 이름을 오름차순으로 정렬합니다.
    #
    # reindex(...)
    #   → 행과 열을 지정한 순서로 다시 배치합니다.
    #
    # columns=["error", "ok"]
    #   → error 열이 먼저, ok 열이 나중에 나오도록 합니다.
    #
    # 특정 에이전트의 실행 시간이 모두 NaN인 경우에도
    # 해당 에이전트가 표에서 빠지지 않게 합니다.
    #
    # 값이 없더라도 0으로 채우지 않고 NaN으로 유지합니다.
    return pivot.reindex(
        index=sorted(clean["agent"].unique()),
        columns=["error", "ok"]
    )


def validate_meta(meta):
    """메타데이터가 올바른 형태인지 검사합니다.

    [메타데이터란?]
    다른 데이터를 설명하기 위한 추가 정보입니다.

    여기서는 에이전트별 토큰 사용 단가를 의미합니다.

    [예시]

    agent      rate_per_1k
    planner    0.5
    coder      1.0
    reviewer   0.8

    rate_per_1k:
    토큰 1,000개를 사용할 때 적용할 가상 단가입니다.

    [검사할 내용]
    1. 필요한 열이 존재하는가?
    2. 에이전트 이름이 비어 있지 않은가?
    3. 동일한 에이전트가 중복되어 있지 않은가?
    4. 단가가 올바른 숫자인가?
    5. 단가가 음수이거나 무한대는 아닌가?
    """

    # [1단계] 필수 열이 있는지 검사합니다.
    require_columns(meta, ["agent", "rate_per_1k"])

    # [2단계] 원본 보호를 위해 데이터를 복사합니다.
    result = meta.copy()

    # [3단계] 에이전트 이름을 통일합니다.
    #
    # astype("string") : 문자열 형태로 변환
    # str.strip()      : 앞뒤 공백 제거
    # str.lower()      : 영어 대문자를 소문자로 변환
    #
    # 예시:
    # " Planner " → "planner"
    # "CODER"     → "coder"
    result["agent"] = (
        result["agent"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # [4단계] 에이전트 이름이 비어 있는지 검사합니다.
    #
    # result.empty
    #   → 데이터 표 전체가 비어 있는지 확인
    #
    # isna()
    #   → 값이 NaN 등 결측값인지 확인
    #
    # eq("")
    #   → 빈 문자열인지 확인
    #
    # any()
    #   → 조건을 만족하는 값이 하나라도 있는지 확인
    #
    # 한 개라도 문제가 있으면 오류를 발생시킵니다.
    if result.empty or result["agent"].isna().any() or result["agent"].eq("").any():
        raise ValueError("메타데이터의 역할 키가 비어 있습니다.")

    # [5단계] 중복된 에이전트 이름이 있는지 검사합니다.
    #
    # duplicated()
    #   → 앞에서 이미 등장한 값인지 확인합니다.
    #
    # 예를 들어 planner가 두 번 등록되어 있으면
    # 어떤 단가를 적용해야 하는지 모호해집니다.
    #
    # 따라서 중복 등록을 허용하지 않습니다.
    if result["agent"].duplicated().any():
        raise ValueError(
            "중복 메타 키: " +
            ", ".join(
                result.loc[result["agent"].duplicated(), "agent"]
            )
        )

    # [6단계] 단가를 숫자로 변환합니다.
    #
    # pd.to_numeric()
    #   → 문자열 등 숫자로 표현된 값을 숫자로 변환
    #
    # errors="coerce"
    #   → 숫자로 변환할 수 없는 값은 NaN으로 처리
    #
    # astype(float)
    #   → 실수형으로 통일
    #
    # 예시:
    # "1.5" → 1.5
    # "abc" → NaN
    result["rate_per_1k"] = pd.to_numeric(
        result["rate_per_1k"],
        errors="coerce"
    ).astype(float)

    # [7단계] 단가가 정상적인 숫자인지 검사합니다.
    #
    # np.isfinite()
    #   → 정상적인 유한 숫자인지 확인합니다.
    #   → NaN, 양의 무한대, 음의 무한대는 False
    #
    # ~
    #   → True와 False를 반대로 바꿉니다.
    #
    # lt(0)
    #   → 0보다 작은 값인지 검사합니다.
    #
    # |
    #   → OR 연산입니다. 조건 하나만 만족해도 True
    #
    # 즉, 정상 숫자가 아니거나 음수이면 오류입니다.
    if (
        ~np.isfinite(result["rate_per_1k"])
        | result["rate_per_1k"].lt(0)
    ).any():
        raise ValueError("단가는 0 이상의 유한한 수치여야 합니다.")

    # 모든 검사를 통과한 메타데이터를 반환합니다.
    return result


def join_meta(clean, meta):
    """실행 기록과 메타데이터를 하나의 표로 결합합니다.

    [예시: 실행 기록]

    agent      tokens
    planner    1000
    planner    2000
    coder      3000

    [예시: 메타데이터]

    agent      rate_per_1k
    planner    0.5
    coder      1.0

    [결합 결과]

    agent      tokens    rate_per_1k
    planner    1000      0.5
    planner    2000      0.5
    coder      3000      1.0

    두 표 모두에 존재하는 agent 열을 기준으로
    각 실행 기록에 적절한 단가를 붙입니다.

    [다대일 결합이란?]
    여러 개의 실행 기록이 하나의 에이전트 정보와
    연결되는 관계를 말합니다.

    실행 기록 여러 개 : 메타데이터 한 개
    """

    # [1단계] 결합할 메타데이터를 먼저 검사합니다.
    meta = validate_meta(meta)

    # [2단계] 실행 기록의 agent에 결측값이 있는지 확인합니다.
    #
    # agent가 NaN이면 어떤 메타데이터와
    # 연결해야 하는지 알 수 없습니다.
    if clean["agent"].isna().any():
        raise ValueError("왼쪽 역할 키가 결측입니다.")

    # [3단계] 두 데이터 표를 결합합니다.
    #
    # merge()
    #   → 두 DataFrame을 연결하는 함수입니다.
    #
    # on="agent"
    #   → agent 열의 값이 같은 데이터를 연결합니다.
    #
    # how="left"
    #   → 왼쪽 clean의 모든 실행 기록을 유지합니다.
    #   → 메타데이터에 없는 에이전트도 일단 남깁니다.
    #
    # validate="many_to_one"
    #   → 여러 실행 기록이 하나의 메타 정보와
    #     연결되는 형태인지 검사합니다.
    #
    # indicator=True
    #   → _merge라는 열을 추가합니다.
    #   → 결합이 정상적으로 이루어졌는지 확인합니다.
    #
    # _merge의 주요 값:
    # both      : 양쪽 표에서 모두 찾음
    # left_only : 왼쪽에만 존재함
    joined = clean.merge(
        meta,
        on="agent",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # [4단계] 결합 전후의 행 개수를 비교합니다.
    #
    # len()은 데이터의 행 개수를 반환합니다.
    #
    # 실행 기록이 100개였다면 결합 후에도
    # 100개여야 합니다.
    #
    # 행이 늘거나 줄면 잘못 결합되었을 수 있습니다.
    if len(joined) != len(clean):
        raise ValueError("결합 후 실행 행 수가 달라졌습니다.")

    # [5단계] 메타데이터와 연결되지 않은 에이전트를 찾습니다.
    #
    # joined["_merge"].ne("both")
    #   → 양쪽 표에서 일치하지 않은 기록을 찾습니다.
    #
    # .loc[조건, "agent"]
    #   → 조건에 해당하는 행의 agent 값을 선택합니다.
    #
    # .unique()
    #   → 중복된 이름을 제거합니다.
    #
    # .tolist()
    #   → 결과를 파이썬 리스트로 변환합니다.
    unmatched = joined.loc[
        joined["_merge"].ne("both"), "agent"
    ].unique().tolist()

    # 메타데이터가 없는 에이전트가 하나라도 있으면
    # 임의의 단가를 적용하지 않고 오류를 발생시킵니다.
    if unmatched:
        raise ValueError("미등록 메타 키: " + ", ".join(unmatched))

    # [6단계] 토큰 사용 비용을 계산합니다.
    #
    # cost_unit = tokens / 1000 * rate_per_1k
    #
    # 예시:
    # tokens = 2000
    # rate_per_1k = 0.5
    #
    # 비용 = 2000 / 1000 * 0.5 = 1.0
    #
    # tokens가 NaN이면 비용도 NaN이 됩니다.
    joined["cost_unit"] = (
        joined["tokens"] / 1000 * joined["rate_per_1k"]
    )

    # 메타데이터와 비용 정보가 추가된 표를 반환합니다.
    return joined


def daily_summary(clean):
    """날짜별 실행 횟수와 평균 실행 시간을 계산합니다.

    [예시]

    day          latency_ms
    2026-10-01   100
    2026-10-01   200
    2026-10-02   300
    NaT          400

    [결과]

    day          runs   measured   mean_ms
    2026-10-01   2      2          150
    2026-10-02   1      1          300
    NaT          1      1          400

    날짜가 없는 기록도 삭제하지 않고 집계합니다.
    """

    # [1단계] 날짜별로 데이터를 묶습니다.
    #
    # groupby("day")
    #   → 같은 날짜의 실행 기록끼리 묶습니다.
    #
    # dropna=False
    #   → 날짜가 NaT인 기록도 집계에 포함합니다.
    #
    # sort=True
    #   → 날짜 기준으로 정렬합니다.
    #
    # [2단계] 날짜별 통계를 계산합니다.
    #
    # runs=("run_id", "size")
    #   → 날짜별 전체 실행 횟수를 계산합니다.
    #   → size는 결측값 여부와 무관하게 행을 셉니다.
    #
    # measured=("latency_ms", "count")
    #   → 실행 시간이 있는 기록의 개수를 계산합니다.
    #   → count는 NaN을 제외합니다.
    #
    # mean_ms=("latency_ms", "mean")
    #   → 날짜별 평균 실행 시간을 계산합니다.
    #   → NaN은 평균 계산에서 제외합니다.
    return clean.groupby(
        "day",
        dropna=False,
        sort=True
    ).agg(
        runs=("run_id", "size"),
        measured=("latency_ms", "count"),
        mean_ms=("latency_ms", "mean")
    )


def cost_audit(joined):
    """계산된 비용이 얼마나 신뢰할 수 있는지 확인합니다.

    [검사 목적]
    전체 실행 기록 중 비용을 계산할 수 있는
    기록이 몇 개인지 확인합니다.

    예를 들어 전체 실행 10건 중 7건만
    tokens가 정상적으로 측정되었다면,
    나머지 3건의 비용은 알 수 없습니다.

    이때 7건의 비용 합계를 전체 10건의
    확정 비용인 것처럼 표현하면 안 됩니다.

    [반환 정보]
    cost_unit  : 계산 가능한 비용의 합계
    measured   : 비용이 계산된 실행 횟수
    runs       : 전체 실행 횟수
    is_partial : 일부 기록의 비용이 누락되었는가?
    unit       : 비용 단위에 대한 설명
    """

    # [1단계] 비용이 존재하는 기록의 개수를 계산합니다.
    #
    # count()
    #   → NaN을 제외한 값의 개수를 계산합니다.
    #
    # int()
    #   → 결과를 파이썬 정수로 변환합니다.
    count = int(joined["cost_unit"].count())

    # [2단계] 계산 가능한 비용을 모두 더합니다.
    #
    # sum()
    #   → 값의 합계를 계산합니다.
    #
    # min_count=1
    #   → 정상적인 비용 값이 최소 1개 있어야
    #     합계를 계산하도록 지정합니다.
    #
    # 모든 값이 NaN이면 합계도 NaN으로 반환합니다.
    #
    # 즉, 비용 정보가 아예 없을 때
    # 합계가 0이라고 잘못 표시하지 않습니다.
    value = float(joined["cost_unit"].sum(min_count=1))

    # [3단계] 비용 계산 결과를 딕셔너리로 반환합니다.
    #
    # len(joined)
    #   → 전체 실행 기록의 개수
    #
    # count < len(joined)
    #   → 비용을 계산하지 못한 기록이 하나라도 있는지 확인
    #
    # True라면 일부 기록만으로 계산한 합계입니다.
    return {
        "cost_unit": value,
        "measured": count,
        "runs": len(joined),
        "is_partial": count < len(joined),
        "unit": "가상 단위/실제 가격 아님"
    }


def weighted_mean(means, counts):
    """서로 다른 집단의 평균을 가중평균으로 합칩니다.

    [일반 평균과 가중평균의 차이]

    A 집단: 10명의 평균 점수 = 90점
    B 집단: 90명의 평균 점수 = 70점

    단순히 두 평균을 다시 평균 내면

    (90 + 70) / 2 = 80점

    하지만 실제 학생은 B 집단에 훨씬 많습니다.

    따라서 각 집단의 학생 수를 고려해야 합니다.

    가중평균 = (90*10 + 70*90) / (10+90)
             = 72점

    이와 같이 집단의 데이터 개수를
    가중치로 사용하는 계산이 가중평균입니다.

    [입력]
    means  : 집단별 평균값을 담은 배열
    counts : 각 집단의 유효 데이터 개수를 담은 배열

    [예시]
    means  = [90, 70]
    counts = [10, 90]
    결과    = 72.0
    """

    # [1단계] 입력값을 NumPy 배열로 변환합니다.
    #
    # dtype=float
    #   → 데이터를 실수 형태로 통일합니다.
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # [2단계] 두 배열의 형태가 올바른지 검사합니다.
    #
    # shape
    #   → 배열의 형태를 의미합니다.
    #
    # ndim
    #   → 배열의 차원 수를 의미합니다.
    #
    # means.shape != counts.shape
    #   → 평균 개수와 가중치 개수가 다른지 검사
    #
    # means.ndim != 1
    #   → 평균 배열이 1차원인지 검사
    #
    # 예시:
    # means  = [90, 70]
    # counts = [10, 90]   → 정상
    #
    # means  = [90, 70]
    # counts = [10]       → 오류
    if means.shape != counts.shape or means.ndim != 1:
        raise ValueError("평균과 개수는 길이가 같은 1차원 배열이어야 합니다.")

    # [3단계] 각 집단의 데이터 개수가 정상인지 검사합니다.
    #
    # np.isfinite(counts)
    #   → NaN이나 무한대가 아닌지 확인합니다.
    #
    # counts < 0
    #   → 개수가 음수인지 확인합니다.
    #
    # counts % 1 != 0
    #   → 개수가 정수인지 확인합니다.
    #
    # 예시:
    # 10 % 1 = 0    → 정수
    # 10.5 % 1 = 0.5 → 정수가 아님
    #
    # 학생 수나 측정 횟수는 음수 또는
    # 소수일 수 없으므로 오류로 처리합니다.
    if (
        ~np.isfinite(counts)
        | (counts < 0)
        | (counts % 1 != 0)
    ).any():
        raise ValueError("측정 개수는 0 이상의 유한한 정수여야 합니다.")

    # [4단계] 측정값이 있는 집단의 평균이 정상인지 검사합니다.
    #
    # counts > 0
    #   → 실제 데이터가 하나 이상 존재하는 집단
    #
    # ~np.isfinite(means)
    #   → 평균이 NaN이나 무한대인 집단
    #
    # 두 조건을 동시에 만족하면 오류입니다.
    #
    # 데이터가 존재하는데 평균을 계산할 수 없다면
    # 올바른 가중평균을 구할 수 없기 때문입니다.
    if ((counts > 0) & ~np.isfinite(means)).any():
        raise ValueError("측정이 있는 그룹의 평균이 유효하지 않습니다.")

    # [5단계] 실제 데이터가 있는 집단만 선택합니다.
    #
    # 예시:
    # counts = [10, 0, 90]
    #
    # valid = [True, False, True]
    #
    # 데이터가 0건인 집단은
    # 가중평균 계산에서 제외합니다.
    valid = counts > 0

    # [6단계] 가중평균을 계산합니다.
    #
    # np.average()
    #   → 평균을 계산하는 NumPy 함수입니다.
    #
    # weights=counts[valid]
    #   → 각 집단의 데이터 개수를 가중치로 사용합니다.
    #
    # valid.any()
    #   → 데이터가 1건 이상인 집단이 존재하는지 확인합니다.
    #
    # 집단이 전부 0건이라면
    # 평균을 계산할 수 없으므로 NaN을 반환합니다.
    return (
        float(np.average(means[valid], weights=counts[valid]))
        if valid.any()
        else np.nan
    )


# ============================================================
# 프로그램 실행부
# ============================================================
#
# 위에서 작성한 함수들을 실제로 실행하는 부분입니다.
#
# 수강생은 먼저 각 함수의 역할과 TODO를 이해하고,
# 그다음 main()을 실행하여 결과를 확인하면 됩니다.
# ============================================================

# Path: 파일이나 폴더의 경로를 다루는 기능입니다.
from pathlib import Path

# JSON 파일 저장 및 데이터 변환을 위한 제공 함수입니다.
from support import write_json, json_safe

# json: 파이썬 데이터를 JSON 문자열로 변환하는 라이브러리입니다.
import json

# 현재 파이썬 파일이 위치한 폴더의 절대 경로를 가져옵니다.
#
# __file__ : 현재 실행 중인 파이썬 파일
# resolve(): 절대 경로로 변환
# parent   : 해당 파일이 들어 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# load_inputs(): 실습 입력 데이터를 불러오는 제공 함수
# strict_json : support.py에서 가져오는 기능
from support import load_inputs, strict_json


def main():
    """실습 데이터를 불러오고 분석 결과를 저장합니다."""

    # [1단계] 결과를 저장할 out 폴더를 생성합니다.
    #
    # exist_ok=True
    #   → 폴더가 이미 존재해도 오류를 발생시키지 않습니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # [2단계] 실습 데이터를 불러옵니다.
    #
    # load_inputs()는 두 개의 값을 반환합니다.
    #
    # clean : 정제된 실행 기록 DataFrame
    # _     : 여기서는 사용하지 않을 두 번째 반환값
    #
    # 파이썬에서는 사용하지 않는 값을
    # 관례적으로 _ 변수에 저장하기도 합니다.
    clean, _ = load_inputs(BASE_DIR)

    # [3단계] 에이전트별 통계를 계산합니다.
    #
    # 현재 aggregate_runs() 내부에는
    # NotImplementedError가 남아 있습니다.
    #
    # 따라서 TODO를 완성하기 전에는
    # 이 위치에서 프로그램 실행이 중단됩니다.
    summary = aggregate_runs(clean)

    # [4단계] 분석 결과를 딕셔너리로 정리합니다.
    #
    # to_dict(orient="records")
    #   → DataFrame의 각 행을 하나의 딕셔너리로 변환하고
    #     전체를 리스트로 묶습니다.
    #
    # summary["runs"].sum()
    #   → 모든 에이전트의 실행 횟수를 합산합니다.
    #
    # summary["measured"].sum()
    #   → 실행 시간이 유효한 기록 수를 합산합니다.
    result = {
        "summary": summary.to_dict(orient="records"),
        "runs": int(summary["runs"].sum()),
        "measured": int(summary["measured"].sum())
    }

    # [5단계] 분석 결과를 JSON 파일로 저장합니다.
    #
    # 저장 위치:
    # 현재 실습 폴더 / out / result.json
    write_json(BASE_DIR / "out/result.json", result)

    # [6단계] 분석 결과를 화면에 출력합니다.
    #
    # json_safe()
    #   → JSON으로 표현하기 어려운 값을
    #     저장 가능한 형태로 변환하는 제공 함수입니다.
    #
    # ensure_ascii=False
    #   → 한글이 깨지거나 유니코드 이스케이프 형태로
    #     표시되지 않도록 합니다.
    #
    # indent=2
    #   → 들여쓰기 두 칸을 사용해 보기 좋게 출력합니다.
    print(
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=2
        )
    )


# 이 파이썬 파일을 직접 실행했을 때만
# main() 함수를 실행한다는 의미입니다.
#
# 다른 파이썬 파일에서 import하는 경우에는
# main()이 자동 실행되지 않습니다.
if __name__ == "__main__":
    main()
