
"""실습 03 / EDA 프로젝트: 집계의 분모와 데이터 결합의 유효성을 확인합니다."""

import numpy as np
import pandas as pd
from support import require_columns


def aggregate_runs(clean):
    """정제된 데이터를 에이전트(agent)별로 묶어 6가지 통계를 계산합니다.

    runs: 전체 실행 횟수
    measured: 실행 시간이 정상적으로 측정된 횟수
    mean_ms: 평균 실행 시간
    median_ms: 실행 시간의 중앙값
    success_rate: 전체 실행 중 성공 비율 (0~1)
    success_pct: 성공률의 백분율 표현 (0~100)

    평균은 NaN을 제외한 유효 측정값만 사용하고,
    성공률은 error를 포함한 전체 실행 횟수를 분모로 사용합니다.
    원본 데이터는 변경하지 않으며 계산 과정에서 반올림하지 않습니다.
    """

    # 필수 열이 존재하는지 검사합니다.
    require_columns(clean)

    # status는 "ok" 또는 "error"만 허용합니다.
    # 알 수 없는 상태를 임의로 실패 처리하면 성공률이 왜곡될 수 있습니다.
    if not clean["status"].isin(["ok", "error"]).all():
        raise ValueError("알 수 없는 상태를 실패로 간주할 수 없습니다.")

    # 원본 보호를 위해 복사본을 생성합니다.
    # status가 "ok"이면 True, "error"이면 False인 열을 추가합니다.
    # bool 평균은 True=1, False=0으로 계산되므로 성공률을 구할 수 있습니다.
    df = clean.copy()
    df["ok"] = df["status"].eq("ok")

    # agent별로 그룹화하여 통계를 계산합니다.
    # size: NaN 포함 전체 개수 / count: NaN 제외 유효 개수
    # mean: 평균 / median: 중앙값
    # as_index=False: agent를 인덱스가 아닌 일반 열로 유지합니다.
    # observed=True: 실제 관측된 범주를 기준으로 집계합니다.
    result = df.groupby("agent", as_index=False, sort=True, observed=True).agg(
        runs=("run_id", "size"),
        measured=("latency_ms", "count"),
        mean_ms=("latency_ms", "mean"),
        median_ms=("latency_ms", "median"),
        success_rate=("ok", "mean"),
    )

    # 성공률(0~1)에 100을 곱하여 백분율(0~100)로 변환합니다.
    result["success_pct"] = result["success_rate"] * 100

    return result


def make_pivot(clean):
    """에이전트(agent)와 실행 상태(status)별 평균 실행 시간 표를 생성합니다.

    행: agent / 열: error, ok / 값: latency_ms 평균
    측정값이 없으면 NaN으로 유지하며 0으로 대체하지 않습니다.
    """

    # pivot_table을 이용하여 agent×status별 평균 실행 시간을 계산합니다.
    # observed=True는 실제 관측된 범주 조합을 기준으로 집계합니다.
    pivot = clean.pivot_table(
        index="agent",
        columns="status",
        values="latency_ms",
        aggfunc="mean",
        observed=True
    )

    # reindex로 agent를 정렬하고 상태 열을 error, ok 순서로 통일합니다.
    # 모든 측정값이 결측인 agent나 관측되지 않은 상태도 NaN으로 표시합니다.
    return pivot.reindex(
        index=sorted(clean["agent"].unique()),
        columns=["error", "ok"]
    )


def validate_meta(meta):
    """데이터 결합 전에 메타데이터의 구조와 값이 올바른지 검증합니다.

    agent: 에이전트 이름(결합 기준 키)
    rate_per_1k: 토큰 1,000개당 가상 단가

    agent의 결측·중복 여부와 단가의 유효성을 확인합니다.
    """

    # 메타데이터에 필수 열이 존재하는지 검사합니다.
    require_columns(meta, ["agent", "rate_per_1k"])

    # 원본을 복사하고 agent 이름의 공백 제거 및 소문자 변환을 수행합니다.
    result = meta.copy()
    result["agent"] = result["agent"].astype("string").str.strip().str.lower()

    # 데이터가 비어 있거나 agent에 결측값·빈 문자열이 있으면 오류 처리합니다.
    if result.empty or result["agent"].isna().any() or result["agent"].eq("").any():
        raise ValueError("메타데이터의 역할 키가 비어 있습니다.")

    # 중복 agent가 존재하면 하나의 역할에 여러 단가가 연결될 수 있으므로 차단합니다.
    if result["agent"].duplicated().any():
        raise ValueError(
            "중복 메타 키: " +
            ", ".join(result.loc[result["agent"].duplicated(), "agent"])
        )

    # 단가를 실수형으로 변환합니다. 숫자로 변환할 수 없는 값은 NaN 처리합니다.
    result["rate_per_1k"] = pd.to_numeric(
        result["rate_per_1k"], errors="coerce"
    ).astype(float)

    # 단가가 NaN, 무한대 또는 음수인 경우 오류를 발생시킵니다.
    # np.isfinite(): 유한한 숫자인지 검사 / lt(0): 0보다 작은지 검사
    if (~np.isfinite(result["rate_per_1k"]) | result["rate_per_1k"].lt(0)).any():
        raise ValueError("단가는 0 이상의 유한한 수치여야 합니다.")

    return result


def join_meta(clean, meta):
    """실행 기록과 메타데이터를 agent 기준으로 결합합니다.

    실행 기록(여러 행) : 메타데이터(하나의 agent당 한 행)의
    다대일(many-to-one) 관계를 검증합니다.

    결합되지 않은 agent가 있으면 오류를 발생시키고,
    정상 결합 시 토큰 사용 비용(cost_unit)을 계산합니다.
    """

    # 결합하기 전에 메타데이터의 키와 단가를 검증합니다.
    meta = validate_meta(meta)

    # 실행 기록의 agent가 NaN이면 정상적인 결합이 불가능합니다.
    if clean["agent"].isna().any():
        raise ValueError("왼쪽 역할 키가 결측입니다.")

    # left merge: 왼쪽 실행 기록을 모두 유지하면서 메타 정보를 연결합니다.
    # validate="many_to_one": 오른쪽 메타 키의 중복을 허용하지 않습니다.
    # indicator=True: 결합 성공 여부를 확인할 _merge 열을 생성합니다.
    joined = clean.merge(
        meta,
        on="agent",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # 결합 전후 실행 기록의 행 개수가 동일한지 확인합니다.
    if len(joined) != len(clean):
        raise ValueError("결합 후 실행 행 수가 달라졌습니다.")

    # _merge가 "both"가 아닌 기록은 메타데이터와 연결되지 않은 경우입니다.
    # 미등록된 agent를 중복 없이 추출합니다.
    unmatched = joined.loc[
        joined["_merge"].ne("both"), "agent"
    ].unique().tolist()

    if unmatched:
        raise ValueError("미등록 메타 키: " + ", ".join(unmatched))

    # 토큰 사용 비용 = 사용 토큰 수 / 1000 × 토큰 1,000개당 단가
    # tokens가 NaN이면 계산 결과도 NaN으로 유지됩니다.
    joined["cost_unit"] = joined["tokens"] / 1000 * joined["rate_per_1k"]

    return joined


def daily_summary(clean):
    """날짜(day)별 실행 횟수, 유효 측정 횟수, 평균 실행 시간을 집계합니다.

    날짜가 NaT(날짜 결측값)인 기록도 삭제하지 않고 집계에 포함합니다.
    """

    # day를 기준으로 그룹화합니다.
    # dropna=False: 날짜가 없는 기록도 하나의 그룹으로 포함합니다.
    # size: 전체 실행 수 / count: NaN 제외 측정 수 / mean: 평균 실행 시간
    return clean.groupby("day", dropna=False, sort=True).agg(
        runs=("run_id", "size"),
        measured=("latency_ms", "count"),
        mean_ms=("latency_ms", "mean")
    )


def cost_audit(joined):
    """계산 가능한 비용의 합계와 데이터 누락 여부를 확인합니다.

    전체 실행 중 일부 기록의 비용이 누락되었다면
    계산된 합계를 전체 비용이 아닌 부분합으로 구분합니다.
    """

    # NaN을 제외하고 비용이 정상적으로 계산된 기록 수를 구합니다.
    count = int(joined["cost_unit"].count())

    # 비용 합계를 계산합니다.
    # min_count=1: 유효 값이 하나도 없으면 0이 아닌 NaN을 반환합니다.
    value = float(joined["cost_unit"].sum(min_count=1))

    # is_partial=True이면 전체 실행 중 비용이 누락된 기록이 있다는 의미입니다.
    # 반환 단위는 실제 청구 금액이 아닌 실습용 가상 단위입니다.
    return {
        "cost_unit": value,
        "measured": count,
        "runs": len(joined),
        "is_partial": count < len(joined),
        "unit": "가상 단위/실제 가격 아님"
    }


def weighted_mean(means, counts):
    """각 집단의 데이터 개수를 고려하여 가중평균을 계산합니다.

    단순 평균: 집단별 평균을 동일한 비중으로 계산
    가중평균: 각 집단의 측정 개수를 반영하여 계산

    예시:
    A집단 평균 90점(10명), B집단 평균 70점(90명)
    단순 평균 = 80점
    가중평균 = (90×10 + 70×90) / 100 = 72점

    서로 중복되지 않는 같은 단위의 집단을 대상으로 계산합니다.
    """

    # 평균과 측정 개수를 NumPy 실수형 배열로 변환합니다.
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # 두 배열의 길이와 차원이 동일한 1차원 배열인지 검사합니다.
    if means.shape != counts.shape or means.ndim != 1:
        raise ValueError("평균과 개수는 길이가 같은 1차원 배열이어야 합니다.")

    # 측정 개수는 음수·소수·NaN·무한대가 아닌 정수여야 합니다.
    # counts % 1 != 0은 소수 부분이 있는지 검사하는 조건입니다.
    if (~np.isfinite(counts) | (counts < 0) | (counts % 1 != 0)).any():
        raise ValueError("측정 개수는 0 이상의 유한한 정수여야 합니다.")

    # 측정 개수가 1 이상인 집단의 평균은 반드시 유효한 숫자여야 합니다.
    if ((counts > 0) & ~np.isfinite(means)).any():
        raise ValueError("측정이 있는 그룹의 평균이 유효하지 않습니다.")

    # 측정 개수가 0인 집단은 가중평균 계산에서 제외합니다.
    valid = counts > 0

    # np.average(): weights에 측정 개수를 전달하여 가중평균을 계산합니다.
    # 모든 집단의 측정 개수가 0이면 계산할 수 없으므로 NaN을 반환합니다.
    return float(np.average(means[valid], weights=counts[valid])) if valid.any() else np.nan


# ============================================================
# 실행부: 입력 데이터 로드 → 에이전트별 집계 → JSON 저장
# ============================================================

from pathlib import Path
from support import write_json, json_safe
import json

# 현재 파이썬 파일이 위치한 폴더의 절대 경로입니다.
BASE_DIR = Path(__file__).resolve().parent

from support import load_inputs, strict_json


def main():
    # 분석 결과를 저장할 out 폴더를 생성합니다. 이미 존재해도 오류가 발생하지 않습니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # 실습용 정제 데이터를 불러옵니다.
    # 두 번째 반환값은 사용하지 않으므로 _ 변수에 저장합니다.
    clean, _ = load_inputs(BASE_DIR)

    # 에이전트별 실행 횟수, 평균, 중앙값, 성공률 등을 계산합니다.
    summary = aggregate_runs(clean)

    # 집계 결과를 JSON 저장이 가능한 딕셔너리로 변환합니다.
    # to_dict(orient="records"): DataFrame의 각 행을 딕셔너리로 변환합니다.
    result = {
        "summary": summary.to_dict(orient="records"),
        "runs": int(summary["runs"].sum()),
        "measured": int(summary["measured"].sum())
    }

    # 분석 결과를 out/result.json 파일에 저장합니다.
    write_json(BASE_DIR / "out/result.json", result)

    # JSON 데이터를 한글이 유지되도록 들여쓰기 2칸으로 출력합니다.
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


# 해당 파일을 직접 실행했을 때만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()
