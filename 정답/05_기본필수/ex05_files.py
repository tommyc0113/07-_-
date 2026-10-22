
"""실습 05: CSV/JSON 왕복 저장 및 자료형·결측값 검증

정제된 실행 기록과 집계 결과를 CSV/JSON으로 저장한 뒤 다시 읽어,
원본과 비교하여 데이터가 올바르게 보존되었는지 확인합니다.

- CSV: 개별 실행 기록 저장
- JSON: 에이전트별 집계 결과 저장
- Round-trip: 저장 → 다시 읽기 → 원본과 비교
- 검증 항목: 행 수, ID, 문자열, 숫자, 결측값, 날짜, 요약 데이터
"""

from pathlib import Path
import numpy as np
import pandas as pd

from support import (
    COLUMNS, read_raw, strict_json,
    staged_output, protect_inputs, require_columns
)
from analyze import validate_meta


def flatten_meta(payload):
    """중첩된 메타 JSON을 agent/rate_per_1k 형태의 DataFrame으로 변환합니다.

    입력: {"agents": [{"agent": "planner", "pricing": {"rate": 0.5}}]}
    출력: agent="planner", rate_per_1k=0.5 형태의 표
    """

    # JSON 최상위가 딕셔너리인지, agents가 비어 있지 않은 리스트인지 검사합니다.
    if not isinstance(payload, dict) or not isinstance(payload.get("agents"), list) or not payload["agents"]:
        raise ValueError("메타 JSON에는 비어 있지 않은 agents 목록이 필요합니다.")

    # 각 에이전트가 딕셔너리이며 pricing.rate 정보를 포함하는지 확인합니다.
    # all()은 모든 항목이 조건을 만족할 때만 True를 반환합니다.
    if not all(isinstance(item, dict) and isinstance(item.get("pricing"), dict)
               and "rate" in item["pricing"] for item in payload["agents"]):
        raise ValueError("각 역할에는 pricing.rate가 필요합니다.")

    # [1단계] 중첩 JSON을 DataFrame으로 변환합니다.
    # sep="_"를 사용하면 pricing.rate가 pricing_rate 열로 펼쳐집니다.
    flat = pd.json_normalize(payload["agents"], sep="_")

    # [2단계] pricing_rate 열을 분석에 사용하는 rate_per_1k로 변경합니다.
    meta = flat.rename(columns={"pricing_rate": "rate_per_1k"})

    # [3단계] 메타데이터의 필수 열, agent 중복, 단가 유효성을 검사합니다.
    return validate_meta(meta)


def read_meta(path):
    """메타 JSON 파일을 읽고 데이터 구조까지 검증합니다.

    strict_json(): JSON 문법 및 비표준 숫자 표현 검사
    flatten_meta(): 중첩 구조 변환 및 메타데이터 검증

    JSON 문법이 올바르더라도 필수 정보가 없으면 오류가 발생합니다.
    """

    return flatten_meta(strict_json(path))


def save_tables(clean, summary, out_dir):
    """정제 데이터는 CSV, 집계 데이터는 JSON 파일로 저장합니다.

    clean   → clean_runs.csv (개별 실행 기록)
    summary → summary.json (에이전트별 집계 결과)

    CSV는 표 형태의 데이터를, JSON은 구조화된 데이터를
    다른 프로그램과 교환할 때 주로 사용합니다.
    """

    # 출력 경로를 Path 객체로 변환하고 폴더가 없다면 생성합니다.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [1단계] 정제 데이터를 CSV 파일로 저장합니다.
    #
    # index=False: DataFrame의 행 인덱스를 별도 열로 저장하지 않습니다.
    # encoding="utf-8-sig": Windows Excel 등에서 한글 인식을 돕습니다.
    # date_format="%Y-%m-%d": 날짜를 연도-월-일 형식으로 저장합니다.
    #
    # 숫자 결측값(NaN)은 기본적으로 빈 셀로 저장됩니다.
    clean.to_csv(
        out_dir / "clean_runs.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d"
    )

    # [2단계] 에이전트별 집계 결과를 JSON 파일로 저장합니다.
    #
    # orient="records": 각 행을 딕셔너리 형태로 변환한 리스트로 저장합니다.
    # force_ascii=False: 한글을 유니코드 이스케이프 없이 저장합니다.
    # indent=2: 들여쓰기 2칸으로 JSON을 보기 좋게 정리합니다.
    # double_precision=10: 실수 저장 시 소수점 이하 최대 10자리까지 사용합니다.
    #
    # Pandas는 NaN 등의 결측값을 JSON 표준의 null로 변환합니다.
    # 문자 "NaN"으로 저장하는 것이 아닙니다.
    summary.to_json(
        out_dir / "summary.json",
        orient="records",
        force_ascii=False,
        indent=2,
        double_precision=10
    )


def read_back(out_dir):
    """저장된 CSV와 JSON 파일을 다시 읽어옵니다.

    CSV의 run_id는 문자열 자료형으로 읽어 앞쪽 0이 사라지지 않도록 합니다.
    JSON은 strict_json()을 사용하여 비표준 NaN/Infinity를 거부합니다.
    """

    # 파일이 저장된 폴더 경로를 Path 객체로 변환합니다.
    out_dir = Path(out_dir)

    # [1단계] CSV를 DataFrame으로 복원합니다.
    #
    # pd.read_csv(): CSV 파일을 DataFrame으로 읽습니다.
    # dtype={"run_id": "string"}: ID를 숫자가 아닌 문자열로 유지합니다.
    #
    # 예: "00123"을 숫자로 읽으면 123으로 변할 수 있으므로
    # 식별자(ID)는 문자열로 명시하는 것이 안전합니다.
    back = pd.read_csv(
        out_dir / "clean_runs.csv",
        encoding="utf-8-sig",
        dtype={"run_id": "string"}
    )

    # [2단계] JSON을 다시 읽습니다.
    # strict_json()은 비표준 NaN/Infinity 표현을 허용하지 않습니다.
    # orient="records"로 저장했으므로 리스트 형태의 결과를 기대합니다.
    records = strict_json(out_dir / "summary.json")

    # CSV 복원 DataFrame과 JSON 복원 데이터를 튜플로 반환합니다.
    return back, records


def verify_roundtrip(clean, summary, back, records):
    """CSV/JSON 저장 전후 데이터가 동일한 의미를 가지는지 검증합니다.

    검증 항목:
    1. CSV 행 개수 및 ID 고유성
    2. JSON 행 개수 및 필수 열
    3. 문자열과 숫자 데이터
    4. NaN 및 NaT 결측값 유지 여부
    5. 날짜 데이터와 집계 결과

    파일의 바이트가 완전히 동일할 필요는 없으며,
    자료형의 표현 차이와 허용 범위 내의 실수 오차는 인정합니다.
    """

    # [1단계] CSV를 다시 읽은 데이터에 필수 열이 존재하는지 확인합니다.
    require_columns(back)

    # 저장 전후 행 개수가 동일하고 run_id가 중복되지 않는지 검사합니다.
    # is_unique는 값이 모두 고유한 경우 True를 반환합니다.
    if len(back) != len(clean) or not back["run_id"].is_unique:
        raise ValueError("CSV 행 수 또는 ID 고유성이 달라졌습니다.")

    # [2단계] JSON이 리스트 형태이며 저장 전 요약 데이터와 행 수가 같은지 검사합니다.
    if not isinstance(records, list) or len(records) != len(summary):
        raise ValueError("JSON은 요약 행 수와 같은 길이의 목록이어야 합니다.")

    # JSON 데이터를 DataFrame으로 복원하고 필수 열을 검사합니다.
    restored = pd.DataFrame(records)
    require_columns(restored, list(summary.columns))

    # [3단계] JSON에서 읽은 숫자 열을 다시 숫자형으로 변환합니다.
    #
    # JSON의 null이 포함된 열은 object 자료형으로 복원될 수 있습니다.
    # is_numeric_dtype(): 원본 열이 숫자형인지 확인합니다.
    # errors="raise": 숫자로 변환할 수 없는 값이 있으면 오류를 발생시킵니다.
    for col in summary.columns:
        if pd.api.types.is_numeric_dtype(summary[col]):
            restored[col] = pd.to_numeric(
                restored[col],
                errors="raise"
            ).astype(float)

    try:
        # [4단계] run_id, agent, status 문자열 열을 저장 전후 비교합니다.
        # 문자열로 자료형을 통일한 후 리스트 단위로 비교합니다.
        # ID의 앞자리 0이 사라지는 등의 변경도 감지할 수 있습니다.
        for col in ["run_id", "agent", "status"]:
            if clean[col].astype("string").tolist() != back[col].astype("string").tolist():
                raise ValueError("CSV 문자열 열이 달라졌습니다: " + col)

        # [5단계] latency_ms와 tokens 숫자 열을 비교합니다.
        #
        # assert_allclose(): 부동소수점의 미세한 오차를 허용하여 비교합니다.
        # equal_nan=True: 동일한 위치의 NaN을 같은 결측값으로 인정합니다.
        for col in ["latency_ms", "tokens"]:
            np.testing.assert_allclose(
                clean[col].to_numpy(float),
                back[col].to_numpy(float),
                equal_nan=True
            )

        # [6단계] 저장 전후 날짜를 동일한 datetime 자료형으로 변환합니다.
        # errors="coerce"는 해석할 수 없는 날짜를 NaT로 처리합니다.
        before_day = pd.to_datetime(
            clean["day"],
            format="%Y-%m-%d",
            errors="coerce"
        )

        after_day = pd.to_datetime(
            back["day"],
            format="%Y-%m-%d",
            errors="coerce"
        )

        # 날짜 값과 NaT 위치가 동일한지 확인합니다.
        # reset_index(drop=True): 행 인덱스를 초기화합니다.
        # check_names=False: Series 이름 차이는 검사하지 않습니다.
        pd.testing.assert_series_equal(
            before_day.reset_index(drop=True),
            after_day.reset_index(drop=True),
            check_names=False
        )

        # [7단계] 저장 전 summary와 JSON에서 복원한 restored를 비교합니다.
        #
        # check_dtype=False: int/float 등 자료형의 차이를 허용합니다.
        # check_exact=False: 실수의 미세한 표현 차이를 허용합니다.
        # rtol=1e-8: 상대 오차 허용 범위
        # atol=1e-8: 절대 오차 허용 범위
        pd.testing.assert_frame_equal(
            summary.reset_index(drop=True),
            restored.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-8,
            atol=1e-8
        )

    # 수치·날짜·요약 비교에서 발생한 오류를 ValueError로 통일합니다.
    # from exc는 원래 오류의 원인 정보를 유지합니다.
    except (AssertionError, TypeError) as exc:
        raise ValueError("저장 전후 수치·날짜·요약이 다릅니다.") from exc

    # [8단계] 모든 검증을 통과하면 검증 결과를 딕셔너리로 반환합니다.
    #
    # csv_rows: 복원된 CSV의 행 개수
    # json_rows: 복원된 JSON의 요약 행 개수
    # latency_missing: 실행 시간에 NaN이 있는 기록 수
    # verified: 검증 통과 여부
    return {
        "csv_rows": len(back),
        "json_rows": len(records),
        "latency_missing": int(back["latency_ms"].isna().sum()),
        "verified": True
    }


def export_tables(clean, summary, out_dir, data_dir):
    """임시 저장 → 파일 재읽기 → 데이터 검증 → 최종 결과 반영을 수행합니다.

    검증 전에 기존 결과 파일을 바로 덮어쓰지 않도록
    임시 저장 영역(staging)을 사용하는 안전한 저장 흐름입니다.
    """

    # [1단계] 출력 경로가 입력 데이터 영역을 침범하지 않도록 검사합니다.
    # 실습 원본 파일을 실수로 덮어쓰는 문제를 방지합니다.
    protect_inputs(out_dir, data_dir)

    # [2단계] staged_output()으로 임시 저장 영역을 준비합니다.
    # with 블록이 정상 종료되면 검증된 결과를 최종 출력 위치에 반영합니다.
    with staged_output(out_dir) as stage:

        # 임시 폴더에 CSV와 JSON을 저장합니다.
        save_tables(clean, summary, stage)

        # 방금 저장한 파일을 다시 읽어옵니다.
        back, records = read_back(stage)

        # 저장 전후 데이터의 일치 여부를 검증합니다.
        # 검증에 실패하면 오류가 발생하여 정상적인 반영 과정으로 진행하지 않습니다.
        result = verify_roundtrip(clean, summary, back, records)

    # 최종 검증 결과를 반환합니다.
    return result


# ============================================================
# 실행부: 데이터 로드 → CSV/JSON 저장 → 검증 결과 출력
# ============================================================

from pathlib import Path
from support import write_json, json_safe
import json

# 현재 파이썬 파일이 위치한 폴더의 절대 경로입니다.
BASE_DIR = Path(__file__).resolve().parent

from support import load_inputs
from analyze import join_meta


def main():
    # [1단계] 결과 저장용 out 폴더를 생성합니다.
    # exist_ok=True이므로 폴더가 이미 존재해도 오류가 발생하지 않습니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # [2단계] 실습 입력 데이터를 불러옵니다.
    # clean: 개별 실행 기록 / summary: 에이전트별 집계 결과
    clean, summary = load_inputs(BASE_DIR)

    # [3단계] CSV/JSON 저장과 Round-trip 검증을 수행합니다.
    # out: 결과 파일 저장 위치 / data: 보호해야 할 원본 입력 폴더
    result = export_tables(
        clean,
        summary,
        BASE_DIR / "out",
        BASE_DIR / "data"
    )

    # [4단계] 최종 검증 결과를 out/result.json으로 저장합니다.
    # summary.json은 집계 결과, result.json은 검증 결과를 저장합니다.
    write_json(BASE_DIR / "out/result.json", result)

    # [5단계] 검증 결과를 JSON 형태로 콘솔에 출력합니다.
    # ensure_ascii=False: 한글 유지 / indent=2: 들여쓰기 2칸
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


# 현재 파이썬 파일을 직접 실행한 경우에만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()
