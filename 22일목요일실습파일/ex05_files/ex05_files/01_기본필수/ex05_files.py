
"""실습 05: CSV/JSON 파일 저장 및 데이터 검증

[실습 목표]
실습 01~04에서 처리한 데이터를 CSV와 JSON 파일로 저장하고,
저장한 파일을 다시 읽어 원본 데이터와 비교합니다.

1. CSV 파일로 데이터 저장하기
2. JSON 파일로 데이터 저장하기
3. CSV/JSON 파일 다시 불러오기
4. 문자열, 숫자, 날짜의 자료형 이해하기
5. NaN(결측값)이 저장 후에도 올바르게 유지되는지 확인하기
6. 저장 전후 데이터가 동일한 의미를 가지는지 검증하기

[주요 용어]

CSV (Comma-Separated Values)
    → 쉼표로 데이터를 구분하는 텍스트 파일 형식입니다.
    → 엑셀과 유사한 행·열 구조의 데이터를 저장할 때 사용합니다.

JSON (JavaScript Object Notation)
    → 데이터를 키(key)와 값(value) 형태로 표현하는 파일 형식입니다.
    → 웹 서비스, API, 데이터 교환 등에 많이 사용합니다.

DataFrame
    → Pandas에서 제공하는 표 형태의 데이터 구조입니다.
    → 엑셀 시트처럼 행과 열로 데이터를 관리합니다.

dtype (Data Type)
    → 데이터의 자료형을 의미합니다.
    → 예: int(정수), float(실수), string(문자열)

NaN (Not a Number)
    → 숫자가 없거나 유효하지 않은 상태를 표현합니다.

NaT (Not a Time)
    → 날짜 또는 시간 데이터의 결측 상태를 표현합니다.

Round-trip (왕복 저장)
    → 데이터를 파일로 저장한 뒤 다시 읽는 과정입니다.
    → 저장 전과 저장 후의 내용이 동일한지 검증합니다.

[중요한 원칙]
파일 저장에 성공했다고 해서 데이터가 올바르게 보존되었다는
의미는 아닙니다.

예를 들어 "00123"이라는 ID를 숫자로 잘못 읽으면
123으로 바뀔 수 있습니다.

또한 NaN을 0으로 변경하면
"측정되지 않음"이라는 원래 의미가 사라집니다.

따라서 저장 → 다시 읽기 → 비교 검증 과정이 필요합니다.
"""


# ============================================================
# 1. 필요한 라이브러리 불러오기
# ============================================================

# 파일 및 폴더의 경로를 관리하는 파이썬 표준 라이브러리입니다.
from pathlib import Path

# NumPy: 숫자 배열 처리, 수치 비교, 결측값 검사 등에 사용합니다.
import numpy as np

# Pandas: CSV/JSON 처리 및 DataFrame 연산에 사용합니다.
import pandas as pd

# support.py에서 제공하는 실습 보조 함수들을 가져옵니다.
#
# COLUMNS
#   → 실습 데이터의 열 이름과 관련된 상수
#
# read_raw
#   → 원본 데이터를 불러오기 위한 제공 함수
#
# strict_json
#   → JSON 파일을 엄격한 규칙으로 읽는 함수
#   → 비표준 NaN, Infinity 등이 포함된 JSON을 거부합니다.
#
# staged_output
#   → 임시 위치에 결과를 저장하고 검증 후 반영하기 위한 기능
#
# protect_inputs
#   → 입력 데이터가 저장 과정에서 덮어써지지 않도록 보호
#
# require_columns
#   → DataFrame에 필수 열이 존재하는지 확인
#
# ※ 일부 가져온 기능은 이 파일에서 직접 사용되지 않을 수 있습니다.
from support import (
    COLUMNS,
    read_raw,
    strict_json,
    staged_output,
    protect_inputs,
    require_columns
)

# 실습 03에서 작성한 메타데이터 검증 함수를 가져옵니다.
from analyze import validate_meta


# ============================================================
# 2. JSON의 중첩 구조를 DataFrame으로 변환
# ============================================================

def flatten_meta(payload):
    """중첩된 JSON 데이터를 평평한 표 형태로 변환합니다.

    [함수의 목적]
    JSON은 데이터 안에 또 다른 데이터가 들어가는
    중첩 구조(Nested Structure)를 표현할 수 있습니다.

    하지만 DataFrame으로 분석할 때는 일반적으로
    각 속성이 하나의 열로 구성된 형태가 편리합니다.

    따라서 중첩된 JSON 구조를 일반적인 표 형태로 변환합니다.

    [입력 예시]

    payload = {
        "agents": [
            {
                "agent": "planner",
                "pricing": {
                    "rate": 0.5
                }
            },
            {
                "agent": "coder",
                "pricing": {
                    "rate": 1.0
                }
            }
        ]
    }

    [변환 결과]

    agent       rate_per_1k
    planner     0.5
    coder       1.0

    pricing 안에 있던 rate 값을 꺼내서
    rate_per_1k라는 열로 변환합니다.

    [주요 함수]
    pd.json_normalize()
        → 중첩된 JSON 데이터를 DataFrame으로 변환

    rename()
        → DataFrame의 열 이름 변경

    validate_meta()
        → 변환된 데이터에 문제가 없는지 검사
    """

    # [1단계] 입력받은 JSON의 기본 구조를 검사합니다.
    #
    # isinstance(payload, dict)
    #   → 입력값이 파이썬 딕셔너리인지 확인합니다.
    #
    # payload.get("agents")
    #   → JSON에 "agents"라는 키가 있는지 확인하고
    #     해당 키의 값을 가져옵니다.
    #
    # isinstance(..., list)
    #   → agents 값이 리스트 형태인지 검사합니다.
    #
    # not payload["agents"]
    #   → agents 리스트가 비어 있는지 확인합니다.
    #
    # 즉, 아래 조건을 모두 만족해야 합니다.
    #
    # ① 최상위 데이터가 딕셔너리여야 합니다.
    # ② agents 항목이 리스트여야 합니다.
    # ③ agents 리스트에 데이터가 최소 1개 있어야 합니다.
    #
    # 조건을 만족하지 않으면 ValueError를 발생시킵니다.
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("agents"), list)
        or not payload["agents"]
    ):
        raise ValueError(
            "메타 JSON에는 비어 있지 않은 agents 목록이 필요합니다."
        )

    # [2단계] agents 내부의 각 항목을 검사합니다.
    #
    # all()
    #   → 모든 항목이 조건을 만족하는지 확인합니다.
    #
    # isinstance(item, dict)
    #   → 개별 에이전트 정보가 딕셔너리인지 검사합니다.
    #
    # isinstance(item.get("pricing"), dict)
    #   → pricing 정보가 딕셔너리 형태인지 확인합니다.
    #
    # "rate" in item["pricing"]
    #   → pricing 안에 rate라는 키가 존재하는지 확인합니다.
    #
    # 예를 들어 아래 데이터는 정상입니다.
    #
    # {"agent": "planner", "pricing": {"rate": 0.5}}
    #
    # 반면 아래 데이터는 rate가 없으므로 오류입니다.
    #
    # {"agent": "planner", "pricing": {}}
    #
    # 모든 에이전트에 pricing.rate 정보가 있어야 합니다.
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("pricing"), dict)
        and "rate" in item["pricing"]
        for item in payload["agents"]
    ):
        raise ValueError("각 역할에는 pricing.rate가 필요합니다.")

    # [3단계] 중첩된 JSON 데이터를 평평한 DataFrame으로 변환합니다.
    #
    # pd.json_normalize()
    #   → JSON 안에 포함된 중첩 딕셔너리 구조를 열로 펼칩니다.
    #
    # payload["agents"]
    #   → JSON에서 에이전트 목록만 선택합니다.
    #
    # sep="_"
    #   → 중첩된 속성 이름을 언더스코어(_)로 연결합니다.
    #
    # 예시:
    # pricing: {"rate": 0.5}
    #          ↓
    # pricing_rate: 0.5
    #
    # 결과적으로 다음과 같은 열이 생성됩니다.
    #
    # agent | pricing_rate
    flat = pd.json_normalize(
        payload["agents"],
        sep="_"
    )

    # [4단계] 열 이름을 변경합니다.
    #
    # rename(columns={기존이름: 새로운이름})
    #
    # pricing_rate → rate_per_1k
    #
    # 다른 분석 함수에서 사용하는 열 이름과
    # 일치시키기 위한 작업입니다.
    meta = flat.rename(
        columns={"pricing_rate": "rate_per_1k"}
    )

    # [5단계] 변환된 메타데이터의 유효성을 검사합니다.
    #
    # validate_meta()는 다음 내용을 검사합니다.
    #
    # ① agent 열이 존재하는가?
    # ② rate_per_1k 열이 존재하는가?
    # ③ agent 값이 비어 있지 않은가?
    # ④ 중복된 agent가 없는가?
    # ⑤ 단가가 0 이상의 유한한 숫자인가?
    #
    # 검증에 성공하면 정제된 DataFrame을 반환합니다.
    return validate_meta(meta)


def read_meta(path):
    """JSON 파일을 읽고 메타데이터의 구조까지 검사합니다.

    [핵심 개념]
    JSON 파일은 문법이 올바르더라도
    프로그램에서 요구하는 데이터 구조와 다를 수 있습니다.

    예를 들어 아래 JSON은 문법적으로 올바릅니다.

    {"hello": "world"}

    하지만 이 실습에서는 agents와 pricing.rate가
    필요하므로 정상적인 메타데이터로 사용할 수 없습니다.

    따라서 JSON 파일을 읽는 것과
    내용의 유효성을 검사하는 것은 별도 과정입니다.
    """

    # [1단계] strict_json(path)
    #   → 파일을 JSON 형식으로 읽습니다.
    #   → NaN, Infinity 같은 비표준 숫자 표현을 거부합니다.
    #
    # [2단계] flatten_meta(...)
    #   → JSON 구조를 검사하고 DataFrame으로 변환합니다.
    #
    # [3단계] 변환된 DataFrame을 반환합니다.
    return flatten_meta(strict_json(path))


# ============================================================
# 3. CSV와 JSON 파일 저장
# ============================================================

def save_tables(clean, summary, out_dir):
    """정제 데이터는 CSV, 집계 데이터는 JSON으로 저장합니다.

    [입력 데이터]

    clean
        → 실습 02에서 정제한 개별 실행 기록 DataFrame

    summary
        → 실습 03에서 계산한 에이전트별 통계 DataFrame

    out_dir
        → 결과 파일을 저장할 폴더 경로

    [저장 결과]

    out_dir/
        clean_runs.csv
        summary.json

    [CSV로 저장하는 이유]
    개별 실행 기록은 행과 열로 구성되어 있기 때문에
    CSV 파일로 저장하면 엑셀 등에서 확인하기 편리합니다.

    [JSON으로 저장하는 이유]
    분석 결과를 리스트와 딕셔너리 형태로 표현하면
    다른 프로그램이나 API에서 활용하기 편리합니다.

    [주의]
    저장 과정에서 DataFrame의 인덱스가 불필요한
    새 열로 저장되지 않도록 해야 합니다.

    또한 NaN을 문자 "NaN"으로 저장하지 않도록
    주의해야 합니다.
    """

    # [1단계] 출력 폴더 경로를 Path 객체로 변환합니다.
    #
    # 예시:
    # out_dir = "out"
    #          ↓
    # Path("out")
    #
    # Path 객체를 사용하면 파일 경로를 쉽게 결합할 수 있습니다.
    out_dir = Path(out_dir)

    # [2단계] 결과를 저장할 폴더를 생성합니다.
    #
    # parents=True
    #   → 필요한 상위 폴더까지 함께 생성합니다.
    #
    # exist_ok=True
    #   → 이미 존재하는 폴더라도 오류를 발생시키지 않습니다.
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # TODO I_CSV: 수강생이 직접 구현할 부분
    # =========================================================
    #
    # [목표]
    # clean DataFrame을 clean_runs.csv 파일로 저장합니다.
    #
    # [구현 순서]
    #
    # ① DataFrame의 to_csv() 함수를 사용합니다.
    #
    #    to_csv()
    #      → DataFrame을 CSV 파일로 저장하는 함수입니다.
    #
    # ② 파일 경로를 지정합니다.
    #
    #    파일 이름: clean_runs.csv
    #    저장 폴더: out_dir
    #
    #    Path 객체와 / 연산자를 이용하면
    #    폴더와 파일 이름을 결합할 수 있습니다.
    #
    # ③ index=False를 지정합니다.
    #
    #    Pandas의 DataFrame에는 행 번호 역할을 하는
    #    인덱스(index)가 존재합니다.
    #
    #    예시:
    #
    #    인덱스  run_id  agent
    #    0       001     planner
    #    1       002     coder
    #
    #    index=True로 저장하면 CSV에 불필요한
    #    인덱스 열이 추가될 수 있습니다.
    #
    #    따라서 index=False를 사용합니다.
    #
    # ④ encoding="utf-8-sig"를 지정합니다.
    #
    #    encoding은 문자를 파일에 저장할 때
    #    사용하는 문자 인코딩 방식을 의미합니다.
    #
    #    utf-8-sig는 UTF-8 파일 앞에 BOM을 포함하여
    #    Windows의 Excel 등에서 한글 인식을 돕습니다.
    #
    # ⑤ date_format="%Y-%m-%d"를 지정합니다.
    #
    #    날짜를 연도-월-일 형식으로 저장합니다.
    #
    #    예시:
    #    2026-10-22
    #
    #    %Y: 네 자리 연도
    #    %m: 두 자리 월
    #    %d: 두 자리 일
    #
    # [참고할 함수]
    # DataFrame.to_csv()
    clean.to_csv(
        out_dir / "clean_runs.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d"
    )
    # 아래 오류는 아직 구현되지 않았음을 의미합니다.
    # 수강생이 코드를 완성하면 제거해야 합니다.

    # =========================================================
    # TODO I_JSON: 수강생이 직접 구현할 부분
    # =========================================================
    #
    # [목표]
    # summary DataFrame을 summary.json 파일로 저장합니다.
    #
    # [구현 순서]
    #
    # ① DataFrame의 to_json() 함수를 사용합니다.
    #
    # ② 파일 이름을 summary.json으로 지정합니다.
    #
    # ③ orient="records"를 지정합니다.
    #
    #    DataFrame의 각 행을 하나의 JSON 객체로 변환하고
    #    전체를 리스트 형태로 저장합니다.
    #
    #    예시:
    #
    #    [
    #      {"agent": "planner", "runs": 10},
    #      {"agent": "coder", "runs": 20}
    #    ]
    #
    # ④ force_ascii=False를 지정합니다.
    #
    #    한글 등의 유니코드 문자를
    #    이스케이프 표현 대신 원래 문자로 저장합니다.
    #
    # ⑤ indent=2를 지정합니다.
    #
    #    JSON 데이터를 들여쓰기 2칸으로 정리합니다.
    #    사람이 파일을 열어 확인하기 편리해집니다.
    #
    # ⑥ 결측값 NaN은 JSON의 null로 저장되어야 합니다.
    #
    #    NaN과 "NaN"은 다릅니다.
    #
    #    NaN
    #      → Pandas의 숫자 결측값
    #
    #    null
    #      → JSON에서 값이 없음을 표현하는 표준 값
    #
    #    "NaN"
    #      → 문자 N, a, N으로 이루어진 문자열
    #
    #    숫자 결측값을 문자열로 바꾸면
    #    나중에 수치 계산에서 문제가 생길 수 있습니다.
    #
    # [참고할 함수]
    # DataFrame.to_json()
    summary.to_json(
        out_dir / "summary.json",
        orient="records",
        force_ascii=False,
        indent=2
    )
    # 아래 오류는 구현을 완료한 뒤 제거합니다.
    

# ============================================================
# 4. 저장된 파일 다시 불러오기
# ============================================================

def read_back(out_dir):
    """저장한 CSV와 JSON 파일을 다시 읽어옵니다.

    [함수의 목적]
    파일이 생성되었다는 사실만으로는 데이터가
    정확하게 저장되었는지 확인할 수 없습니다.

    따라서 저장한 파일을 다시 읽어
    실제로 어떤 자료형과 값이 복원되는지 확인합니다.

    [주의할 점 1: ID의 자료형]

    예를 들어 다음 ID가 있다고 가정합니다.

    "000123"

    이 값을 숫자로 읽으면 123이 됩니다.

    숫자 크기는 같지만 ID의 앞쪽 0이 사라지므로
    원래 식별자를 복원할 수 없습니다.

    따라서 run_id는 반드시 문자열로 읽어야 합니다.

    [주의할 점 2: JSON 결측값]

    JSON 표준에서는 결측값을 null로 표현합니다.
    NaN, Infinity는 JSON의 표준 숫자 값이 아닙니다.

    [반환값]

    back
        → CSV 파일을 다시 읽은 DataFrame

    records
        → JSON 파일에서 읽은 리스트 형태의 데이터
    """

    # [1단계] 입력받은 폴더 경로를 Path 객체로 변환합니다.
    out_dir = Path(out_dir)

    # =========================================================
    # TODO I_READ: 수강생이 직접 구현할 부분
    # =========================================================
    #
    # [목표]
    # 앞에서 저장한 CSV와 JSON 파일을 다시 불러옵니다.
    #
    # [구현 순서]
    #
    # ① pd.read_csv()로 clean_runs.csv를 읽습니다.
    #
    #    pd.read_csv()
    #      → CSV 파일을 DataFrame으로 변환하는 함수입니다.
    #
    # ② run_id 열의 자료형을 문자열로 지정합니다.
    #
    #    dtype={"run_id": "string"}
    #
    #    dtype은 Data Type(자료형)의 약자입니다.
    #
    #    run_id를 문자열로 명시하면
    #    앞쪽의 0이 사라지는 문제를 방지할 수 있습니다.
    #
    #    예:
    #    "000123" → "000123"
    #
    # ③ CSV를 읽은 결과를 back 변수에 저장합니다.
    #
    # ④ strict_json()으로 summary.json 파일을 읽습니다.
    #
    #    일반적인 JSON 데이터 구조는
    #    파이썬의 dict 또는 list로 변환됩니다.
    #
    #    이번 실습에서는 orient="records"로 저장했으므로
    #    리스트 형태의 결과를 기대합니다.
    #
    # ⑤ JSON을 읽은 결과를 records 변수에 저장합니다.
    #
    # ⑥ 두 결과를 튜플로 반환합니다.
    #
    #    반환 순서: (back, records)
    #
    # [참고할 함수]
    # pd.read_csv()
    # strict_json()
    
    back = pd.read_csv(
        out_dir / "clean_runs.csv",
        dtype={"run_id": "string"}
    )

    # ② JSON 파일 읽기
    records = strict_json(out_dir / "summary.json")

    # ③ 두 결과 반환
    return back, records
    # 아래 오류는 수강생이 구현한 뒤 제거합니다.

# ============================================================
# 5. 저장 전후 데이터가 동일한지 검증
# ============================================================

def verify_roundtrip(clean, summary, back, records):
    """파일 저장 전후의 데이터가 올바르게 유지되었는지 검증합니다.

    [입력]

    clean
        → CSV로 저장하기 전의 정제 DataFrame

    summary
        → JSON으로 저장하기 전의 요약 DataFrame

    back
        → CSV를 다시 읽어 복원한 DataFrame

    records
        → JSON을 다시 읽어 복원한 데이터

    [검증 내용]

    1. CSV의 행 개수가 동일한가?
    2. 실행 ID(run_id)가 중복되지 않았는가?
    3. 문자열 데이터가 동일한가?
    4. 숫자 데이터가 동일한가?
    5. NaN이 올바르게 유지되었는가?
    6. 날짜 데이터가 동일한가?
    7. JSON 요약 데이터가 동일한가?

    [중요한 개념]

    저장 전후의 파일이 글자 단위로 완전히
    동일할 필요는 없습니다.

    예를 들어 다음 값들은 숫자로서 동일합니다.

    100
    100.0

    또한 JSON의 null은 다시 읽으면
    파이썬의 None 등으로 표현될 수 있습니다.

    따라서 파일의 바이트가 아니라
    데이터의 의미와 값이 유지되었는지 검증합니다.
    """

    # [1단계] CSV에서 복원한 DataFrame에
    # 필요한 열이 존재하는지 검사합니다.
    #
    # require_columns()는 필수 열 누락을 확인합니다.
    require_columns(back)

    # [2단계] CSV의 행 개수와 ID 고유성을 확인합니다.
    #
    # len(back)
    #   → 파일에서 읽은 데이터의 행 개수
    #
    # len(clean)
    #   → 저장하기 전 데이터의 행 개수
    #
    # back["run_id"].is_unique
    #   → run_id 값에 중복이 없는지 확인합니다.
    #
    # 행 수가 달라졌거나 ID에 중복이 생기면 오류입니다.
    if len(back) != len(clean) or not back["run_id"].is_unique:
        raise ValueError("CSV 행 수 또는 ID 고유성이 달라졌습니다.")

    # [3단계] JSON 데이터의 기본 형태를 확인합니다.
    #
    # isinstance(records, list)
    #   → JSON을 읽은 결과가 리스트인지 확인합니다.
    #
    # len(records) != len(summary)
    #   → 저장 전후 요약 데이터의 행 개수를 비교합니다.
    #
    # 이번 실습의 JSON은 records 형식으로 저장하므로
    # 리스트 형태여야 하며 행 개수도 동일해야 합니다.
    if not isinstance(records, list) or len(records) != len(summary):
        raise ValueError("JSON은 요약 행 수와 같은 길이의 목록이어야 합니다.")

    # [4단계] JSON 데이터를 DataFrame으로 복원합니다.
    #
    # records는 다음과 같은 리스트 형태일 수 있습니다.
    #
    # [
    #     {"agent": "planner", "runs": 10},
    #     {"agent": "coder", "runs": 20}
    # ]
    #
    # pd.DataFrame()을 사용하면 다시 표 형태로 변환됩니다.
    restored = pd.DataFrame(records)

    # 복원된 DataFrame에 원래 summary의 열이 모두 존재하는지 검사합니다.
    #
    # list(summary.columns)
    #   → 저장 전 요약 데이터의 열 이름을 리스트로 가져옵니다.
    require_columns(restored, list(summary.columns))

    # [5단계] JSON에서 읽은 숫자 열을 숫자형으로 복원합니다.
    #
    # JSON의 null이 포함된 열은 데이터에 따라
    # object 등의 자료형으로 읽힐 수 있습니다.
    #
    # 따라서 저장 전 숫자형이었던 열을 찾아
    # 다시 숫자형으로 변환합니다.
    for col in summary.columns:

        # is_numeric_dtype()
        #   → 해당 열의 자료형이 숫자형인지 확인합니다.
        if pd.api.types.is_numeric_dtype(summary[col]):

            # pd.to_numeric()
            #   → 데이터를 숫자형으로 변환합니다.
            #
            # errors="raise"
            #   → 숫자로 변환할 수 없는 값이 있으면 오류 발생
            #
            # astype(float)
            #   → 실수형으로 통일합니다.
            #
            # JSON의 null은 숫자 변환 시 NaN으로
            # 처리될 수 있습니다.
            restored[col] = pd.to_numeric(
                restored[col],
                errors="raise"
            ).astype(float)

    # [6단계] 저장 전후 값을 본격적으로 비교합니다.
    #
    # try:
    #   → 비교 과정에서 발생할 수 있는 오류를 처리합니다.
    try:

        # ----------------------------------------------------
        # 6-1. 문자열 열 비교
        # ----------------------------------------------------
        #
        # run_id, agent, status 세 열의 값이
        # 저장 전후 동일한지 확인합니다.
        #
        # astype("string")
        #   → 문자열 자료형으로 통일합니다.
        #
        # tolist()
        #   → 각 열을 파이썬 리스트로 변환합니다.
        #
        # 두 리스트가 다르면 ValueError를 발생시킵니다.
        #
        # 예:
        # 저장 전 ["001", "002"]
        # 저장 후 ["1", "2"]
        #
        # 이러한 경우 ID가 변형되었으므로 오류입니다.
        for col in ["run_id", "agent", "status"]:
            if (
                clean[col].astype("string").tolist()
                != back[col].astype("string").tolist()
            ):
                raise ValueError(
                    "CSV 문자열 열이 달라졌습니다: " + col
                )

        # ----------------------------------------------------
        # 6-2. 숫자 열 비교
        # ----------------------------------------------------
        #
        # latency_ms와 tokens가 저장 전후 동일한지 확인합니다.
        #
        # np.testing.assert_allclose()
        #   → 두 숫자 배열이 허용 오차 범위 내에서
        #     동일한지 검사하는 함수입니다.
        #
        # to_numpy(float)
        #   → Pandas 열을 NumPy 실수 배열로 변환합니다.
        #
        # equal_nan=True
        #   → 같은 위치에 NaN이 있으면 동일하게 판단합니다.
        #
        # 일반적으로 NaN == NaN은 False이므로
        # 결측값 비교에는 별도 옵션이 필요합니다.
        for col in ["latency_ms", "tokens"]:
            np.testing.assert_allclose(
                clean[col].to_numpy(float),
                back[col].to_numpy(float),
                equal_nan=True
            )

        # ----------------------------------------------------
        # 6-3. 날짜 열 비교
        # ----------------------------------------------------
        #
        # 날짜는 저장 전과 저장 후의 자료형이
        # 다를 수 있으므로 같은 날짜형으로 변환합니다.
        #
        # pd.to_datetime()
        #   → 문자열 등을 날짜형으로 변환합니다.
        #
        # format="%Y-%m-%d"
        #   → 연도-월-일 형식으로 해석합니다.
        #
        # errors="coerce"
        #   → 해석할 수 없는 날짜는 NaT로 처리합니다.
        #
        # NaT는 날짜의 결측값을 의미합니다.
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

        # 두 날짜 Series가 동일한지 검사합니다.
        #
        # pd.testing.assert_series_equal()
        #   → 두 Pandas Series의 값을 비교합니다.
        #
        # reset_index(drop=True)
        #   → 행 인덱스를 0부터 다시 설정합니다.
        #
        # check_names=False
        #   → Series 이름의 차이는 검사하지 않습니다.
        #
        # 날짜 값과 NaT의 위치가 유지되었는지 확인합니다.
        pd.testing.assert_series_equal(
            before_day.reset_index(drop=True),
            after_day.reset_index(drop=True),
            check_names=False
        )

        # ----------------------------------------------------
        # 6-4. JSON 요약 데이터 비교
        # ----------------------------------------------------
        #
        # 저장 전 summary와 JSON에서 복원한 restored를
        # DataFrame 단위로 비교합니다.
        #
        # pd.testing.assert_frame_equal()
        #   → 두 DataFrame의 구조와 값을 비교합니다.
        #
        # check_dtype=False
        #   → int64와 float64 같은 자료형 차이를 허용합니다.
        #   → 단, 값 자체가 같은지는 계속 검사합니다.
        #
        # check_exact=False
        #   → 부동소수점 수치의 미세한 차이를 허용합니다.
        #
        # rtol=1e-8
        #   → 상대 오차 허용 범위
        #
        # atol=1e-8
        #   → 절대 오차 허용 범위
        #
        # JSON 저장 과정에서 소수점 표현이 약간 달라져도
        # 실질적으로 같은 값이면 통과하도록 설정합니다.
        pd.testing.assert_frame_equal(
            summary.reset_index(drop=True),
            restored.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-8,
            atol=1e-8
        )

    # [7단계] 비교 과정에서 발생한 AssertionError나
    # TypeError를 사용자에게 이해하기 쉬운 오류로 변환합니다.
    #
    # AssertionError
    #   → assert_allclose 등 검증 함수에서
    #     비교 결과가 다를 때 발생할 수 있는 오류
    #
    # TypeError
    #   → 자료형이 적절하지 않을 때 발생하는 오류
    #
    # from exc
    #   → 원래 오류의 원인 정보를 유지합니다.
    #
    # 참고:
    # try 내부에서 직접 발생시킨 ValueError는
    # 이 except에서 잡지 않고 그대로 전달됩니다.
    except (AssertionError, TypeError) as exc:
        raise ValueError(
            "저장 전후 수치·날짜·요약이 다릅니다."
        ) from exc

    # [8단계] 모든 검증을 통과하면 결과를 반환합니다.
    #
    # csv_rows
    #   → CSV에서 복원한 실행 기록 수
    #
    # json_rows
    #   → JSON에서 복원한 요약 기록 수
    #
    # latency_missing
    #   → 실행 시간에 NaN이 존재하는 기록 수
    #
    # verified
    #   → 모든 검증을 통과했는지 여부
    #
    # True는 지정된 검증 항목을 통과했다는 의미입니다.
    return {
        "csv_rows": len(back),
        "json_rows": len(records),
        "latency_missing": int(back["latency_ms"].isna().sum()),
        "verified": True
    }


# ============================================================
# 6. 파일 저장부터 검증까지 전체 과정 실행
# ============================================================

def export_tables(clean, summary, out_dir, data_dir):
    """CSV/JSON 저장부터 검증까지 수행하는 통합 함수입니다.

    [전체 처리 과정]

    1. 입력 데이터 보호
            ↓
    2. 임시 출력 폴더 준비
            ↓
    3. CSV/JSON 파일 저장
            ↓
    4. 저장한 파일 다시 읽기
            ↓
    5. 저장 전후 데이터 비교
            ↓
    6. 검증 성공 시 최종 결과 반영

    [임시 저장을 사용하는 이유]

    기존 결과 파일을 바로 덮어쓰면
    저장 도중 오류가 발생했을 때
    정상적인 기존 파일까지 손상될 수 있습니다.

    따라서 임시 위치에 파일을 먼저 생성하고,
    검증이 성공한 경우에만 최종 결과로 반영합니다.

    이러한 방식은 안전한 파일 저장을 위한
    기본적인 처리 방법입니다.
    """

    # [1단계] 출력 폴더가 입력 데이터 영역을
    # 침범하지 않는지 검사합니다.
    #
    # protect_inputs()
    #   → 실습 원본 데이터가 잘못된 저장 경로 때문에
    #     덮어써지는 문제를 방지하기 위한 제공 함수입니다.
    protect_inputs(out_dir, data_dir)

    # [2단계] staged_output()을 이용해 임시 출력 영역을 준비합니다.
    #
    # with 문
    #   → 특정 작업 영역을 시작하고 종료할 때
    #     필요한 정리 작업을 자동으로 수행하도록 돕습니다.
    #
    # as stage
    #   → 임시 출력 경로를 stage 변수로 사용합니다.
    #
    # 아래 단계들은 최종 폴더가 아니라
    # 임시 출력 위치에서 먼저 수행됩니다.
    with staged_output(out_dir) as stage:

        # [3단계] CSV와 JSON 파일을 임시 폴더에 저장합니다.
        #
        # clean   → clean_runs.csv
        # summary → summary.json
        save_tables(clean, summary, stage)

        # [4단계] 방금 저장한 두 파일을 다시 읽습니다.
        #
        # back    → CSV에서 읽은 DataFrame
        # records → JSON에서 읽은 데이터
        back, records = read_back(stage)

        # [5단계] 저장 전후 데이터가 동일한지 검사합니다.
        #
        # 검증에 실패하면 ValueError 등이 발생하여
        # 정상적인 결과 반영 단계로 진행하지 않습니다.
        result = verify_roundtrip(
            clean,
            summary,
            back,
            records
        )

    # [6단계] with 블록이 정상적으로 종료되면
    # staged_output의 제공된 처리 흐름에 따라
    # 검증된 결과가 최종 출력 폴더에 반영됩니다.
    #
    # 검증 결과 딕셔너리를 반환합니다.
    return result


# ============================================================
# 7. 프로그램 실행부
# ============================================================
#
# 앞에서 작성한 함수들을 사용하여
# CSV/JSON 저장 및 검증 과정을 실제로 실행합니다.
#
# 이번 실습에서는 다음 함수를 중심으로 진행합니다.
#
# save_tables()
#     → CSV와 JSON 파일 저장
#
# read_back()
#     → 저장된 파일 다시 읽기
#
# verify_roundtrip()
#     → 저장 전후 데이터 검증
#
# export_tables()
#     → 위 과정을 하나로 묶어 실행
#
# ============================================================


# 파일 경로를 관리하기 위한 라이브러리입니다.
from pathlib import Path

# JSON 파일 저장과 JSON 호환 데이터 변환을 위한 제공 함수입니다.
from support import write_json, json_safe

# 파이썬 데이터를 JSON 문자열로 변환하는 표준 라이브러리입니다.
import json

# 현재 실행 중인 파이썬 파일의 폴더 경로를 가져옵니다.
#
# __file__
#   → 현재 파이썬 파일의 경로
#
# resolve()
#   → 절대 경로로 변환합니다.
#
# parent
#   → 해당 파일이 들어 있는 폴더를 반환합니다.
BASE_DIR = Path(__file__).resolve().parent

# 실습에서 사용할 정제 데이터와 집계 데이터를 불러옵니다.
from support import load_inputs

# 실습 03에서 작성한 메타데이터 결합 함수를 가져옵니다.
# 현재 main()에서는 직접 호출하지 않습니다.
from analyze import join_meta


def main():
    """실습 데이터를 불러와 CSV/JSON으로 저장하고 검증합니다."""

    # [1단계] 결과 저장용 out 폴더를 생성합니다.
    #
    # BASE_DIR / "out"
    #   → 현재 실습 폴더 아래의 out 폴더
    #
    # exist_ok=True
    #   → 이미 폴더가 존재해도 오류를 발생시키지 않습니다.
    (BASE_DIR / "out").mkdir(exist_ok=True)

    # [2단계] 실습 입력 데이터를 불러옵니다.
    #
    # load_inputs(BASE_DIR)는 두 개의 데이터를 반환합니다.
    #
    # clean
    #   → 정제된 개별 실행 기록 DataFrame
    #
    # summary
    #   → 에이전트별 집계 결과 DataFrame
    #
    # 실습 02와 실습 03에서 다룬 데이터입니다.
    clean, summary = load_inputs(BASE_DIR)

    # [3단계] CSV/JSON 저장 및 검증을 실행합니다.
    #
    # export_tables()에 전달하는 인자는 다음과 같습니다.
    #
    # clean
    #   → CSV로 저장할 데이터
    #
    # summary
    #   → JSON으로 저장할 데이터
    #
    # BASE_DIR / "out"
    #   → 결과 파일 저장 위치
    #
    # BASE_DIR / "data"
    #   → 보호해야 할 입력 데이터 위치
    #
    # 모든 검증이 성공하면 결과 딕셔너리를 반환합니다.
    result = export_tables(
        clean,
        summary,
        BASE_DIR / "out",
        BASE_DIR / "data"
    )

    # [4단계] 최종 검증 결과를 JSON 파일로 저장합니다.
    #
    # 저장 위치:
    # 현재 실습 폴더 / out / result.json
    #
    # 앞에서 저장한 summary.json과는 다른 파일입니다.
    #
    # summary.json
    #   → 에이전트별 집계 데이터
    #
    # result.json
    #   → 저장 및 복원 검증 결과
    write_json(
        BASE_DIR / "out/result.json",
        result
    )

    # [5단계] 최종 검증 결과를 콘솔에 출력합니다.
    #
    # json_safe()
    #   → JSON으로 표현 가능한 값으로 변환합니다.
    #
    # json.dumps()
    #   → 파이썬 데이터를 JSON 문자열로 변환합니다.
    #
    # ensure_ascii=False
    #   → 한글을 유니코드 이스케이프 없이 표시합니다.
    #
    # indent=2
    #   → 들여쓰기를 2칸 적용해 보기 좋게 출력합니다.
    print(
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=2
        )
    )


# ============================================================
# 8. 프로그램 시작
# ============================================================

# __name__ == "__main__"
#   → 현재 파이썬 파일을 직접 실행했는지 확인합니다.
#
# 직접 실행한 경우:
#   python exercise05.py
#   → main() 함수가 실행됩니다.
#
# 다른 파일에서 import한 경우:
#   from exercise05 import save_tables
#   → main()은 자동으로 실행되지 않습니다.
#
# 이러한 구조는 함수를 다른 프로그램에서도
# 재사용할 수 있도록 하기 위한 일반적인 파이썬 작성 방식입니다.
if __name__ == "__main__":
    main()
