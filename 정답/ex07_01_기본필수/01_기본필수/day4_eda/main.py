
"""실습 07: 전체 데이터 분석 파이프라인 통합

원본 데이터 읽기 → 데이터 정제 → 통계 집계 → 그래프/파일 생성 → 검증

실습 02~05에서 작성한 함수들을 연결하여
전체 분석 과정을 하나의 프로그램으로 실행합니다.

- basic: 정제, 집계, CSV/JSON 저장, 막대그래프, 히스토그램
- applied: basic 기능 + 메타데이터 결합, 피벗, 산점도
- advanced: applied 기능 + 날짜별 분석, 선그래프, 엄격한 토큰 검사
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

# 실습 공통 기능: 원본 읽기, JSON 저장, 입력 보호, 임시 저장 등
from support import (
    read_raw, write_json, json_safe,
    protect_inputs, staged_output, DEFAULT_AGENTS
)

# 실습 02: 데이터 정제 및 품질 검사
from eda.clean import clean_runs, quality_report, DataContractError

# 실습 03: 통계 집계, 피벗 테이블, 메타데이터 결합, 비용 분석
from eda.analyze import (
    aggregate_runs, make_pivot, join_meta,
    daily_summary, cost_audit
)

# 실습 04: 막대그래프, 히스토그램, 산점도, 선그래프
from eda.plot import plot_bar, plot_hist, plot_scatter, plot_daily

# 실습 05: CSV/JSON 저장, 재읽기, 왕복 검증 및 메타데이터 로드
from eda.io_utils import save_tables, read_back, verify_roundtrip, read_meta


# 현재 파이썬 파일이 위치한 폴더의 절대 경로입니다.
BASE_DIR = Path(__file__).resolve().parent

# 실습 난이도 설정
# basic: 기본필수 / applied: 응용 / advanced: 심화
LEVEL = "basic"


def run_pipeline(input_path, out_dir, meta_path=None):
    """원본 데이터부터 최종 분석 결과까지 전체 파이프라인을 실행합니다.

    input_path: 분석할 원본 CSV 파일 경로
    out_dir: 결과를 저장할 출력 폴더
    meta_path: 에이전트 메타데이터 경로(선택)

    저장 과정에서 오류가 발생하면 최종 결과 반영을 중단합니다.
    원본 파일과 기존 출력 결과를 보호하는 구조입니다.
    """

    # [1단계] 입력 및 출력 경로를 절대 경로로 변환합니다.
    # resolve()를 이용해 상대 경로로 인한 경로 혼동을 줄입니다.
    input_path, out_dir = (
        Path(input_path).resolve(),
        Path(out_dir).resolve()
    )

    # 출력 위치가 원본 데이터 폴더나 입력 파일을 침범하지 않도록 검사합니다.
    # 분석 결과 저장 과정에서 원본 파일이 덮어써지는 문제를 방지합니다.
    protect_inputs(out_dir, BASE_DIR / "data", input_path, meta_path)

    # [2단계] 원본 CSV 파일의 SHA-256 해시값을 계산합니다.
    # read_bytes(): 파일 내용을 바이트 단위로 읽습니다.
    # hexdigest(): 계산된 해시를 16진수 문자열로 변환합니다.
    #
    # 분석 종료 시 해시를 다시 계산하여 원본 변경 여부를 확인합니다.
    original_hash = hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()

    # [3단계] 메타데이터 경로가 지정되었다면 JSON을 읽고 검증합니다.
    # 경로가 없다면 None으로 설정합니다.
    meta = read_meta(meta_path) if meta_path is not None else None

    # 응용/심화 수준에서는 메타데이터가 필수입니다.
    # 메타데이터에는 에이전트별 토큰 단가 등의 정보가 포함됩니다.
    if LEVEL != "basic" and meta is None:
        raise ValueError("응용/심화 프로젝트에는 역할 메타데이터 경로가 필요합니다.")

    # [4단계] 허용할 에이전트 목록을 설정합니다.
    # 메타데이터가 있으면 해당 agent 목록을 사용하고,
    # 없으면 DEFAULT_AGENTS의 기본 목록을 사용합니다.
    allowed = tuple(meta["agent"]) if meta is not None else DEFAULT_AGENTS

    # ========================================================
    # [5단계] 핵심 데이터 처리 파이프라인 (PIPE_CORE)
    # ========================================================

    # 원본 읽기 → 정제 → 집계 → 품질 검사 순서로 실행합니다.
    # 각 함수의 출력값이 다음 분석 단계의 입력값으로 사용됩니다.

    # 원본 CSV를 DataFrame으로 읽습니다.
    raw = read_raw(input_path)

    # 원본 데이터를 정제합니다.
    # allowed_agents: 허용된 에이전트 목록
    # strict_tokens=True: 분수 형태의 토큰 수를 NaN으로 처리합니다.
    # 심화 수준에서만 엄격한 토큰 검사를 활성화합니다.
    clean = clean_runs(
        raw,
        allowed_agents=allowed,
        strict_tokens=(LEVEL == "advanced")
    )

    # 정제된 데이터를 에이전트별로 집계합니다.
    # 전체 실행 수, 유효 측정 수, 평균, 중앙값, 성공률 등을 계산합니다.
    summary = aggregate_runs(clean)

    # 정제 전후 데이터를 기반으로 품질 검사 결과를 생성합니다.
    # 데이터 정제 결과와 결측 현황 등을 확인하는 데 사용합니다.
    quality = quality_report(raw, clean)

    # ========================================================
    # [6단계] 정제 및 집계 결과의 일관성 검증
    # ========================================================

    # 첫 번째 불변식(Invariant) 검사:
    #
    # ① 정제된 run_id가 모두 고유해야 합니다.
    # ② 에이전트별 실행 횟수의 합계가 정제 데이터의 행 수와 같아야 합니다.
    #
    # is_unique: 중복값이 없으면 True
    # len(clean): 정제 데이터의 전체 행 개수
    if not clean["run_id"].is_unique or int(summary["runs"].sum()) != len(clean):
        raise ValueError("정제/집계의 행 수 불변식이 깨졌습니다.")

    # 두 번째 불변식 검사:
    #
    # 에이전트별 유효 측정 수의 합계와
    # clean에서 NaN을 제외한 latency_ms 개수가 같아야 합니다.
    #
    # count()는 NaN을 제외하고 개수를 계산합니다.
    if int(summary["measured"].sum()) != int(clean["latency_ms"].count()):
        raise ValueError("유효 측정 개수가 집계와 일치하지 않습니다.")

    # [7단계] 메타데이터가 있다면 실행 기록과 결합합니다.
    #
    # join_meta()는 agent를 기준으로 다대일 결합을 수행하고
    # 토큰 사용량과 단가를 이용해 cost_unit을 계산합니다.
    # 메타데이터가 없다면 joined는 None입니다.
    joined = join_meta(clean, meta) if meta is not None else None

    # ========================================================
    # [8단계] 임시 저장 및 결과 생성
    # ========================================================

    # staged_output()을 이용하여 임시 폴더에 먼저 결과를 생성합니다.
    #
    # 저장 → 다시 읽기 → 검증 → 그래프 생성 과정을 모두 마친 후
    # with 블록이 정상 종료되면 최종 출력 폴더에 결과를 반영합니다.
    #
    # 중간에 오류가 발생하면 정상적인 결과 반영을 진행하지 않습니다.
    with staged_output(out_dir) as stage:

        # ----------------------------------------------------
        # [8-1] CSV/JSON 파일 저장 및 왕복 검증 (PIPE_FILES)
        # ----------------------------------------------------

        # 정제 데이터는 CSV, 집계 결과는 JSON으로 임시 저장합니다.
        #
        # 출력 파일:
        # clean_runs.csv
        # summary.json
        save_tables(clean, summary, stage)

        # 저장한 두 파일을 다시 읽습니다.
        # back: CSV에서 복원한 DataFrame
        # records: JSON에서 복원한 리스트 데이터
        back, records = read_back(stage)

        # 저장 전후 데이터를 비교합니다.
        #
        # 행 수, ID, 문자열, 숫자, 결측값, 날짜, 요약 데이터 등을 검사합니다.
        # 검증에 성공하면 결과 딕셔너리를 반환합니다.
        roundtrip = verify_roundtrip(clean, summary, back, records)

        # ----------------------------------------------------
        # [8-2] 기본 그래프 생성 (PIPE_CHARTS)
        # ----------------------------------------------------

        # 막대그래프와 히스토그램을 생성합니다.
        #
        # plot_bar(): 에이전트별 평균 실행 시간 비교
        # plot_hist(): 실행 시간의 구간별 분포
        #
        # 생성 이미지는 stage/charts 폴더에 저장합니다.
        # 반환된 수치 정보는 chart_audit 딕셔너리에 보관합니다.
        chart_audit = {
            "bar": plot_bar(summary, stage / "charts"),
            "hist": plot_hist(clean, stage / "charts")
        }

        # ----------------------------------------------------
        # [8-3] 응용/심화 추가 분석 (PIPE_EXTRA)
        # ----------------------------------------------------

        # basic이 아닌 경우 산점도와 피벗 테이블을 추가 생성합니다.
        # applied와 advanced에서 실행됩니다.
        if LEVEL != "basic":

            # 토큰 수와 실행 시간의 관계를 나타내는 산점도를 생성합니다.
            # 결과는 chart_audit의 scatter 항목에 추가합니다.
            chart_audit["scatter"] = plot_scatter(
                clean, stage / "charts"
            )

            # 에이전트×상태별 평균 실행 시간의 피벗 테이블을 CSV로 저장합니다.
            # 출력 파일: pivot.csv
            make_pivot(clean).to_csv(
                stage / "pivot.csv",
                encoding="utf-8-sig"
            )

            # 메타데이터가 결합된 실행 기록을 CSV로 저장합니다.
            # 출력 파일: joined_runs.csv
            # index=False: DataFrame의 인덱스 열을 제외합니다.
            joined.to_csv(
                stage / "joined_runs.csv",
                index=False,
                encoding="utf-8-sig",
                date_format="%Y-%m-%d"
            )

        # ----------------------------------------------------
        # [8-4] 심화 수준 날짜별 분석
        # ----------------------------------------------------

        # advanced에서는 날짜별 통계와 선그래프를 추가합니다.
        if LEVEL == "advanced":

            # 날짜별 실행 횟수, 유효 측정 수, 평균 실행 시간을 저장합니다.
            # 날짜가 없는 기록도 집계에 포함됩니다.
            # 출력 파일: daily.csv
            daily_summary(clean).to_csv(
                stage / "daily.csv",
                encoding="utf-8-sig",
                date_format="%Y-%m-%d"
            )

            # 날짜별 평균 실행 시간의 변화를 선그래프로 생성합니다.
            # 출력 파일: charts/daily_latency.png
            chart_audit["daily"] = plot_daily(
                clean, stage / "charts"
            )

        # ====================================================
        # [9단계] 전체 실행 시간의 통계 계산
        # ====================================================

        # latency_ms에서 NaN을 제거하고 NumPy 배열로 변환합니다.
        # 평균, 중앙값, 백분위수는 유효 측정값만 사용합니다.
        values = clean["latency_ms"].dropna().to_numpy()

        # 주요 분석 지표를 facts 딕셔너리에 정리합니다.
        facts = {

            # 정제 데이터의 품질 검사 결과
            "quality": quality,

            # 전체 유효 실행 시간의 평균
            # 유효 측정값이 없으면 NaN으로 반환합니다.
            "overall_mean_ms": (
                float(values.mean()) if len(values) else np.nan
            ),

            # 전체 유효 실행 시간의 중앙값
            # 중앙값은 극단적으로 큰 값의 영향을 평균보다 적게 받습니다.
            "overall_median_ms": (
                float(np.median(values)) if len(values) else np.nan
            ),

            # P90: 실행 시간의 90백분위수
            # np.quantile(..., .9)는 90% 지점의 값을 계산합니다.
            # method="linear"는 선형 보간 방식입니다.
            "p90_ms": (
                float(np.quantile(values, .9, method="linear"))
                if len(values) else np.nan
            ),

            # 히스토그램의 구간별 실행 기록 개수
            "hist_counts": chart_audit["hist"]["counts"],

            # 메타데이터가 있다면 비용 합계와 누락 여부를 계산합니다.
            # 없으면 비용 정보를 None으로 기록합니다.
            "cost": cost_audit(joined) if joined is not None else None,

            # 분석 시작 시 계산한 원본 CSV의 SHA-256 해시값
            "input_sha256": original_hash,

            # 실습 데이터가 합성 데이터임을 나타내는 메타 정보
            "synthetic_data": True
        }

        # ====================================================
        # [10단계] 분석 및 검증 결과 JSON 저장
        # ====================================================

        # 데이터 품질 검사 결과
        write_json(stage / "quality.json", quality)

        # CSV/JSON 왕복 저장 검증 결과
        write_json(stage / "roundtrip_check.json", roundtrip)

        # 그래프에 사용한 수치와 생성 결과 정보
        write_json(stage / "chart_audit.json", chart_audit)

        # 전체 평균, 중앙값, P90, 비용, 원본 해시 등의 핵심 통계
        write_json(stage / "facts.json", facts)

        # ====================================================
        # [11단계] 원본 파일 변경 여부 최종 검사
        # ====================================================

        # 원본 파일을 다시 읽어 SHA-256 해시를 계산합니다.
        # 분석 시작 시 계산한 original_hash와 비교하여
        # 실행 도중 파일 내용이 변경되었는지 확인합니다.
        #
        # 변경되었다면 분석 결과의 일관성을 보장할 수 없으므로
        # 결과 반영을 중단합니다.
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != original_hash:
            raise ValueError("실행 중 원본이 바뀌어 결과 게시를 중단합니다.")

    # ========================================================
    # [12단계] 파이프라인 실행 결과 반환
    # ========================================================

    # 모든 과정이 정상적으로 완료되면 실행 결과를 반환합니다.
    #
    # raw_rows: 원본 데이터 행 수
    # clean_rows: 정제 후 행 수
    # summary_rows: 에이전트별 집계 결과 행 수
    # measured: 실행 시간이 유효한 기록 수
    # charts: 생성한 그래프 개수
    # out: 결과 저장 폴더 이름
    # roundtrip_verified: CSV/JSON 왕복 검증 성공 여부
    return {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "summary_rows": len(summary),
        "measured": int(clean["latency_ms"].count()),
        "charts": (
            2 if LEVEL == "basic"
            else (3 if LEVEL == "applied" else 4)
        ),
        "out": out_dir.name,
        "roundtrip_verified": roundtrip["verified"]
    }


def main():
    """명령행 인자를 처리하고 전체 분석 파이프라인을 실행합니다.

    실행 예시:
    python main.py
    python main.py --input data/agent_runs.csv --out out
    python main.py --help
    """

    # [1단계] 명령행 인자를 처리할 ArgumentParser를 생성합니다.
    # description=__doc__는 파일 상단 설명을 도움말에 사용합니다.
    parser = argparse.ArgumentParser(description=__doc__)

    # --input: 원본 CSV 경로 (기본값: data/agent_runs.csv)
    parser.add_argument(
        "--input",
        default="data/agent_runs.csv"
    )

    # --meta: 메타데이터 JSON 경로
    # basic은 기본값 None, 응용/심화는 기본 메타 파일을 지정합니다.
    parser.add_argument(
        "--meta",
        default=None if LEVEL == "basic" else "data/agents_meta.json"
    )

    # --out: 분석 결과 저장 폴더
    # 원본 데이터가 있는 data 폴더는 출력 경로로 사용할 수 없습니다.
    parser.add_argument(
        "--out",
        default="out",
        help="main.py 기준 결과 폴더. data는 금지"
    )

    # [2단계] 사용자가 전달한 명령행 옵션을 읽습니다.
    # 옵션이 없으면 위에서 설정한 기본값을 사용합니다.
    args = parser.parse_args()

    # [3단계] 입력 경로를 프로젝트 폴더 기준으로 변환하는 보조 함수입니다.
    # 절대 경로는 그대로 사용하고 상대 경로는 BASE_DIR과 결합합니다.
    def here(text):
        path = Path(text)
        return path if path.is_absolute() else BASE_DIR / path

    # [4단계] 전체 파이프라인을 실행합니다.
    # 입력 CSV, 출력 폴더, 선택적 메타데이터 경로를 전달합니다.
    try:
        result = run_pipeline(
            here(args.input),
            here(args.out),
            here(args.meta) if args.meta else None
        )

    # 데이터 계약 위반 시 오류 메시지와 문제 행 정보를 출력합니다.
    # 예: 필수 열 누락, 허용되지 않은 에이전트 등
    except DataContractError as exc:
        print("중단:", str(exc))
        print(exc.issues.to_string(index=False))
        return 1

    # 일반적인 값 오류, 파일 처리 오류, 문자 인코딩 오류를 처리합니다.
    # type(exc).__name__은 발생한 예외의 종류를 반환합니다.
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"중단: {type(exc).__name__}: {exc}")
        return 1

    # [5단계] 정상 완료 시 실행 결과를 JSON 형태로 출력합니다.
    # ensure_ascii=False: 한글 유지 / indent=2: 들여쓰기 2칸
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))

    # 종료 코드 0은 정상 종료를 의미합니다.
    return 0


# 현재 파이썬 파일을 직접 실행한 경우에만 main() 함수를 호출합니다.
# SystemExit는 main()의 반환값을 프로그램 종료 코드로 사용합니다.
# 0: 정상 종료 / 1: 오류 발생
if __name__ == "__main__":
    raise SystemExit(main())
