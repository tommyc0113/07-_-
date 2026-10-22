"""실습 07: 원본 -> 정제 -> 집계 -> 그림/파일 -> 검증을 한 번에 연결합니다."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from support import read_raw, write_json, json_safe, protect_inputs, staged_output, DEFAULT_AGENTS
from eda.clean import clean_runs, quality_report, DataContractError
from eda.analyze import aggregate_runs, make_pivot, join_meta, daily_summary, cost_audit
from eda.plot import plot_bar, plot_hist, plot_scatter, plot_daily
from eda.io_utils import save_tables, read_back, verify_roundtrip, read_meta

BASE_DIR = Path(__file__).resolve().parent
LEVEL = "advanced"


def run_pipeline(input_path, out_dir, meta_path=None):
    """경로 -> 검증된 결과 요약. 입력/기존 결과는 실패 때문에 빈 표로 바뀌지 않습니다."""
    input_path, out_dir = Path(input_path).resolve(), Path(out_dir).resolve()
    protect_inputs(out_dir, BASE_DIR / "data", input_path, meta_path)
    original_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    meta = read_meta(meta_path) if meta_path is not None else None
    if LEVEL != "basic" and meta is None:
        raise ValueError("응용/심화 프로젝트에는 역할 메타데이터 경로가 필요합니다.")
    allowed = tuple(meta["agent"]) if meta is not None else DEFAULT_AGENTS
    # 해설 PIPE_CORE: read_raw -> clean_runs -> aggregate_runs -> quality_report로 연결해 raw/clean/summary/quality를 만드세요.
    # 해설 PIPE_CORE: 함수들의 입력/출력 계약을 맞춰 이어 붙입니다.
    raw = read_raw(input_path)
    clean = clean_runs(raw, allowed_agents=allowed, strict_tokens=(LEVEL == "advanced"))
    summary = aggregate_runs(clean)
    quality = quality_report(raw, clean)
    if not clean["run_id"].is_unique or int(summary["runs"].sum()) != len(clean):
        raise ValueError("정제/집계의 행 수 불변식이 깨졌습니다.")
    if int(summary["measured"].sum()) != int(clean["latency_ms"].count()):
        raise ValueError("유효 측정 개수가 집계와 일치하지 않습니다.")
    joined = join_meta(clean, meta) if meta is not None else None
    # stage 안에서만 결과를 생성합니다. 모든 검사가 끝나야 out이 교체됩니다.
    with staged_output(out_dir) as stage:
        # 해설 PIPE_FILES: stage에 save_tables 후 read_back/verify_roundtrip을 호출해 roundtrip 사전을 만드세요.
        save_tables(clean, summary, stage)
        back, records = read_back(stage)
        roundtrip = verify_roundtrip(clean, summary, back, records)
        # 해설 PIPE_CHARTS: stage/charts에 plot_bar/plot_hist를 호출해 chart_audit={"bar":..., "hist":...}를 만드세요.
        chart_audit = {"bar": plot_bar(summary, stage / "charts"),
                       "hist": plot_hist(clean, stage / "charts")}
        if LEVEL != "basic":
            # TODO PIPE_EXTRA: 산점도 감사 정보를 추가하고 pivot.csv, joined_runs.csv를 stage에 저장하세요.
            raise NotImplementedError("TODO PIPE_EXTRA")
        if LEVEL == "advanced":
            daily_summary(clean).to_csv(stage / "daily.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
            chart_audit["daily"] = plot_daily(clean, stage / "charts")
        values = clean["latency_ms"].dropna().to_numpy()
        facts = {"quality": quality, "overall_mean_ms": float(values.mean()) if len(values) else np.nan,
                 "overall_median_ms": float(np.median(values)) if len(values) else np.nan,
                 "p90_ms": float(np.quantile(values, .9, method="linear")) if len(values) else np.nan,
                 "hist_counts": chart_audit["hist"]["counts"],
                 "cost": cost_audit(joined) if joined is not None else None,
                 "input_sha256": original_hash, "synthetic_data": True}
        write_json(stage / "quality.json", quality)
        write_json(stage / "roundtrip_check.json", roundtrip)
        write_json(stage / "chart_audit.json", chart_audit)
        write_json(stage / "facts.json", facts)
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != original_hash:
            raise ValueError("실행 중 원본이 바뀌어 결과 게시를 중단합니다.")
    return {"raw_rows": len(raw), "clean_rows": len(clean), "summary_rows": len(summary),
            "measured": int(clean["latency_ms"].count()), "charts": 2 if LEVEL=="basic" else (3 if LEVEL=="applied" else 4),
            "out": out_dir.name, "roundtrip_verified": roundtrip["verified"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/agent_runs.csv")
    parser.add_argument("--meta", default=None if LEVEL=="basic" else "data/agents_meta.json")
    parser.add_argument("--out", default="out", help="main.py 기준 결과 폴더. data는 금지")
    args = parser.parse_args()
    def here(text):
        path = Path(text)
        return path if path.is_absolute() else BASE_DIR / path
    try:
        result = run_pipeline(here(args.input), here(args.out), here(args.meta) if args.meta else None)
    except DataContractError as exc:
        print("중단:", str(exc))
        print(exc.issues.to_string(index=False))
        return 1
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"중단: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
