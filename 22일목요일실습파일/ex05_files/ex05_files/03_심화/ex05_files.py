"""실습 05: CSV/JSON 왕복 저장 및 자료형·결측의 의미를 확인합니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from support import COLUMNS, read_raw, strict_json, staged_output, protect_inputs, require_columns
from analyze import validate_meta


def flatten_meta(payload):
    """{agents:[{agent, pricing:{rate}}]} -> agent/rate_per_1k 표."""
    if not isinstance(payload, dict) or not isinstance(payload.get("agents"), list) or not payload["agents"]:
        raise ValueError("메타 JSON에는 비어 있지 않은 agents 목록이 필요합니다.")
    if not all(isinstance(item, dict) and isinstance(item.get("pricing"), dict)
               and "rate" in item["pricing"] for item in payload["agents"]):
        raise ValueError("각 역할에는 pricing.rate가 필요합니다.")
    # 해설 I_FLATTEN: json_normalize(payload["agents"], sep="_") 후 pricing_rate를 rate_per_1k로 바꾸고 validate_meta 결과를 반환하세요.
    flat = pd.json_normalize(payload["agents"], sep="_")
    meta = flat.rename(columns={"pricing_rate": "rate_per_1k"})
    return validate_meta(meta)


def read_meta(path):
    """문법이 맞아도 내용의 스키마를 따로 검사합니다."""
    return flatten_meta(strict_json(path))


def save_tables(clean, summary, out_dir):
    """정제 CSV와 요약 JSON을 지정한 임시/출력 폴더에 씁니다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 해설 I_CSV: clean_runs.csv를 index=False, encoding="utf-8-sig", date_format="%Y-%m-%d"로 저장하세요.
    # 해설 I_CSV: 날짜는 문자로, NaN은 빈 셀로 저장됩니다. 인덱스 열은 제외합니다.
    clean.to_csv(out_dir / "clean_runs.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    # 해설 I_JSON: summary.json을 orient="records", force_ascii=False, indent=2로 저장하세요. NaN을 문자열로 쓰지 않습니다.
    # 해설 I_JSON: pandas는 결측 수치를 JSON null로 변환합니다.
    summary.to_json(out_dir / "summary.json", orient="records", force_ascii=False, indent=2, double_precision=10)


def read_back(out_dir):
    """CSV ID는 문자열로 명시하고 JSON은 비표준 NaN/Infinity를 거부합니다."""
    out_dir = Path(out_dir)
    # 해설 I_READ: CSV run_id는 string dtype으로, JSON은 strict_json으로 다시 읽어 (back, records)를 반환하세요.
    back = pd.read_csv(out_dir / "clean_runs.csv", encoding="utf-8-sig", dtype={"run_id": "string"})
    records = strict_json(out_dir / "summary.json")
    return back, records


def verify_roundtrip(clean, summary, back, records):
    """저장 전후 행/열/ID/수치/날짜/요약을 비교합니다. 글자 byte 일치는 요구하지 않습니다."""
    # TODO I_VERIFY: 행 수·ID·문자열·숫자/결측·날짜·요약 값을 저장 전후 대조하세요. 다르면 ValueError, 같으면 csv_rows/json_rows/latency_missing/verified 사전을 반환하세요.
    raise NotImplementedError("TODO I_VERIFY")


def export_tables(clean, summary, out_dir, data_dir):
    """제공 흐름: 임시 저장 -> 재읽기 검사 -> 성공 시 기존 결과 교체."""
    protect_inputs(out_dir, data_dir)
    with staged_output(out_dir) as stage:
        save_tables(clean, summary, stage)
        back, records = read_back(stage)
        result = verify_roundtrip(clean, summary, back, records)
    return result


# 실행부는 제공 코드입니다. 핵심 함수의 TODO를 먼저 완성하세요.
from pathlib import Path
from support import write_json, json_safe
import json
BASE_DIR = Path(__file__).resolve().parent
from support import load_inputs
from analyze import join_meta

def main():
    (BASE_DIR/"out").mkdir(exist_ok=True)
    clean,summary=load_inputs(BASE_DIR)
    meta=read_meta(BASE_DIR/"data/agents_meta.json")
    joined=join_meta(clean,meta)
    result=export_tables(clean,summary,BASE_DIR/"out",BASE_DIR/"data")
    small=pd.read_csv(BASE_DIR/"data/fixtures/leading_zero_ids.csv",encoding="utf-8-sig",dtype={"run_id":"string"})
    result["leading_zero_ids"]=small["run_id"].tolist()
    result["joined_rows"]=len(joined)
    joined.to_csv(BASE_DIR/"out/joined_runs.csv",index=False,encoding="utf-8-sig",date_format="%Y-%m-%d")
    write_json(BASE_DIR/"out/result.json",result)
    print(json.dumps(json_safe(result),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
