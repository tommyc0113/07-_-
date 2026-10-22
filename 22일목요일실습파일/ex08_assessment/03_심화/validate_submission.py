"""실습 08: 제출 결과의 수치와 파일을 확인합니다. 사람의 평가를 대신 채점하지 않습니다."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
from PIL import Image


def validate_tables(clean, summary):
    """정제표와 요약표의 관계를 검사하고 오류 문장 목록을 반환합니다."""
    # TODO V_TABLES: 필수 열·ID·역할 그룹을 검사하고 직접 다시 계산한 전체/측정/평균/중앙값/성공률과 요약을 비교하세요. 오류 문자열 목록을 반환하세요.
    raise NotImplementedError("TODO V_TABLES")


def validate_join(clean, joined):
    """결합 행 수·ID·매칭·비용을 검사합니다."""
    # TODO V_JOIN: 결합 행 수·run_id 순서·_merge==both·토큰×가상단가를 검사하세요. 오류 문자열 목록을 반환하세요.
    raise NotImplementedError("TODO V_JOIN")


def verify_png(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "PNG" and min(image.size) >= 200
    except (OSError, ValueError):
        return False


def inspect_project(project, level="basic"):
    """실제 제출 폴더를 읽기만 합니다. Git 생성·커밋·코드 실행은 하지 않습니다."""
    project = Path(project).resolve()
    required = ["main.py", "data/agent_runs.csv", "out/clean_runs.csv", "out/summary.json",
                "out/charts/latency_by_agent.png", "out/charts/latency_hist.png",
                "reports/eda_report.md", "README.md", "requirements.txt", ".gitignore"]
    if level != "basic":
        required += ["data/agents_meta.json", "out/joined_runs.csv", "out/pivot.csv", "out/charts/tokens_latency.png"]
    errors = ["파일 누락: " + name for name in required if not (project / name).is_file()]
    if errors:
        return {"automatic_errors": errors, "manual_review": ["누락 자료 보완 후 재검사"]}
    clean = pd.read_csv(project / "out/clean_runs.csv", dtype={"run_id": "string"}, encoding="utf-8-sig")
    def reject(value):
        raise ValueError("비표준 JSON 수치: " + value)
    records = json.loads((project / "out/summary.json").read_text(encoding="utf-8"), parse_constant=reject)
    errors += validate_tables(clean, pd.DataFrame(records))
    if level != "basic":
        joined = pd.read_csv(project / "out/joined_runs.csv", dtype={"run_id": "string"}, encoding="utf-8-sig")
        errors += validate_join(clean, joined)
    for path in (project / "out/charts").glob("*.png"):
        if not verify_png(path):
            errors.append("잘못된 이미지: " + path.name)
    report = (project / "reports/eda_report.md").read_text(encoding="utf-8")
    if "TODO" in report or "[작성]" in report:
        errors.append("분석 보고서에 미작성 표시가 남음")
    for path in [project/"main.py", *sorted((project/"eda").glob("*.py"))]:
        if "raise NotImplementedError" in path.read_text(encoding="utf-8"):
            errors.append("미구현 코드: " + str(path.relative_to(project)))
    if not (project / "tests").is_dir():
        errors.append("tests 폴더 또는 테스트 기록이 없음")
    return {"automatic_errors": errors,
            "manual_review": ["원본 보존·정제 정책 설명", "보고서의 근거 3개·한계·개선안", "그림 축·단위·표본 수의 적절성",
                              "코드리뷰와 개인 기여", "실제 Git 커밋 2개 이상·변경 의도·최종 상태"],
            "notice": "자동 검사 통과는 100점 또는 최종 제출 승인과 동일하지 않습니다."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--level", choices=["basic", "applied", "advanced"], default="basic")
    args = parser.parse_args()
    try:
        result = inspect_project(args.project, args.level)
    except (OSError, ValueError) as exc:
        print("검사 실패:", exc)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result["automatic_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
