"""실습 07: 전처리 구성·실행·결과·보고서의 위치를 나눈 작은 프로젝트입니다."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import build_pipeline

# 제공: 경로 처리는 학습 과제가 아닙니다. 다른 작업 폴더에서 실행해도 이 파일을 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
data = pd.read_csv(HERE / "data" / "requests.csv")

# 제공: 원본 100행의 순서와 두 seed를 유지하여 전날과 같은 60/20/20을 만듭니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# preprocessing.py는 처리 순서만 구성합니다. 실제 학습은 train으로만 여기에서 수행합니다.
pipeline = build_pipeline()
pipeline.fit(train[features], train["failed"])
# TODO H07-1: 학습된 pipeline으로 valid의 입력 4열을 예측하세요.
prediction = [1] * len(valid)

# 제공: 각 예측을 원래 요청 ID와 연결합니다. 원본 requests.csv는 변경하지 않습니다.
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction
result.to_csv(OUTPUTS / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 제공: 세 묶음의 ID와 묶음 안의 순서(position)를 남깁니다. test의 특징·정답은 모델에 전달하지 않습니다.
manifest_parts = []
for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
    manifest_parts.append(pd.DataFrame({
        "request_id": part["request_id"].tolist(),
        "split": split_name,
        "position": range(len(part)),
    }))
manifest = pd.concat(manifest_parts, ignore_index=True)
manifest.to_csv(OUTPUTS / "split_manifest.csv", index=False, encoding="utf-8-sig")

# 제공: 학습된 열 이름과 전처리+모델을 저장합니다. 다른 사람이 보낸 joblib 파일은 열지 않습니다.
feature_names = pipeline["pre"].get_feature_names_out().tolist()
(OUTPUTS / "feature_names.json").write_text(
    json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
joblib.dump(pipeline, OUTPUTS / "pipeline.joblib")

accuracy = (result["failed"] == result["predicted"]).mean()
found = ((result["failed"] == 1) & (result["predicted"] == 1)).sum()
print("분할:", len(train), len(valid), len(test))
print("변환 특성 수:", len(feature_names))
print("valid 정확도:", round(accuracy, 2))
print("실패 발견:", int(found), "/", int((result["failed"] == 1).sum()))
print("저장: outputs/validation_predictions.csv")
print("저장: outputs/split_manifest.csv, outputs/feature_names.json")
print("저장: outputs/pipeline.joblib")
print("08교시: restore.py 실행 후 reports/model_review.md를 작성합니다.")
