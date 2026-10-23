"""실습 08: 먼저 같은 폴더의 main.py를 완성하고 실행한 뒤, 이 파일을 별도로 실행합니다."""
from pathlib import Path
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
data = pd.read_csv(HERE / "data" / "requests.csv")

# 제공: 저장한 분할표를 읽어 같은 ID를 같은 순서로 꺼냅니다. 새로 분할하거나 학습하지 않습니다.
manifest = pd.read_csv(OUTPUTS / "split_manifest.csv")
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
valid = data.set_index("request_id").loc[valid_ids].reset_index()
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# 직접 실행한 main.py가 같은 환경에서 만든 파일만 복원합니다.
restored = joblib.load(OUTPUTS / "pipeline.joblib")
saved = pd.read_csv(OUTPUTS / "validation_predictions.csv")
# TODO H08-1: 복원한 restored로 valid 입력을 예측하세요. fit은 호출하지 마세요.
prediction = saved["predicted"].to_numpy() + 1

# 제공: 예측 숫자뿐 아니라 요청 ID의 순서도 함께 대조합니다.
same_ids = valid["request_id"].tolist() == saved["request_id"].tolist()
same = (prediction == saved["predicted"].to_numpy()).all()
print("요청 ID 순서 동일:", same_ids)
print("저장 전후 예측 동일:", bool(same))
print("valid 예측 건수:", len(prediction))
print("복원한 모델에 fit을 다시 호출하지 않았습니다.")
