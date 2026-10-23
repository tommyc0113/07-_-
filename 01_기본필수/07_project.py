"""07~08 기본필수: 이 파일 하나를 완성하고 같은 파일로 재실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.

# 제공: 앞에서 배운 입력 선택과 전처리 구조를 그대로 연결합니다.
features = ["input_chars", "task_type"]
X_train = train[features]
y_train = train["failed"]
X_valid = valid[features]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# Dummy는 가장 많은 정답만 반복하는 비교 출발점입니다. 실제 AI 성능이 아닙니다.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
# TODO B07-1: pipeline.fit(X_train, y_train)으로 train에서만 학습하세요. 
pipeline.fit(X_train, y_train)
# TODO B07-2: pipeline.predict(X_valid)로 검증 입력을 예측하세요. 
prediction = pipeline.predict(X_valid)

# 제공: 예측을 원래 ID·정답과 나란히 놓고, 이 파일 옆에 저장합니다.
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction
result.to_csv(HERE / "valid_predictions.csv", index=False, encoding="utf-8-sig")
accuracy = (result["failed"] == result["predicted"]).mean()
print("분할:", len(train), len(valid), len(test))
print(result.to_string(index=False))
print("valid 정확도:", round(accuracy, 2))
print("저장: valid_predictions.csv / test는 사용하지 않았습니다.")
