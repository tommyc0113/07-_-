"""실습 05 응용: PPT와 같은 XGBoost 80회차 분류 모델입니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
pre = ColumnTransformer([  # 숫자형과 범주형 특성 각각 전처리
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),  # 숫자형 특성 2개의 결측값을 중앙값으로 대체
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),  # 범주형 데이터의 결측값을 최빈값으로 대체
                      
                      # 학습 시 보지 못한 범주가 나타나도 오류가 발생하지 않도록 설정합니다.
                        # sparse_output=False는 결과를 일반 배열 형태로 반환합니다. 
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, log_loss

estimator = XGBClassifier(
    n_estimators=80,          # 학습에 사용할 트리 개수
    max_depth=2,              # 각 트리의 최대 깊이
    learning_rate=0.05,       # 각 트리의 학습 기여도를 조절하는 학습률
    subsample=0.8,             # 각 트리 학습 시 사용할 학습 데이터의 비율
    colsample_bytree=1.0,      # 각 트리 학습 시 사용할 특성의 비율
    reg_lambda=1.0,            # 과적합을 줄이기 위한 L2 정규화 강도
    objective="binary:logistic",  # 이진 분류 확률을 예측하도록 설정
    eval_metric="logloss",     # 학습 과정에서 사용할 로그 손실 평가 지표
    tree_method="hist",        # 히스토그램 기반 트리 학습 방식
    device="cpu",              # CPU를 사용해 학습
    random_state=42,           # 실행 결과를 재현하기 위한 난수 시드
    n_jobs=1                  # 병렬 작업에 사용할 스레드 수
)

model = Pipeline([("pre", clone(pre)), ("model", estimator)])
# TODO A05: Pipeline을 학습하고 valid를 예측하세요.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# TODO 끝
probability = model.predict_proba(X_valid)[:, 1]  # classes_ = [0, 1] 확인 후 실패1의 열 사용
print("클래스 순서:", model.classes_.tolist())
tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
result = pd.DataFrame([{"train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                       "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn,
                       "valid_logloss": log_loss(y_valid, probability, labels=[0, 1])}])
result.to_csv(OUT / "05_xgboost.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.")
