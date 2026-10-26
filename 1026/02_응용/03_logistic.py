"""실습 03 응용: PPT와 같은 100건에서 실패1의 예측을 확인합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
# 숫자형 데이터 전처리 파이프라인, 결측치를 각 숫자 열의 중앙값으로 대체, 평균 0 표준편차 1을 기준으로 숫자 데이터를 표준화
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# 범주형 데이터 전처리 파이프라인, 결측치를 각 범주형 열에서 가장 자주 등장하는 값으로 대체, 범주형 값을 숫자 벡터로 변환하는 원-핫 인코딩
# 처음 보는 범주는 오류 없이 처리하고 결과를 일반 배열로 반환
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

# 열의 종류에 따라 서로 다른 전처리 적용
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 전처리와 로지스틱 회귀 모델을 하나의 파이프라인으로 연결
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, max_iter=1000))])
# TODO A03: 실패 여부로 학습하고 X_valid를 예측하세요.
# 힌트: 실패1의 정답 열은 failed입니다.
y_train = train["failed"]
# 학습한 모델로 검증 데이터(X_valid)의 실패 여부를 예측
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
print("클래스 순서:", model.classes_)
print("앞 4건 실패 확률:", model.predict_proba(X_valid)[:4, 1].round(3))
print("혼동행렬 / 행=실제, 열=예측, 순서=[0,1]")
print(confusion_matrix(valid["failed"], prediction, labels=[0, 1]))
print("valid F1:", round(f1_score(valid["failed"], prediction, zero_division=0), 3))
