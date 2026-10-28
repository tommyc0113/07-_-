"""실습 04 응용: 행별 특성 두 개만 추가하고 같은 Logistic·5-fold로 비교합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()

# request_id를 기준으로 학습 데이터와 검증 데이터를 분리하고 복사합니다.
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 제공: 모든 평균·중앙값·범주 목록은 Pipeline.fit의 학습 부분에서만 결정됩니다.
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.base import clone
# TODO A04: train·valid에 log_input_chars와 chars_per_slot을 추가하세요. 자기 행만 사용하고 NaN은 다음 Pipeline에서 처리합니다.

for table in (X_train, X_valid):

    # 파생 특성 1: 글자 수를 로그 변환합니다.
    # np.log1p(x)는 log(1 + x)를 계산합니다.
    # 각 행의 input_chars 값만 사용합니다.
    table["log_input_chars"] = np.log1p(table["input_chars"])
    
    # 파생 특성 2: 글자 수를 허용 도구 수 + 1로 나눕니다.
    # 분모에 1을 더하므로 allowed_tools가 0이어도 계산할 수 있습니다.
    table["chars_per_slot"] = table["input_chars"] / (1.0 + table["allowed_tools"])

# TODO 끝

# 정답 클래스의 비율을 유지하면서 학습 데이터를 5개 그룹으로 나눕니다.
# shuffle=True: 데이터를 무작위로 섞습니다.
# random_state=42: 실행할 때마다 동일한 분할을 재현합니다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

# 비교할 모델과 모델별 수치형 입력 특성을 지정합니다.
choices = {"Logistic": NUM, "Logistic_FE": NUM + ["log_input_chars", "chars_per_slot"]}

rows = []
for name, numeric_columns in choices.items():
    current_pre = clone(pre)
    # 제공: 파생 열이 있을 때 수치 경로에만 추가합니다. 범주 경로는 바꾸지 않습니다.
    # 수치형 전처리 경로에서 사용할 특성 목록을 지정합니다.
    # 범주형 특성의 전처리 설정은 그대로 유지합니다.
    current_pre.transformers[0] = ("num", current_pre.transformers[0][1], numeric_columns)

    # 전처리와 Logistic Regression을 하나의 Pipeline으로 연결합니다.
    # C=1: 로지스틱 회귀의 정규화 설정
    # max_iter=1000: 최대 학습 반복 횟수
    model = Pipeline([("pre", current_pre), ("model", LogisticRegression(C=1, max_iter=1000))])

    # 동일한 5개 폴드를 사용해 교차검증 F1 점수를 계산합니다.
    # 각 폴드에서 4개 그룹으로 학습하고 나머지 1개 그룹으로 평가합니다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=1)
    
    # 전체 학습 데이터로 모델을 다시 학습합니다.
    model.fit(X_train, y_train)


   # 모델 이름, 교차검증 평균·표준편차, 검증 데이터 F1 점수를 저장합니다.
    rows.append({
        "model": name,
        "CV_F1": scores.mean(),  # 5개 폴드 F1 점수의 평균
        "CV_std": scores.std(),  # 폴드별 F1 점수의 표준편차
        "valid_F1": f1_score(
            y_valid,
            model.predict(X_valid),
            zero_division=0
        )  # 검증 데이터의 F1 점수
    })
result = pd.DataFrame(rows)
result.to_csv(OUT / "feature_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(4).to_string(index=False))
print("CV는 선택용, valid는 진단용입니다. test는 아직 보지 않습니다.")
