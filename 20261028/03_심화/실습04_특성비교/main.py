"""실습 04 심화: 행별 파생 규칙까지 Pipeline에 포함해 두 표현을 비교합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
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
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

def add_features(X):
    """원본을 바꾸지 않고 파생 열 두 개를 추가한 새 표를 반환합니다."""
    out = X.copy()
    # TODO H04: log1p와 비율 두 열을 추가하세요. 분모는 실제 도구 사용량이 아니라 1+allowed_tools입니다.
    # 파생 특성 1: 글자 수를 로그 변환합니다.
    # log1p(x)는 log(1 + x)를 계산합니다.
    out["log_input_chars"] = np.log1p(out["input_chars"])

    # 파생 특성 2: 글자 수를 1 + 허용 도구 수로 나눕니다.
    # 분모에 1을 더해 0으로 나누는 문제를 방지합니다.
    out["chars_per_slot"] = (
        out["input_chars"] / (1.0 + out["allowed_tools"])
    )
    # TODO 끝
    return out

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
rows = []
for name, use_derived in [("Logistic", False), ("Logistic_FE", True)]:
    current_pre = clone(pre) # 원본 전처리기를 복사해 모델별로 독립적으로 사용합니다.
    steps = []
    if use_derived:

        # 학습과 예측 과정에서 add_features 함수를 실행합니다.
        # validate=False는 DataFrame 입력을 사용할 때 유용합니다.
        steps.append(("features", FunctionTransformer(add_features, validate=False)))
        
        # 수치형 전처리 대상에 파생 특성 두 개를 추가합니다.
        # 기존 수치형 특성 NUM에 새 특성 이름을 결합합니다.
        # 범주형 특성의 전처리 설정은 변경하지 않습니다.
        current_pre.transformers[0] = ("num", current_pre.transformers[0][1], NUM + ["log_input_chars", "chars_per_slot"])
    steps += [("pre", current_pre), ("model", LogisticRegression(C=1, max_iter=1000))]
    model = Pipeline(steps)
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=1)
    model.fit(X_train, y_train)
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0),
                 "features_after": model[:-1].transform(X_train).shape[1]})
result = pd.DataFrame(rows)
result.to_csv(OUT / "feature_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(4).to_string(index=False))
print("행별 규칙은 통계를 학습하지 않습니다. 대체·표준화·인코딩은 fold 학습 자료에서만 fit합니다.")
