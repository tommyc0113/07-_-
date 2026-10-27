"""실습 06 심화: 같은 회귀 KFold에서 음수 RMSE 부호를 되돌립니다."""
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

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from preprocessing import build_model

folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
candidates = [("Tree", DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestRegressor(
                n_estimators=100,      # 결정 트리 100개 사용
                max_depth=4,           # 각 트리의 최대 깊이를 4로 제한
                min_samples_leaf=2,    # 리프 노드마다 최소 2개의 샘플 유지
                max_features=1.0,      # 각 분할에서 전체 특성 사용
                random_state=42,       # 결과 재현을 위한 난수 시드
                n_jobs=1               # 작업자 1개 사용
            )),

            ("XGB", XGBRegressor(
                n_estimators=80,                  # 트리 80개 사용
                max_depth=2,                      # 각 트리의 최대 깊이를 2로 제한
                learning_rate=0.05,               # 학습률 설정
                subsample=0.8,                    # 각 트리 학습에 데이터의 80% 사용
                colsample_bytree=1.0,             # 각 트리에서 전체 특성 사용
                reg_lambda=1.0,                   # L2 정규화 강도
                objective="reg:squarederror",     # 제곱 오차를 최소화하는 회귀 목적 함수
                eval_metric="rmse",               # 평가 지표로 RMSE 사용
                tree_method="hist",               # 히스토그램 기반 트리 학습 방식
                device="cpu",                     # CPU로 학습
                random_state=42,                  # 결과 재현을 위한 난수 시드
                n_jobs=1                          # 작업자 1개 사용
            ))]

rows, detail = [], []
for name, estimator in candidates:
    model = build_model(estimator)
    # TODO C06a: 회귀 정답으로 neg_root_mean_squared_error 점수를 구하세요.
    negative = cross_val_score(
        model,
        X_train,
        train["elapsed_ms"],
        cv=folds,
        scoring="neg_root_mean_squared_error"
    )
    # TODO 끝
    # TODO C06b: 음수 scorer 값을 양수 RMSE로 되돌리세요.
    rmse = -negative
    # TODO 끝
    rows.append({"model": name, "CV_RMSE_ms": rmse.mean(), "CV_std_ms": rmse.std()})
    for number, value in enumerate(rmse, start=1):
        detail.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "RMSE_ms": value})
result = pd.DataFrame(rows)
result.to_csv(OUT / "regression_cv.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(detail).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("CV_RMSE는 fold별 RMSE의 단순 평균입니다. 전체 예측을 합친 RMSE와 다를 수 있습니다.")
