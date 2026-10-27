"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
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
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
candidates = [(
                # 1. Dummy 모델: 가장 많이 등장하는 클래스로만 예측하는 기준 모델
                # 다른 모델의 성능이 단순한 기준 모델보다 좋은지 비교할 때 사용
                "Dummy", DummyClassifier(strategy="most_frequent")),
              (
                "Tree", DecisionTreeClassifier(max_depth=3, # 트리의 최대 깊이를 3으로 제한
                                               min_samples_leaf=3, # 리프 노드에 최소 3개의 샘플이 있도록 설정
                                               random_state=42)), # 결과 재현을 위한 난수 시드
            ("Forest", RandomForestClassifier( n_estimators=100, #결정 트리 100개 사용
                                            max_depth=4, #각 트리의 최대 깊이를 4로 제한
                                            min_samples_leaf=2, #리프 노드당 최소 샘플 수를 2로 설정
                                            max_features="sqrt", #각 분할에서 전체 특성 수의 제곱근만큼 후보 특성 사용
                                            random_state=42, #결과 재현을 위한 난수 시드
                                            n_jobs=1, #CPU 작업에 작업자 1개 사용
            )), 

            # 4. XGBoost: 이전 트리의 오차를 보완하도록 트리를 순차적으로 학습하는 모델
            # n_estimators=80: 트리 80개 사용
            # max_depth=2: 각 트리의 최대 깊이를 2로 제한
            # learning_rate=0.05: 각 트리의 학습 결과를 반영하는 비율
            # subsample=0.8: 각 트리 학습에 학습 데이터의 80%를 무작위로 사용
            # colsample_bytree=1.0: 트리마다 전체 특성을 사용
            # reg_lambda=1.0: L2 정규화 강도
            # objective="binary:logistic": 이진 분류 확률을 예측하는 목적 함수
            # eval_metric="logloss": 평가 지표로 로그 손실 사용
            # tree_method="hist": 히스토그램 기반 트리 학습 방식
            # device="cpu": CPU를 사용해 학습
            # random_state=42: 결과 재현을 위한 난수 시드
            # n_jobs=1: CPU 작업에 작업자 1개 사용
            ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
rows, cv_rows = [], []
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring="f1"
    )

    # TODO A07b: valid의 예측을 prediction에 넣으세요.
    prediction = model.predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    
    # 현재 모델의 성능 평가 결과
    rows.append({"model": name, 
                 "CV_F1": scores.mean(),  # 교차검증 F1 점수 평균
                 "CV_std": scores.std(),  # 교차검증 F1 점수 표준편차
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),  # 전체 train 데이터에 대한 F1 점수
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})  # valid 데이터에 대한 F1 점수

    # 교차검증 각 회차의 성능을 cv_rows 리스트에 저장
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name,
                        "fold": number,  # 교차검증 회차 번호
                        "fit_n": 48,  # 해당 회차의 학습 샘플 수
                        "evaluate_n": 12,  # 해당 회차의 평가 샘플 수
                        "F1": value  # 해당 회차의 F1 점수
                    })
        
    predictions[name] = prediction
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
