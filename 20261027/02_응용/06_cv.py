"""실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
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
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

folds = list(
    StratifiedKFold(
        n_splits=5,       # 데이터를 5개 fold로 분할합니다.
        shuffle=True,     # 분할 전에 데이터를 섞습니다.
        random_state=42   # 분할 결과를 재현할 수 있도록 시드를 고정합니다.
    ).split(X_train, y_train)
)

# 비교할 모델 3개와 각 모델의 하이퍼파라미터를 설정합니다.
candidates = [
    (
        "Tree",  # 첫 번째 모델: 결정 트리
        DecisionTreeClassifier(
            max_depth=3,        # 트리의 최대 깊이를 3으로 제한합니다.
            min_samples_leaf=3, # 리프 노드에 최소 3개 샘플이 있도록 설정합니다.
            random_state=42     # 난수 시드를 고정합니다.
        )
    ),

    (
        "Forest",  # 두 번째 모델: 랜덤 포레스트
        RandomForestClassifier(
            n_estimators=100,    # 트리 100개를 사용합니다.
            max_depth=4,         # 각 트리의 최대 깊이를 4로 제한합니다.
            min_samples_leaf=2,  # 리프 노드의 최소 샘플 수를 2로 설정합니다.
            max_features="sqrt",# 분할할 때 고려할 특성 수를 제곱근 기준으로 설정합니다.
            random_state=42,     # 난수 시드를 고정합니다.
            n_jobs=1             # 작업 스레드를 1개 사용합니다.
        )
    ),

    (
        "XGB",  # 세 번째 모델: XGBoost
        XGBClassifier(
            n_estimators=80,           # 부스팅에 사용할 트리 개수
            max_depth=2,               # 각 트리의 최대 깊이
            learning_rate=0.05,        # 학습률
            subsample=0.8,             # 각 트리 학습에 사용할 샘플 비율
            colsample_bytree=1.0,      # 각 트리에 사용할 특성 비율
            reg_lambda=1.0,             # L2 정규화 강도
            objective="binary:logistic",# 이진 분류 목적 함수
            eval_metric="logloss",      # 로그 손실 평가 지표
            tree_method="hist",         # 히스토그램 기반 트리 학습 방식
            device="cpu",               # CPU로 학습합니다.
            random_state=42,            # 난수 시드를 고정합니다.
            n_jobs=1                    # 작업 스레드를 1개 사용합니다.
        )
    )
]

rows = []
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A06: 전처리까지 포함한 model의 fold별 F1을 계산하세요.
    scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=folds,  # 미리 설정한 교차검증 분할 방식 사용
    scoring="f1"  # 각 fold의 F1 점수를 계산
)
    # TODO 끝
    print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))
    for number, value in enumerate(scores, start=1):
        rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
print("사용 자료: train60 내부만 / valid20·test20 미사용")
