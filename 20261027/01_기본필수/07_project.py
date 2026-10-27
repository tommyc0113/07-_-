"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score

# 제공: 전날과 같은 입력 2열입니다. 실행시간과 failed는 입력에 넣지 않습니다.
features = ["input_chars", "task_type"]
X_train, X_valid = train[features], valid[features]
pre = ColumnTransformer([
    ("num", "passthrough", ["input_chars"]),  # 별도 변환 없이 그대로 사용
    # 학습 시 없었던 범주가 나와도 오류가 발생하지 않도록 설정, 결과를 일반 배열 형태로 반환
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])

# 실패여부를 기준으로 데이터를 3개 그룹으로 나누어 교차검증
# 데이터를 무작위로 섞어 실행할때마다 같은 방식으로 분할
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),  # 최대 길이가 3인 결정 트리 모델
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]  # 트리 20개를 사용하는 랜덤 포레스트 모델
rows, cv_rows = [], []
predictions = valid[["request_id", "failed"]].copy()  # 검증 데이터의 요청id와 실제 실패 여부를 복사

for name, estimator in candidates:
    # 제공: 전처리까지 Pipeline에 포함하여 fold 학습 부분에서만 fit하게 합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07a: 전처리까지 연결한 model을 train에서 교차검증하세요.
    scores = cross_val_score (model,
        X_train,
        y_train,
        cv=folds,
        scoring="f1"
    )
    # TODO 끝
    # TODO B07b: 전체 train으로 학습하고 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝

    # 모델별 성능 평가 결과물 rows 리스트에 추가
    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
    
    # 교차검증 각 회차의 F1 점수를 하나씩 저장
    for number, value in enumerate(scores, start=1):
        cv_rows.append({
            "model": name,  # 모델 이름
            "fold": number,  # 교차검증 회차 번호(1, 2, 3)
            "F1": value  # 해당 회차의 F1 점수
        })
    predictions[name] = prediction  # 검증 데이터에 해당 모델의 예측 결과를 새로운 열로 추가
    if name == "Forest":
        importance = pd.DataFrame({
            # 전처리 후 생성된 특성 이름
            "feature": model["pre"].get_feature_names_out(),

            # 랜덤 포레스트 모델이 계산한 특성별 중요도
            "importance": model["model"].feature_importances_
        })

# 제공: 입력 파일을 덮어쓰지 않고 결과 4개를 같은 폴더에 만듭니다.
comparison = pd.DataFrame(rows)
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print(comparison.round(3).to_string(index=False))
print("중요도")
print(importance.round(3).to_string(index=False))
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
