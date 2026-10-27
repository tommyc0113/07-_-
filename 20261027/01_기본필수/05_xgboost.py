"""실습 05: XGBoost도 같은 fit·predict 흐름으로 실행합니다."""
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

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 제공: 학습 회차 80·깊이 2·보정 크기 0.05를 고정합니다. CPU로 실행합니다.
model = XGBClassifier(
    n_estimators=80,          # 부스팅 학습에 사용할 트리 개수
    max_depth=2,              # 각 트리의 최대 깊이
    learning_rate=0.05,       # 각 트리의 학습 기여도를 조절하는 학습률
    objective="binary:logistic",  # 이진 분류 문제로 설정
    eval_metric="logloss",    # 학습 시 사용할 평가 지표
    tree_method="hist",       # 히스토그램 기반의 효율적인 트리 학습 방식
    random_state=42,          # 실행 결과를 재현하기 위한 난수 시드
    n_jobs=1                  # 사용할 작업 스레드 수
)
# TODO B05: train으로 학습하고 valid를 예측하세요.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# TODO 끝

# 검증 데이터의 예측 결과를 리스트 형태로 출력
print("valid 예측:", prediction.tolist())

# 실제 정답(y_valid)과 예측 결과(prediction)를 비교해 F1 점수를 계산
# zero_division=0은 정밀도 또는 재현율 계산에서 분모가 0인 경우 0을 반환
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
print("학습 목적: 실패 분류 / 확인 지표: F1")
