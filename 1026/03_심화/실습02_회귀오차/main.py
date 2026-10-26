"""실습 02 심화: 기존 응용 수준의 회귀 비교와 오류 사례입니다."""
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

from preprocessing import build_model
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

rows = []
linear_prediction = None
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = build_model(estimator)  # 전처리와 모델이 결합된 파이프라인 생성
    # TODO H02: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: build_model은 전처리와 모델이 묶인 Pipeline을 반환합니다.
    model.fit(X_train, train["elapsed_ms"]) # 학습데이터로 회귀 모델을 학습
    prediction = model.predict(X_valid) # 검증데이터의 실행 시간을 예측

    # 실제값과 예측값을 비교해 평가 지표를 계산합니다.
    rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                 "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction),
                 "R2": r2_score(valid["elapsed_ms"], prediction)})
    if name == "Linear":
        linear_prediction = prediction
# 제공: 실제값-예측값이 잔차입니다. 음수라면 실제보다 크게 예측한 것입니다.
errors = valid[["elapsed_ms"]].copy() # 검증 데이터의 실제 실행시간만 복사
errors["predicted_ms"] = linear_prediction # 선형회귀 모델의 예측 실행시간을 추가
errors["residual_ms"] = errors["elapsed_ms"] - errors["predicted_ms"] 
errors["absolute_error_ms"] = errors["residual_ms"].abs() # 잔차의 절댓값으로 예측 오차의 크기 계산
worst = errors.sort_values("absolute_error_ms", ascending=False).head(3) # 절대 오차가 큰 순서대로 상위 3건을 추출
comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
worst.to_csv(out / "largest_errors.csv", encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("Linear의 큰 잔차 3건")
print(worst.round(3).to_string())
