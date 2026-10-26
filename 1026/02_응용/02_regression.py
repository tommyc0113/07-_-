"""실습 02 응용: 같은 valid 20건에서 두 회귀 모델을 비교합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")

# split 열의 값이 "train"인 행만 선택
# position을 기준으로 정렬한 뒤 request_id 열만 추출
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]

# split 열의 값이 "valid"인 행만 선택
# position을 기준으로 정렬한 뒤 request_id 열만 추출
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]

# raw 데이터에서 train_ids에 해당하는 행을 선택해 학습 데이터 생성
# request_id를 인덱스로 사용하므로 해당 ID의 행을 가져옴
train = raw.loc[train_ids]
valid = raw.loc[valid_ids] # raw 데이터에서 valid_ids에 해당하는 행을 선택해 검증 데이터 생성

features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

rows = []
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A02: 회귀 정답으로 학습하고 같은 valid를 예측하세요.
    # 힌트: fit의 정답은 train["elapsed_ms"]입니다.
    
    # 학습 데이터의 입력 특성과 실제 실행 시간을 사용한 모델 학습
    model.fit(train[features], train["elapsed_ms"])

    # 검증 데이터의 입력 특성을 이용해 실행 시간 예측
    prediction = model.predict(valid[features])

    # 모델이름, MAE, RMSE를 딕셔너리로 만들어 결과 목록에 추가
    # MAE: 실제 실행 시간과 예측 실행 시간의 오차 평균
    # RMSE: 평균 제곱근 오차 
    rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                 "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})

comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("저장: outputs/regression_comparison.csv")
