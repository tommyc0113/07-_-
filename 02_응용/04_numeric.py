"""04 응용: 결측 대체와 숫자 변환을 짧은 Pipeline으로 연결합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data" / "requests.csv")
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.

numeric_columns = ["input_chars", "allowed_tools"]
# 숫자형 데이터에 적용할 전처리 파이프라인을 생성합니다.
numeric = Pipeline([
    # 1단계: 결측값(NaN)을 각 열의 중앙값으로 대체합니다.
    ("fill", SimpleImputer(strategy="median")),

    # 2단계: 데이터를 평균 0, 표준편차 1을 기준으로 표준화합니다.
    ("scale", StandardScaler())
])

# 학습 데이터의 숫자형 2개 열로 전처리 기준을 학습하고 변환합니다.
train_ready = numeric.fit_transform(train[numeric_columns])

# TODO A04-1: 학습된 numeric으로 valid의 숫자 2열을 변환하세요.
# train에서 학습한 중앙값과 표준화 기준을 그대로 적용합니다.
valid_ready = numeric.transform(valid[numeric_columns])
print("중앙값:", numeric["fill"].statistics_.tolist())
print("train/valid 모양:", train_ready.shape, valid_ready.shape)
print("valid 처음 2행:", valid_ready[1])
print(valid_ready[:2].round(3))

