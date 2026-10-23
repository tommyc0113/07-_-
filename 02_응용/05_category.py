"""05 응용: 결측은 최빈값으로, 처음 보는 종류는 0/1 열의 0으로 처리합니다."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

train = pd.DataFrame({"task_type": ["qa", "qa", "research", np.nan]})
valid = pd.DataFrame({"task_type": ["summary", np.nan]})

# 범주형 데이터 전처리 파이프라인을 생성합니다.
categorical = Pipeline([
    # 결측값을 가장 자주 등장하는 값으로 대체합니다.
    ("fill", SimpleImputer(strategy="most_frequent")),

    # 문자를 원-핫 인코딩으로 변환합니다.
    # handle_unknown="ignore": 학습 때 보지 못한 범주는 오류 없이 처리합니다.
    # sparse_output=False: 결과를 일반 배열로 반환합니다.
    ("encode", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

# 학습 데이터로 결측값 대체 기준과 인코딩 기준을 학습합니다.
categorical.fit(train)

# TODO A05-1: 학습된 categorical로 valid를 변환하세요.
# valid에 fit()을 다시 실행하지 않고, 학습된 기준만 적용합니다.
valid_ready = categorical.transform(valid)
print("대체값:", categorical["fill"].statistics_.tolist()) # 학습 데이터에서 결측값을 대체할 때 사용한 값
print("열 순서:", categorical["encode"].categories_[0].tolist()) # 원-핫 인코딩에 사용된 범주 순서
print("valid 변환:", valid_ready.tolist()) # 검증 데이터의 실제 변환 결과

