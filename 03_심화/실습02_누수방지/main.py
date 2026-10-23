"""02 심화(기존 응용): 결측 대체와 표준화의 학습 상태를 비교합니다."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train = pd.DataFrame({"value": [10, 20, np.nan]})
valid = pd.DataFrame({"value": [100, np.nan]})
processor = Pipeline([("fill", SimpleImputer(strategy="mean")),
                      ("scale", StandardScaler())])

processor.fit(train) # 학습 데이터(train)를 사용해 평균값과 표준화 기준을 학습

before = processor["fill"].statistics_.copy()
# TODO H02-1: valid에서는 transform만 사용해 두 처리 단계를 적용하세요.
valid_ready = valid.to_numpy()
after = processor["fill"].statistics_.copy()
print("학습 기준 전/후:", before.tolist(), after.tolist())
print("기준 유지:", np.array_equal(before, after))
print("valid 변환:", valid_ready.round(3).tolist())

