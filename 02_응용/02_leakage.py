"""02 응용: 결측 평균 대체를 train에서만 학습합니다."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# NaN은 결측 개념을 배우기 위해 의도적으로 넣은 빈 값입니다.
train = pd.DataFrame({"value": [10, 20, np.nan]})
valid = pd.DataFrame({"value": [100, np.nan]})
imputer = SimpleImputer(strategy="mean")  # 결측값을 평균으로 대체하는 객체를 생성합니다.
imputer.fit(train)
# TODO A02-1: imputer.transform(valid)로 train의 평균을 적용하세요.
valid_ready = imputer.transform(valid)
print("train에서 배운 값:", imputer.statistics_.tolist())
print("valid 변환:", valid_ready.tolist())
print("valid에서 fit 또는 fit_transform을 다시 호출하지 않습니다.")


print(valid_ready)


