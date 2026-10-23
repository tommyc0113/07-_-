"""06 기본필수: 숫자 1열과 문자 1열을 함께 변환합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 학습 데이터(train)를 생성합니다.
train = pd.DataFrame({
    "input_chars": [100, 200, 300],          # 입력 글자 수(숫자형 데이터)
    "task_type": ["qa", "summary", "qa"]      # 작업 유형(범주형 데이터)
})

# 검증 데이터(valid)를 생성합니다.
valid = pd.DataFrame({
    "input_chars": [400, 500],               # 검증할 입력 글자 수
    "task_type": ["summary", "qa"]            # 검증할 작업 유형
})

# 제공: 숫자에는 표준화, 문자에는 원-핫 변환을 지정합니다.
preprocessor = ColumnTransformer([
    # 숫자형 열 input_chars에 표준화를 적용합니다.
    # StandardScaler: 평균 0, 표준편차 1을 기준으로 변환합니다.
    ("num", StandardScaler(), ["input_chars"]),

    # 범주형 열 task_type에 원-핫 인코딩을 적용합니다.
    # OneHotEncoder: qa, summary 같은 문자를 0과 1로 변환합니다.
    # sparse_output=False: 결과를 일반 배열 형태로 반환합니다.
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

preprocessor.fit(train)
# TODO B06-1: preprocessor의 transform으로 valid를 변환하세요.
valid_ready = preprocessor.transform(valid)

print("변환 전:", valid.shape)
print("변환 후:", valid_ready.shape)
print(valid_ready.round(3))
