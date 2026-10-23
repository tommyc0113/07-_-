"""05 심화(기존 응용): 범주형 2열을 6열로 만들고 새 범주를 처리합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

columns = ["task_type", "priority"]
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
train_ready = categorical.fit_transform(train[columns])
valid_ready = categorical.transform(valid[columns])
new_request = pd.DataFrame({"task_type": ["translation"], "priority": ["urgent"]})
# TODO H05-1: 새 요청 new_request를 transform하세요.
new_ready = categorical.transform(new_request)
print("최빈값:", categorical["fill"].statistics_.tolist())
print("train/valid 모양:", train_ready.shape, valid_ready.shape)
print("새 요청:", new_ready.tolist())
