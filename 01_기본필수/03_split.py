"""03 기본필수: 20행을 train 12행, valid 4행, test 4행으로 나눕니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")

# test_size=0.2는 전체 20행 중 4행을 따로 보관한다는 뜻입니다.
# TODO B03-1: test 비율을 전체의 20%로 바꾸세요. 소수로 적습니다.
test_ratio = 0.2
train_valid, test = train_test_split(
    data, test_size=test_ratio, random_state=42, stratify=data["failed"])
# 남은 16행의 25%인 4행이 valid입니다. 아래 코드는 제공합니다.
# 동일한 분할 결과 유지, failed의 0과 1 비율을 최대한 유지
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

print("train:", len(train))
print("valid:", len(valid))
print("test:", len(test))
print("valid ID:", valid["request_id"].tolist())
print("test는 오늘 학습과 예측에 사용하지 않습니다.")
