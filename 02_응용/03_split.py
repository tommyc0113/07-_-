"""03 응용: 전체 표를 함께 나누면 입력과 정답이 따로 섞이지 않습니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

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

print("분할:", len(train), len(valid), len(test))
# TODO A03-1: valid의 정답 개수를 세도록 자료를 바꾸세요.
valid_counts = (train["failed"]==1).value_counts().sort_index()
print("valid 정답 개수:", valid_counts.to_dict())
print("seed: 42 / 42")
