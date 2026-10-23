"""01 심화(기존 응용): 입력은 고정하고 분류/회귀의 정답을 구분합니다."""
from pathlib import Path
import pandas as pd

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data" / "requests.csv")
features = ["input_chars", "allowed_tools", "task_type", "priority"]

X = data[features]
y_classification = data["failed"]
# TODO H01-1: 실행시간 회귀에서 예측할 정답을 선택하세요.
y_regression = data["failed"]
print("X:", X.shape)
print("분류 정답:", y_classification.head(3).tolist())
print("회귀 정답(ms):", y_regression.head(3).tolist())
print("elapsed_ms는 실행 전 입력이 아니라, 실행시간 회귀에서 예측할 정답입니다.")
