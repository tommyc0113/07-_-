"""01 응용: 원래 PPT의 100행 데이터로 입력을 선택합니다."""
from pathlib import Path
import pandas as pd

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data" / "requests.csv")

# 제공: 결측은 이후 전처리에서 다룹니다. 지금은 정답 유입을 막습니다.
# TODO A01-1: 문자 입력 task_type과 priority도 추가해 4개 입력을 선택하세요.
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X = data[features]
y = data["failed"]
print("X/y 모양:", X.shape, y.shape)
print("사용:", features)
print("제외: request_id(식별용), failed(정답), status·elapsed_ms·retry_count(실행 후)")
