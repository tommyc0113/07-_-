"""실습 08: main.py가 만든 state_dict만 로드합니다. 새 프로세스 CPU에서 복원합니다."""
from pathlib import Path
import csv
import torch
from model import SimpleANN

base = Path(__file__).resolve().parent

# points.csv 파일에서 split 값이 valid인 행만 가져옵니다.
# newline=""은 CSV 파일을 읽을 때 줄바꿈 처리를 안정적으로 합니다.
with (base / "data" / "points.csv").open(encoding="utf-8",newline="") as f:
    valid_rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]

# 검증 데이터의 x1, x2를 입력 텐서로 변환합니다.
# 저장 당시와 같은 데이터 및 행 순서를 사용해야 비교가 정확합니다.
X = torch.tensor([[float(r["x1"]),float(r["x2"])] for r in valid_rows])

model = SimpleANN()
# 직접 저장한 model_state.pt를 map_location="cpu", weights_only=True로 읽고 모델에 넣으세요.
# 먼저 같은 모델 구조를 만들었습니다. 인터넷에서 받은 출처 불명 파일은 읽지 마세요.
# >>> H10
# TODO H10: 위 요구사항을 읽고 이 블록을 완성하세요.
# map_location="cpu": 저장 장치와 관계없이 CPU에서 읽습니다.
# weights_only=True: 지원되는 PyTorch 버전에서 가중치 사전을 불러옵니다.
state = torch.load(
    base / "outputs" / "model_state.pt",
    map_location="cpu",
    weights_only=True
)

# 저장된 파라미터를 현재 모델에 적용합니다.
# 모델 구조와 파라미터의 이름 및 형태가 일치해야 합니다.
model.load_state_dict(state)
# <<< H10
model.eval()

with torch.no_grad():

    # 각 데이터에 대한 클래스별 점수를 계산합니다.
    # argmax(dim=1)은 가장 높은 점수를 가진 클래스 번호를 선택합니다.
    # tolist()는 텐서를 파이썬 리스트로 변환합니다.
    predictions = model(X).argmax(dim=1).tolist()
    
with (base / "outputs" / "validation_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
    saved = list(csv.DictReader(f))
print("ID 순서 동일:", [r["id"] for r in valid_rows] == [r["id"] for r in saved])
print("예측 동일:", predictions == [int(r["predicted"]) for r in saved])
print("서로 다른 CPU/GPU의 마지막 비트까지 동일함을 보장하는 검사는 아닙니다.")
