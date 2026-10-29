"""실습 02: 모양 변환과 장치 이동 후 같은 장치에서 연산합니다."""
import torch
from torch import nn

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
values = torch.arange(12, dtype=torch.float32)
# 12개 원소를 (3,4)로 바꾸고 네 열에 [1,2,3,4]를 더하세요.
# 이후 앞의 두 열을 신경망 입력으로 사용합니다.
# >>> H02
# TODO H02: 위 요구사항을 읽고 이 블록을 완성하세요.
matrix = values.reshape(3, 4)
bias = torch.tensor([1,2,3,4]) # 각 열에 더할 보정값을 텐서로 생성합니다.
shifted = matrix+bias # 브로드캐스팅을 사용하여 모든 행에 [1, 2, 3, 4]를 더합니다.
# <<< H02
# 보정된 행렬에서 모든 행의 앞 두 열만 선택하고,
# 선택한 데이터를 CPU 또는 GPU로 이동합니다.
X = shifted[:, :2].to(device)
# 정답 레이블 3개를 정수형(long) 텐서로 생성하고,
# 입력 데이터 X와 같은 장치로 이동합니다.
y = torch.tensor([0, 0, 1], dtype=torch.long).to(device)
# 입력 특성 2개를 받아 출력값 2개를 만드는 선형 신경망을 생성하고,
# 모델을 선택한 장치로 이동합니다.
model = nn.Linear(2, 2).to(device)
logits = model(X)
print("행렬:", matrix.tolist())
print("보정:", shifted.tolist())
print("입력/출력:", tuple(X.shape), tuple(logits.shape))
print("device 일치:", X.device == y.device == next(model.parameters()).device)
