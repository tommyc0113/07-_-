"""실습 04: 2 → 8 → 2 신경망의 순전파를 확인합니다."""
import torch
from torch import nn

torch.manual_seed(42)
# Sequential의 마지막 Linear를 수정해 출력 클래스 수가 2가 되게 하세요.
# 첫 Linear(2, 8)와 ReLU는 그대로 두세요. 모델 정의는 미리 제공했습니다.
# >>> B04
# TODO B04: 위 요구사항을 읽고 이 블록을 완성하세요.
# 입력 특성 2개 → 은닉층 뉴런 8개 → 출력 클래스 2개
model = nn.Sequential(
    nn.Linear(2, 8),  # 입력 2개를 받아 8개로 변환
    nn.ReLU(),        # 음수는 0으로, 양수는 그대로 출력
    nn.Linear(8, 1)   
)
# <<< B04
# 데이터 4개, 각 데이터는 특성 2개로 구성
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0], [2.0, 0.0]])
logits = model(X) # 신경망에 입력 데이터를 전달해 순전파 실행
print("입력:", tuple(X.shape))
print("출력:", tuple(logits.shape))
print("파라미터 수:", sum(p.numel() for p in model.parameters()))
# logits는 원시 점수입니다. 이 시점에는 학습하지 않았습니다.
