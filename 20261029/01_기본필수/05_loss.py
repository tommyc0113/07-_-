"""실습 05: 정답 인덱스와 logits로 손실을 계산합니다."""
import torch
from torch import nn

logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]]) # 각 행은 하나의 데이터에 대한 두 클래스의 예측 점수입니다.
target = torch.tensor([0, 1], dtype=torch.long) # 각 데이터의 실제 정답 클래스 인덱스
loss_fn = nn.CrossEntropyLoss() # 교차 엔트로피 손실 함수 생성
# loss_fn에 logits와 target을 순서대로 넣어 loss를 계산하세요.
# logits에 softmax를 먼저 적용하지 마세요.
# >>> B05
# TODO B05: 위 요구사항을 읽고 이 블록을 완성하세요.
loss = loss_fn(logits, target)
# <<< B05
print("logits:", tuple(logits.shape))
print("target:", tuple(target.shape), target.dtype)
print("loss:", round(loss.item(), 6))
