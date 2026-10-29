"""실습 06: 작은 정상 자료로 학습 순서만 익힙니다. 여기는 train 오차 확인용입니다."""
import torch
from torch import nn

torch.manual_seed(42) # 실행할 때마다 같은 초기 가중치를 사용하도록 난수 시드를 고정합니다.
# 입력 데이터 8개, 각 데이터의 특성은 2개입니다.
X = torch.tensor([[-2., -1.], [-1., -1.], [-1., 0.], [0., -1.],
                  [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
# 각 입력 데이터의 실제 정답 클래스입니다.
# 앞의 4개는 클래스 0, 뒤의 4개는 클래스 1입니다.
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
# 신경망 구성: 입력 2개 → 은닉층 8개 → ReLU → 출력 2개
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss() # 예측 결과와 실제 정답의 차이를 계산하는 손실 함수입니다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # SGD 최적화 알고리즘을 사용하고 학습률은 0.1로 설정합니다.
print("학습 전 loss:", round(loss_fn(model(X), y).item(), 6))
model.train()
for epoch in range(80): # 전체 데이터로 학습을 80회 반복합니다.
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    # 현재 loss의 gradient를 계산하세요.
    # 파라미터 값은 아직 바뀌지 않으며 .grad에 기울기가 들어갑니다.
    # >>> B06
    # TODO B06: 위 요구사항을 읽고 이 블록을 완성하세요.
    loss.backward()
    # <<< B06
    # 현재 gradient를 사용해 파라미터를 한 번 갱신하세요.
    # optimizer의 step 메서드를 호출합니다.
    # >>> B07
    # TODO B07: 위 요구사항을 읽고 이 블록을 완성하세요.
    optimizer.step()
    # <<< B07
model.eval()
with torch.no_grad():
    logits = model(X)
    print("학습 후 loss:", round(loss_fn(logits, y).item(), 6))
    print("train 정확도:", round((logits.argmax(dim=1) == y).float().mean().item(), 4))
print("주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.")
