from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 학습률 후보 두 개를 리스트로 쓰세요: 0.001, 0.01.
# 모델·seed·20 epochs·AdamW·weight_decay=0은 그대로 둡니다.
# >>> B03
# TODO B03: 위 요구사항에 맞게 이 블록만 완성합니다.
learning_rates = [0.001, 0.01]
loss_fn = nn.CrossEntropyLoss()

# learning_rates에 저장될 학습률을 하나씩 가져옴
for lr in learning_rates:

    # 학습률 후보마다 동일한 초기 가중치 사용
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    
    # AdamW Optimizer 생성
    # lr : 현재 실험의 학습률
    # weight_decay=0.0 : 가중치 감쇠 사용 안 함
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(X_train), y_train).backward() # 학습 데이터를 이용해 손실 계산 후 역전파
        optimizer.step() # 계산된 기울기를 이용해 가중치 업데이트
        # 갱신이 끝난 모델의 valid loss를 기록합니다. valid로 backward하지 않습니다.
        model.eval() # 갱신이 완료된 모델을 평가 모드로 변경
        
        with torch.no_grad():
            losses.append(loss_fn(model(X_valid), y_valid).item())
    print("lr:", lr, "처음 5개:", [round(x, 6) for x in losses[:5]])
    print("마지막 valid loss:", round(losses[-1], 6))
