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

loss_fn = nn.CrossEntropyLoss()
for name in ["SGD", "AdamW"]:
    # 후보마다 같은 초기 가중치에서 다시 시작합니다. 앞 모델을 이어 학습하지 않습니다.
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    if name == "SGD":
        # SGD의 학습할 파라미터는 model.parameters()입니다. lr=0.01, momentum=0.0으로 만드세요.
        # 학습률은 기본값에 기대지 않고 명시하며, 두 Optimizer 모두 weight_decay=0.0으로 비교합니다.
        # >>> B01
        # TODO B01: 위 요구사항에 맞게 이 블록만 완성합니다.
        optimizer = torch.optim.SGD(

            # 갱신할 모델의 파라미터를 전달합니다.
            # 현재 모델에서는 각 Linear 층의 가중치(weight)와 편향(bias)이 해당합니다.
            # ReLU처럼 학습할 파라미터가 없는 층은 갱신 대상이 없습니다.
            model.parameters(),

            # 학습률(learning rate): 기울기를 이용해 파라미터를 얼마나 이동시킬지 정합니다.
            # momentum과 weight_decay가 0이면 다음 식으로 갱신합니다.
            # 새로운 파라미터 = 기존 파라미터 - 학습률 × 현재 기울기
            # 너무 크면 손실이 진동하거나 발산할 수 있고, 너무 작으면 학습이 느릴 수 있습니다.
            lr=0.01,

            # 모멘텀(momentum): 이전 갱신 방향을 누적하여 현재 갱신에 반영하는 정도입니다.
            # 같은 방향의 기울기가 이어지면 이동을 가속하고,
            # 방향이 반복해서 바뀌는 구간에서는 진동을 줄이는 데 도움이 될 수 있습니다.
            # 0.0은 모멘텀을 사용하지 않는다는 뜻이며, 0.9 등이 자주 사용됩니다.
            momentum=0.0,

            # 가중치 감쇠(weight decay): 파라미터가 지나치게 커지는 것을 억제하는 규제입니다.
            # PyTorch의 SGD에서는 기울기에 "weight_decay × 현재 파라미터"를 더합니다.
            # 이는 L2 규제에 해당하며, 과적합을 줄이는 데 도움이 될 수 있습니다.
            # 0.0은 적용하지 않는다는 뜻이고, 너무 크게 설정하면 과소적합이 생길 수 있습니다.
            # 여기서는 model.parameters() 전체를 전달했으므로 편향에도 동일하게 적용됩니다.
            weight_decay=0.0)
                
    else:
        # AdamW를 같은 lr=0.01로 만드세요. weight_decay=0.0을 반드시 적으세요.
        # 이번에는 감쇠 효과가 아닌 갱신 규칙을 비교합니다. 이 공통 lr이 두 알고리즘의 최적 lr이라는 뜻은 아닙니다.
        # >>> B02
        # TODO B02: 위 요구사항에 맞게 이 블록만 완성합니다.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
        # <<< B02
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        valid_loss = loss_fn(model(X_valid), y_valid).item()
    print(name, "20 epochs / valid loss:", round(valid_loss, 6))
