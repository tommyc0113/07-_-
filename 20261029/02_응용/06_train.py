"""실습 06: 80 epoch 학습하고 10 epoch마다 같은 시점의 지표를 기록합니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
Xv, yv = X_valid.to(device), y_valid.to(device)
Xt, yt = X_train.to(device), y_train.to(device)
history = []

for epoch in range(81):  # epoch 0~80까지 총 81번 확인
    if epoch > 0:        # epoch 0은 학습 전 상태이므로 학습하지 않음
        model.train()    # 모델을 학습 모드로 설정

        for xb, yb in loader:  # 학습 데이터를 배치 단위로 가져옴
            xb, yb = xb.to(device), yb.to(device)  # 선택한 장치로 이동

            optimizer.zero_grad()  # 이전 배치의 기울기 초기화

            loss = loss_fn(model(xb), yb)  # 예측 및 손실 계산

            # 현재 batch의 loss로 역전파하고 optimizer를 한 번 갱신하세요.
            # zero_grad와 순전파·손실 계산은 위에 제공되어 있습니다.
            # >>> A06
            # TODO A06: 위 요구사항을 읽고 이 블록을 완성하세요.
            loss.backward() # 현재 batch의 loss로 기울기를 계산합니다.
            optimizer.step() # 계산된 기울기를 사용해 모델 파라미터를 갱신합니다.
            # <<< A06
    # epoch=0은 학습 전입니다. 로그는 갱신이 끝난 동일 모델로 측정합니다.
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad(): # 평가 중에는 기울기를 계산하지 않음
            train_loss = loss_fn(model(Xt), yt).item()
            logits = model(Xv)
            valid_loss = loss_fn(logits, yv).item()
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        history.append([epoch, train_loss, valid_loss, accuracy])
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    writer.writerows(history)
