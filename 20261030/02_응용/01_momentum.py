from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
loss_fn = nn.CrossEntropyLoss()
# 두 설정은 momentum만 0.0과 0.9로 다르게 만드세요.
# optimizer=SGD, lr=0.05, 나머지 설정은 동일하게 유지합니다.
# >>> A01
# TODO A01: 위 요구사항에 맞게 이 블록만 완성합니다.
configs = [
    {
        'name': 'momentum0',
        'optimizer': 'SGD',
        'lr': 0.05,
        'momentum': 0.0
    },
    {
        'name': 'momentum09',
        'optimizer': 'SGD',
        'lr': 0.05,
        'momentum': 0.9
    }
]
# <<< A01
summary = []

# configs에 저장된 설정을 하나씩 가져와 학습 진행
for config in configs:
    torch.manual_seed(42)  # 모든 후보 모델이 동일한 초기 가중치로 시작하도록 시드 고정
    # 설정에 width가 있으면 사용하고, 없으면 기본값 32 사용
    # width는 은닉층(Hidden Layer)의 뉴런 개수
    width = config.get("width", 32)

    # 첫 번째 입력층 생성
    # 입력 데이터 2개(x1, x2)를 width개의 뉴런으로 변환
    layers = [nn.Linear(2, width)]

    # Batch Normalization 사용 여부 확인
    if config.get("bn", False):
        # 각 층의 출력값을 정규화하여 학습 안정화
        layers.append(nn.BatchNorm1d(width))

    # ReLU 활성화 함수와 Dropout 추가
    # Dropout 비율은 설정값을 사용하고, 없으면 0.0
    # 마지막 Linear는 은닉층에서 은닉층으로 연결
    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, width)
    ])

    # 두 번째 은닉층에도 Batch Normalization 적용 가능
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # 두 번째 은닉층의 활성화 함수와 Dropout
    # 마지막 출력층은 2개 클래스를 분류하도록 구성
    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, 2)
    ])

    # 리스트에 저장한 층들을 하나의 Sequential 모델로 구성
    # device에 모델을 올림 (CPU 또는 GPU)
    model = nn.Sequential(*layers).to(device)
    
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    # 학습 데이터를 DataLoader로 구성
    # batch_size=32 : 한 번에 32개 데이터 학습
    # shuffle=True : 학습할 때 데이터 순서를 섞음
    # generator : 모든 후보가 동일한 셔플 순서를 사용하도록 시드 고정
    # num_workers=0 : 데이터를 불러오는 작업에 별도 프로세스를 사용하지 않음
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # Optimizer 종류에 따라 SGD 또는 AdamW 선택
    if config.get("optimizer", "AdamW") == "SGD":

        # SGD 옵티마이저 설정
        # lr : 학습률
        # momentum : 이전 기울기의 누적 비율
        # weight_decay=0.0 : 가중치 감쇠 사용 안 함
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.0),
            weight_decay=0.0
        )

    else:
        # AdamW 옵티마이저 설정
        # lr : 설정에 지정된 학습률
        # weight_decay=0.0 : 가중치 감쇠 사용 안 함
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.0
        )
    history = []
    
    # 총 20번 반복 학습
    for epoch in range(1, 20 + 1):

        # 모델을 학습 모드로 변경
        # Dropout과 BatchNorm이 학습 모드로 동작
        model.train()

        # DataLoader에서 미니배치 데이터를 하나씩 가져옴
        for xb, yb in loader:

            # 입력 데이터와 정답 데이터를 CPU 또는 GPU로 이동
            xb, yb = xb.to(device), yb.to(device)

            # 이전 단계에서 계산된 기울기 초기화
            optimizer.zero_grad()

            # 모델의 예측값과 실제 정답을 비교하여 손실 계산
            loss = loss_fn(model(xb), yb)

            # 역전파를 통해 각 파라미터의 기울기 계산
            loss.backward()

            # 계산된 기울기를 이용해 모델의 가중치 업데이트
            optimizer.step()

        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        model.eval()

        # 평가 단계에서는 기울기 계산을 하지 않음
        with torch.no_grad():

            # 전체 학습 데이터의 손실 계산
            # eval 모드에서 계산하므로 Dropout이 적용되지 않음
            train_loss = loss_fn(
                model(X_train.to(device)),
                y_train.to(device)
            ).item()

            # 검증 데이터에 대한 예측값 계산
            logits = model(X_valid)

            # 검증 데이터의 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 가장 높은 예측 확률을 가진 클래스 선택
            # 실제 정답과 비교하여 정확도 계산
            valid_acc = (
                logits.argmax(1) == y_valid
            ).float().mean().item()

        # 현재 Epoch의 학습 손실, 검증 손실, 정확도 저장
        history.append([
            epoch,
            train_loss,
            valid_loss,
            valid_acc
        ])

    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]   # 검증 손실이 가장 낮았던 Epoch의 결과 선택
    name = config["name"]
    table.to_csv(out / f"01_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"01_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "01_comparison.csv", index=False)
print("생성: outputs/01_*.csv, *.png / test 미사용")
