"""학습 한 epoch와 평가를 분리합니다. 빈 데이터·불량 자료는 이번 단계의 과제가 아닙니다."""
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, count = 0.0, 0
    # DataLoader에서 batch 단위로 입력 데이터와 정답을 가져옵니다.
    for xb, yb in loader:
        # 입력 데이터와 정답을 학습에 사용할 장치(CPU 또는 GPU)로 이동합니다.
        xb, yb = xb.to(device), yb.to(device)

        # 이전 batch에서 계산한 gradient를 초기화합니다.
        optimizer.zero_grad()

        # 모델의 예측값(logits)을 구하고 실제 정답과 비교해 손실을 계산합니다.
        loss = loss_fn(model(xb), yb)
        # loss.backward() 후 optimizer.step()을 호출하세요.
        # 매 batch마다 초기화하고 새로운 gradient로 갱신해야 합니다.
        # >>> H07
        # TODO H07: 위 요구사항을 읽고 이 블록을 완성하세요.
        loss.backward()   # 손실을 기준으로 각 파라미터의 gradient 계산
        optimizer.step()  # 계산된 gradient를 이용해 모델 파라미터 갱신
        # <<< H07
        total += loss.item() * len(yb)
        count += len(yb)
    # 마지막 작은 batch도 실제 샘플 수만큼 반영합니다.
    return total / count


def evaluate(model, X, y, loss_fn, device):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = loss_fn(logits, y.to(device)).item()
        # logits에서 클래스 인덱스를 구하고 CPU로 옮겨 prediction에 저장하세요.
        # 평가에서는 backward와 optimizer.step을 호출하지 않습니다.
        # >>> H08
        # TODO H08: 위 요구사항을 읽고 이 블록을 완성하세요.
        prediction = logits.argmax(dim=1).cpu()
        # <<< H08
    return loss, (prediction == y.cpu()).float().mean().item(), prediction
