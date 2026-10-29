"""실습 08 추가: pytest 없이 표준 unittest로 두 검증을 직접 완성합니다."""
import sys
from pathlib import Path
import unittest
import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import SimpleANN
from train import train_one_epoch, evaluate

# 학습 및 평가 함수의 동작을 검사하는 테스트 클래스입니다.
# unittest.TestCase를 상속하면 assertEqual, assertTrue 등의 검증 함수를 사용할 수 있습니다.
class TrainingChecks(unittest.TestCase):
    def setUp(self):
        # 테스트마다 난수 시드를 고정하여 결과를 재현하기 쉽게 합니다.
        torch.manual_seed(42)
        
        # CPU에서 사용할 스레드 수를 1개로 설정합니다.
        # 작은 테스트의 실행 환경을 단순하게 유지합니다.
        torch.set_num_threads(1)

        # 테스트용 신경망 모델을 생성
        self.model = SimpleANN()
        
        # 테스트용 입력 데이터 7건을 생성합니다.
        # 각 데이터는 x1, x2 두 개의 특성을 가집니다.
        self.X = torch.tensor([[1.,2.], [-1.,0.], [2.,1.], [0.,0.], [1.,-1.], [-2.,1.], [3.,2.]])
        
        # 각 입력 데이터에 대응하는 실제 클래스 정답입니다.
        # 정답 클래스는 0 또는 1입니다.
        self.y = torch.tensor([1,0,1,0,1,0,1])

    def test_last_batch_weighting(self):
        # 입력 7건을 batch_size=3으로 나눕니다.
        # batch 크기는 3, 3, 1이 됩니다.
        # 마지막 batch에는 데이터가 1건만 들어갑니다.
        loader = DataLoader(TensorDataset(self.X, self.y), batch_size=3, shuffle=False)
        
        # 분류 모델의 손실을 계산하는 함수입니다.
        loss_fn = nn.CrossEntropyLoss()
        # lr=0은 검사용입니다. 가중치를 고정해야 배치 분할 효과만 비교할 수 있습니다.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)

        # 전체 7건을 한 번에 모델에 입력하여 기준 손실을 계산합니다.
        # CrossEntropyLoss의 기본 reduction은 mean이므로
        # 7건 전체의 평균 손실이 반환됩니다.
        expected = loss_fn(self.model(self.X), self.y).item()

        # 학습 함수를 실행하여 배치 단위로 계산한 손실을 얻습니다.
        # train_one_epoch는 모델 학습과 epoch 손실 계산을 수행합니다.
        actual = train_one_epoch(self.model, loader, loss_fn, optimizer, torch.device("cpu"))
        
        # H11 요구: 7건을 3·3·1로 나눈 손실이 전체 손실과 소수점5자리까지 같음을 검사하세요.
        # 수치를 출력만 하지 말고 self.assertAlmostEqual을 사용합니다.
        # >>> H11
        # TODO H11: actual과 expected가 같은지 검사하세요.
        self.assertAlmostEqual(actual, expected, places=5)
        # <<< H11

    def test_evaluation_does_not_learn(self):
        # 모델의 파라미터에 기존 gradient를 만들어 둡니다.
        # 평가 함수가 이 gradient를 변경하거나 초기화하는지 검사하기 위함입니다.
        self.model(self.X).sum().backward()

        # 평가 전 모델 파라미터를 깊은 복사로 저장합니다.
        # 이후 파라미터가 바뀌었는지 비교할 기준입니다.
        before = copy.deepcopy(self.model.state_dict())

        # 각 파라미터의 현재 gradient도 복사해 둡니다.
        # 평가 전후 gradient가 같은지 검사하는 데 사용합니다.
        grads = [p.grad.clone() for p in self.model.parameters()]
        loss, accuracy, prediction = evaluate(self.model, self.X, self.y, nn.CrossEntropyLoss(), torch.device("cpu"))
        # H12 요구: 평가 전후 각 Parameter 값과 기존 grad가 모두 같은지 assertTrue로 검사하세요.
        # 추가: prediction에 requires_grad가 없음을 assertFalse로 확인합니다.
        # 모델의 eval 모드 전환은 정상입니다. 이 단계는 모드 복구까지 요구하지 않습니다.
        # >>> H12
        # TODO H12: 가중치·gradient·예측의 그래프 여부를 검사하세요.
        for before_param, after_param in zip(
            before.values(), self.model.state_dict().values()
        ):
            self.assertTrue(torch.equal(before_param, after_param))

        for param, grad_before in zip(self.model.parameters(), grads):
            self.assertTrue(torch.equal(param.grad, grad_before))

        self.assertFalse(prediction.requires_grad)
        # <<< H12

if __name__ == "__main__":
    unittest.main(verbosity=2)
