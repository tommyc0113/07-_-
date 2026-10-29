"""실습 02: reshape와 broadcasting을 확인합니다."""
import torch

a = torch.arange(6, dtype=torch.float32)
# 원소 6개를 2행 3열로 바꾸어 matrix에 저장하세요.
# 아래 bias는 각 열에 더해집니다.
# >>> A02
# TODO A02: 위 요구사항을 읽고 이 블록을 완성하세요.
matrix = a.reshape(2,3)
# <<< A02
bias = torch.tensor([10., 20., 30.])  # 각 열에 더할 보정값 3개를 실수형 텐서로 생성합니다.
result = matrix + bias
print("변환 전:", tuple(a.shape)) # 원본 텐서 a의 모양(shape)을 튜플로 출력합니다.
print("행렬:", tuple(matrix.shape), matrix.tolist()) # 변환된 행렬의 모양과 실제 데이터를 출력합니다.
print("보정 후:", tuple(result.shape), result.tolist()) # 보정값을 더한 결과의 모양과 실제 데이터를 출력합니다.

# CUDA를 사용할 수 있으면 첫 번째 GPU(cuda:0)를 선택하고,
# 사용할 수 없으면 CPU를 선택합니다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
moved = result.to(device)
print("이동 장치:", moved.device)
print("CPU에서 확인:", moved.cpu().tolist())
