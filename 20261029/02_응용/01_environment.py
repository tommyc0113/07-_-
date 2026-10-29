"""실습 01: CUDA가 있으면 사용하고 없으면 CPU로 진행합니다."""
from pathlib import Path
import torch

out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
# CUDA를 사용할 수 있으면 cuda:0, 아니면 cpu를 선택하세요.
# 선택된 device는 아래 연산과 기록에 함께 사용합니다.
# >>> A01
# TODO A01: 위 요구사항을 읽고 이 블록을 완성하세요.
device = torch.device("cpu")
# <<< A01
x = torch.tensor([1., 2.]).to(device)  # 선택한 장치(CPU 또는 GPU)로 텐서를 이동합니다.
# PyTorch와 GPU 환경 정보를 딕셔너리에 저장합니다.
info = {
    "torch": str(torch.__version__),  # 설치된 PyTorch 버전
    "cuda_build": torch.version.cuda,  # PyTorch가 빌드될 때 사용한 CUDA 버전
    "cuda_available": torch.cuda.is_available(),  # CUDA 사용 가능 여부
    "selected_device": str(device)  # 실제 선택한 장치
}
# GPU를 사용하는 경우에만 GPU 정보를 추가합니다.
if device.type == "cuda":
    # GPU 모델명 확인
    info["gpu_name"] = torch.cuda.get_device_name(device)

    # GPU 전체 메모리 용량을 GiB 단위로 변환하고 소수점 둘째 자리까지 표시
    info["vram_gib"] = round(
        torch.cuda.get_device_properties(device).total_memory / 1024**3,
        2
    )
for key, value in info.items():
    print(f"{key}: {value}")
print("연산:", (x + 1).cpu().tolist())
(out / "gpu_check.txt").write_text("\n".join(f"{k}: {v}" for k, v in info.items()), encoding="utf-8")
