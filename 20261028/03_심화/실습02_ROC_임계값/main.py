"""실습 02 심화: ROC 좌표와 두 임계값 표를 별도로 기록합니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
y = np.array([0, 0, 1, 1]) # 실제 정답 데이터: 0은 성공, 1은 실패
score = np.array([0.10, 0.40, 0.35, 0.80]) # 각 데이터가 실패일 가능성을 나타내는 연속함수

# FPR: 실제 성공을 실패로 잘못 예측한 비율
# TPR: 실제 실패를 올바르게 찾아낸 비율(재현율)
# thresholds: 각 ROC 좌표에 해당하는 분류 임계값
fpr, tpr, thresholds = roc_curve(y, score)
pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": thresholds}).to_csv(OUT / "roc_points.csv", index=False, encoding="utf-8-sig")
rows = []
for threshold in [0.5, 0.3]:
    # TODO H02: 현재 임계값보다 점수가 클 때만 양성1로 바꾸세요.
    pred = np.zeros(len(y), dtype=int)  # 모든 데이터를 성공(0)으로 초기화
    pred[score > threshold] = 1  # 점수가 현재 임계값보다 클 때만 실패(1)로 변경
    # TODO 끝
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, pred),
                 "cost": fp+5*fn, "AUC": roc_auc_score(y, score)})
result = pd.DataFrame(rows)
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("ROC 첫 임계값 inf는 어떤 값도 양성으로 정하지 않는 시작점입니다. 입력 오류가 아닙니다.")
print("이 원리 예제는 최종 분류기의 임계값을 변경하는 실험이 아닙니다.")
