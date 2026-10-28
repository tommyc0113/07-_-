"""실습 02 응용: 가상 확률로 임계값만 바꿉니다. 최종 모델의 임계값을 탐색하는 코드가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
y = np.array([0, 0, 1, 1])  # 0은 성공, 1은 실패
score = np.array([0.10, 0.40, 0.35, 0.80])  # 각 데이터가 실패일 가능성을 나타내는 가상의 확률
rows = []  # 임계값별 평가 결과를 저장할 빈 리스트

# 임계값 0.5와 0.3을 각각 적용하여 비교
for threshold in [0.5, 0.3]:
    # TODO A02: 확률이 threshold보다 클 때만 실패1로 바꾸세요. 같은 값은 성공0입니다.
    prediction = np.zeros(len(y), dtype=int)  # 모든 예측값을 0으로 초기화
    prediction[score>threshold] = 1 # 실패확률이 임계값보다 클 때만 1로 예측
    # TODO 끝
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    
    # 현재 임계값의 평가 지표와 가정 비용을 딕셔너리로 저장
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, prediction),
                 "assumed_cost": fp + 5*fn, "AUC": roc_auc_score(y, score)})
result = pd.DataFrame(rows)
print(result.round(3).to_string(index=False))
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
