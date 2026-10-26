"""FP·FN 사례와 제한된 C 실험 | PPT 34 · 85쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
from pathlib import Path
import sys
# 터미널의 현재 폴더와 무관하게 이 실습의 배포 루트를 찾습니다.
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'course_support.py').is_file())
sys.path.insert(0,str(ROOT))
import argparse
import numpy as np
import pandas as pd
from course_support import (NUM,CAT,FEATURES,DATA,load_data,load_checkpoint,
    make_pre,build,reg_metrics,cls_metrics,snapshot,emit,read_table)
from sklearn.linear_model import LogisticRegression

def error_examples(X,y,pred,prob):
    """FP와 FN을 ID 오름차순으로 각각 1건 선택해 입력·실제·예측·실패확률을 포함한 표 반환.
    오류가 없는 유형은 행을 만들지 않는다. index를 섞거나 valid 행을 삭제하지 않는다.
    """
    # TODO D6-03-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-03-H1: 구현 후 다시 실행하세요.")

def compare_C(d,candidates):
    """C만 바꾼 모든 후보의 valid F1·FP·FN 기록. C 오름차순을 동점 규칙으로 사용한다."""
    # TODO D6-03-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-03-H2: 구현 후 다시 실행하세요.")

def solve():
    d=load_data(); m=build(LogisticRegression(C=1.0,max_iter=1000)).fit(d.X_train,d.yc_train)
    prob=m.predict_proba(d.X_valid)[:,list(m['model'].classes_).index(1)]
    table=compare_C(d,[0.1,1.0,10.0])
    return {'error_cases':error_examples(d.X_valid,d.yc_valid,m.predict(d.X_valid),prob),
            'C_comparison':table,'selected_C':float(table.sort_values(['F1','C'],ascending=[False,True]).iloc[0]['C']),
            'fixed':'동일 분할·standard·max_iter=1000; 사전 F1 최대·동점 작은 C'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    args=parser.parse_args()
    # 교육 환경의 CPU 수에 따른 과도한 병렬화를 줄입니다. 계산 정의는 바뀌지 않습니다.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result=solve()
    emit(result,args.output)

if __name__=='__main__':
    main()
