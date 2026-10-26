"""지표 선택과 train-only 이상치 정책 | PPT 18~23쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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
from sklearn.linear_model import LinearRegression

def iqr_mask(y_train):
    """train y의 Q1,Q3와 1.5*IQR로 경계를 정한다. (keep bool 배열, [lower,upper]) 반환.
    valid y를 인자로 받지 않으므로 valid 결과를 본 뒤 행을 지울 수 없다.
    """
    # TODO D6-02-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-02-H1: 구현 후 다시 실행하세요.")

def compare_policy(X_train,y_train,X_valid,y_valid):
    """원래 train/사전 IQR 정책 train으로 학습. 동일 valid에 대한 결과 표를 반환한다.
    model: all_train, train_iqr. n_train, MAE_ms, RMSE_ms, R2도 기록.
    """
    # TODO D6-02-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-02-H2: 구현 후 다시 실행하세요.")

def solve():
    # 별도 원리 예제입니다. 공통 100건 데이터에서 이상치를 삭제하지 않습니다.
    X=np.arange(1,7).reshape(-1,1); y=np.array([100,200,300,400,500,5000])
    Xv=np.array([[7],[8],[9]]); yv=np.array([700,800,900])
    table,bounds=compare_policy(X,y,Xv,yv)
    custom={name:reg_metrics([100,200,300,400],p) for name,p in
            {'A':[100,200,300,560],'B':[150,250,350,450]}.items()}
    return {'policy_comparison':table,'train_iqr_bounds':bounds,'custom_metric_tradeoff':custom,
            'valid_kept':len(Xv),'note':'개선은 이 교육용 자료의 결과이며 이상치 자동삭제를 일반화하지 않습니다.'}


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
