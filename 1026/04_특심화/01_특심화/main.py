"""이상치 민감도와 외삽 | PPT 13~14 · 40쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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

def sensitivity(X,original_y,changed_y,queries):
    """original/changed 두 모델의 coef·intercept·queries 예측을 dict로 반환한다.
    원본 y를 덮어쓰지 않는다. 두 모델은 새로운 객체여야 한다.
    """
    # TODO D6-01-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-01-H1: 구현 후 다시 실행하세요.")

def extrapolation_mask(X,queries):
    """각 query가 하나라도 train의 열별 최소·최대 밖이면 True인 (m,) bool 배열 반환."""
    # TODO D6-01-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-01-H2: 구현 후 다시 실행하세요.")

def solve():
    X=np.array([[1],[2],[3],[4],[5]]); y=np.array([52,61,69,80,89])
    changed=y.copy(); changed[-1]=200  # 교육용으로 한 관측을 의도적으로 변경
    queries=np.array([[3],[6],[10]])
    return {'models':sensitivity(X,y,changed,queries),
            'queries':queries.ravel(),'extrapolation':extrapolation_mask(X,queries),
            'original_y_preserved':y,'warning':'이 작은 적합 예제는 일반화 평가가 아닙니다.'}


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
