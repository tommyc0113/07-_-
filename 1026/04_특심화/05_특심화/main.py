"""거리 기여와 전체 배열 표준화 | PPT 47~52쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def distance_components(a,b,n_numeric=2):
    """변환된 두 행의 차이를 제곱해 수치/범주/전체 제곱거리 dict로 반환한다.
    key: numeric_squared, categorical_squared, total_squared.
    전체는 두 성분의 합이며 원래 유클리드 거리는 그 제곱근이다.
    """
    # TODO D6-05-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-05-H1: 구현 후 다시 실행하세요.")

def compare_whole_scaling(d):
    """기존 수치-only와 전체 배열(원핫 포함) 표준화를 별도 실험명으로 비교한다.
    반환: experiment, valid_F1 표. 전체 배열 경로는 make_pre('none')→StandardScaler→KNN5.
    """
    # TODO D6-05-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-05-H2: 구현 후 다시 실행하세요.")

def solve():
    d=load_data(); pre=make_pre().fit(d.X_train)
    train_z=pre.transform(d.X_train); valid_z=pre.transform(d.X_valid)
    return {'distance_contribution':distance_components(train_z[0],valid_z[0]),
            'id_pair':[d.X_train.index[0],d.X_valid.index[0]],
            'alternative_scaling':compare_whole_scaling(d),
            'note':'범주 원핫의 거리 가중도 바뀌는 별도 실험. 기존 none/standard/minmax 비교와 섞지 않음'}


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
