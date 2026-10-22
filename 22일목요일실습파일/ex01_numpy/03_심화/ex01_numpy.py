"""실습 01: axis, broadcasting, 결측과 분모가 0인 비율."""
import numpy as np


def axis_means(a):
    """비어 있지 않은 2차원 유한 배열 -> (열 평균, 행 평균). 원본은 변경하지 않습니다."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or 0 in a.shape or not np.isfinite(a).all():
        raise ValueError("평균 입력은 비어 있지 않은 2차원 유한 배열이어야 합니다.")
    # 해설 N_AXIS: axis=0/axis=1 평균을 순서대로 반환하세요. 결과 shape는 열 수/행 수입니다.
    return a.mean(axis=0), a.mean(axis=1)


def calibrate(a, column_scale, row_scale):
    """(행,열) 배열과 (열,)·(행,) 계수 -> 각각 독립된 보정 결과 두 개."""
    a = np.asarray(a, dtype=float)
    col, row = np.asarray(column_scale, dtype=float), np.asarray(row_scale, dtype=float)
    # TODO N_SHAPE: a는 2차원, col은 (열 수,), row는 (행 수,)인지 검사하고 아니면 ValueError. 유한한 값만 허용합니다.
    raise NotImplementedError("TODO N_SHAPE")
    # 해설 N_SCALE: 역할별 a*col, 묶음별 a*row[:, None] 결과를 튜플로 반환하세요.
    return a * col, a * row[:, None]


def valid_mean(latency):
    """1차원 측정 -> (정제 배열, 유효 수, 평균). 전체 결측이면 평균 NaN."""
    a = np.asarray(latency, dtype=float)
    if a.ndim != 1:
        raise ValueError("측정값은 1차원 배열이어야 합니다.")
    # TODO N_VALID: isfinite와 >=0 조건을 &로 결합. 잘못된 값은 NaN, 유효 수를 세고 0건이면 평균도 NaN으로 반환하세요.
    raise NotImplementedError("TODO N_VALID")


def safe_rate(ok, total):
    """같은 길이의 1차원 횟수 -> 비율. 0/0은 NaN, 잘못된 횟수는 ValueError."""
    ok, total = np.asarray(ok, dtype=float), np.asarray(total, dtype=float)
    if ok.ndim != 1 or ok.shape != total.shape:
        raise ValueError("성공/전체 횟수는 길이가 같은 1차원 배열이어야 합니다.")
    # 유한성을 먼저 검사합니다. NaN/무한대에 나머지 연산을 수행하지 않습니다.
    if not np.isfinite(ok).all() or not np.isfinite(total).all():
        raise ValueError("횟수는 유한한 정수여야 합니다.")
    if ((ok < 0) | (total < 0) | (ok > total)
        | (ok % 1 != 0) | (total % 1 != 0)).any():
        raise ValueError("횟수는 정수이고 0 <= 성공 <= 전체여야 합니다.")
    # TODO N_RATE: NaN float 배열을 만든 뒤 np.divide(..., out=..., where=total>0)로 계산해 반환하세요.
    raise NotImplementedError("TODO N_RATE")


# 실행부는 제공 코드입니다. 핵심 함수의 TODO를 먼저 완성하세요.
from pathlib import Path
from support import write_json, json_safe
import json
BASE_DIR = Path(__file__).resolve().parent

def main():
    (BASE_DIR/"out").mkdir(exist_ok=True)
    a=np.array([[100,300,200],[120,500,250]])
    col,row=axis_means(a)
    result={"shape":list(a.shape),"column_means":col.tolist(),"row_means":row.tolist()}
    c,r=calibrate(a,[1.,.8,1.1],[1.,.5])
    clean,n,mean=valid_mean([100,-5,np.nan,300])
    result.update(column_adjusted=c.tolist(),row_adjusted=r.tolist(),valid_count=n,valid_mean=mean,cleaned=clean.tolist())
    result["success_rates"]=safe_rate([3,0,2],[4,0,3]).tolist()
    result["all_missing_mean"]=valid_mean([np.nan,np.inf,-1])[2]
    write_json(BASE_DIR/"out/result.json",result)
    print(json.dumps(json_safe(result),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
