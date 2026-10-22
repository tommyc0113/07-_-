# 실습 03 · 집계·피벗·결합 · 응용

**PPT 33쪽 / 25분 / 수강생용**

## 목표·입력
data/clean_runs.csv와 summary.json은 이전 실습의 정답 실행으로 만든 시작점이다. 정제 정답 소스가 아니라 데이터 체크포인트이다.

## 완료 기준
역할별 runs4/measured3. 평균120/566.6667/266.6667, 성공률0.75/0.5/1. 결합12행. 가상 비용0.436은 측정11건의 부분합.

## 진행
기본필수부터 수행하고 선택 확장합니다. 같은 단계의 제공 코드는 경로·읽기·이전 단원 재사용을 위한 것으로, 직접 완성할 부분은 아래 TODO입니다. 난이도별로 다른 독립 시작점을 제공하므로 앞 과제의 미완성 때문에 다음 과제가 막히지 않습니다. 07과 08은 같은 프로젝트를 계속 사용합니다.

## 구현 위치와 힌트
- `A_PIVOT`: agent×status 평균 pivot_table(observed=True)을 만들고 역할/상태 열을 reindex하세요. NaN은 0으로 채우지 않습니다.
- `A_JOIN`: left merge(validate="many_to_one", indicator=True). 행 수와 _merge를 검사하고 미연결 시 ValueError. cost_unit을 계산하세요.

## 함수 계약
함수 위 docstring의 매개변수·반환값·오류 조건을 따릅니다. 반환 형식을 바꾸기보다 계산을 채웁니다. `NotImplementedError`는 의도적인 미완성 표시이며 문법 오류가 아닙니다. 정답의 결과는 expected_output.txt/expected_outputs에 제공하되 검사 기준 자체를 수정하지 않습니다.

## 실행·검사 (역할 폴더에서)
```text
python tools/run.py --exercise 03 --level applied
python tools/check.py --exercise 03 --level applied
```
직접 실행 파일: `ex03_aggregate.py`. 실행 위치가 달라도 파일 위치를 기준으로 data/out을 찾습니다. 검사는 임시 복사본에서 이루어져 본인의 결과를 바꾸지 않습니다.

## 확장 반환값
cost_audit는 cost_unit, measured, runs, is_partial, unit 사전입니다. 모두결측은 cost_unit=NaN, measured=0입니다. weighted_mean은 같은 단위의 중복없는 집단에 대해서만 사용합니다.
