# 실습 01 — 이상치 민감도와 외삽 / 심화

PPT 13~14 · 40쪽 · 실습 시간 20분. 기본필수를 먼저 완료한 뒤 다음 단계를 선택합니다.

## 목표

원본을 복사한 뒤 마지막 정답89를200으로 바꾸어 민감도를 관찰합니다. 3·6·10시간 입력의 외삽 여부를 기록합니다.


## 진행 순서

먼저 예상 값을 학습기록에 적습니다. 해당 코드의 TODO 함수에 입력·반환값 계약을 확인하고 구현합니다.
한 번에 많은 설정을 바꾸지 말고 현재 단계의 고정 조건을 유지합니다.
실행 후 `expected_output.txt`와 `expected_outputs/`의 수치·shape를 대조하고 자동 검사를 실행합니다.
마지막에 학습기록의 해석 질문에 답합니다. 점수만 기록하고 끝내지 않습니다.

역할 폴더에서 실행합니다.

```bash
python run.py 01 advanced
python tools/check.py --exercise 01 --level advanced
```

직접 실행 파일: `ex01_linear/03_심화/main.py`. 실행 결과: `outputs/result.json`와 결과에 포함된 표의 CSV.
01~06을 실행해도 이전 단원 파일을 수정하거나 원본 데이터를 덮어쓰지 않습니다.

## 구현할 함수

### `main.py :: sensitivity()`

original/changed 두 모델의 coef·intercept·queries 예측을 dict로 반환한다.
원본 y를 덮어쓰지 않는다. 두 모델은 새로운 객체여야 한다.

### `main.py :: extrapolation_mask()`

각 query가 하나라도 train의 열별 최소·최대 밖이면 True인 (m,) bool 배열 반환.


## 완료 기준과 제출

README의 목표 수치·반환 형태를 확인하고 `학습기록.md`를 작성합니다.
코드·직접 생성한 결과·검사 로그가 서로 같은 실행에서 나온 것인지 확인합니다.
07~08은 `reports/model_choice.md`와 Git 변경 이력도 제출합니다. test 점수는 제출하지 않습니다.

## 자주 하는 실수

X의 형태/열 순서를 바꾸거나, X와 y를 따로 정렬하거나, valid로 fit하지 마세요.
실수 출력의 마지막 자리 차이는 허용 오차로 비교하고 학습용 예제와 검증용 자료를 구분하세요.
`NotImplementedError`는 TODO 미완성 표시입니다. `expected_outputs`를 코드의 입력으로 읽지 마세요.
