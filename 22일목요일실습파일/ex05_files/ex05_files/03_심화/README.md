# 실습 05 · CSV·JSON 왕복 저장과 확인 · 심화

**PPT 52쪽 / 25분 / 수강생용**

## 목표·입력
정제 12행·요약 3행 체크포인트와 중첩 agents_meta.json. leading_zero_ids.csv는 독립된 2행 예제다.

## 완료 기준
clean_runs.csv 12행, summary.json 3행, latency 결측 3. ID 001/002 문자열 보존. dates는 저장 시 문자열, 재분석 전 datetime 복원.

## 진행
기본필수부터 수행하고 선택 확장합니다. 같은 단계의 제공 코드는 경로·읽기·이전 단원 재사용을 위한 것으로, 직접 완성할 부분은 아래 TODO입니다. 난이도별로 다른 독립 시작점을 제공하므로 앞 과제의 미완성 때문에 다음 과제가 막히지 않습니다. 07과 08은 같은 프로젝트를 계속 사용합니다.

## 구현 위치와 힌트
- `I_VERIFY`: 행 수·ID·문자열·숫자/결측·날짜·요약 값을 저장 전후 대조하세요. 다르면 ValueError, 같으면 csv_rows/json_rows/latency_missing/verified 사전을 반환하세요.

## 함수 계약
함수 위 docstring의 매개변수·반환값·오류 조건을 따릅니다. 반환 형식을 바꾸기보다 계산을 채웁니다. `NotImplementedError`는 의도적인 미완성 표시이며 문법 오류가 아닙니다. 정답의 결과는 expected_output.txt/expected_outputs에 제공하되 검사 기준 자체를 수정하지 않습니다.

## 실행·검사 (역할 폴더에서)
```text
python tools/run.py --exercise 05 --level advanced
python tools/check.py --exercise 05 --level advanced
```
직접 실행 파일: `ex05_files.py`. 실행 위치가 달라도 파일 위치를 기준으로 data/out을 찾습니다. 검사는 임시 복사본에서 이루어져 본인의 결과를 바꾸지 않습니다.

## 실패 정책
invalid_encoding.csv/empty.csv/header_only.csv/missing_columns.csv/broken.json은 의도된 실패 입력입니다. 정상 자료로 바꾸지 않습니다. tests가 임시 폴더에서 이 예외들을 확인합니다. `staged_output`은 제공 안전장치이며 학습의 핵심은 변환과 재읽기 검증입니다.
verify_roundtrip의 정상 반환: {"csv_rows":12,"json_rows":3,"latency_missing":3,"verified":True}.
