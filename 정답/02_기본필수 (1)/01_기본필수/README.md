# 실습 02 · 데이터 정제와 품질 기록 · 기본필수

**PPT 23쪽 / 25분 / 강사용**

## 목표·입력
data/agent_runs.csv: PPT 16~17쪽과 같은 합성 로그 13행. R03은 완전 중복.

## 완료 기준
13→12행; latency 유효 9/결측 3(R08·R09·R10); tokens 결측 1; 날짜 결측 1(R11); ok9/error3.

## 진행
기본필수부터 수행하고 선택 확장합니다. 같은 단계의 제공 코드는 경로·읽기·이전 단원 재사용을 위한 것으로, 직접 완성할 부분은 아래 TODO입니다. 난이도별로 다른 독립 시작점을 제공하므로 앞 과제의 미완성 때문에 다음 과제가 막히지 않습니다. 07과 08은 같은 프로젝트를 계속 사용합니다.

## 구현 위치와 힌트
- `C_COPY`: 모든 열이 같은 행만 제거해 복사하세요. ID 공백을 제거하고 agent/status는 공백 제거 후 소문자로 통일하세요.
- `C_NUMERIC`: latency_ms/tokens를 to_numeric(errors="coerce")로 변환. 음수·무한대는 NaN. strict_tokens이면 분수 토큰도 결측 처리하세요.

## 함수 계약
함수 위 docstring의 매개변수·반환값·오류 조건을 따릅니다. 반환 형식을 바꾸기보다 계산을 채웁니다. `NotImplementedError`는 의도적인 미완성 표시이며 문법 오류가 아닙니다. 정답의 결과는 expected_output.txt/expected_outputs에 제공하되 검사 기준 자체를 수정하지 않습니다.

## 실행·검사 (역할 폴더에서)
```text
python tools/run.py --exercise 02 --level basic
python tools/check.py --exercise 02 --level basic
```
직접 실행 파일: `ex02_clean.py`. 실행 위치가 달라도 파일 위치를 기준으로 data/out을 찾습니다. 검사는 임시 복사본에서 이루어져 본인의 결과를 바꾸지 않습니다.

## quality_report 사전 키
raw_rows, clean_rows, exact_duplicates_removed, unique_ids, latency_valid, latency_missing, latency_missing_ids, tokens_valid, tokens_missing, day_missing, day_missing_ids, ok, error, policy.
기본필수에서는 day를 원래 문자열로 두지만 quality_report는 날짜 유효성도 계산해 제공합니다. 응용에서 날짜 변환과 보고서 함수를 직접 구현합니다. audit_contract는 issue/run_id/source_row 표이며 문제없으면 해당 열을 가진 0행 표입니다.
