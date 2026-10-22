# 실습 07 · EDA 프로젝트 통합 · 기본필수

**PPT 72쪽 / 25분 / 강사용**

## 목표·입력
day4_eda/data/agent_runs.csv가 유일한 기본 분석 원본. 응용부터 agents_meta.json 결합. 원본파일을 변경하지 않는다.

## 완료 기준
한 번 실행해 정제 12행·요약 3행·PNG2개(응용3/심화4)·왕복확인을 생성. 전체평균317.7778/중앙값250. reports는 본인이 작성.

## 진행
기본필수부터 수행하고 선택 확장합니다. 같은 단계의 제공 코드는 경로·읽기·이전 단원 재사용을 위한 것으로, 직접 완성할 부분은 아래 TODO입니다. 난이도별로 다른 독립 시작점을 제공하므로 앞 과제의 미완성 때문에 다음 과제가 막히지 않습니다. 07과 08은 같은 프로젝트를 계속 사용합니다.

## 구현 위치와 힌트
- `PIPE_CORE`: read_raw -> clean_runs -> aggregate_runs -> quality_report로 연결해 raw/clean/summary/quality를 만드세요.
- `PIPE_FILES`: stage에 save_tables 후 read_back/verify_roundtrip을 호출해 roundtrip 사전을 만드세요.
- `PIPE_CHARTS`: stage/charts에 plot_bar/plot_hist를 호출해 chart_audit={"bar":..., "hist":...}를 만드세요.

## 함수 계약
함수 위 docstring의 매개변수·반환값·오류 조건을 따릅니다. 반환 형식을 바꾸기보다 계산을 채웁니다. `NotImplementedError`는 의도적인 미완성 표시이며 문법 오류가 아닙니다. 정답의 결과는 expected_output.txt/expected_outputs에 제공하되 검사 기준 자체를 수정하지 않습니다.

## 실행·검사 (역할 폴더에서)
```text
python tools/run.py --exercise 07 --level basic
python tools/check.py --exercise 07 --level basic
```
직접 실행 파일: `day4_eda/main.py`. 실행 위치가 달라도 파일 위치를 기준으로 data/out을 찾습니다. 검사는 임시 복사본에서 이루어져 본인의 결과를 바꾸지 않습니다.

## 25분에 끝내는 순서
5분: 함수입출력 확인 → 12분: 연결 TODO 완성 → 5분: 기준값/파일 확인 → 3분: 보고서초안. 모든 함수의 새작성보다 이미 배운 함수 재사용에 집중합니다. 08에서35분간검증·보고·Git정리를 계속합니다.
