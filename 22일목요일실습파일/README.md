# 2026.10.22 Day4 실습 · 수강생용

**실습 01~08을 기본필수·응용·심화의 세 단계로 제공합니다.** 수업 중에는 기본필수를 우선하고, 응용·심화는 선택합니다. 강사용은 완성된 정답 코드와 설명 주석, 수강생용은 함수의 입력·출력 규칙과 TODO가 있는 미완성 코드입니다.

## 1. 시작하기
압축을 완전히 해제한 뒤 **현재 역할 폴더를 VS Code로 여세요.** `PPT_실습_대응표.md`에서 실습을 찾고 해당 난이도의 README를 읽습니다. 이전 날짜의 ZIP은 필요하지 않습니다. 다음 실습에 필요한 정제 데이터와 이전 단원에서 재사용할 코드를 함께 제공합니다.

기본필수에서 막혔다는 이유로 다음 실습을 중단할 필요는 없습니다. 실습 03~05에는 정제 데이터 체크포인트가 있습니다. 단, **07과 08은 같은 프로젝트를 계속 사용**합니다.

## 2. 개발환경 준비
Python 3.13 가상환경이 있으면 재사용할 수 있습니다. 새로 준비하는 경우 아래 명령을 현재 역할 폴더에서 실행합니다.

**Windows PowerShell**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/check_environment.py
```
`py`가 없다면 설치된 Python 버전을 확인하고 `python -m venv .venv`를 사용합니다. VS Code의 Python 인터프리터도 이 폴더의 `.venv`로 선택하세요. 실행기를 직접 호출하므로 PowerShell 실행 정책을 변경할 필요가 없습니다.

**macOS / Linux**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/check_environment.py
```
`requirements.txt`는 검수한 NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8을 고정합니다. 후속 머신러닝 수업의 버전과 임의로 혼합하지 마세요. 패키지 설치에는 네트워크가 필요하지만 설치 후 데이터 분석과 Git 실습은 로컬에서 수행합니다. GPU·LLM 호출·유료 API는 사용하지 않습니다.

## 3. 실행과 검사
아래 `python`은 선택한 가상환경의 실행기를 뜻합니다. 가상환경을 활성화하지 않았다면 Windows는 `.\.venv\Scripts\python.exe`, macOS/Linux는 `.venv/bin/python`으로 바꿔 실행하세요.

```text
python tools/run.py --exercise 01 --level basic
python tools/check.py --exercise 01 --level basic
python tools/run.py --exercise 04 --level applied
python tools/check.py --all --level basic --report reports/basic_checks.json
python tools/check.py --all --syntax-only
```
난이도 값은 `basic`(기본필수), `applied`(응용), `advanced`(심화)입니다. `NotImplementedError`는 직접 완성할 TODO를 표시합니다. **문법 검사 통과와 기능 구현 완료는 다릅니다.**

자동 검사는 임시 복사본에서 실행하므로 본인의 원본과 `out` 결과를 덮어쓰지 않습니다. `--all`에서 Git 활동은 `MANUAL`로 기록하며 실제 저장소는 `tools/check_git.py`로 별도 확인합니다.

## 4. 파일의 역할
| 파일·폴더 | 용도 |
|---|---|
| 난이도별 `README.md` | 목표, 구현 위치, 실행 방법, 완료 기준 |
| Python 소스 | 강사용 정답 또는 수강생용 TODO |
| `data/agent_runs.csv` | PPT와 같은 합성 로그 원본 13행 |
| `data/fixtures/` | 빈 파일, 결측, ID 충돌, 손상 JSON 등 테스트 입력 |
| `data/clean_runs.csv`, `data/summary.json` | 앞 실습이 미완성이어도 사용할 수 있는 체크포인트 |
| `expected_output.txt`, `expected_outputs/` | 정답을 실제 실행해 만든 비교용 출력·파일 |
| `checks.py` | 해당 단계의 공개 검사기를 실행하는 도구 |
| 테스트·해석·보고서 양식 | 예상과 실제 결과, 수정 이유를 기록할 문서 |

정답 프로그램은 `expected_outputs`를 읽어 결과를 복사하지 않습니다. 학습자가 생성한 결과는 `out`에 저장됩니다. 실습 06은 Git 활동이므로 코드 실행 결과 대신 명령과 상태를 기록합니다. 그래프 축은 영문, 안내와 설명 주석은 한국어입니다.

## 5. 공통 데이터와 정답 기준
원본 13행 중 완전히 같은 R03 한 행만 제거하면 **정제 12행**입니다. latency는 유효 9건·결측 3건, tokens는 유효 11건·결측 1건, 날짜 결측은 1건입니다. 성공 9건과 실패 3건을 모두 보존합니다.

역할별 평균은 planner 120ms, researcher 약 566.67ms, writer 약 266.67ms이며 성공률은 각각 75%, 50%, 100%입니다. 평균의 분모와 성공률의 분모는 다릅니다. **0은 유효한 관측이며 결측과 구분합니다.** 확장 분석의 가상 비용 0.436은 측정된 11건의 부분합으로, 실제 LLM 가격이 아닙니다.

## 6. Git → 프로젝트 통합 → 제출
실습 06에서 선택한 `ex07_project/선택난이도/day4_eda`를 로컬 저장소로 사용합니다. 실습 07에서 기능을 연결하고 실습 08에서 **동일한 폴더**의 결과 검증·보고서·Git 기록을 완성합니다. 중간에 프로젝트를 새로 복사하여 개발 이력을 잃지 마세요.

```text
python tools/run.py --exercise 07 --level basic
python tools/run.py --exercise 08 --level basic
python tools/check_git.py --project "ex07_project/01_기본필수/day4_eda" --level basic
```
실습 08의 `run`은 본인의 제출 결과를 읽기만 합니다. `check`는 작성한 검증 함수 자체를 테스트합니다. 보고서 작성, 동료 리뷰, 실제 Git 활동의 완료 여부와는 구분합니다. 제출 경로와 방식은 강사가 현장에서 안내합니다.

## 7. 평가와 개인정보
기본필수만으로 공통 100점 기준을 충족할 수 있습니다. 자동 검사 통과가 100점 자동 부여를 뜻하지는 않습니다. `평가_루브릭.md`와 `코드리뷰_체크리스트.md`로 설명·해석·기여를 함께 확인합니다.

만족도는 이름 없이 별도로 수집하며 개인 제출 ZIP에는 넣지 않습니다. 가상환경, 캐시, 실제 비밀키와 개인정보는 제출하지 마세요. Git 저장소를 공유할 때는 커밋 작성자 정보도 확인합니다.

## 8. 검수 범위
검수 환경은 Python 3.13.5·NumPy 2.3.5·pandas 2.2.3·Matplotlib 3.10.8·Linux입니다. **Windows 및 macOS에서 직접 실행한 검증은 하지 않았습니다.** 상세 실행 기록은 강사용의 `검수_결과.md`와 `reports/`에 있습니다. PPT는 이 ZIP에 중복 포함하지 않았습니다.
