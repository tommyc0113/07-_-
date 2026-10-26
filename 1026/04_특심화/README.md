# Day6 실습 — 회귀·분류 모델 비교

**강의일: 2026년 10월 26일(월) / 머신러닝 모델링 및 성능평가 2/4**
대응 PPT: `20261026월_머신러닝_회귀_분류_KNN_모델비교_Day6.pptx`.
실습은 01~08, 각 기본필수·응용·심화로 구성됩니다. 단계는 사람의 등급이 아니라 과제의 깊이입니다.
기본필수만으로 공통 100점 기준을 충족할 수 있으며, 08은 새로운 프로젝트가 아니라 07의 검증·설명 단계입니다.

## 1. 설치 및 첫 실행

압축은 ZIP 안에서 실행하지 말고 **전체 해제**합니다. VS Code에서 현재 역할 폴더를 열고 터미널 위치를 확인합니다.
이미 준비된 Python 3.13 가상환경이 있다면 해당 인터프리터를 사용합니다. 수업 직전 무조건 최신 버전으로 올리지 않습니다.

### Windows PowerShell — 활성화 없이 실행하는 방법

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe check_env.py
.\.venv\Scripts\python.exe run.py 01 basic
.\.venv\Scripts\python.exe tools/check.py --exercise 01 --level basic
```

VS Code의 `Python: Select Interpreter`에서 `.venv`를 선택합니다. `py -3.13`이 없다면 Python 3.13 설치 여부를 확인합니다.
활성화 정책에 막혔을 때 운영체제의 보안 정책을 변경할 필요 없이 위의 인터프리터 직접 경로를 사용합니다.

### macOS / Linux

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python check_env.py
.venv/bin/python run.py 01 basic
```

이후 안내의 `python`은 선택한 가상환경의 Python을 뜻합니다. 설치에는 인터넷이 필요하지만 **설치 후 수업은 오프라인으로 실행**됩니다.
LLM API, GPU, API 키, 외부 데이터 다운로드, GitHub 계정이 필요하지 않습니다.
검증 환경은 Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0입니다.
Windows/macOS 안내는 제공하지만 실제 실행 검증은 Linux에서 수행했습니다.

## 2. 8교시의 실습 진행

| 실습 | 시간 | 주제 | 기본필수 완료 기준 |
|---|---:|---|---|
| 01 | 20분 | 선형회귀 | X·y 형태, 계수 9.3·절편 42.3·예측 98.1 설명 |
| 02 | 25분 | 회귀 지표 | MAE·RMSE의 단위, A/B 선택 차이, 공통 valid 결과 |
| 03 | 20분 | 로지스틱회귀 | 확률과 클래스, TN/FP/FN/TP=10/4/4/2 |
| 04 | 25분 | KNN | 이웃 4·2·1과 거리, 분류 0·회귀 23.33 |
| 05 | 25분 | 스케일링 | 수치/범주 경로, (60,8)·(20,8), train 통계 |
| 06 | 25분 | K 후보 비교 | 1·3·5·9·15를 같은 valid에서 비교, 사전 선택 규칙 |
| 07 | 25분 | 비교 프로젝트 | 두 비교표, K 탐색 기록, valid 예측 저장 |
| 08 | 35분 | 검증·설명 | 재실행, 오류 사례, 보고서, Git 변경 이력 |

설명·시연 200분 + 실습·리뷰 200분. 응용·심화는 기본필수 뒤 선택하며 같은 시간 안에 3단계를 모두 완료할 필요는 없습니다.
페어 실습에서는 작성자와 검토자를 교대하고 마지막 선택 근거는 각자 설명합니다.

## 3. 실행 및 검사

```bash
python run.py --list
python run.py 03 basic
python run.py 06 applied
python tools/check.py --exercise 06 --level applied
python tools/check.py --syntax
python run.py 07 basic
python run.py 08 basic
```

`run.py`는 경로 선택을 돕는 실행기입니다. 각 `main.py`도 직접 실행할 수 있습니다.
`tools/check.py`는 기능을 확인하는 검사 도구이며 보고서 문장·학습 태도·Git 기여를 자동 채점하지 않습니다.
수강생 코드는 완성 전 검사에서 실패하는 것이 정상입니다. 실패 이름과 TODO ID부터 확인합니다.
모든 함수를 구현한 뒤 `python tools/check.py`로 전체 자동 검사를 수행합니다.

| 폴더/파일 | 용도 |
|---|---|
| `ex01_*` ~ `ex06_*` | 독립 실습 코드, 단계별 README·학습기록·예상 결과 |
| `ex07_compare/<단계>/day6_compare` | 단독 복사해도 실행 가능한 최종 프로젝트 |
| `ex08_validate/<단계>` | 동일 프로젝트 검증 실행기와 제출 활동 |
| `data/requests.csv` | Day5와 바이트 단위로 같은 합성 요청 100건 |
| `data/split_manifest.csv` | Day5 원본 분할 소속 |
| `data/split_order.csv` | 분할 내부 행 순서: PPT 확률 예시와 KNN 재현에 사용 |
| `data/checkpoints` | 앞 실습 없이 사용할 수 있는 미전처리 train·valid와 K 탐색 기록 |
| `data/fixtures` | 새 범주·누락 열·비정상 입력 검사 자료 |
| `course_support.py` | Day5 전처리·경로·파일 출력·재사용 지표 등 제공 코드 |
| `tools/check.py` | 선택 실습/단계의 자동 확인 |

## 4. 데이터와 평가 정책

입력은 `input_chars`, `allowed_tools`, `task_type`, `priority` 네 열입니다.
`elapsed_ms`는 회귀 정답, `failed`는 분류 정답입니다. 둘 다 입력에 넣지 않습니다.
`request_id`는 행 정렬용 식별자이고 `status`, `retry_count`는 실행 후 정보라 입력에서 제외합니다.
train 60건에서만 전처리와 모델을 학습하고 valid 20건으로 비교합니다. test 20건은 최종 평가까지 보관합니다.
`load_data()`는 test의 특성·정답을 모델 함수에 반환하지 않습니다. 전체 파일 읽기는 ID 무결성 확인을 포함한 준비 과정이지 test 모델 평가가 아닙니다.

작은 원리 예제(E01~E04·E10~E14)와 공통 합성 요청 100건을 구분합니다.
원리 예제의 손계산 결과를 공통 데이터 모델의 일반화 성능이라고 부르지 않습니다.
KNN의 거리 동률 때문에 원본뿐 아니라 **train 행 순서도 유지**해야 합니다.
K 실험은 standard·uniform·p=2로 고정합니다. 스케일링 실험에서 가장 좋았던 설정으로 몰래 바꾸지 않습니다.

## 5. 미완성 코드와 예상 결과 사용법

코드의 `TODO D6-...`는 직접 구현할 부분입니다. 함수 위 문서 문자열에 입력·반환값·힌트가 있습니다.
강사용의 동일 ID는 정답 해설입니다. 수강생에게는 강사용 폴더를 배포하지 않습니다.
`expected_output.txt`와 `expected_outputs`는 강사용 코드를 실제 실행해 만든 **대조용** 결과입니다.
이를 그대로 복사하거나 코드에서 읽어 정답을 출력하는 대신 함수의 계산을 구현합니다.
오차 허용 범위가 있는 실수는 반올림 전 값을 비교하고, 화면의 자리수 차이만으로 실패라 판단하지 않습니다.

## 6. 최종 산출물

`day6_compare/outputs/`의 `regression_comparison.csv`, `classification_comparison.csv`,
`knn_search.csv`, `validation_predictions.csv`와 `reports/model_choice.md`, 실행 코드·README·환경 기록·Git 이력을 제출합니다.
기본필수의 K 탐색 기록은 실습06 결과 또는 명시된 제공 체크포인트를 재사용해도 됩니다.
응용·심화는 K 탐색 재실행과 직접 생성한 Pipeline 저장·복원을 연결합니다.
ZIP에는 미리 학습한 `.joblib` 파일을 넣지 않았습니다. 필요할 때 본인 코드로 생성합니다.
시험용 test 점수 열을 만들지 않습니다. 미평가는 누락이 아니라 오늘의 정책입니다.

상세 기준은 `평가_기준.md`, 환경 문제는 `문제해결_안내.md`, 출처는 `공식_참고문서.md`를 참고합니다.
