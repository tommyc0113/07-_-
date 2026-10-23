# 실습 07·08 · 전처리·분할 Pipeline 프로젝트

[심화 전체 안내](../README.md) · 수강생용

**목표:** 같은 100건을 같은 ID로 나누고, train에서만 학습하여 valid를 예측합니다. 실행·구성·데이터·결과·보고서를 구분하고 저장한 Pipeline을 새 Python 실행에서 복원합니다.

## 먼저 읽기

이 프로젝트의 직접 구현은 `main.py`의 **H07-1**과 `restore.py`의 **H08-1**, 총 두 곳입니다. `preprocessing.py`, 분할, CSV·JSON 저장, 모델 저장 코드는 제공합니다. 구조가 나뉘었다고 더 어려운 예외처리·테스트를 작성하지 않습니다.

## 전체 구조와 순서

```text
day5_ml/
├─ README.md
├─ main.py                       # 실습07: 읽기 → 분할 → 학습 → 예측 → 저장
├─ preprocessing.py              # 제공: build_pipeline() 한 개
├─ restore.py                    # 실습08: 직접 저장한 모델 복원 → 예측 대조
├─ requirements.txt
├─ data/
│  └─ requests.csv               # 원본 100건
├─ expected/                     # 비교용, 코드의 입력으로 쓰지 않음
│  ├─ main_output.txt
│  ├─ restore_output.txt
│  ├─ validation_predictions.csv
│  ├─ split_manifest.csv
│  └─ feature_names.json
├─ outputs/                      # 처음에는 비어 있음
│  ├─ validation_predictions.csv # main.py 실행 후 생성
│  ├─ split_manifest.csv
│  ├─ feature_names.json
│  └─ pipeline.joblib            # 직접 생성한 파일만 복원
└─ reports/
   └─ model_review.md            # 본인의 설명을 작성
```

**경로와 명령은 이 `day5_ml` 폴더의 터미널 기준**입니다. 기존 환경을 사용합니다. 이 프로젝트만 다른 위치에 복사해도 `data/`와 코드·모듈을 함께 유지하면 실행할 수 있습니다. 필요한 경우에만 `python -m pip install -r requirements.txt`로 검증 버전을 설치합니다.

```bash
python main.py
python restore.py
```

`restore.py`는 `main.py`의 결과를 읽으므로 순서를 바꾸지 않습니다. 기존 결과는 같은 이름으로 다시 생성하며 원본 CSV를 수정하지 않습니다.

## 제공 모듈은 무엇을 하나요?

`preprocessing.py`의 `build_pipeline()`은 **아직 학습하지 않은 새 Pipeline**을 반환합니다. 숫자 2열은 중앙값 대체→표준화, 문자 2열은 최빈값 대체→One-hot으로 처리하고 다수 클래스 기준선을 연결합니다. `fit`은 이 함수 안이 아니라 `main.py`에서 train 60건으로 호출합니다.

숫자 2 + 작업 종류 3 + 우선순위 3 = 변환 특성 8개입니다. valid와 새 입력에는 이 상태를 그대로 재사용합니다. test 20건은 오늘 모델에 전달하지 않습니다.

## 실습 07 · 역할별 파일로 나누어 저장하기

**목표:** 기존 응용의 최소 모듈 분리와 저장을 수행합니다.

**열 파일:** `main.py` · PPT 실습 안내 75쪽

### 진행 순서

1. `preprocessing.py`가 새 Pipeline을 만드는 제공 모듈인지 읽습니다.
2. `main.py`의 H07-1만 완성하고 실행합니다.
3. `outputs/`에 예측표·분할표·특성명·모델 파일이 생겼는지 확인합니다.

```bash
python main.py
```

### 직접 완성할 부분

| TODO | 수행할 일 |
|---|---|
| `H07-1` | 학습된 pipeline으로 valid를 예측하세요. |

**힌트:** `preprocessing.py`는 제공 모듈입니다. 새 함수·패키지·설정 파일을 만들 필요가 없습니다. `predict`에는 `valid[features]`를 전달합니다. 분할·저장 코드는 제공합니다.

### 예상 출력

완성한 강사용 코드를 실제 실행한 결과입니다. 표시 자리수나 표의 공백보다 값·행 수·열의 의미를 확인합니다.

```text
분할: 60 20 20
변환 특성 수: 8
valid 정확도: 0.7
실패 발견: 0 / 6
저장: outputs/validation_predictions.csv
저장: outputs/split_manifest.csv, outputs/feature_names.json
저장: outputs/pipeline.joblib
08교시: restore.py 실행 후 reports/model_review.md를 작성합니다.
```

**설명 질문:** 왜 전처리와 모델을 함께 저장하나요?

**완료 기준:** 지정한 TODO를 실제 연산·메서드 호출로 완성하고, 예상 결과를 재현한 뒤 설명 질문에 답합니다. 출력 숫자를 직접 넣어 결과만 맞추지 않습니다.


---

## 실습 08 · 새 실행에서 모델 복원

**목표:** 기존 응용의 새 프로세스 복원과 재현을 확인합니다.

**열 파일:** `restore.py` · PPT 실습 안내 78쪽

### 진행 순서

1. 완성한 `main.py`를 먼저 실행합니다.
2. `restore.py`의 H08-1을 완성하고 별도 명령으로 실행합니다.
3. ID 순서·예측이 모두 동일한지 확인하고 보고서에 기록합니다.

```bash
python restore.py
```

### 직접 완성할 부분

| TODO | 수행할 일 |
|---|---|
| `H08-1` | 복원한 restored로 valid 입력을 예측하세요. fit은 호출하지 마세요. |

**힌트:** restored는 joblib.load로 불러온 학습된 Pipeline입니다.

### 예상 출력

완성한 강사용 코드를 실제 실행한 결과입니다. 표시 자리수나 표의 공백보다 값·행 수·열의 의미를 확인합니다.

```text
요청 ID 순서 동일: True
저장 전후 예측 동일: True
valid 예측 건수: 20
복원한 모델에 fit을 다시 호출하지 않았습니다.
```

**설명 질문:** 복원 뒤 fit을 다시 해야 하나요?

**완료 기준:** 지정한 TODO를 실제 연산·메서드 호출로 완성하고, 예상 결과를 재현한 뒤 설명 질문에 답합니다. 출력 숫자를 직접 넣어 결과만 맞추지 않습니다.


## 산출물 내용

| 파일 | 확인할 내용 |
|---|---|
| `outputs/validation_predictions.csv` | valid 20행. `request_id`, 실제값 `failed`, 예측값 `predicted` |
| `outputs/split_manifest.csv` | 100개 ID, `split`, 해당 묶음 안의 0부터 시작하는 `position`. train 60 / valid 20 / test 20 |
| `outputs/feature_names.json` | 학습된 변환 특성 이름 8개 |
| `outputs/pipeline.joblib` | 직접 학습한 전처리+기준선 상태. 참고 결과에서 복사하지 않음 |
| `reports/model_review.md` | 입력·분할·학습 기준·0.70과 실패0/6의 의미·복원 확인 |

PPT의 프로젝트 경로·핵심 파일명을 유지했습니다. 별도 `checks.py`와 복잡한 검증 코드는 이번 조정판에서 요구하지 않습니다. 기본필수·응용 파일명은 `valid_predictions.csv`이고, 이 프로젝트는 PPT와 같은 `validation_predictions.csv`라는 점을 구분합니다.

## 08교시 완료·제출 기준

실행 결과를 `expected/`와 대조하고, 예측 숫자뿐 아니라 ID 순서도 같음을 확인합니다. 보고서의 빈 칸을 자신의 말로 작성합니다. **정확도 0.70이어도 실제 실패 6건을 모두 놓치는 기준선**이라는 한계를 적습니다. 수치가 같아도 `.predict`를 실제로 호출했는지 코드를 확인합니다.

소스 파일·원본·결과 CSV/JSON·보고서를 제출합니다. 모델 파일은 본인 실행·복원 확인용으로 보관하고, 다른 사람이 보내 준 모델을 열어 확인하지 않습니다. 개인 발표에서 문제 정의·train-only fit·test 보관을 설명합니다.

### 선택: Git으로 구조의 변경을 기록하기

기관에서 Git 사용을 안내한 경우에만 진행합니다. 한 번만 프로젝트에서 `git init`하고 `git status`로 개인정보·모델 파일이 포함되지 않는지 확인합니다. `.gitignore`는 제공됩니다. 코드와 문서를 명시해 저장합니다.

```bash
git init
git add .gitignore README.md requirements.txt main.py preprocessing.py restore.py data reports
git commit -m "Add preprocessing practice"
```

이후 보고서를 자신의 내용으로 수정한 뒤 다시 `git add reports/model_review.md`와 `git commit`을 수행할 수 있습니다. commit을 원격 업로드라고 생각하지 않습니다. Git 계정·인증 설정 때문에 본 실습을 중단하지 않습니다.

## 복원 주의

`joblib`은 신뢰할 수 없는 파일을 불러오면 코드가 실행될 수 있습니다. **자신의 main.py로 같은 환경에서 만든 outputs/pipeline.joblib만** 불러옵니다. `restore.py`에서 다시 fit하지 않습니다. 저장·복원 환경의 버전은 `requirements.txt`를 함께 보관합니다.
