# 실습 04 · 저장·복원과 변경 이력 남기기 · 응용

**PPT 실습 40쪽 · 설명 33~41쪽 · 전체 단계 선택 실습 25분**

## 목표
저장·복원 값의 동등성을 확인하고 README 변경을 커밋·diff로 검토합니다. 계정 접속이 안 되면 제공된 로컬 비교 도구와 기록지로 대체합니다.

## 진행 순서
먼저 동작을 예측하고, 아래 TODO를 구현합니다. 직접 실행 → 공개 검사 → 페어 설명/기록 순서로 마칩니다.
TODO가 아닌 입출력 틀·검사 도구·이전 단계 구현은 제공 코드입니다. 파일 간 책임을 바꾸지 않고 핵심 처리부터 완성합니다.

| 수정할 파일 | TODO/해설 번호 | 구현할 내용 |
|---|---|---|
| `main.py` | `EQUAL` | 두 Idea 목록을 각각 List[Dict]로 바꿔 값이 같은지 비교하세요. 객체의 is 비교가 아닙니다. |

## 실행
가상환경의 Python으로 실행합니다. 현재 난이도 폴더에서:

```text
python main.py
```

역할 폴더(강사용 또는 수강생용)에서 제공 런처를 사용할 때:

```text
python tools/run.py --exercise 04 --level applied
python tools/check.py --exercise 04 --level applied
```

`checks.py`는 기능 확인용 제공 테스트입니다. 검사 도구는 임시 복사본에서 실행하여 원본 소스와 결과물을 변경하지 않습니다.
출력 기대값은 `expected_output.txt`, 생성 파일 예시는 `expected_outputs/`를 확인합니다. 구현을 하드코딩해 예시 한 건만 맞추지 마세요.

## 생성할 결과
- `data/ideas.json`

### 재실행 확인
새 작업 폴더에서 `python main.py save`를 실행한 뒤 프로세스가 끝나면 `python main.py load`를 다시 실행합니다.
기존 파일이 있으면 save는 덮어쓰지 않습니다. 안전한 새 경로 예: `python main.py save --workspace ./trial_02` 후 `python main.py load --workspace ./trial_02`.
기본 실행 `python main.py`는 파일이 없을 때만 샘플을 저장하고 읽습니다.

### 데이터·실패 규칙
JSON에는 name, problem, roles, difficulty를 가진 Dict의 List를 저장합니다. 객체는 값이 같아도 서로 다른 인스턴스입니다.
처음 읽을 파일이 없으면 빈 목록, JSON 손상은 ValueError, 권한 오류는 PermissionError를 호출자에게 전달합니다.
전체 검사 후 현재 목록을 교체합니다. io_support.py는 강사가 제공한 안전 저장 보조 코드이며 작성 시험 대상이 아닙니다.

### 협업
`협업_활동기록.md`와 공통 GitHub_활동안내.md를 작성합니다. 자동 검사만으로 실제 커밋/PR이 완료되었다고 표시하지 않습니다.


## 완료 확인
구현한 범위의 자동 테스트가 통과하고, 테스트_기록.md에 실제 명령/결과를 기록했는지 확인합니다.
코드를 설명하지 못한 채 출력만 맞춘 경우에는 페어에게 입력 → 처리 → 반환/저장의 순서를 설명해 봅니다.

강사용: 각 `해설` 번호는 수강생 TODO와 대응합니다. `예상 출력`은 실제 정답 실행 결과이며 학생 기록의 대필용으로 배포하지 않습니다.
