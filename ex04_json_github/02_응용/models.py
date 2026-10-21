"""Idea의 데이터 규칙. 모델은 입력창·화면 출력·파일 입출력을 담당하지 않습니다."""
DIFFICULTIES = ("기본필수", "응용", "심화")

def name_key(value):
    """비교용 문자열. 표시할 원문과 비교할 키를 분리합니다."""
    return value.strip().casefold()

def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: 비어 있지 않은 문자열이 필요합니다")
    return value.strip()

class Idea:
    def __init__(self, name, problem, roles, difficulty):
        self.name = require_text(name, "name")
        self.problem = require_text(problem, "problem")
        self.difficulty = require_text(difficulty, "difficulty")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty: 기본필수/응용/심화 중 하나여야 합니다")
        if not isinstance(roles, list) or not roles:
            raise ValueError("roles: 비어 있지 않은 List가 필요합니다")
        cleaned, seen = [], set()
        for role in roles:
            value = require_text(role, "role")
            key = name_key(value)
            if key in seen:
                raise ValueError("roles: 같은 역할을 중복 등록할 수 없습니다")
            cleaned.append(value)
            seen.add(key)
        # 입력 List와 다른 새 List를 소유합니다. 문자열 원소는 불변입니다.
        self.roles = cleaned

    def summary(self):
        return f"{self.name}: 역할 {len(self.roles)}개"

def to_record(idea):
    """JSON에 저장 가능한 Dict를 반환합니다. 역할 List도 복사합니다."""
    # 해설 TO-RECORD: Idea의 네 속성을 같은 키의 Dict로 반환하고 roles를 복사하세요.
    return {"name": idea.name, "problem": idea.problem,
            "roles": list(idea.roles), "difficulty": idea.difficulty}

def from_record(record):
    """필수 키를 확인한 뒤 새 Idea를 만듭니다. 추가 키는 사용하지 않습니다."""
    # 해설 FROM-RECORD: Dict와 필수 키를 검사한 뒤 Idea 객체로 복원하세요.
    required = {"name", "problem", "roles", "difficulty"}
    if not isinstance(record, dict):
        raise ValueError("레코드는 Dict여야 합니다")
    if not required <= record.keys():
        missing = ", ".join(sorted(required - record.keys()))
        raise ValueError(f"필수 키 누락: {missing}")
    return Idea(record["name"], record["problem"],
                record["roles"], record["difficulty"])
