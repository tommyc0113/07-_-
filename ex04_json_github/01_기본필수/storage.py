"""JSON 읽기·전체 검사·저장. 실패를 빈 목록 또는 성공으로 위장하지 않습니다."""
import json
from pathlib import Path
from models import from_record, to_record, name_key
from io_support import write_json_atomic
REJECT_DUPLICATES = True

def validate_records(records):
    # 해설 VALIDATE: 최상위 List·각 레코드·중복 이름을 전체 검사하고 독립적인 레코드 목록을 반환하세요.
    if not isinstance(records, list):
        raise ValueError("최상위 데이터는 List여야 합니다")
    validated, seen = [], set()
    for index, record in enumerate(records, start=1):
        try:
            idea = from_record(record)
        except ValueError as error:
            raise ValueError(f"레코드 {index}: {error}") from error
        key = name_key(idea.name)
        if REJECT_DUPLICATES and key in seen:
            raise ValueError(f"레코드 {index}: 이름 중복: {idea.name}")
        seen.add(key)
        validated.append(to_record(idea))
    return validated

def load_records(path):
    # 해설 LOAD: UTF-8 JSON을 읽으세요. 파일 없음만 []로 처리하고 손상은 ValueError로 전달한 뒤 전체 검사하세요.
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        # '처음 실행하면 빈 목록'이라는 수업의 명시적인 정책입니다.
        return []
    except json.JSONDecodeError as error:
        raise ValueError("JSON 손상: 원본 보존") from error
    # PermissionError, UnicodeDecodeError 등은 숨기지 않고 호출자에게 전달합니다.
    return validate_records(records)

def save_records(path, records):
    # 해설 SAVE: 전체 검사를 먼저 수행하고 제공된 write_json_atomic(path, records)로 저장하세요. 실패를 숨기지 마세요.
    validated = validate_records(records)
    write_json_atomic(Path(path), validated)
    # 성공 메시지는 함수가 정상 반환된 뒤 main에서만 출력합니다.
