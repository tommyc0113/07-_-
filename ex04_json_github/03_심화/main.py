"""저장 -> 프로세스 종료 -> 다시 읽기의 동등성을 확인하는 제공 실행기."""
import argparse
import json
from pathlib import Path
from models import from_record, to_record
from storage import load_records, save_records
BASE = Path(__file__).resolve().parent

def records_equal(first, second):
    # 해설 EQUAL: 두 Idea 목록을 각각 List[Dict]로 바꿔 값이 같은지 비교하세요. 객체의 is 비교가 아닙니다.
    return [to_record(idea) for idea in first] == [to_record(idea) for idea in second]

def replace_current(current, path):
    # 해설 REPLACE: 읽기·검사·객체 복원을 새 목록에서 끝낸 뒤에만 current[:]를 교체하세요.
    records = load_records(path)
    restored = [from_record(record) for record in records]
    current[:] = restored
    return current

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="demo", choices=["demo", "save", "load"])
    parser.add_argument("--workspace", type=Path, default=BASE)
    args = parser.parse_args()
    path = args.workspace.resolve() / "data" / "ideas.json"
    original = [from_record(record) for record in json.loads((BASE / "fixtures" / "valid_ideas.json").read_text(encoding="utf-8"))]
    try:
        if args.command in {"demo", "save"}:
            if path.exists():
                if args.command == "save":
                    print("샘플 저장 중단: 이미 있는 파일은 덮어쓰지 않습니다.")
                    return 2
                print("기존 파일 사용: 덮어쓰지 않습니다.")
            else:
                save_records(path, [to_record(idea) for idea in original])
                print("저장 완료: data/ideas.json")
            if args.command == "save":
                return 0
        restored = [from_record(record) for record in load_records(path)]
        print(f"복원: {len(restored)}개")
        print([idea.name for idea in restored])
        print("샘플과 값 일치:", records_equal(original, restored))
    except (ValueError, OSError) as error:
        print(f"처리 실패: {error}; 현재 데이터와 원본을 확인하세요.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
