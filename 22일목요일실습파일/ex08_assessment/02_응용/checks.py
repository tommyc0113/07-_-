"""역할 폴더의 공개 검사기로 연결하는 제공 실행기입니다."""
from pathlib import Path
import subprocess,sys
root=next(p for p in Path(__file__).resolve().parents if (p/"manifest.json").is_file())
raise SystemExit(subprocess.call([sys.executable,str(root/"tools/check.py"),"--exercise","08","--level","applied"]))
