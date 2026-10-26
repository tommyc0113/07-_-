"""실습08: 새 프로젝트를 만들지 않고 07의 동일 프로젝트를 검증합니다."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'course_support.py').is_file())
PROJECT=ROOT/'ex07_compare/03_심화/day6_compare'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true')
    p.add_argument('--trust-local',action='store_true')
    a=p.parse_args()
    args=[sys.executable,str(PROJECT/'verify_outputs.py')]
    if a.restore: args.append('--restore')
    if a.trust_local: args.append('--trust-local')
    return subprocess.run(args,check=False).returncode
if __name__=='__main__': raise SystemExit(main())
