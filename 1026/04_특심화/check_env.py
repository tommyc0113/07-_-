"""설치 상태를 읽어 확인할 뿐 자동 설치·업그레이드하지 않습니다."""
from pathlib import Path
import sys,platform,json,importlib.metadata,argparse

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--strict',action='store_true');a=p.parse_args()
    expected=json.loads((Path(__file__).resolve().parent/'environment_reference.json').read_text(encoding='utf-8'))
    print('실행 Python:',sys.executable)
    missing=[];different=[]
    for name,version in expected.items():
        try: actual=platform.python_version() if name=='python' else importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name);print(name,'미설치');continue
        print(f'{name}: {actual} (검증 기준 {version})')
        if actual!=version: different.append(name)
    if missing: print('requirements-lock.txt로 가상환경에 설치하세요:',missing);return 1
    if different: print('기준 환경과 다름. 동일 수치 재현이 필요하면 별도 검증 환경을 사용하세요:',different)
    else: print('검증 기준 버전과 일치합니다.')
    return 1 if a.strict and different else 0

if __name__=='__main__':raise SystemExit(main())
