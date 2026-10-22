"""현재 가상환경의 Python으로 선택한 실습을 실행합니다."""
from pathlib import Path
import argparse,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exercise',choices=[f'{x:02}' for x in range(1,9)],required=True)
p.add_argument('--level',choices=['basic','applied','advanced'],default='basic')
a,extra=p.parse_known_args()
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));item=manifest[a.exercise+'_'+a.level]
stage=ROOT/item['path']
if a.exercise=='06':
    print('활동 안내:',stage/'활동안내.md')
    print('Git은 실제07프로젝트에서 직접 실행합니다. 명령 안내를 읽고 tools/check_git.py로 확인하세요.')
    raise SystemExit(0)
cmd=[sys.executable,str(stage/item['entry'])]
if a.exercise=='08':cmd+=['--project',str(ROOT/manifest['07_'+a.level]['path']/'day4_eda'),'--level',a.level]
cmd+=extra
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
raise SystemExit(subprocess.call(cmd,cwd=(stage/item['entry']).parent,env=env))
