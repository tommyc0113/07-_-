"""실제 로컬 저장소 읽기 전용 검사. init/add/commit/switch/push는 수행하지 않습니다."""
from pathlib import Path
import argparse,json,subprocess,shutil


def inspect_git(project,level='basic'):
    project=Path(project).resolve()
    if shutil.which('git') is None:raise ValueError('Git 실행기가 없습니다. 설치 상태를 먼저 확인하세요.')
    def git(*args):
        r=subprocess.run(['git','-C',str(project),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if r.returncode:raise ValueError(r.stderr.strip() or 'Git 명령 실패')
        return r.stdout.strip()
    top=Path(git('rev-parse','--show-toplevel')).resolve()
    if top!=project:raise ValueError('선택 폴더 자체가 저장소 루트가 아닙니다. 상위 저장소를 사용하지 마세요.')
    errors=[]
    count=int(git('rev-list','--count','HEAD'))
    branch=git('branch','--show-current')
    status=git('status','--short')
    tracked=git('ls-files').splitlines()
    commits=git('log','--format=%h %s','-8').splitlines()
    if count<2:errors.append('목적 있는 커밋2개 이상이 필요합니다.')
    if not {'README.md','.gitignore'}.issubset(set(tracked)):errors.append('README.md/.gitignore가 추적되어야 합니다.')
    forbidden=[f for f in tracked if '.venv' in Path(f).parts or '__pycache__' in Path(f).parts or Path(f).name=='.env' or f.endswith('.pyc')]
    if forbidden:errors.append('추적하면 안 되는 파일: '+', '.join(forbidden))
    if level!='basic' and 'feature/eda' not in git('branch','--format=%(refname:short)').splitlines():errors.append('feature/eda 브랜치가 없습니다.')
    if git('check-ignore','--no-index','.env')!='.env':errors.append('.env 제외 규칙을 확인하세요.')
    return {'errors':errors,'commits':count,'branch':branch,'recent_log':commits,'status_short':status,
            'manual_review':['두 커밋이 실제 의미 있는 서로 다른 변경인지','diff와 diff --cached 설명','Git상태에 남은 변경/미추적파일과 최종제출범위'],
            'notice':'네트워크/원격 업로드는 하지 않았습니다. 커밋수만으로10점 만점을 자동부여하지 않습니다.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--level',choices=['basic','applied','advanced'],default='basic');a=p.parse_args()
    try:r=inspect_git(a.project,a.level)
    except ValueError as exc:print('Git 확인 미완료:',exc);return 1
    print(json.dumps(r,ensure_ascii=False,indent=2));return int(bool(r['errors']))

if __name__=='__main__':raise SystemExit(main())
