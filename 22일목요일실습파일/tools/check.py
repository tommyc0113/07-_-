"""선택한 실습을 임시 복사본에서 검사합니다. 본인 결과/저장소를 덮어쓰지 않습니다."""
from pathlib import Path
import argparse, ast, json, os, shutil, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exercise',choices=[f'{x:02}' for x in range(1,9)])
    p.add_argument('--level',choices=['basic','applied','advanced'])
    p.add_argument('--all',action='store_true')
    p.add_argument('--syntax-only',action='store_true')
    p.add_argument('--list',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args();manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    if a.list:
        for key,r in manifest.items():print(key,r['path'],'PPT',r['slide'])
        return 0
    if not a.all and not a.exercise:p.error('--exercise 또는 --all을 지정하세요.')
    chosen=[(k,v) for k,v in manifest.items() if (a.all or k[:2]==a.exercise) and (a.level is None or k.endswith('_'+a.level))]
    results=[]
    for key,r in chosen:
        ex,level=key.split('_',1);stage=ROOT/r['path']
        if ex=='06' and not a.syntax_only:
            if a.all:
                print(key,'MANUAL: 실제 본인 Git 저장소는 --exercise 06으로 별도 확인')
                results.append({'exercise':ex,'level':level,'status':'MANUAL','tests_run':0});continue
            project=ROOT/manifest['07_'+level]['path']/'day4_eda'
            code=subprocess.call([sys.executable,str(ROOT/'tools/check_git.py'),'--project',str(project),'--level',level])
            results.append({'exercise':ex,'level':level,'status':'PASS' if code==0 else 'FAIL','tests_run':0});continue
        if a.syntax_only:
            try:
                paths=[f for f in stage.rglob('*.py') if not any(x in f.parts for x in ['__pycache__','.venv'])]
                for f in paths:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
                info={'exercise':ex,'level':level,'status':'SYNTAX_ONLY','files':len(paths),'tests_run':0}
            except SyntaxError as err:info={'exercise':ex,'level':level,'status':'FAIL','error':str(err),'tests_run':0}
            results.append(info);print(key,info['status']);continue
        with tempfile.TemporaryDirectory(prefix='day4_check_') as temp:
            tmp=Path(temp);copy=tmp/'task'
            shutil.copytree(stage,copy,ignore=shutil.ignore_patterns('expected_outputs','out','out_extended','.git','.venv','__pycache__'))
            env=os.environ.copy();env.update(DAY4_EXERCISE=ex,DAY4_LEVEL=level,DAY4_STAGE=str(copy),DAY4_RESULT=str(tmp/'result.json'),PYTHONIOENCODING='utf-8',MPLBACKEND='Agg')
            try:
                run=subprocess.run([sys.executable,str(ROOT/'tools/tests_public.py')],cwd=tmp,env=env,capture_output=True,text=True,encoding='utf-8',timeout=180)
                info=json.loads((tmp/'result.json').read_text()) if (tmp/'result.json').exists() else {'exercise':ex,'level':level,'tests_run':0,'error':'테스트 초기화 실패'}
                info['status']='PASS' if run.returncode==0 else 'FAIL';info['log']=run.stdout+run.stderr
            except subprocess.TimeoutExpired:
                info={'exercise':ex,'level':level,'status':'FAIL','tests_run':0,'error':'180초 제한 초과'}
            results.append(info);print(f"{key}: {info['status']} / tests={info.get('tests_run',0)}",flush=True)
            if info['status']=='FAIL':print(info.get('log',info.get('error','')))
    if a.report:
        path=a.report if a.report.is_absolute() else ROOT/a.report
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print('기록:',path)
    failed=sum(r['status']=='FAIL' for r in results)
    print('총 실행 테스트:',sum(r.get('tests_run',0) for r in results),'실패 단계:',failed)
    print('MANUAL은 미검사 활동입니다. SYNTAX_ONLY는 기능 완료 판정이 아닙니다.')
    return int(failed>0)

if __name__=='__main__':raise SystemExit(main())
