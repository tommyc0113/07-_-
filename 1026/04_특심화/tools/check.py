"""Day6 자동 확인. --exercise 01 --level basic / 기본값: 전체 단계.
예상 결과를 복사해서 답을 만들지 않습니다. 임의 입력·경계 입력도 함께 검사합니다.
보고서의 내용·구두 설명·실제 Git 기여는 강사가 별도로 확인합니다.
"""
from pathlib import Path
import argparse,contextlib,importlib.util,io,json,os,runpy,shutil,subprocess,sys,tempfile,ast
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from course_support import (load_data,load_checkpoint,check_features,validate_k,read_table,
    make_pre,build,snapshot,environment,write_json,FEATURES,DATA,reg_metrics,cls_metrics)
REGISTRY=json.loads((ROOT/'exercise_manifest.json').read_text(encoding='utf-8'))

class Checks:
    def __init__(self): self.rows=[]
    def check(self,name,fn):
        try:
            value=fn()
            if isinstance(value,(bool,np.bool_)) and not value:
                raise AssertionError('조건이 False입니다.')
            self.rows.append({'check':name,'passed':True})
            print('[PASS]',name)
            return True
        except Exception as exc:
            self.rows.append({'check':name,'passed':False,'error':f'{type(exc).__name__}: {exc}'})
            print('[FAIL]',name,'->',type(exc).__name__,str(exc)[:250])
            return False
    def equal(self,name,a,b): return self.check(name,lambda:a==b)
    def close(self,name,a,b,tol=1e-9):
        return self.check(name,lambda:np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True))
    def raises(self,name,fn,kind=ValueError):
        def action():
            try: fn()
            except kind: return
            raise AssertionError(f'{kind.__name__}가 발생하지 않았습니다.')
        return self.check(name,action)


def syntax_checks(c):
    paths=sorted(ROOT.rglob('*.py'))
    paths=[p for p in paths if not any(x in p.parts for x in ['.venv','__pycache__'])]
    def action():
        for p in paths: compile(p.read_text(encoding='utf-8'),str(p),'exec')
    c.check(f'Python 소스 {len(paths)}개 문법 검사',action)
    return len(paths)


def data_checks(c):
    d=load_data()
    c.equal('데이터/분할 크기',(len(d.X_train),len(d.X_valid),len(d.test_ids)),(60,20,20))
    c.equal('입력 열',list(d.X_train),FEATURES)
    c.equal('두 문제 y 정렬',d.X_train.index.equals(d.yc_train.index) and d.X_train.index.equals(d.yr_train.index),True)
    c.equal('train/valid ID 독립',len(set(d.X_train.index)&set(d.X_valid.index)),0)
    c.equal('test/개발 ID 독립',len(set(d.test_ids)&(set(d.X_train.index)|set(d.X_valid.index))),0)
    c.equal('ID 전체 합집합',len(set(d.X_train.index)|set(d.X_valid.index)|set(d.test_ids)),100)
    c.equal('valid 클래스 수',d.yc_valid.value_counts().sort_index().tolist(),[14,6])
    c.close('train 중앙값',d.X_train[['input_chars','allowed_tools']].median(),[1315,3])
    Xtr,Xva,ytr,yva,yrtr,yrva=load_checkpoint()
    c.check('독립 체크포인트/원본 일치',lambda:pd.testing.assert_frame_equal(Xtr,d.X_train))
    c.close('체크포인트 회귀 정답',yrva,d.yr_valid)
    c.equal('체크포인트 valid 순서',Xva.index.tolist(),d.X_valid.index.tolist())
    for fn in ['missing_priority.csv','forbidden_status.csv','infinite_numeric.csv','negative_numeric.csv','duplicate_id.csv','empty_input.csv']:
        frame=read_table(DATA/'fixtures'/fn).set_index('request_id')
        c.raises('입력 거부/'+fn,lambda f=frame:check_features(f))
    for k in [0,61,1.5,True]: c.raises('K 거부/'+str(k),lambda k=k:validate_k(k,60))
    for k in [1,60]: c.equal('K 허용/'+str(k),validate_k(k,60),k)
    all_missing=d.X_train.copy();all_missing['input_chars']=np.nan
    c.raises('전체 결측 train 정책',lambda:check_features(all_missing,training=True))
    return d


def check_exercise(c,record):
    n=record['exercise'];level=record['level'];prefix=f'{n:02d}/{level}'
    holder={}
    def run():
        ns=runpy.run_path(str(ROOT/record['script']))
        holder['ns']=ns
        holder['result']=ns['solve']()
    if not c.check(prefix+'/실행',run): return
    ns=holder['ns'];res=holder['result']
    eq=lambda label,a,b:c.equal(prefix+'/'+label,a,b)
    cl=lambda label,a,b,tol=1e-9:c.close(prefix+'/'+label,a,b,tol)
    er=lambda label,fn:c.raises(prefix+'/'+label,fn)
    d=load_data()
    if n==1 and level=='basic':
        eq('X/y 형태',(res['X_shape'],res['y_shape']),([5,1],[5]))
        cl('기울기',res['coef'],[9.3]);cl('절편',res['intercept'],42.3);cl('6시간',res['prediction_at_6'],[98.1])
        model,p=ns['fit_and_predict'](np.array([[0],[1],[2]]),np.array([1,3,5]),np.array([[4],[5]]))
        cl('임의 입력에 적용',p,[9,11]);er('1차원 query는 오류',lambda:model.predict([4]))
    elif n==1 and level=='applied':
        cl('두 계수',res['coef'],[0.5,100]);cl('절편',res['intercept'],500)
        cl('예측900',res['prediction_ms'],[900]);eq('손계산 일치',res['calculation_equal'],True)
        cl('일반화된 행렬계산',ns['hand_prediction']([2,3],5,[[1,2],[4,5]]),[13,28])
    elif n==1:
        cl('원래 기울기',res['models']['original']['coef'],[9.3])
        eq('외삽 여부',res['extrapolation'].tolist(),[False,True,True])
        eq('원본 미변경',res['original_y_preserved'].tolist(),[52,61,69,80,89])
        c.check(prefix+'/이상치 효과',lambda:res['models']['changed']['coef']!=res['models']['original']['coef'])
        eq('2차원 외삽 검사',ns['extrapolation_mask']([[0,0],[10,10]],[[5,11],[5,5]]).tolist(),[True,False])
    elif n==2 and level=='basic':
        cl('MAE 손계산',res['toy']['MAE_ms'],100/3);cl('MSE 손계산',res['toy']['MSE_ms2'],4400/3)
        cl('RMSE 손계산',res['toy']['RMSE_ms'],np.sqrt(4400/3));cl('R2 손계산',res['toy']['R2'],.78)
        c.check(prefix+'/A/B 선택 다름',lambda:res['A']['MAE_ms']<res['B']['MAE_ms'] and res['A']['RMSE_ms']>res['B']['RMSE_ms'])
        cl('공통 Linear RMSE',res['linear_valid']['RMSE_ms'],640.00,.005)
        eq('valid20',res['validation_count'],20)
        cl('완전 정답 입력',list(ns['measure']([1,2,3],[1,2,3]).values()),[0,0,0,1])
    elif n==2 and level=='applied':
        t=res['regression_comparison'];e=res['largest_errors']
        eq('후보',t.model.tolist(),['Dummy','Linear']);cl('RMSE',t.RMSE_ms,[658.75,640.00],.005)
        eq('잔차3건',len(e),3);cl('잔차 부호',e.residual_ms,e.actual_ms-e.predicted_ms)
        eq('절대오차 순서',e.abs_error_ms.is_monotonic_decreasing,True)
        eq('valid 삭제 없음',res['valid_rows_removed'],0)
        test=ns['largest_residuals'](pd.Series([10,20],index=['x','y']),[30,19],1)
        eq('임의 ID 보존',test.request_id.tolist(),['x'])
    elif n==2:
        table=res['policy_comparison']
        eq('train만 행 변경',table.n_train.tolist(),[6,5]);eq('valid3 유지',res['valid_kept'],3)
        mask,bounds=ns['iqr_mask'](np.array([1,2,3,4,5,50]));eq('train IQR 판정',mask.tolist(),[True]*5+[False])
        c.check(prefix+'/지표에 따른 선택차이',lambda:res['custom_metric_tradeoff']['A']['MAE_ms']<res['custom_metric_tradeoff']['B']['MAE_ms'] and res['custom_metric_tradeoff']['A']['RMSE_ms']>res['custom_metric_tradeoff']['B']['RMSE_ms'])
        c.check(prefix+'/교육용 정책 비교 완료',lambda:table.loc[1,'RMSE_ms']<table.loc[0,'RMSE_ms'])
    elif n==3 and level=='basic':
        eq('양성 클래스',res['classes'].tolist(),[0,1]);cl('확률 앞4',res['first4_probability'],[.141,.185,.601,.745])
        eq('예측 앞4',res['first4_prediction'].tolist(),[0,0,1,1])
        eq('혼동행렬',res['confusion_matrix'].tolist(),[[10,4],[4,2]])
        cl('F1',res['metrics']['F1'],1/3);cl('Accuracy',res['metrics']['Accuracy'],.6)
        m=ns['fit_logistic'](d.X_train,d.yc_train);before=snapshot(m['pre'])
        ns['probability_and_class'](m,d.X_valid);eq('예측 중 통계 불변',snapshot(m['pre']),before)
    elif n==3 and level=='applied':
        t=res['threshold_comparison']
        eq('FP',t.FP.tolist(),[1,2]);eq('FN',t.FN.tolist(),[1,0]);eq('TP',t.TP.tolist(),[2,3])
        cl('같은 정확도',t.Accuracy,[4/6,4/6]);eq('사전 비용',t.cost.tolist(),[6,2])
        er('불가능한 확률',lambda:ns['threshold_table']([0,1],[.2,1.5],[.5]))
    elif n==3:
        t=res['C_comparison'];e=res['error_cases']
        cl('C 후보',t.C,[.1,1,10]);cl('C F1',t.F1,[0,1/3,1/3]);cl('동점 작은 C',res['selected_C'],1)
        eq('FP/FN 각 1건',e.error_type.tolist(),['FP','FN']);eq('행수',len(e),2)
        c.check(prefix+'/오류 ID는valid',lambda:set(e.request_id)<=set(d.X_valid.index))
        er('음수 C',lambda:ns['compare_C'](d,[-1]))
    elif n==4 and level=='basic':
        t=res['neighbors'];eq('이웃',[int(x) for x in t.x],[4,2,1]);cl('거리',t.distance,[.8,1.2,2.2])
        eq('분류0',res['classification'],0);cl('회귀',res['regression'],70/3)
        eq('투표 일치',res['classification'],res['hand_vote']);cl('평균 일치',res['regression'],res['hand_mean'])
    elif n==4 and level=='applied':
        t=res['weight_comparison'];cl('uniform',t.prediction.iloc[0],70/3)
        cl('distance',t.prediction.iloc[1],28.05970149253731);cl('손계산 일치',t.prediction,t.hand)
        cl('단일 거리0',res['at_training_point'].loc[1,'prediction'],40)
        cl('다중 거리0',ns['weighted_average']([0,0,1],[2,8,999]),5)
        er('음수 거리',lambda:ns['weighted_average']([-1,1],[1,2]))
        er('빈 입력',lambda:ns['weighted_average']([],[]))
    elif n==4:
        t=res['distance_choice'];eq('p별 이웃',t.neighbor_index.tolist(),[0,1]);eq('p별 분류',t.prediction.tolist(),[0,1])
        cl('p별 거리',t.distance,[3,np.sqrt(8)])
        for k in [0,4,1.5,True]:er('K경계/'+str(k),lambda k=k:ns['guard_k'](k,3))
        eq('K상한 허용',ns['guard_k'](3,3),3)
        eq('동률 관찰2개',len(res['tie_order_observation']),2)
    elif n==5 and level=='basic':
        ex=res['small_examples'];eq('이웃 변경',(ex['raw_prediction'],ex['scaled_prediction']),(0,1))
        cl('MinMax 새값',ex['minmax_500'],2);cl('표준화',ex['standard_values'],[-1.225,0,1.225])
        eq('shape',(res['train_shape'],res['valid_shape']),([60,8],[20,8]));eq('유한값',res['all_finite'],True)
        cl('중앙값',res['train_statistics']['numeric_imputer'],[1315,3]);eq('특성8',len(res['feature_names']),8)
        pre=ns['make_preprocessor']().fit(d.X_train);before=snapshot(pre)
        new=read_table(DATA/'new_requests.csv').set_index('request_id');z=pre.transform(new)
        eq('새 입력 shape',z.shape,(2,8));cl('미지 범주 블록0',z[1,2:],np.zeros(6));eq('통계 불변',snapshot(pre),before)
    elif n==5 and level=='applied':
        t=res['scaling_comparison'];eq('스케일 후보',t.scale.tolist(),['none','standard','minmax'])
        cl('F1 3종',t.valid_F1,[.4,.2,.4]);eq('K고정',t.K.tolist(),[5]*3)
        eq('통계불변',t.statistics_unchanged.tolist(),[True]*3)
        eq('범주 경로 동일',t.categorical_steps.tolist(),['imputer,encoder']*3)
    elif n==5:
        dd=res['distance_contribution'];cl('성분 합',dd['numeric_squared']+dd['categorical_squared'],dd['total_squared'])
        unit=ns['distance_components']([1,2,1,0],[0,0,0,1]);cl('임의 성분',[unit['numeric_squared'],unit['categorical_squared'],unit['total_squared']],[5,2,7])
        eq('별도 실험명',res['alternative_scaling'].experiment.tolist(),['numeric_only','whole_including_onehot'])
        cl('원래 standard F1',res['alternative_scaling'].valid_F1.iloc[0],.2)
    elif n==6 and level=='basic':
        t=res['classification_K'];eq('K후보',t.K.tolist(),[1,3,5,9,15])
        cl('valid F1',t.valid_F1,[.4,6/11,.2,0,0]);cl('K1 train',t.train_F1.iloc[0],1)
        eq('선택3',res['selected_K'],3)
        tie=pd.DataFrame({'K':[9,3,1],'valid_F1':[.6,.6,.2]});eq('동점정책',ns['select_k'](tie),3)
    elif n==6 and level=='applied':
        t=res['knn_search'];eq('10개후보',len(t),10);eq('선택',res['selected_K'],{'classification':3,'regression':15})
        cl('회귀RMSE',t.loc[t.task=='regression','valid_score'],[944.68,795.22,744.99,707.65,641.79],.005)
        cl('분류F1',t.loc[t.task=='classification','valid_score'],[.4,6/11,.2,0,0])
        eq('고정스케일',set(t.scale),{'standard'});eq('고정가중치',set(t.weights),{'uniform'})
        tied=t.copy();tied['valid_score']=1;eq('문제별 동점 작은K',ns['select_both'](tied),{'classification':1,'regression':1})
    elif n==6:
        t=res['weights_comparison'];eq('두후보',t.weights.tolist(),['uniform','distance'])
        eq('base 미학습',res['clone_check']['base_unfitted'],True)
        eq('설정 분리',(res['clone_check']['base_K'],res['clone_check']['trial_K']),(5,3))
        c.check(prefix+'/선택은후보중하나',lambda:res['selected_weights'] in {'uniform','distance'})
        cl('uniform참조',t.valid_F1.iloc[0],6/11)

@contextlib.contextmanager
def project_context(project):
    names=['course_io','preprocessing','checks','evaluate','search','persist','verify_outputs']
    saved={n:sys.modules.get(n) for n in names}
    for n in names:sys.modules.pop(n,None)
    sys.path.insert(0,str(project))
    try:
        main=runpy.run_path(str(project/'main.py'))
        verify=runpy.run_path(str(project/'verify_outputs.py'))
        yield main,verify
    finally:
        sys.path.remove(str(project))
        for n in names:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]


def project_checks(c,record,deep=True):
    project=(ROOT/record['script']).parent
    prefix='07/'+record['level'];d=load_data();holder={};trace=[];refs=[]
    from sklearn.pipeline import Pipeline
    original_fit=Pipeline.fit
    def audited_fit(obj,X,y=None,**kw):
        if 'pre' in obj.named_steps and 'model' in obj.named_steps:
            if not isinstance(X,pd.DataFrame) or X.index.tolist()!=d.X_train.index.tolist():
                raise AssertionError('fit 입력에 train 이외의 행 또는 다른 순서가 포함되었습니다.')
            trace.append(X.index.tolist());refs.append(obj)
        return original_fit(obj,X,y,**kw)
    with tempfile.TemporaryDirectory(prefix='day6_check_') as temp,project_context(project) as (main,verify):
        out=Path(temp)/'outputs'
        def run():
            with patch.object(Pipeline,'fit',audited_fit),contextlib.redirect_stdout(io.StringIO()):
                holder['r']=main['run_project'](project/'data',out)
        if not c.check(prefix+'/실행 및 train-only fit 감시',run):return
        r=holder['r'];reg=r['regression'];cls=r['classification'];pred=r['predictions'];meta=r['metadata']
        c.equal(prefix+'/후보3+3',(len(reg),len(cls)),(3,3))
        c.equal(prefix+'/예측120',len(pred),120)
        c.equal(prefix+'/분류선택',r['selection']['classification']['model'],'KNN_k3')
        c.equal(prefix+'/회귀선택',r['selection']['regression']['model'],'Linear')
        c.close(prefix+'/분류F1',cls.F1,[0,1/3,6/11])
        c.close(prefix+'/회귀RMSE',reg.RMSE_ms,[658.75,640,641.79],.005)
        c.equal(prefix+'/test모델미사용',meta['test_used_for_model'],False)
        c.equal(prefix+'/실제fit횟수',len(trace),6 if record['level']=='basic' else 16)
        c.equal(prefix+'/독립Pipeline객체',len({id(m) for m in refs}),len(refs))
        c.equal(prefix+'/독립전처리객체',len({id(m['pre']) for m in refs}),len(refs))
        c.equal(prefix+'/K탐색행수',len(r['search']),10)
        c.equal(prefix+'/통계불변',all(t['statistics_unchanged'] for t in meta['fit_trace']),True)
        c.equal(prefix+'/testID예측없음',bool(set(pred.request_id)&set(d.test_ids)),False)
        # 검증기는 저장한 실제 예측에서 지표를 독립적으로 다시 계산합니다.
        def check_outputs():
            with contextlib.redirect_stdout(io.StringIO()): rows=verify['verify_output'](out,project/'data')
            if not rows or not all(x['passed'] for x in rows):raise AssertionError('결과 검증 실패')
        c.check(prefix+'/저장파일재계산36항목',check_outputs)
        # y를 고정하지 않고 metric 구현 자체를 작은 임의 입력으로 검사합니다.
        evaluation=sys.modules['evaluate']
        c.close(prefix+'/회귀지표 일반입력',evaluation.regression_metrics([1,2,3],[1,2,3])['RMSE_ms'],0)
        c.close(prefix+'/분류 zero_division',evaluation.classification_metrics([0,0],[0,0])['F1'],0)
        c.equal(prefix+'/분류 양성오류',evaluation.classification_metrics([0,1],[1,0])['FN'],1)
        checks=sys.modules['checks']
        # 저장용이 아닌 별도 학습 모델로 경계 입력의 계약을 점검합니다.
        from sklearn.neighbors import KNeighborsClassifier
        fitted=build(KNeighborsClassifier(n_neighbors=3)).fit(d.X_train,d.yc_train)
        new=read_table(DATA/'new_requests.csv').set_index('request_id')
        c.equal(prefix+'/미지범주8열',checks.prediction_contract(fitted,new)['n_features'],8)
        c.raises(prefix+'/필수열누락',lambda:checks.prediction_contract(fitted,new.drop(columns=['priority'])))
        c.raises(prefix+'/금지열',lambda:checks.prediction_contract(fitted,new.assign(status='error')))
        c.raises(prefix+'/과도한K',lambda:checks.before_model(KNeighborsClassifier(n_neighbors=61),d.X_train,d.yc_train,d.X_valid,d.yc_valid))
        missing=read_table(DATA/'fixtures/all_numeric_missing_new.csv').set_index('request_id')
        c.equal(prefix+'/새수치전체결측',checks.prediction_contract(fitted,missing)['statistics_unchanged'],True)
        if record['level']!='basic':
            c.equal(prefix+'/자체저장2모델',set(meta['persistence']),{'regression','classification'})
            c.raises(prefix+'/신뢰확인없는복원거부',lambda:verify['verify_saved'](out,False))
            def new_process():
                env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                p=subprocess.run([sys.executable,str(project/'verify_outputs.py'),'--output-dir',str(out),'--restore','--trust-local'],
                                 capture_output=True,text=True,env=env,cwd=temp,timeout=30)
                if p.returncode:raise AssertionError(p.stderr+p.stdout)
                result=json.loads((out/'restore_checks.json').read_text(encoding='utf-8'))
                if result!={'regression':True,'classification':True}:raise AssertionError(result)
            c.check(prefix+'/새프로세스두모델복원',new_process)
        if deep:
            def poison_test():
                copy=Path(temp)/'test를 바꾼 입력';shutil.copytree(project/'data',copy)
                raw=pd.read_csv(copy/'requests.csv');mask=raw.request_id.isin(d.test_ids)
                # test의 특성/두 y를 극단값으로 바꿔도 개발 결과는 바뀌면 안 됩니다.
                raw.loc[mask,['input_chars','allowed_tools']]=999999
                raw.loc[mask,'failed']=7;raw.loc[mask,'elapsed_ms']=-999
                raw.loc[mask,['task_type','priority']]='TEST_ONLY_CATEGORY'
                raw.to_csv(copy/'requests.csv',index=False)
                with contextlib.redirect_stdout(io.StringIO()):changed=main['run_project'](copy,Path(temp)/'changed')
                pd.testing.assert_frame_equal(changed['classification'],cls)
                pd.testing.assert_frame_equal(changed['regression'],reg)
                pd.testing.assert_frame_equal(changed['predictions'],pred)
            c.check(prefix+'/test오염에도개발결과불변',poison_test)
            def broken_prediction():
                broken=Path(temp)/'tampered';shutil.copytree(out,broken)
                t=pd.read_csv(broken/'validation_predictions.csv');t.loc[0,'y_pred']+=1000
                t.to_csv(broken/'validation_predictions.csv',index=False)
                return verify['verify_output'](broken,project/'data')
            c.raises(prefix+'/변조된예측파일탐지',broken_prediction,AssertionError)
            def broken_manifest():
                copy=Path(temp)/'bad_manifest';shutil.copytree(project/'data',copy)
                frame=pd.read_csv(copy/'split_manifest.csv');frame=pd.concat([frame,frame.iloc[[0]]]);frame.to_csv(copy/'split_manifest.csv',index=False)
                return main['run_project'](copy,Path(temp)/'bad_out')
            c.raises(prefix+'/분할중복탐지',broken_manifest)


def existing_submission(c,record):
    level=record['level'];project=ROOT/'ex07_compare'/Path(record['folder']).name/'day6_compare'
    prefix='08/'+level
    with project_context(project) as (main,verify):
        def action():
            with contextlib.redirect_stdout(io.StringIO()):verify['verify_output']()
        c.check(prefix+'/실습07결과파일검사',action)
        c.check(prefix+'/보고서파일존재',lambda:(project/'reports/model_choice.md').is_file())
        print('안내: 보고서 빈칸·구두 설명·Git 이력은 직접 확인하세요.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exercise',type=int,choices=range(1,9))
    p.add_argument('--level',choices=['basic','applied','advanced'])
    p.add_argument('--syntax',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args();c=Checks()
    with threadpool_limits(limits=1):
        if a.syntax:syntax_checks(c)
        else:
            if a.exercise is None:data_checks(c)
            selected=[r for r in REGISTRY if (a.exercise is None or r['exercise']==a.exercise) and (a.level is None or r['level']==a.level)]
            # 전체 검사는 07을 임시 폴더에 생성·검증하므로 08의 기존 outputs를 요구하지 않습니다.
            for r in selected:
                if r['exercise']<=6:check_exercise(c,r)
                elif r['exercise']==7:project_checks(c,r)
                elif a.exercise==8:existing_submission(c,r)
    passed=sum(x['passed'] for x in c.rows);failed=len(c.rows)-passed
    report=a.report or ROOT/'outputs'/f"checks_{a.exercise or 'all'}_{a.level or 'all'}{'_syntax' if a.syntax else ''}.json"
    write_json(report,{'environment':environment(),'checks':len(c.rows),'passed':passed,'failed':failed,'items':c.rows})
    print(f'\n자동 검사 {len(c.rows)}건: 통과 {passed} / 실패 {failed}')
    print('검사 기록:',report)
    return 0 if failed==0 else 1

if __name__=='__main__':raise SystemExit(main())
