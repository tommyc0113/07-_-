"""공개 기능 검사. 정답 소스가 아니라 입력/출력 계약을 검사합니다."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import numpy as np
import pandas as pd
from PIL import Image

EX = os.environ['DAY4_EXERCISE']
LEVEL = os.environ['DAY4_LEVEL']
HERE = Path(os.environ['DAY4_STAGE']).resolve()
IS_MORE = LEVEL != 'basic'
IS_ADV = LEVEL == 'advanced'
ENTRY = {'01':'ex01_numpy.py','02':'ex02_clean.py','03':'ex03_aggregate.py',
         '04':'ex04_plot.py','05':'ex05_files.py','07':'day4_eda/main.py',
         '08':'validate_submission.py'}[EX]
CODE = HERE / ENTRY
BASE = CODE.parent
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('exercise_target',CODE)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name]=M
spec.loader.exec_module(M)


def raw():
    return pd.read_csv(BASE/'data/agent_runs.csv',encoding='utf-8-sig',dtype='string',keep_default_na=False)


def clean_fixture():
    df=pd.read_csv(BASE/'data/clean_runs.csv',encoding='utf-8-sig',dtype={'run_id':'string','agent':'string','status':'string'})
    for col in ['latency_ms','tokens']:df[col]=df[col].astype(float)
    df['day']=pd.to_datetime(df['day'],format='%Y-%m-%d',errors='coerce')
    return df


def summary_fixture():
    return pd.DataFrame(json.loads((BASE/'data/summary.json').read_text(encoding='utf-8')))


def meta_fixture():
    return pd.DataFrame({'agent':['planner','researcher','writer'],'rate_per_1k':[.2,.4,.3]})


class NumpyCases(unittest.TestCase):
    def test_axis_values(self):
        c,r=M.axis_means([[100,300,200],[120,500,250]])
        np.testing.assert_allclose(c,[110,400,225]);np.testing.assert_allclose(r,[200,290])
    def test_axis_shapes(self):
        c,r=M.axis_means([[1,2,3],[4,5,6]])
        self.assertEqual(c.shape,(3,));self.assertEqual(r.shape,(2,))
    def test_general_input(self):
        c,r=M.axis_means([[2,8]])
        np.testing.assert_allclose(c,[2,8]);np.testing.assert_allclose(r,[5])
    def test_axis_unchanged(self):
        a=np.arange(6.).reshape(2,3);before=a.copy();M.axis_means(a);np.testing.assert_array_equal(a,before)
    def test_bad_axis_input(self):
        for a in ([1,2],np.empty((0,3)),[[np.nan,2]]):
            with self.subTest(a=str(a)),self.assertRaises(ValueError):M.axis_means(a)
    def test_broadcasting(self):
        c,r=M.calibrate([[100,300,200],[120,500,250]],[1,.8,1.1],[1,.5])
        np.testing.assert_allclose(c,[[100,240,220],[120,400,275]])
        np.testing.assert_allclose(r,[[100,300,200],[60,250,125]])
    def test_calibrate_unchanged(self):
        a=np.ones((2,3));before=a.copy();M.calibrate(a,[2,3,4],[5,6]);np.testing.assert_array_equal(a,before)
    def test_valid_mean(self):
        c,n,v=M.valid_mean([100,-5,np.nan,300]);self.assertEqual(n,2);self.assertEqual(v,200)
        np.testing.assert_allclose(c,[100,np.nan,np.nan,300],equal_nan=True)
    def test_zero_is_valid(self):
        c,n,v=M.valid_mean([0,np.inf,-np.inf,-1]);self.assertEqual(n,1);self.assertEqual(v,0)
    def test_safe_rate(self):
        r=M.safe_rate([3,0,2],[4,0,3]);np.testing.assert_allclose(r,[.75,np.nan,2/3],equal_nan=True)
    def test_all_missing_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            c,n,v=M.valid_mean([np.nan,-1,np.inf]);self.assertEqual(n,0);self.assertTrue(np.isnan(v))
            self.assertTrue(np.isnan(M.safe_rate([0],[0])[0]))
    def test_bad_shapes(self):
        for a,c,r in [([[1,2,3],[4,5,6]],[1,2],[1,2]),([[1,2]],[1,2],[1,2])]:
            with self.subTest(),self.assertRaises(ValueError):M.calibrate(a,c,r)
    def test_invalid_counts(self):
        for a,b in [([2],[1]),([-1],[2]),([1.5],[3]),([np.inf],[4]),([1],[1,2])]:
            with self.subTest(),self.assertRaises(ValueError):M.safe_rate(a,b)
    def test_empty_measurements(self):
        c,n,v=M.valid_mean([]);self.assertEqual(n,0);self.assertTrue(np.isnan(v))


class CleanCases(unittest.TestCase):
    def result(self):return M.clean_runs(raw(),parse_dates=IS_MORE,strict_tokens=IS_ADV)
    def test_row_counts_and_id(self):
        r=self.result();self.assertEqual(len(r),12);self.assertTrue(r.run_id.is_unique)
    def test_roles(self):self.assertEqual(set(self.result().agent),{'planner','researcher','writer'})
    def test_numeric_missing(self):
        r=self.result();self.assertEqual(r.loc[r.latency_ms.isna(),'run_id'].tolist(),['R08','R09','R10']);self.assertEqual(r.tokens.isna().sum(),1)
    def test_error_rows_and_900_retained(self):
        r=self.result();self.assertEqual(r.status.eq('error').sum(),3);self.assertEqual(r.loc[r.run_id.eq('R11'),'latency_ms'].iloc[0],900)
    def test_raw_unchanged(self):
        a=raw();b=a.copy(deep=True);M.clean_runs(a);pd.testing.assert_frame_equal(a,b)
    def test_zero_negative_infinite(self):
        a=raw();a.loc[0,'latency_ms']='0';a.loc[1,'latency_ms']='inf';r=M.clean_runs(a)
        self.assertEqual(r.loc[r.run_id.eq('R01'),'latency_ms'].iloc[0],0);self.assertTrue(pd.isna(r.loc[r.run_id.eq('R02'),'latency_ms'].iloc[0]))
    def test_all_missing(self):
        a=raw();a['latency_ms']='';r=M.clean_runs(a);self.assertEqual(len(r),12);self.assertEqual(r.latency_ms.count(),0)
    def test_required_columns(self):
        with self.assertRaises(ValueError):M.clean_runs(raw().drop(columns='tokens'))
    def test_dates(self):
        r=M.clean_runs(raw());self.assertEqual(r.loc[r.day.isna(),'run_id'].tolist(),['R11']);self.assertTrue(pd.api.types.is_datetime64_any_dtype(r.day))
    def test_quality_values(self):
        a=raw();q=M.quality_report(a,M.clean_runs(a))
        for k,v in {'raw_rows':13,'clean_rows':12,'exact_duplicates_removed':1,'unique_ids':12,'latency_valid':9,'latency_missing':3,'tokens_valid':11,'tokens_missing':1,'day_missing':1,'ok':9,'error':3}.items():
            with self.subTest(key=k):self.assertEqual(q[k],v)
    def test_quality_ids(self):
        a=raw();q=M.quality_report(a,M.clean_runs(a));self.assertEqual(q['latency_missing_ids'],['R08','R09','R10']);self.assertEqual(q['day_missing_ids'],['R11'])
    def test_quality_on_small_input(self):
        a=raw().iloc[:2].copy();q=M.quality_report(a,M.clean_runs(a));self.assertEqual(q['clean_rows'],2);self.assertEqual(q['latency_missing'],0)
    def test_conflicting_id_both_rows(self):
        a=raw();extra=a.iloc[[0]].copy();extra['latency_ms']='999';a=pd.concat([a,extra],ignore_index=True)
        issues=M.audit_contract(a);self.assertEqual(issues.run_id.tolist(),['R01','R01']);self.assertEqual(issues.source_row.tolist(),[1,14])
        with self.assertRaises(M.DataContractError):M.clean_runs(a)
    def test_unknown_role(self):
        a=raw();a.loc[0,'agent']='reviewer';self.assertIn('unknown_agent',M.audit_contract(a).issue.tolist())
        with self.assertRaises(M.DataContractError):M.clean_runs(a)
    def test_unknown_status(self):
        a=raw();a.loc[0,'status']='pending'
        with self.assertRaises(M.DataContractError):M.clean_runs(a)
    def test_blank_id(self):
        a=raw();a.loc[0,'run_id']=' '
        with self.assertRaises(M.DataContractError):M.clean_runs(a)
    def test_allowed_extension(self):
        a=raw();a.loc[0,'agent']='reviewer';r=M.clean_runs(a,allowed_agents=['planner','researcher','writer','reviewer']);self.assertIn('reviewer',r.agent.tolist())
    def test_fractional_tokens(self):
        a=raw();a.loc[0,'tokens']='2.5';r=M.clean_runs(a,strict_tokens=True);self.assertTrue(pd.isna(r.loc[r.run_id.eq('R01'),'tokens'].iloc[0]))
    def test_empty_input(self):
        with self.assertRaises(ValueError):M.clean_runs(raw().iloc[:0])


class AggregateCases(unittest.TestCase):
    def test_reference_summary(self):
        s=M.aggregate_runs(clean_fixture()).set_index('agent');self.assertEqual(s.runs.tolist(),[4,4,4]);self.assertEqual(s.measured.tolist(),[3,3,3])
        np.testing.assert_allclose(s.mean_ms,[120,1700/3,800/3]);np.testing.assert_allclose(s.median_ms,[120,500,250])
    def test_success_denominator(self):
        s=M.aggregate_runs(clean_fixture());np.testing.assert_allclose(s.success_rate,[.75,.5,1]);np.testing.assert_allclose(s.success_pct,[75,50,100])
    def test_total_counts(self):
        s=M.aggregate_runs(clean_fixture());self.assertEqual(s.runs.sum(),12);self.assertEqual(s.measured.sum(),9)
    def test_no_mutation(self):
        a=clean_fixture();b=a.copy();M.aggregate_runs(a);pd.testing.assert_frame_equal(a,b)
    def test_subset(self):
        s=M.aggregate_runs(clean_fixture().iloc[:2]);self.assertEqual(s.runs.sum(),2);self.assertEqual(len(s),2)
    def test_all_missing_group(self):
        a=clean_fixture();a['latency_ms']=np.nan;s=M.aggregate_runs(a);self.assertTrue(s.mean_ms.isna().all());self.assertEqual(s.measured.sum(),0);self.assertEqual(s.runs.sum(),12)
    def test_bad_status(self):
        a=clean_fixture();a.loc[0,'status']='pending'
        with self.assertRaises(ValueError):M.aggregate_runs(a)
    def test_pivot(self):
        p=M.make_pivot(clean_fixture());self.assertTrue(pd.isna(p.loc['planner','error']));self.assertTrue(pd.isna(p.loc['writer','error']));self.assertEqual(p.loc['researcher','ok'],400);self.assertEqual(p.loc['researcher','error'],900)
    def test_join_cardinality(self):
        j=M.join_meta(clean_fixture(),meta_fixture());self.assertEqual(len(j),12);self.assertTrue(j['_merge'].eq('both').all());self.assertAlmostEqual(j.cost_unit.sum(),.436);self.assertEqual(j.cost_unit.count(),11)
    def test_duplicate_meta(self):
        meta=meta_fixture();meta=pd.concat([meta,meta.iloc[:1]])
        with self.assertRaises(ValueError):M.join_meta(clean_fixture(),meta)
    def test_unmatched_meta(self):
        with self.assertRaises(ValueError):M.join_meta(clean_fixture(),meta_fixture().iloc[:2])
    def test_missing_meta_key(self):
        meta=meta_fixture();meta.loc[0,'agent']=None
        with self.assertRaises(ValueError):M.join_meta(clean_fixture(),meta)
    def test_pivot_all_missing(self):
        a=clean_fixture();a['latency_ms']=np.nan;p=M.make_pivot(a);self.assertEqual(p.shape,(3,2));self.assertTrue(p.isna().all().all())
    def test_daily_unknown(self):
        d=M.daily_summary(clean_fixture());self.assertEqual(d.runs.sum(),12);self.assertEqual(d.loc[d.index.isna(),'runs'].iloc[0],1)
        self.assertEqual(d.loc[pd.Timestamp('2026-10-21'),'measured'],6);self.assertEqual(d.loc[pd.Timestamp('2026-10-22'),'measured'],2)
    def test_partial_cost(self):
        a=M.cost_audit(M.join_meta(clean_fixture(),meta_fixture()));self.assertEqual(a['measured'],11);self.assertTrue(a['is_partial']);self.assertAlmostEqual(a['cost_unit'],.436)
    def test_all_missing_cost(self):
        a=M.join_meta(clean_fixture(),meta_fixture());a['cost_unit']=np.nan;r=M.cost_audit(a);self.assertEqual(r['measured'],0);self.assertTrue(np.isnan(r['cost_unit']))
    def test_weighted_means(self):
        self.assertEqual(M.weighted_mean([100,300],[1,3]),250);self.assertTrue(np.isnan(M.weighted_mean([np.nan],[0])))
    def test_bad_weighted_inputs(self):
        for a,b in [([100],[1,2]),([100],[-1]),([np.nan],[1])]:
            with self.subTest(),self.assertRaises(ValueError):M.weighted_mean(a,b)


class PlotCases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
    def tearDown(self):
        M.plt.close('all');self.tmp.cleanup()
    def assert_png(self,name):
        p=self.out/name;self.assertTrue(p.is_file())
        with Image.open(p) as im:im.load();self.assertEqual(im.format,'PNG');self.assertGreater(im.width,700);self.assertGreater(im.height,300)
    def test_bar_image(self):
        a=M.plot_bar(summary_fixture(),self.out);self.assert_png('latency_by_agent.png');self.assertEqual(a['measured'],[3,3,3])
    def test_bar_axis_and_values(self):
        figures=[]
        with mock.patch.object(M,'_save',side_effect=lambda fig,*a:figures.append(fig)):M.plot_bar(summary_fixture(),self.out)
        ax=figures[0].axes[0];np.testing.assert_allclose([b.get_height() for b in ax.patches],[120,1700/3,800/3]);self.assertEqual(ax.get_ylim()[0],0);self.assertIn('ms',ax.get_ylabel());self.assertTrue(ax.get_title())
    def test_hist_reference(self):
        a=M.plot_hist(clean_fixture(),self.out);self.assert_png('latency_hist.png');self.assertEqual(a['counts'],[3,4,1,0,1]);self.assertEqual(a['n'],9);self.assertEqual(a['missing'],3)
    def test_hist_large_input_not_silently_lost(self):
        a=clean_fixture();a.loc[0,'latency_ms']=1500;r=M.plot_hist(a,self.out);self.assertEqual(sum(r['counts']),9);self.assertGreaterEqual(r['edges'][-1],1500)
    def test_all_missing_chart(self):
        a=clean_fixture();a['latency_ms']=np.nan;r=M.plot_hist(a,self.out);self.assertEqual(r['n'],0);self.assertEqual(sum(r['counts']),0);self.assert_png('latency_hist.png')
    def test_bar_all_missing(self):
        s=summary_fixture();s['mean_ms']=np.nan;s['measured']=0;r=M.plot_bar(s,self.out);self.assertEqual(r['measured'],[0,0,0]);self.assert_png('latency_by_agent.png')
    def test_scatter_nine(self):
        r=M.plot_scatter(clean_fixture(),self.out);self.assertEqual(r['n'],9);self.assertEqual(r['excluded'],3);self.assert_png('tokens_latency.png')
    def test_scatter_pair_policy(self):
        a=clean_fixture();a.loc[0,'tokens']=np.nan;r=M.plot_scatter(a,self.out);self.assertEqual(r['n'],8)
    def test_daily_known_and_unknown(self):
        r=M.plot_daily(clean_fixture(),self.out);self.assertEqual(r['unknown_date_runs'],1);self.assertEqual(r['known_dates'],2);self.assert_png('daily_latency.png')
    def test_daily_gap_not_imputed(self):
        a=clean_fixture().iloc[:2].copy();a['day']=pd.to_datetime(['2026-10-21','2026-10-23']);figures=[]
        with mock.patch.object(M,'_save',side_effect=lambda fig,*a:figures.append(fig)):M.plot_daily(a,self.out)
        y=figures[0].axes[0].lines[0].get_ydata();self.assertEqual(len(y),3);self.assertTrue(np.isnan(y[1]))
    def test_daily_all_unknown(self):
        a=clean_fixture();a['day']=pd.NaT;r=M.plot_daily(a,self.out);self.assertEqual(r['unknown_date_runs'],12);self.assertEqual(r['known_dates'],0)


class FilesCases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/'out'
    def tearDown(self):self.tmp.cleanup()
    def export(self):return M.export_tables(clean_fixture(),summary_fixture(),self.out,BASE/'data')
    def test_reference_roundtrip(self):
        r=self.export();self.assertEqual(r['csv_rows'],12);self.assertEqual(r['json_rows'],3);self.assertTrue(r['verified'])
    def test_csv_encoding_and_columns(self):
        self.export();self.assertTrue((self.out/'clean_runs.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
        b,r=M.read_back(self.out);self.assertNotIn('Unnamed: 0',b.columns);self.assertEqual(b.latency_ms.isna().sum(),3)
    def test_json_records(self):
        self.export();b,r=M.read_back(self.out);self.assertIsInstance(r,list);self.assertEqual(r[0]['agent'],'planner');self.assertEqual(r[0]['success_pct'],75)
    def test_all_missing_standard_null(self):
        a=clean_fixture();a['latency_ms']=np.nan;s=summary_fixture();s['measured']=0;s['mean_ms']=np.nan;s['median_ms']=np.nan
        M.export_tables(a,s,self.out,BASE/'data');text=(self.out/'summary.json').read_text();self.assertIn('null',text);self.assertNotIn('NaN',text)
    def test_input_unchanged(self):
        p=BASE/'data/clean_runs.csv';before=p.read_bytes();self.export();self.assertEqual(p.read_bytes(),before)
    def test_leading_zero_ids(self):
        a=clean_fixture();a['run_id']=pd.Series([f'{i:03}' for i in range(1,len(a)+1)],dtype='string')
        M.export_tables(a,summary_fixture(),self.out,BASE/'data');b,r=M.read_back(self.out);self.assertEqual(b.run_id.iloc[0],'001')
    def test_numeric_roundtrip_tampering(self):
        self.export();b,r=M.read_back(self.out);b.loc[0,'latency_ms']=999
        with self.assertRaises(ValueError):M.verify_roundtrip(clean_fixture(),summary_fixture(),b,r)
    def test_summary_roundtrip_tampering(self):
        self.export();b,r=M.read_back(self.out);r[0]['runs']=100
        with self.assertRaises(ValueError):M.verify_roundtrip(clean_fixture(),summary_fixture(),b,r)
    def test_json_flatten(self):
        payload=json.loads((BASE/'data/agents_meta.json').read_text());meta=M.flatten_meta(payload);self.assertEqual(meta.rate_per_1k.tolist(),[.2,.4,.3])
    def test_bad_meta_schema(self):
        with self.assertRaises(ValueError):M.flatten_meta({'agents':[{'agent':'planner'}]})
    def test_duplicate_meta(self):
        with self.assertRaises(ValueError):M.read_meta(BASE/'data/fixtures/duplicate_meta.json')
    def test_broken_and_nonstandard_json(self):
        for filename in ['broken.json','nonstandard.json']:
            with self.subTest(filename=filename),self.assertRaises(ValueError):M.read_meta(BASE/'data/fixtures'/filename)
    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError,'필수 열'):M.read_raw(BASE/'data/fixtures/missing_columns.csv')
    def test_empty_file(self):
        with self.assertRaisesRegex(ValueError,'완전히 빈'):M.read_raw(BASE/'data/fixtures/empty.csv')
    def test_header_only(self):
        with self.assertRaisesRegex(ValueError,'헤더만'):M.read_raw(BASE/'data/fixtures/header_only.csv')
    def test_invalid_encoding(self):
        with self.assertRaises(UnicodeError):M.read_raw(BASE/'data/fixtures/invalid_encoding.csv')
    def test_existing_output_preserved_when_validation_fails(self):
        self.export();before={p.name:p.read_bytes() for p in self.out.iterdir()}
        with mock.patch.object(M,'verify_roundtrip',side_effect=ValueError('검증실패 주입')):
            with self.assertRaises(ValueError):self.export()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})
    def test_output_not_data(self):
        with self.assertRaises(ValueError):M.export_tables(clean_fixture(),summary_fixture(),BASE/'data',BASE/'data')
    def test_replace_failure_rolls_back(self):
        self.export();before={p.name:p.read_bytes() for p in self.out.iterdir()};import support
        real=support.os.replace
        def fail_second(src,dest):
            if Path(src).name.startswith('.eda-stage-'):
                raise PermissionError('잠긴 결과 폴더 상황을 모사')
            return real(src,dest)
        with mock.patch.object(support.os,'replace',side_effect=fail_second):
            with self.assertRaises(PermissionError):self.export()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})


class PipelineCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.meta=BASE/'data/agents_meta.json' if IS_MORE else None
        cls.result=M.run_pipeline(BASE/'data/agent_runs.csv',cls.root/'reference',cls.meta)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def snapshot(self,path):return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    def test_pipeline_counts(self):
        self.assertEqual(self.result['raw_rows'],13);self.assertEqual(self.result['clean_rows'],12);self.assertEqual(self.result['summary_rows'],3);self.assertEqual(self.result['measured'],9)
    def test_quality(self):
        q=json.loads((self.root/'reference/quality.json').read_text());self.assertEqual(q['ok'],9);self.assertEqual(q['error'],3);self.assertEqual(q['day_missing'],1)
    def test_facts_mean_median(self):
        q=json.loads((self.root/'reference/facts.json').read_text());self.assertAlmostEqual(q['overall_mean_ms'],2860/9);self.assertEqual(q['overall_median_ms'],250);self.assertAlmostEqual(q['p90_ms'],580)
    def test_png_outputs(self):
        paths=list((self.root/'reference/charts').glob('*.png'));self.assertEqual(len(paths),2 if not IS_MORE else (4 if IS_ADV else 3))
        for path in paths:
            with Image.open(path) as im:im.verify()
    def test_csv_and_json(self):
        b=pd.read_csv(self.root/'reference/clean_runs.csv');r=json.loads((self.root/'reference/summary.json').read_text());self.assertEqual(len(b),12);self.assertEqual(len(r),3);self.assertEqual(b.latency_ms.isna().sum(),3)
    def test_raw_sha(self):
        f=json.loads((self.root/'reference/facts.json').read_text());self.assertEqual(f['input_sha256'],hashlib.sha256((BASE/'data/agent_runs.csv').read_bytes()).hexdigest())
    def test_bad_inputs_preserve_previous(self):
        out=self.root/'protected';out.mkdir(exist_ok=True);(out/'sentinel.txt').write_text('keep')
        for name in ['empty.csv','header_only.csv','missing_columns.csv','conflicting_id.csv','unknown_agent.csv','unknown_status.csv']:
            before=self.snapshot(out)
            with self.subTest(name=name),self.assertRaises(ValueError):M.run_pipeline(BASE/'data/fixtures'/name,out,self.meta)
            self.assertEqual(self.snapshot(out),before)
    def test_all_missing_pipeline(self):
        out=self.root/'all_missing';r=M.run_pipeline(BASE/'data/fixtures/all_missing_latency.csv',out,self.meta)
        self.assertEqual(r['measured'],0);summary=json.loads((out/'summary.json').read_text());self.assertTrue(all(row['mean_ms'] is None for row in summary))
    def test_other_cwd(self):
        result=subprocess.run([sys.executable,str(CODE),'--out',str(self.root/'other_cwd')],cwd=self.root,text=True,capture_output=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr+result.stdout);self.assertTrue((self.root/'other_cwd/summary.json').exists())
    def test_cli_invalid_input_nonzero(self):
        result=subprocess.run([sys.executable,str(CODE),'--input','data/fixtures/empty.csv','--out',str(self.root/'should_not_exist')],cwd=self.root,text=True,capture_output=True,encoding='utf-8')
        self.assertNotEqual(result.returncode,0);self.assertFalse((self.root/'should_not_exist').exists())
    def test_source_overlap_refused(self):
        with self.assertRaises(ValueError):M.run_pipeline(BASE/'data/agent_runs.csv',BASE/'data',self.meta)
    def test_join_and_pivot(self):
        j=pd.read_csv(self.root/'reference/joined_runs.csv');p=pd.read_csv(self.root/'reference/pivot.csv')
        self.assertEqual(len(j),12);self.assertAlmostEqual(j.cost_unit.sum(),.436);self.assertTrue(j['_merge'].eq('both').all());self.assertEqual(len(p),3)
    def test_invalid_meta_preserves_results(self):
        before=self.snapshot(self.root/'reference')
        for name in ['duplicate_meta.json','unregistered_meta.json','broken.json']:
            with self.subTest(name=name),self.assertRaises(ValueError):M.run_pipeline(BASE/'data/agent_runs.csv',self.root/'reference',BASE/'data/fixtures'/name)
            self.assertEqual(self.snapshot(self.root/'reference'),before)
    def test_extended_input(self):
        out=self.root/'extended';r=M.run_pipeline(BASE/'data/fixtures/extended_runs.csv',out,BASE/'data/fixtures/extended_meta.json')
        self.assertEqual(r['raw_rows'],19);self.assertEqual(r['clean_rows'],18);self.assertEqual(r['summary_rows'],4)
        charts=json.loads((out/'chart_audit.json').read_text());self.assertEqual(sum(charts['hist']['counts']),r['measured'])
    def test_daily_unknown(self):
        d=pd.read_csv(self.root/'reference/daily.csv');self.assertEqual(d['runs'].sum(),12);self.assertEqual(d['day'].isna().sum(),1)
    def test_strict_tokens(self):
        out=self.root/'tokens';M.run_pipeline(BASE/'data/fixtures/fractional_infinite_tokens.csv',out,self.meta)
        d=pd.read_csv(out/'clean_runs.csv');self.assertEqual(d.tokens.isna().sum(),3)


class AssessmentCases(unittest.TestCase):
    def test_accepts_reference(self):self.assertEqual(M.validate_tables(clean_fixture(),summary_fixture()),[])
    def test_rejects_wrong_runs(self):
        s=summary_fixture();s.loc[0,'runs']=3;self.assertTrue(M.validate_tables(clean_fixture(),s))
    def test_rejects_wrong_measurements(self):
        s=summary_fixture();s.loc[0,'measured']=4;self.assertTrue(M.validate_tables(clean_fixture(),s))
    def test_rejects_mean(self):
        s=summary_fixture();s.loc[0,'mean_ms']=0;self.assertTrue(M.validate_tables(clean_fixture(),s))
    def test_rejects_success_denominator(self):
        s=summary_fixture();s['success_rate']=1.;s['success_pct']=100.;self.assertTrue(M.validate_tables(clean_fixture(),s))
    def test_rejects_missing_schema(self):self.assertTrue(M.validate_tables(clean_fixture().drop(columns='status'),summary_fixture()))
    def test_rejects_duplicate_id(self):
        a=clean_fixture();a.loc[0,'run_id']='R02';self.assertTrue(M.validate_tables(a,summary_fixture()))
    def test_missing_not_zero(self):
        a=clean_fixture();a['latency_ms']=np.nan;s=summary_fixture();s['measured']=0;s['mean_ms']=np.nan;s['median_ms']=np.nan
        self.assertEqual(M.validate_tables(a,s),[]);s['mean_ms']=0;self.assertTrue(M.validate_tables(a,s))
    def joined(self):
        a=clean_fixture();j=a.merge(meta_fixture(),on='agent',how='left',indicator=True);j['cost_unit']=j.tokens/1000*j.rate_per_1k;return j
    def test_accepts_join(self):self.assertEqual(M.validate_join(clean_fixture(),self.joined()),[])
    def test_rejects_join_row_inflation(self):
        j=self.joined();j=pd.concat([j,j.iloc[:1]]);self.assertTrue(M.validate_join(clean_fixture(),j))
    def test_rejects_unmatched(self):
        j=self.joined();j['_merge']=j['_merge'].astype('string');j.loc[0,'_merge']='left_only';self.assertTrue(M.validate_join(clean_fixture(),j))
    def test_rejects_cost_tamper(self):
        j=self.joined();j.loc[0,'cost_unit']=99;self.assertTrue(M.validate_join(clean_fixture(),j))


BASIC_NAMES={
'01':['test_axis_values','test_axis_shapes','test_general_input','test_axis_unchanged','test_bad_axis_input'],
'02':['test_row_counts_and_id','test_roles','test_numeric_missing','test_error_rows_and_900_retained','test_raw_unchanged','test_zero_negative_infinite','test_all_missing','test_required_columns'],
'03':['test_reference_summary','test_success_denominator','test_total_counts','test_no_mutation','test_subset','test_all_missing_group','test_bad_status'],
'04':['test_bar_image','test_bar_axis_and_values','test_hist_reference','test_hist_large_input_not_silently_lost','test_all_missing_chart','test_bar_all_missing'],
'05':['test_reference_roundtrip','test_csv_encoding_and_columns','test_json_records','test_all_missing_standard_null','test_input_unchanged','test_leading_zero_ids','test_numeric_roundtrip_tampering','test_summary_roundtrip_tampering'],
'07':['test_pipeline_counts','test_quality','test_facts_mean_median','test_png_outputs','test_csv_and_json','test_raw_sha','test_bad_inputs_preserve_previous','test_all_missing_pipeline','test_other_cwd','test_cli_invalid_input_nonzero','test_source_overlap_refused'],
'08':['test_accepts_reference','test_rejects_wrong_runs','test_rejects_wrong_measurements','test_rejects_mean','test_rejects_success_denominator','test_rejects_missing_schema','test_rejects_duplicate_id','test_missing_not_zero']}
APPLIED_NAMES={
'01':['test_broadcasting','test_calibrate_unchanged','test_valid_mean','test_zero_is_valid'],
'02':['test_dates','test_quality_values','test_quality_ids','test_quality_on_small_input'],
'03':['test_pivot','test_join_cardinality','test_duplicate_meta','test_unmatched_meta','test_missing_meta_key','test_pivot_all_missing'],
'04':['test_scatter_nine','test_scatter_pair_policy'],
'05':['test_json_flatten','test_bad_meta_schema','test_duplicate_meta','test_broken_and_nonstandard_json'],
'07':['test_join_and_pivot','test_invalid_meta_preserves_results'],
'08':['test_accepts_join','test_rejects_join_row_inflation','test_rejects_unmatched','test_rejects_cost_tamper']}
CLASSES={'01':NumpyCases,'02':CleanCases,'03':AggregateCases,'04':PlotCases,'05':FilesCases,'07':PipelineCases,'08':AssessmentCases}


def suite():
    cls=CLASSES[EX]
    names=unittest.defaultTestLoader.getTestCaseNames(cls) if IS_ADV else BASIC_NAMES[EX]+(APPLIED_NAMES[EX] if IS_MORE else [])
    return unittest.TestSuite(cls(name) for name in names)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(suite())
    info={'exercise':EX,'level':LEVEL,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful()}
    if os.environ.get('DAY4_RESULT'):
        Path(os.environ['DAY4_RESULT']).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
