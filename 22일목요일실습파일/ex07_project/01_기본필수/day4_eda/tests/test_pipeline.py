"""제출용 독립 테스트: 이 프로젝트만 복사해도 실행됩니다."""
from pathlib import Path
import hashlib,json,tempfile,unittest
import pandas as pd
from main import run_pipeline, BASE_DIR, LEVEL
from eda.clean import clean_runs
from eda.analyze import aggregate_runs
from support import read_raw


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)/'out'
        self.meta=BASE_DIR/'data/agents_meta.json' if LEVEL!='basic' else None
    def tearDown(self):self.temp.cleanup()
    def test_reference(self):
        result=run_pipeline(BASE_DIR/'data/agent_runs.csv',self.out,self.meta)
        self.assertEqual(result['clean_rows'],12);self.assertEqual(result['summary_rows'],3)
        self.assertTrue((self.out/'charts/latency_hist.png').is_file())
        summary=pd.DataFrame(json.loads((self.out/'summary.json').read_text(encoding='utf-8')))
        self.assertEqual(summary.runs.sum(),12);self.assertEqual(summary.measured.sum(),9)
    def test_all_missing(self):
        r=read_raw(BASE_DIR/'data/fixtures/all_missing_latency.csv')
        c=clean_runs(r);s=aggregate_runs(c)
        self.assertEqual(s.measured.sum(),0);self.assertTrue(s.mean_ms.isna().all())
    def test_conflict_stops_and_preserves_output(self):
        self.out.mkdir();(self.out/'keep.txt').write_text('old',encoding='utf-8')
        path=BASE_DIR/'data/fixtures/conflicting_id.csv';before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):run_pipeline(path,self.out,self.meta)
        self.assertEqual((self.out/'keep.txt').read_text(),'old');self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):run_pipeline(BASE_DIR/'data/fixtures/empty.csv',self.out,self.meta)
        self.assertFalse(self.out.exists())

if __name__=='__main__':unittest.main(verbosity=2)
