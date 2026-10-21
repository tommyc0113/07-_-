"""공개 확인 테스트. 구현 예시가 아니라 입력·출력 계약을 확인합니다."""
import unittest
import tempfile
import subprocess
import sys
import os
import json
import copy
import io
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import main as target
import models
import storage
import io_support
class JsonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"ideas.json"
        self.records=json.loads((HERE/"fixtures"/"valid_ideas.json").read_text(encoding="utf-8"))
    def test_conversion(self):
        idea=models.from_record(self.records[0])
        self.assertEqual(models.to_record(idea),self.records[0])
    def test_export_roles_copy(self):
        idea=models.from_record(self.records[0]); record=models.to_record(idea); record["roles"].append("X")
        self.assertNotIn("X",idea.roles)
    def test_round_trip(self):
        storage.save_records(self.path,self.records)
        self.assertEqual(storage.load_records(self.path),self.records)
        self.assertIn("문서 도우미",self.path.read_text(encoding="utf-8"))
    def test_first_run_missing(self):
        self.assertEqual(storage.load_records(self.path),[])
        self.assertFalse(self.path.exists())
    def test_empty_json(self):
        storage.save_records(self.path,[])
        self.assertEqual(storage.load_records(self.path),[])
    def test_reopen_new_process(self):
        storage.save_records(self.path,self.records)
        script="from pathlib import Path; from storage import load_records; import sys; print(len(load_records(Path(sys.argv[1]))))"
        p=subprocess.run([sys.executable,"-B","-c",script,str(self.path)],cwd=HERE,capture_output=True,text=True,encoding="utf-8",timeout=15)
        self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(p.stdout.strip(),"2")
    def test_missing_keys(self):
        with self.assertRaises(ValueError): models.from_record({"name":"x"})
    def test_save_failure_not_hidden(self):
        with patch.object(io_support.os,"replace",side_effect=PermissionError("모의 오류")):
            with self.assertRaises(PermissionError): storage.save_records(self.path,self.records)
        self.assertFalse(self.path.exists())
    def test_failed_replace_preserves_old_file(self):
        storage.save_records(self.path,self.records); before=self.path.read_bytes()
        changed=copy.deepcopy(self.records); changed[0]["name"]="새 이름"
        with patch.object(io_support.os,"replace",side_effect=PermissionError("모의 오류")):
            with self.assertRaises(PermissionError): storage.save_records(self.path,changed)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(list(self.path.parent.glob(".ideas_*.tmp")),[])

if __name__ == "__main__":
    unittest.main(verbosity=2)
