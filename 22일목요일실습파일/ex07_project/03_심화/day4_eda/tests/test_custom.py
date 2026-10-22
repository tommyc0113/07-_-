"""심화 추가 테스트: 확장 역할과 메타키 충돌 정책을 검증합니다."""
import unittest
from pathlib import Path
from main import BASE_DIR
from support import read_raw
from eda.clean import clean_runs
from eda.analyze import join_meta
from eda.io_utils import read_meta

class CustomTests(unittest.TestCase):
    def test_extended_roles(self):
        raw=read_raw(BASE_DIR/"data/fixtures/extended_runs.csv")
        meta=read_meta(BASE_DIR/"data/fixtures/extended_meta.json")
        clean=clean_runs(raw,allowed_agents=tuple(meta["agent"]),strict_tokens=True)
        # TODO TEST_EXTENDED: 정제18행, 역할4개, 결합후18행을 assert로 검사하세요.
        raise NotImplementedError("TODO TEST_EXTENDED")
