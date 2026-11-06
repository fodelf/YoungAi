"""离线回归：混域锚、长短锚预算、留出跳题与配对统计。不加载模型、不调用 API。"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "gguf-tools/scripts"
sys.path.insert(0, str(SCRIPTS))
import vloop_heldout as heldout
import gap_scan as gap


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.training = [{"ids": [1, 2]}, {"text": "training"}]
        self.questions = [{"id": f"q{i}", "src": family, "q": f"question:{i};"}
                          for i, family in enumerate(("general", "code", "math", "writing"))]
        self.rows = [{"prefix": q["q"], "text": "answer"} for q in self.questions]

    def test_dropped_first_and_middle_questions_do_not_shift_families(self):
        records = [(10, 1.0, -1.0, 3, 10), (20, 2.0, 0.1, 3, 11), (15, 1.5, 0.2, 5, 14)]
        mapping = heldout.map_segments(records, self.training, self.rows, self.questions, gap.FAMS)
        self.assertEqual(mapping, {0: (1, "q1", "code"), 1: (1, "q1", "code"), 2: (3, "q3", "writing")})
        families, _ = gap.fam_agg(records, mapping)
        self.assertEqual(families["code"][:2], [3.0, 30])
        self.assertEqual(families["general"][1], 0)

    def test_prefix_collision_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "多道题"):
            heldout.fam_of_prefix("question:0; question:1;", self.questions)

    def test_out_of_range_document_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "不在留出范围"):
            heldout.map_segments([(10, 1, -1, 20, 3)], self.training, self.rows, self.questions, gap.FAMS)

    def test_pair_rejects_changed_sample_or_token_count(self):
        original = [(10, 1.0, -1.0, 3, 10)]
        for wrong in [(11, 1.0, -1.0, 3, 10), (10, 1.0, -1.0, 3, 11)]:
            with self.assertRaisesRegex(ValueError, "不能配对"):
                heldout.validate_pair(original, [wrong])

    def test_old_five_column_dump_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heldout.tsv"
            path.write_text("text\t1\t10\t1.0\t-1.0\n")
            with self.assertRaisesRegex(ValueError, "缺真实篇号"):
                heldout.read_heldout(path)

    def test_standard_error_uses_documents_not_segments(self):
        before = [(10, 2., -1., 3, 10), (10, 2., -1., 3, 11), (20, 2., -1., 5, 14)]
        after = [(10, 1., -1., 3, 10), (10, 1., -1., 3, 11), (20, 3., -1., 5, 14)]
        paired = heldout.paired_gain(before, after)
        self.assertEqual(paired["documents"], 2)
        self.assertAlmostEqual(paired["gain"], 0.025)
        self.assertAlmostEqual(paired["se"], 0.075)

    def test_real_round71_regression(self):
        fixture = Path(__file__).parent / "fixtures/vloop"
        before, after = (heldout.read_heldout(fixture / f"vl071_e0{i}.tsv") for i in (0, 1))
        paired = heldout.paired_gain(before, after)
        self.assertEqual((paired["documents"], paired["segments"]), (13, 68))
        self.assertAlmostEqual(paired["gain"], -0.000711163908773, places=12)
        self.assertAlmostEqual(paired["se"], 0.000310935826865, places=12)
        self.assertAlmostEqual(paired["z"], -2.28717261675, places=9)
        families = ["general", "general", "general", "code", "math", "math", "math", "math", "writing",
                    "writing", "data", "writing", "writing", "code", "svgchar", "code", "svgchar"]
        questions = [{"id": f"q{i}", "src": family, "q": f"question:{i};"} for i, family in enumerate(families)]
        rows = [{"prefix": q["q"], "text": "answer"} for q in questions]
        mapping = heldout.map_segments(before, [{"ids": [1, 2]}] * 59, rows, questions, gap.FAMS)
        f0, _ = gap.fam_agg(before, mapping)
        f1, _ = gap.fam_agg(after, mapping)
        self.assertAlmostEqual(100 * (f1["writing"][0] / f0["writing"][0] - 1), -0.48621661768, places=8)
        self.assertEqual(f1["code"][1], 19708)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_epoch_follows_judged_checkpoint_including_fallback(self):
        run = self.root / "pt-vl066"
        run.mkdir()
        for epoch in (0, 1, 2):
            (run / f"heldout_e{epoch:02d}.tsv").touch()
        ledger = self.root / "ledger.tsv"
        for epoch in (2, 1):
            ledger.write_text(f"66\tvl066\t0-39\tpt\tgguf/v41/posttrain/pt-vl066/ckpt_e{epoch:02d}v17\n")
            self.assertEqual(heldout.selected_epoch("vl066", run, ledger), epoch)
        ledger.write_text("")
        with self.assertRaisesRegex(ValueError, "不能猜"):
            heldout.selected_epoch("vl066", run, ledger)

    def test_exposure_counts_only_questions_merged_into_training(self):
        think = self.root / "r071/think"
        think.mkdir(parents=True)
        questions = [{"id": f"q{i}", "q": f"question:{i};", "src": family}
                     for i, family in enumerate(("code", "math", "writing"))]
        (think / "q.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
        training = [{"text": "answer", "prefix": "question:1;<think>"},
                    {"text": "short answer", "prefix": "question:0;"}, {"ids": [1, 2]}]
        with patch.object(gap, "VL", str(self.root)):
            actual = gap.exposure("vl071", training)
        self.assertEqual(actual, dict.fromkeys(gap.FAMS, 0) | {"math": 1})

    def test_old_ledger_cannot_create_next_quota(self):
        (self.root / "ledger.tsv").write_text("tag\tkl0\tkl1\nvl071\t0.0968\t0.0975\n")
        args = argparse.Namespace(budget=60, minq=4, maxq=24)
        with patch.object(gap, "GAP", str(self.root)):
            with self.assertRaisesRegex(ValueError, "旧的猜测映射"):
                gap.cmd_next(args)
        self.assertFalse((self.root / "next.json").exists())

    def test_rebuild_cannot_overwrite_existing_ledger(self):
        ledger = self.root / "ledger.tsv"
        ledger.write_text("existing ledger\n")
        with self.assertRaisesRegex(ValueError, "避免覆盖"):
            gap.cmd_rebuild(argparse.Namespace(out=str(self.root)))
        self.assertEqual(ledger.read_text(), "existing ledger\n")

    def test_readonly_scan_writes_nothing(self):
        fixture = Path(__file__).parent / "fixtures/vloop"
        run = self.root / "gguf/v41/posttrain/pt-vl071"
        run.mkdir(parents=True)
        for epoch in (0, 1):
            (run / f"heldout_e{epoch:02d}.tsv").write_bytes((fixture / f"vl071_e{epoch:02d}.tsv").read_bytes())
        families = ["general", "general", "general", "code", "math", "math", "math", "math", "writing",
                    "writing", "data", "writing", "writing", "code", "svgchar", "code", "svgchar"]
        questions = [{"id": f"q{i}", "src": family, "q": f"question:{i};"} for i, family in enumerate(families)]
        inputs = ([{"ids": [1, 2]}] * 59, [{"prefix": q["q"], "text": "answer"} for q in questions], questions)
        args = argparse.Namespace(tag="vl071", dir=str(run), epoch=None, read_only=True)
        before = set(self.root.rglob("*"))
        with patch.object(gap, "run_inputs", return_value=inputs), patch.object(gap, "VL", str(self.root)), \
             patch.object(gap, "GAP", str(self.root / "gap")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gap.cmd_scan(args), 0)
        self.assertEqual(set(self.root.rglob("*")), before)


if __name__ == "__main__":
    unittest.main()
