"""选锚与循环失败回归；临时目录里运行，不加载模型、不调用 API。"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "gguf-tools/scripts"
sys.path.insert(0, str(SCRIPTS))
import vloop_anchors as anchors


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def add_anchor(self, number, size=8, source="code_rust"):
        directory = self.root / f"r{number:03d}" / "fp"
        directory.mkdir(parents=True)
        (directory.parent / "seeds.jsonl").write_text(json.dumps({"src": source}) + "\n")
        (directory / "rows.jsonl").write_text(json.dumps({"ids": list(range(size))}) + "\n")
        (directory / "teacher.ptf").write_bytes(b"fixture: selection only checks presence")
        return directory

    def test_mixed_history_keeps_code_and_current_with_real_budget(self):
        for number, size, source in [(46, 8, "code_c"), (47, 8, "code_rust"), (50, 12, "code_python"),
                                     (57, 12, "text_news"), (58, 12, "text_history"), (71, 8, "code_go")]:
            self.add_anchor(number, size, source)
        chosen, skipped = anchors.select_anchors(self.root, 71, 24, require_current=True)
        self.assertEqual([Path(x["path"]).parent.name for x in chosen], ["r050", "r071"])
        self.assertEqual(sum(x["tokens"] for x in chosen), 20)
        self.assertTrue(any("r057" in p and "text" in reason for p, reason in skipped))
        self.assertTrue(any("r058" in p and "text" in reason for p, reason in skipped))

    def test_full_sequences_are_unchanged_and_future_round_is_excluded(self):
        old = self.add_anchor(47, 12)
        now = self.add_anchor(71, 8)
        self.add_anchor(72, 4)
        before = {p: p.read_bytes() for d in (old, now) for p in d.iterdir()}
        chosen, _ = anchors.select_anchors(self.root, 71, 20, require_current=True)
        self.assertEqual([Path(x["path"]).parent.name for x in chosen], ["r047", "r071"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_does_not_fetch_ancient_small_anchor_to_fill_budget(self):
        self.add_anchor(13, 4)
        self.add_anchor(47, 8)
        self.add_anchor(50, 12)
        self.add_anchor(71, 8)
        chosen, _ = anchors.select_anchors(self.root, 71, 24, require_current=True)
        self.assertEqual([Path(x["path"]).parent.name for x in chosen], ["r050", "r071"])

    def test_missing_current_cannot_be_replaced_by_old_data(self):
        self.add_anchor(47)
        with self.assertRaisesRegex(ValueError, "本轮锚缺失"):
            anchors.select_anchors(self.root, 71, 16, require_current=True)

    def test_wrong_domain_current_fails(self):
        self.add_anchor(47)
        self.add_anchor(71, source="text_news")
        with self.assertRaisesRegex(ValueError, "本轮锚不能使用"):
            anchors.select_anchors(self.root, 71, 16, require_current=True)

    def test_oversized_current_fails_instead_of_truncating_teacher_sequence(self):
        self.add_anchor(71, size=32)
        with self.assertRaisesRegex(ValueError, "超过"):
            anchors.select_anchors(self.root, 71, 16, require_current=True)

    def test_unknown_history_and_empty_teacher_are_not_used(self):
        bad = self.add_anchor(69)
        (bad / "teacher.ptf").write_bytes(b"")
        self.add_anchor(70, source="unknown")
        self.add_anchor(71)
        chosen, skipped = anchors.select_anchors(self.root, 71, 32, require_current=True)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(len(skipped), 2)

    def test_general_mode_can_explicitly_keep_both_sources(self):
        self.add_anchor(56)
        self.add_anchor(57, source="text_news")
        chosen, _ = anchors.select_anchors(self.root, 57, 16, "any", True)
        self.assertEqual([x["domain"] for x in chosen], ["code", "text"])

    def test_failed_model_wrapper_cannot_report_loop_done(self):
        scripts = self.root / "gguf-tools/scripts"
        scripts.mkdir(parents=True)
        script = scripts / "vstep_loop_spark.sh"
        script.write_text((SCRIPTS / script.name).read_text().replace('ROOT="$HOME/ds4-main"', 'ROOT=' + shlex.quote(str(self.root)), 1))
        (scripts / "with_model_stopped_spark.sh").write_text("echo '[wrap fixture] filtered output'\nexit 7\n")
        loop = self.root / "gguf/go-onebit/evo/vloop"
        (loop / "r072").mkdir(parents=True)
        (loop / "r072/GEN_DONE").write_text("prepared\n")
        (self.root / "checkpoint").mkdir()
        (self.root / "checkpoint/base.fnv").write_text("fixture\n")
        for name, value in {"current": "checkpoint", "current.kld": "0.20733", "current.same": "85.94",
                            "base.same": "85.94", "ledger.tsv": "round\ttag\n71\tvl071", "LOOP_DONE": "old"}.items():
            (loop / name).write_text(value + "\n")
        before = (loop / "ledger.tsv").read_bytes()
        result = subprocess.run(["bash", str(script), "1", "checkpoint", "10", "3", "pt"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("第 72 轮失败", result.stdout)
        self.assertFalse((loop / "LOOP_DONE").exists())
        self.assertEqual((loop / "ledger.tsv").read_bytes(), before)
        self.assertEqual((loop / "current").read_text().strip(), "checkpoint")



if __name__ == "__main__":
    unittest.main()
