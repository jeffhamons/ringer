"""Checked artifacts and CLI completion are independent observations."""
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ringer


class TerminalParserTests(unittest.TestCase):
    def setUp(self):
        self.engine = ringer.EngineConfig("grok", "grok", (), (), ())
        self.command = ["grok", "--output-format", "streaming-messages-json", "-p", "spec"]

    def parse(self, value, command=None, engine=None):
        return ringer.parse_worker_terminal_state(
            value, engine or self.engine, self.command if command is None else command)

    def test_real_terminal_error_shape_is_fixed_metadata(self):
        result = {"type": "result", "is_error": True, "subtype": "error_max_turns",
                  "num_turns": 3, "result": "untrusted secret thought"}
        self.assertEqual(("ERROR", "error_max_turns", 3), self.parse(json.dumps(result)))
        stream = json.dumps({"type": "assistant", "message": result}) + "\n" + json.dumps(result)
        self.assertEqual(("ERROR", "error_max_turns", 3), self.parse(stream))

    def test_success_and_unknown_error_have_safe_fixed_values(self):
        self.assertEqual(("COMPLETE", None, 2), self.parse(json.dumps(
            {"type": "result", "is_error": False, "subtype": "success", "num_turns": 2})))
        self.assertEqual(("ERROR", "OTHER_ERROR", None), self.parse(json.dumps(
            {"type": "result", "is_error": True, "subtype": "raw private payload", "num_turns": True})))

    def test_nested_quoted_malformed_and_untyped_errors_are_unknown(self):
        result = {"type": "result", "is_error": True, "subtype": "error_max_turns"}
        for value in ({"type": "user", "message": {"content": [result]}},
                      {"type": "text", "text": json.dumps(result)},
                      [result], dict(result, is_error="true"),
                      dict(result, type=["result"]),
                      {"type": "result", "is_error": False, "subtype": "not-success"}):
            with self.subTest(value=value):
                self.assertEqual(("UNKNOWN", None, None), self.parse(json.dumps(value)))
        self.assertEqual(("UNKNOWN", None, None), self.parse('{"type":"result",'))

    def test_cli_protocol_is_gated_by_engine_and_actual_argv(self):
        payload = json.dumps({"type": "result", "is_error": True, "subtype": "error_max_turns"})
        for command in (["grok", "-p", "--output-format", "json"],
                        ["grok", "--output-format", "text"], ["grok", "-p", payload]):
            self.assertEqual(("UNKNOWN", None, None), self.parse(payload, command=command))
        for name in ("opencode", "codex", "mock"):
            engine = ringer.EngineConfig(name, name, (), (), ())
            self.assertEqual(("UNKNOWN", None, None), self.parse(payload, engine=engine))
        self.assertEqual(("UNKNOWN", None, None), self.parse("data: " + payload + "\n\ndata: [DONE]\n"))
        self.assertEqual(("ERROR", "error_max_turns", None),
                         self.parse(payload, command=["grok", "--output-format=json", "-p", "spec"]))


class WorkerTerminalIntegrationTests(unittest.TestCase):
    def runner(self, root, worker_source, *, engine="grok", attempts=2):
        worker = root / "worker.py"
        worker.write_text(worker_source, encoding="utf-8")
        cfg = root / "config.toml"
        cfg.write_text(
            f'state_dir = {json.dumps(str(root / "state"))}\n'
            '[artifact]\nenabled = false\n'
            f'[engines.{engine}]\nbin = {json.dumps(sys.executable)}\n'
            f'args_template = [{json.dumps(str(worker))}, "--output-format", "json"]\n'
            'sandbox_args = []\nfull_access_args = []\n', encoding="utf-8")
        manifest = ringer.Manifest.from_obj({
            "run_name": "terminal-fixture", "workdir": str(root / "tasks"),
            "max_parallel": 1, "worktrees": False,
            "tasks": [{"key": "t", "engine": engine, "spec": "local fixture",
                       "verified": "file exists", "check": "test -s report.txt",
                       "expect_files": ["report.txt"], "max_attempts": attempts}]})
        return ringer.RingerRunner(manifest, ringer.AppConfig.load(cfg), "test", dashboard_enabled=False)

    def run_task(self, runner):
        with mock.patch.dict(os.environ, {"RINGER_NO_SELF_UPDATE": "1"}), contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(runner._run_task(runner.runtimes[0]))

    def test_checked_pair_pass_keeps_terminal_error_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.runner(root,
                "import json\nfrom pathlib import Path\n"
                "Path('report.txt').write_text('valid artifact')\n"
                "print(json.dumps({'type':'result','is_error':True,'subtype':'error_max_turns','num_turns':3}))\n")
            self.run_task(runner)
            runtime = runner.runtimes[0]
            task = runner.state_writer.snapshot()["tasks"][0]
            row = json.loads(runner.config.eval.jsonl_path.read_text().splitlines()[0])
            self.assertEqual("PASS", runtime.final_verdict)
            self.assertEqual(1, runtime.attempts)
            self.assertTrue(runtime.deliverables, "checked artifact still gets harvested")
            for record in (task, row):
                self.assertEqual(0, record["worker_returncode"])
                self.assertFalse(record["worker_timed_out"])
                self.assertEqual("ERROR", record["worker_terminal_status"])
                self.assertEqual("error_max_turns", record["worker_terminal_subtype"])
                self.assertEqual(3, record["worker_terminal_num_turns"])
                self.assertEqual(0, record["check_returncode"])
                self.assertFalse(record["check_timed_out"])

    def test_billed_steps_are_retained_before_engine_down_fast_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.runner(root,
                "import json,sys\n"
                "print(json.dumps({'type':'step_finish','part':{'cost':0.125,'tokens':{'input':12,'output':3}}}))\n"
                "print('402 Payment Required: usage balance exhausted')\n"
                "sys.exit(1)\n", engine="opencode")
            self.run_task(runner)
            runtime = runner.runtimes[0]
            self.assertEqual("ENGINE_DOWN", runtime.final_verdict)
            self.assertEqual(0.125, runtime.cost_usd)
            self.assertEqual(1, runtime.model_steps)
            row = json.loads(runner.config.eval.jsonl_path.read_text().splitlines()[0])
            self.assertEqual("ENGINE_DOWN", row["verdict"])
            self.assertEqual(0.125, row["cost_usd"])
            self.assertEqual("opencode", row["worker_engine"])


if __name__ == "__main__":
    unittest.main()
