#!/usr/bin/env python3
"""Engine spawn failures are infrastructure evidence, not model evidence.

Observed in the field (2026-07-14): an engine wrapper script exited 127
("command not found") on every task because the underlying binary was not on
ringer's PATH. Each attempt wrote a FAIL row to the per-model log — 36 rows —
silently dragging an innocent model's pass rate from 1.00 to 0.40, and every
task burned its retry on a failure a retry cannot fix.

These tests pin the fix:
  * verdict is INFRA, not FAIL, when the worker never ran
  * no model-log row is written for such attempts
  * the retry is skipped (attempts == 1)
  * the run summary names the failure class and the fix
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import (  # noqa: E402
    VerifyResult,
    WorkerResult,
    verdict_for,
    worker_never_ran,
    worker_zero_token_early_exit,
)
from ringer import AppConfig, ArtifactConfig, EngineConfig, EvalConfig, Manifest, RingerRunner, load_engines  # noqa: E402


def toml_string(value: object) -> str:
    return json.dumps(str(value))


FAILING_VERIFY = VerifyResult(
    ok=False, check_returncode=1, check_timed_out=False, raw_output_excerpt=""
)
PASSING_VERIFY = VerifyResult(
    ok=True, check_returncode=0, check_timed_out=False, raw_output_excerpt=""
)


class WorkerNeverRanTests(unittest.TestCase):
    def test_command_not_found_without_tokens_is_infra(self) -> None:
        worker = WorkerResult(returncode=127, timed_out=False, tokens=None)
        self.assertTrue(worker_never_ran(worker))
        self.assertEqual("INFRA", verdict_for(worker, FAILING_VERIFY))

    def test_not_executable_without_tokens_is_infra(self) -> None:
        worker = WorkerResult(returncode=126, timed_out=False, tokens=None)
        self.assertTrue(worker_never_ran(worker))
        self.assertEqual("INFRA", verdict_for(worker, FAILING_VERIFY))

    def test_infra_wins_over_a_passing_check(self) -> None:
        # A check may pass against artifacts left by an earlier run; a model
        # that never ran must not be credited with that PASS.
        worker = WorkerResult(returncode=127, timed_out=False, tokens=None)
        self.assertEqual("INFRA", verdict_for(worker, PASSING_VERIFY))

    def test_exit_127_with_token_usage_is_model_evidence(self) -> None:
        # If the engine reported token usage, a model ran — a late 127 from a
        # wrapper's cleanup must still count as a model FAIL, not INFRA.
        worker = WorkerResult(returncode=127, timed_out=False, tokens=512)
        self.assertFalse(worker_never_ran(worker))
        self.assertEqual("FAIL", verdict_for(worker, FAILING_VERIFY))

    def test_ordinary_failure_without_tokens_is_not_infra(self) -> None:
        # Plenty of engines crash with rc=1 and no parseable token line; that
        # is still model evidence and must keep reaching the log.
        worker = WorkerResult(returncode=1, timed_out=False, tokens=None)
        self.assertFalse(worker_never_ran(worker))
        self.assertEqual("FAIL", verdict_for(worker, FAILING_VERIFY))

    def test_spawn_exception_before_exec_never_ran(self) -> None:
        worker = WorkerResult(
            returncode=None, timed_out=False, tokens=None, error="spawn failed"
        )
        self.assertTrue(worker_never_ran(worker))
        # verdict stays ERROR (the error branch wins); the shared
        # worker_never_ran gate is what keeps it out of the model log.
        self.assertEqual("ERROR", verdict_for(worker, FAILING_VERIFY))

    def test_timeout_is_model_evidence(self) -> None:
        worker = WorkerResult(returncode=None, timed_out=True, tokens=None)
        self.assertFalse(worker_never_ran(worker))
        self.assertEqual("TIMEOUT", verdict_for(worker, FAILING_VERIFY))


class ZeroTokenEarlyExitTests(unittest.TestCase):
    """A fast, tokenless, non-timeout failure on a token-reporting engine is
    infrastructure evidence (kept out of the model log); grok is exempt."""

    def engine(self, name: str, *, reports_tokens: bool = True) -> tuple[str, EngineConfig]:
        engines = load_engines({})
        base = next(iter(engines.values()))
        return name, __import__("dataclasses").replace(
            base, token_regex=r"tokens used:?\s*([0-9,]+)" if reports_tokens else None, token_format=None
        )

    def check(self, worker: WorkerResult, name: str = "opencode", *, reports_tokens: bool = True, duration_s: float = 5.0) -> bool:
        engine_name, engine = self.engine(name, reports_tokens=reports_tokens)
        return worker_zero_token_early_exit(worker, engine_name, engine, duration_s=duration_s)

    def test_fast_zero_token_failure_is_infra(self) -> None:
        self.assertTrue(self.check(WorkerResult(returncode=1, timed_out=False, tokens=None)))
        self.assertTrue(self.check(WorkerResult(returncode=1, timed_out=False, tokens=0)))

    def test_narrow_exclusions(self) -> None:
        failing = WorkerResult(returncode=1, timed_out=False, tokens=None)
        self.assertFalse(self.check(failing, "grok"))
        self.assertFalse(self.check(failing, reports_tokens=False))
        self.assertFalse(self.check(failing, duration_s=300.0))
        self.assertFalse(self.check(WorkerResult(returncode=1, timed_out=False, tokens=900)))
        self.assertFalse(self.check(WorkerResult(returncode=None, timed_out=True, tokens=None)))
        self.assertFalse(self.check(WorkerResult(returncode=0, timed_out=False, tokens=None)))


class ZeroTokenRowTests(unittest.TestCase):
    """A fast tokenless FAIL is written but flagged evidence_excluded; other
    verdicts (TIMEOUT, ENGINE_DOWN) on the same worker shape are never touched."""

    def log_one(self, verdict: str) -> dict[str, object]:
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_path = root / "runs.jsonl"
            engine = EngineConfig(
                name="opencode",
                bin="opencode",
                args_template=("run", "--model", "{model}", "{spec}"),
                full_access_args=(),
                sandbox_args=(),
                token_regex=r"tokens used:?\s*([0-9,]+)",
                token_format=None,
            )
            config = AppConfig(
                path=None,
                identity_default=None,
                state_dir=root / "state",
                dashboard_port_base=8787,
                hud_port=8700,
                hud_app_path=None,
                allow_full_access=False,
                eval=EvalConfig(backend="jsonl", jsonl_path=log_path),
                engines={"opencode": engine},
                artifact=ArtifactConfig(
                    enabled=False,
                    out_template=str(root / "live.html"),
                    report_template=str(root / "report.html"),
                    index_out=root / "index.html",
                ),
            )
            manifest = Manifest.from_obj(
                {
                    "run_name": "zero-token-row",
                    "workdir": str(root / "work"),
                    "tasks": [
                        {
                            "key": "t",
                            "spec": "Create output.txt with the requested value and keep the change scoped.",
                            "check": "test -s output.txt",
                            "expect_files": ["output.txt"],
                            "verified": "output.txt exists",
                            "engine": "opencode",
                            "model": "openrouter/z-ai/glm-5.3",
                        }
                    ],
                }
            )
            runner = RingerRunner(manifest, config=config, identity="tester", dashboard_enabled=False)
            runtime = runner.runtimes[0]
            runner._log_attempt(
                runtime,
                runtime.task.spec,
                False,
                WorkerResult(returncode=1, timed_out=False, tokens=None),
                VerifyResult(ok=False, check_returncode=1, check_timed_out=False, raw_output_excerpt="x"),
                verdict,
                5000,
            )
            lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            self.assertEqual(1, len(lines), f"{verdict}: row must be written, not dropped")
            return json.loads(lines[0])

    def test_fast_zero_token_fail_is_written_but_excluded(self) -> None:
        payload = self.log_one("FAIL")
        self.assertIs(payload["evidence_excluded"], True)
        self.assertTrue(payload["invalidation_reason"])
        from ringer import model_log_row_is_invalidated

        self.assertTrue(model_log_row_is_invalidated(payload))

    def test_timeout_and_engine_down_rows_are_not_excluded(self) -> None:
        for verdict in ("TIMEOUT", "ENGINE_DOWN"):
            payload = self.log_one(verdict)
            self.assertNotEqual(True, payload.get("evidence_excluded"), verdict)
            self.assertFalse(payload.get("invalidation_reason"), verdict)


class SpawnFailureEndToEndTests(unittest.TestCase):
    def test_spawn_failure_skips_model_log_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            home = root / "home"
            ringer_home = root / "ringer-home"
            state_dir = root / "state"
            workdir = root / "work"
            config_path = root / "config.toml"
            manifest_path = root / "manifest.json"
            model_log = root / "runs.jsonl"

            home.mkdir()
            ringer_home.mkdir()

            # The engine is a wrapper that cannot find its real binary — the
            # exact shape of the field failure (opencode-sandboxed.sh: 127).
            config_path.write_text(
                "\n".join(
                    [
                        f"state_dir = {toml_string(state_dir)}",
                        "",
                        "[eval]",
                        'backend = "jsonl"',
                        f"jsonl_path = {toml_string(model_log)}",
                        "",
                        "[artifact]",
                        "enabled = false",
                        "",
                        "[engines.broken]",
                        'bin = "/bin/sh"',
                        "args_template = [",
                        '  "-c",',
                        '  "echo wrapper: engine not found on PATH; exit 127",',
                        '  "wrapper",',
                        '  "{spec}",',
                        "]",
                        "sandbox_args = []",
                        "full_access_args = []",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            manifest_path.write_text(
                json.dumps(
                    {
                        "run_name": "spawn-failure-test",
                        "workdir": str(workdir),
                        "max_parallel": 1,
                        "worktrees": False,
                        "tasks": [
                            {
                                "key": "spawn-task",
                                "engine": "broken",
                                "spec": (
                                    "You will never run: the engine wrapper exits 127 "
                                    "before any model is invoked. This spec exists so "
                                    "the manifest is well-formed."
                                ),
                                "check": (
                                    "test -f never-created.txt || "
                                    "{ echo FAIL: worker never ran; exit 1; }"
                                ),
                                "expect_files": ["never-created.txt"],
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["HOME"] = str(home)
            env["RINGER_HOME"] = str(ringer_home)
            env["XDG_CONFIG_HOME"] = str(root / "xdg-config")

            proc = subprocess.run(
                [
                    sys.executable,
                    "ringer.py",
                    "run",
                    str(manifest_path),
                    "--config",
                    str(config_path),
                    "--no-dashboard",
                    "--identity",
                    "spawn-failure-test",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )

            combined_output = proc.stdout + proc.stderr

            # Verdict INFRA, exactly one attempt — the retry was not spent.
            self.assertRegex(
                combined_output,
                re.compile(r"^spawn-task\s+fail\s+INFRA\s+1\s+", re.MULTILINE),
                combined_output,
            )

            # The summary names the failure class and points at the fix.
            self.assertIn("engine spawn failures (INFRA)", combined_output)

            # No model-log row was written for an attempt with no model.
            if model_log.exists():
                rows = [
                    json.loads(line)
                    for line in model_log.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                offending = [r for r in rows if r.get("task_key") == "spawn-task"]
                self.assertEqual([], offending, rows)

            # The worker log says why nothing was logged and why no retry ran.
            worker_log = (workdir / "spawn-task" / "worker.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("no model evidence for this attempt", worker_log)
            self.assertIn("Not retrying", worker_log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
