#!/usr/bin/env python3
"""`check_timeout_s`: per-task budget for the check command.

Regression proof for making the check timeout configurable. The default was
a hardcoded 60s that silently killed any check running a real build/test
suite (a ~770-test pytest run measured at 140-240s reported as TIMEOUT even
though the underlying work was correct). The default is now 300s, overridable
per task; 0 is the sentinel meaning "use CHECK_TIMEOUT_S".
"""
from __future__ import annotations

import sys
import asyncio
import contextlib
import os
import shlex
import signal
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import CHECK_TIMEOUT_S, TaskSpec, Verifier  # noqa: E402


@unittest.skipUnless(os.name == "posix", "process groups require POSIX")
class CheckCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_kills_and_reaps_actual_check_and_child(self) -> None:
        await self.assert_cancelled_check_cleanup(timeout=60, cancel_delay=0)

    async def test_cancellation_during_timeout_cleanup_kills_check_and_child(self) -> None:
        # Allow real Python startup under concurrent test load before provoking
        # the timeout grace phase; 50ms could terminate the fixture before ready.
        await self.assert_cancelled_check_cleanup(timeout=1, cancel_delay=1.1)

    async def assert_cancelled_check_cleanup(self, timeout, cancel_delay) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            fixture = path / "check.py"
            fixture.write_text(
                "import os, signal, subprocess, sys, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
                "Path('pids').write_text(f'{os.getpid()} {child.pid}')\n"
                "time.sleep(60)\n", encoding="utf-8")
            command = "exec " + shlex.join([sys.executable, str(fixture)])
            task = asyncio.create_task(Verifier._run_check(command, path, timeout=timeout))
            pids = []
            try:
                for _ in range(100):
                    if (path / "pids").exists():
                        pids = [int(pid) for pid in (path / "pids").read_text().split()]
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(2, len(pids), "fixture must actually launch its child")
                await asyncio.sleep(0.05)  # child installs its SIGTERM handler
                await asyncio.sleep(cancel_delay)
                task.cancel()
                started = time.monotonic()
                await asyncio.sleep(0.05)
                task.cancel()  # repeated shutdown must not cancel cleanup
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
                self.assertLess(time.monotonic() - started, 9)
                for _ in range(100):
                    if all(not self._alive(pid) for pid in pids):
                        break
                    await asyncio.sleep(0.02)
                self.assertTrue(all(not self._alive(pid) for pid in pids),
                                "cancelling verification must not leave check or child alive")
            finally:
                if pids:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pids[0], signal.SIGKILL)
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    async def test_check_stdin_is_closed_and_output_is_drained(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            command = "exec " + shlex.join([
                sys.executable, "-c", "import sys; print('closed' if sys.stdin.read() == '' else 'open')"])
            rc, timed_out, output = await Verifier._run_check(command, Path(root), timeout=5)
            self.assertEqual(0, rc)
            self.assertFalse(timed_out)
            self.assertEqual("closed\n", output)

    async def test_check_timeout_still_reports_timeout_and_reaps(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            command = "exec " + shlex.join([sys.executable, "-c", "import time; time.sleep(60)"])
            rc, timed_out, output = await Verifier._run_check(command, Path(root), timeout=0.05)
            self.assertNotEqual(0, rc)
            self.assertTrue(timed_out)
            self.assertIn("check timed out after 0.05s", output)


class CheckTimeoutTests(unittest.TestCase):
    def _task(self, **extra: object) -> dict[str, object]:
        base: dict[str, object] = {
            "key": "t",
            "spec": "a self-contained spec long enough to pass validation " * 2,
            "check": "true",
        }
        base.update(extra)
        return base

    def test_default_is_zero_meaning_use_global(self) -> None:
        # 0 is the sentinel; the check path resolves it to CHECK_TIMEOUT_S.
        task = TaskSpec.from_obj(self._task())
        self.assertEqual(0, task.check_timeout_s)
        self.assertEqual(CHECK_TIMEOUT_S, task.check_timeout_s or CHECK_TIMEOUT_S)

    def test_global_default_is_300(self) -> None:
        # The whole point of the fix: 60s silently killed real build/test
        # suites, so the floor is now 300s.
        self.assertEqual(300, CHECK_TIMEOUT_S)

    def test_explicit_value_parses_and_wins(self) -> None:
        task = TaskSpec.from_obj(self._task(check_timeout_s=900))
        self.assertEqual(900, task.check_timeout_s)
        self.assertEqual(900, task.check_timeout_s or CHECK_TIMEOUT_S)

    def test_negative_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "check_timeout_s must not be negative"):
            TaskSpec.from_obj(self._task(check_timeout_s=-1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
