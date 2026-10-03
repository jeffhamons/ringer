"""A worker wrapper owns descendants even when they start a new session."""
import asyncio
import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ringer


HELPER_SOURCE = r'''
import os, signal, subprocess, sys, time
from pathlib import Path
stopping = False
def stop(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
server = subprocess.Popen([sys.executable, '-c',
    "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('server.pid').write_text(str(os.getpid())); time.sleep(60)"],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
client = subprocess.Popen([sys.executable, '-c',
    "import os,time; from pathlib import Path; Path('client.pid').write_text(str(os.getpid())); time.sleep(60)"])
Path('helper.pid').write_text(str(os.getpid()))
while not stopping:
    time.sleep(0.02)
client.terminate()
client.wait(timeout=5)
server.terminate()
try:
    server.wait(timeout=5)
except subprocess.TimeoutExpired:
    server.kill()
    server.wait(timeout=5)
sys.exit(241)
'''


@unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
class WorkerCleanupTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, root):
        helper = root / "helper.py"
        helper.write_text(HELPER_SOURCE, encoding="utf-8")
        engine = ringer.EngineConfig("fixture", sys.executable, (str(helper),), (), ())
        runner = object.__new__(ringer.RingerRunner)
        runner.config = SimpleNamespace(engines={"fixture": engine}, allow_full_access=False,
                                        steering=ringer.SteeringConfig())
        runner.lock = threading.RLock()
        runner.active_processes = {}
        return runner

    def runtime(self, root, timeout=1):
        root.mkdir()
        return ringer.TaskRuntime(ringer.TaskSpec(key="fixture", spec="local only", engine="fixture",
                                                timeout_s=timeout, max_attempts=1, check="true"),
                                  root, root / "worker.log")

    async def assert_gone(self, root):
        paths = [root / (name + ".pid") for name in ("helper", "server", "client")]
        self.assertTrue(all(path.exists() for path in paths), "fixture must start every process")
        pids = [int(path.read_text()) for path in paths]
        for _ in range(100):
            if all(not self.alive(pid) for pid in pids):
                return
            await asyncio.sleep(0.02)
        self.fail("worker completion left an owned helper, client, or separate-session server alive")

    @staticmethod
    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    def cleanup_fixture(root):
        for path in root.glob("*.pid"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(path.read_text()), signal.SIGKILL)

    async def test_timeout_cleans_separate_server_and_next_job_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.runner(root)
            runtime = self.runtime(root / "first")
            try:
                with mock.patch.object(ringer.RingerRunner, "_write_worker_output", lambda _fh, _chunk: None):
                    result = await runner._run_worker(runtime, runtime.task.spec, 1)
                self.assertTrue(result.timed_out)
                await self.assert_gone(runtime.taskdir)
                self.assertEqual({}, runner.active_processes)
                # A subsequent fresh task uses the same runner without stale
                # ownership or infrastructure state from the stopped job.
                next_dir = root / "next"
                next_runtime = self.runtime(next_dir)
                engine = runner.config.engines["fixture"]
                runner.config.engines["fixture"] = ringer.EngineConfig(
                    engine.name, sys.executable, ("-c", "from pathlib import Path; Path('done').write_text('ready')"), (), ())
                result = await runner._run_worker(next_runtime, next_runtime.task.spec, 1)
                self.assertEqual(0, result.returncode)
                self.assertFalse(result.timed_out)
                self.assertEqual("ready", (next_dir / "done").read_text())
            finally:
                self.cleanup_fixture(runtime.taskdir)

    async def test_shutdown_cleans_descendants_and_leaves_unrelated_group_alive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.runner(root)
            runtime = self.runtime(root / "shutdown", timeout=60)
            outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                         start_new_session=True, stdout=subprocess.DEVNULL)
            worker = asyncio.create_task(runner._run_worker(runtime, runtime.task.spec, 1))
            try:
                for _ in range(100):
                    if (runtime.taskdir / "server.pid").exists() and (runtime.taskdir / "client.pid").exists():
                        break
                    await asyncio.sleep(0.02)
                await runner.kill_all_workers()
                await asyncio.wait_for(worker, 10)
                await self.assert_gone(runtime.taskdir)
                self.assertIsNone(outsider.poll(), "cleanup must not target unrelated process groups")
                self.assertEqual({}, runner.active_processes)
            finally:
                self.cleanup_fixture(runtime.taskdir)
                if not worker.done():
                    worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                outsider.kill()
                outsider.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
