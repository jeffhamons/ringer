#!/usr/bin/env python3
"""Persistent HUD opener behavior."""
from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ringer  # noqa: E402
from ringer import AppConfig, ArtifactConfig, EvalConfig, ensure_hud_running  # noqa: E402


def config(root: Path) -> AppConfig:
    return AppConfig(
        path=None,
        identity_default=None,
        state_dir=root / "state",
        dashboard_port_base=8787,
        hud_port=8700,
        hud_app_path=None,
        allow_full_access=False,
        eval=EvalConfig(backend="jsonl", jsonl_path=root / "eval.jsonl"),
        engines={},
        artifact=ArtifactConfig(
            enabled=False,
            out_template=str(root / "live.html"),
            report_template=str(root / "report.html"),
            index_out=root / "index.html",
        ),
    )


class HudSingleTabTests(unittest.TestCase):
    def setUp(self) -> None:
        heartbeat = mock.patch.object(ringer, "hud_has_recent_viewer", return_value=False)
        heartbeat.start()
        self.addCleanup(heartbeat.stop)

    def test_already_alive_does_not_open_browser(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            opened: list[str] = []
            original_alive = ringer.hud_is_healthy
            original_open = ringer.open_in_browser
            original_popen = ringer.subprocess.Popen
            try:
                ringer.hud_is_healthy = lambda _port, _config: True
                ringer.open_in_browser = opened.append

                def fail_popen(*_args: object, **_kwargs: object) -> object:
                    raise AssertionError("should not spawn a HUD that is already alive")

                ringer.subprocess.Popen = fail_popen
                ensure_hud_running(config(Path(temp)), open_browser=True)
            finally:
                ringer.hud_is_healthy = original_alive
                ringer.open_in_browser = original_open
                ringer.subprocess.Popen = original_popen
            self.assertEqual([], opened)

    def test_dead_then_alive_opens_once_after_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            opened: list[str] = []
            spawned: list[object] = []
            alive_results = iter([False, True, True])
            original_alive = ringer.hud_is_healthy
            original_open = ringer.open_in_browser
            original_popen = ringer.subprocess.Popen
            try:
                ringer.hud_is_healthy = lambda _port, _config: next(alive_results)
                ringer.open_in_browser = opened.append

                def fake_popen(*args: object, **_kwargs: object) -> object:
                    spawned.append(args)
                    return mock.Mock(poll=mock.Mock(return_value=None))

                ringer.subprocess.Popen = fake_popen
                ensure_hud_running(config(Path(temp)), open_browser=True)
            finally:
                ringer.hud_is_healthy = original_alive
                ringer.open_in_browser = original_open
                ringer.subprocess.Popen = original_popen
            self.assertEqual(1, len(spawned))
            self.assertEqual(["http://127.0.0.1:8700"], opened)


if __name__ == "__main__":
    unittest.main(verbosity=2)
