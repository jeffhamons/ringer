"""Dashboard identity/lifecycle regressions; real servers bind ephemeral ports."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from urllib.request import urlopen

import ringer
from tests.test_hud_single_tab import config


class HudIdentityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.config = config(Path(temp.name))

    def server(self):
        server = ringer.PersistentHudServer(self.config.state_dir, 0, open_viewer=False)
        server.model_log_path = self.config.eval.jsonl_path
        server.start()
        self.addCleanup(server.stop)
        return server

    def test_real_identity_and_matching_reuse(self):
        server = self.server()
        self.config = replace(self.config, hud_port=server.port)
        with urlopen(f'http://127.0.0.1:{server.port}/api/identity') as response:
            self.assertEqual(ringer.hud_identity(self.config.state_dir, self.config.eval.jsonl_path), json.load(response))
        with mock.patch.object(ringer.subprocess, 'Popen') as spawn:
            ringer.ensure_hud_running(self.config, open_browser=False)
            self.assertEqual(0, ringer.run_persistent_hud(self.config, port=None, open_viewer=False))
            spawn.assert_not_called()

    def test_real_mismatch_rejected_by_both_entry_points(self):
        server = self.server()
        other = replace(self.config, hud_port=server.port, state_dir=self.config.state_dir / 'other')
        with mock.patch.object(ringer.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                ringer.ensure_hud_running(other, open_browser=False)
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                ringer.run_persistent_hud(other, port=None, open_viewer=False)
            spawn.assert_not_called()
        self.assertTrue(ringer.hud_is_healthy(server.port, self.config))

    def test_model_log_mismatch_rejected(self):
        server = self.server()
        other = replace(self.config, eval=replace(self.config.eval, jsonl_path=Path('/different.jsonl')))
        with self.assertRaisesRegex(RuntimeError, 'incompatible'):
            ringer.hud_is_healthy(server.port, other)

    def test_legacy_service_rejected(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        class Legacy(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_error(404)
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Legacy)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                ringer.hud_is_healthy(server.server_port, self.config)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_spawn_forwards_config_and_effective_paths(self):
        for path in (None, self.config.state_dir.parent / 'explicit.toml'):
            cfg = replace(self.config, path=path)
            child = mock.Mock()
            child.poll.return_value = None
            with mock.patch.object(ringer, 'hud_is_healthy', side_effect=[False, True]), mock.patch.object(ringer.subprocess, 'Popen', return_value=child) as spawn:
                ringer.ensure_hud_running(cfg, open_browser=False)
            argv = spawn.call_args.args[0]
            self.assertEqual(str(cfg.state_dir.resolve()), argv[argv.index('--state-dir') + 1])
            self.assertEqual(str(cfg.eval.jsonl_path.resolve()), argv[argv.index('--model-log-path') + 1])
            self.assertEqual(path is not None, '--config' in argv)
            if path:
                self.assertEqual(str(path.resolve()), argv[argv.index('--config') + 1])
            parsed = ringer.build_parser().parse_args(argv[2:])
            self.assertEqual(cfg.state_dir.resolve(), parsed.state_dir)

    def test_real_detached_spawn_preserves_explicit_and_programmatic_paths(self):
        import os
        import socket
        import subprocess
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                with socket.socket() as reservation:
                    reservation.bind(('127.0.0.1', 0))
                    port = reservation.getsockname()[1]
                root = self.config.state_dir.parent
                path = root / 'custom.toml'
                path.write_text(f'state_dir = "{root / "custom-state"}"\n[eval]\njsonl_path = "{root / "custom-log.jsonl"}"\n[update]\nauto = false\n')
                cfg = replace(self.config, path=path if explicit else None, hud_port=port,
                              state_dir=root / 'effective-state',
                              eval=replace(self.config.eval, jsonl_path=root / 'effective.jsonl'))
                children = []
                real_popen = subprocess.Popen
                def spawn(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    return child
                home = root / 'home'
                home.mkdir(exist_ok=True)
                with mock.patch.dict(os.environ, {'HOME': str(home), 'RINGER_NO_SELF_UPDATE': '1'}), mock.patch.object(ringer.subprocess, 'Popen', side_effect=spawn):
                    try:
                        ringer.ensure_hud_running(cfg, open_browser=False)
                        self.assertEqual(1, len(children))
                        self.assertTrue(ringer.hud_is_healthy(port, cfg))
                        with urlopen(f'http://127.0.0.1:{port}/api/runs') as response:
                            self.assertEqual([], json.load(response)['runs'])
                        self.assertTrue((cfg.state_dir / 'hud.log').exists())
                    finally:
                        for child in children:
                            child.terminate()
                            child.wait(timeout=5)

    def test_spawn_failure_is_not_success(self):
        with mock.patch.object(ringer, 'hud_is_healthy', return_value=False), mock.patch.object(ringer.subprocess, 'Popen', side_effect=OSError('cannot launch')), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(OSError, 'cannot launch'):
                ringer.ensure_hud_running(self.config, open_browser=False)
            self.assertNotIn('Ringside:', output.getvalue())

    def test_timeout_cleans_up_only_spawned_child(self):
        child = mock.Mock()
        child.poll.return_value = None
        with mock.patch.object(ringer, 'hud_is_healthy', return_value=False), mock.patch.object(ringer.subprocess, 'Popen', return_value=child), mock.patch.object(ringer.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'did not become ready'):
                ringer.ensure_hud_running(self.config, open_browser=False)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=2)

    def test_exited_child_reports_failure(self):
        child = mock.Mock()
        child.poll.return_value = 2
        with mock.patch.object(ringer, 'hud_is_healthy', return_value=False), mock.patch.object(ringer.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'exited before readiness'):
                ringer.ensure_hud_running(self.config, open_browser=False)
        child.terminate.assert_not_called()
