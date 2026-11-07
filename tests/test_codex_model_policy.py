#!/usr/bin/env python3
"""Policy (Jeff, 2026-11-07): the Codex engine may only run gpt-6.1-sol."""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ringer  # noqa: E402
from ringer import AppConfig, Manifest, lint_manifest, validate_manifest_engines  # noqa: E402

SPEC = (
    "Create the requested artifact in the current working directory, keep the change scoped, "
    "and make the check command able to explain any failure clearly."
)
CHECK = (
    "test -s output.txt && grep -q 'ready' output.txt || "
    "{ echo 'FAIL: output.txt missing'; exit 1; }"
)


def make_manifest(root: str, **task_fields: object) -> Manifest:
    task = {
        "key": "one",
        "spec": SPEC,
        "check": CHECK,
        "expect_files": ["output.txt"],
        "verified": "output.txt exists and contains ready",
        **task_fields,
    }
    return Manifest.from_obj(
        {"run_name": "policy", "workdir": str(Path(root) / "work"), "tasks": [task]}
    )


def config_with_default(model_default: str) -> AppConfig:
    config = AppConfig.load(None)
    engine = replace(config.engines["codex"], model_default=model_default)
    return replace(config, engines={**config.engines, "codex": engine})


class CodexModelPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def errors(self, findings: list[str]) -> list[str]:
        return [f for f in findings if f.startswith("ERROR:") and "gpt-6.1-sol" in f]

    def test_allowlist_constant(self) -> None:
        self.assertEqual(("gpt-6.1-sol",), ringer.CODEX_ALLOWED_MODELS)

    def test_lint_errors_on_older_task_model(self) -> None:
        for model in ("gpt-5.5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-6-astra"):
            manifest = make_manifest(self.root, model=model)
            findings = lint_manifest(manifest, config=config_with_default("gpt-6.1-sol"))
            self.assertTrue(self.errors(findings), (model, findings))

    def test_lint_errors_on_older_config_default(self) -> None:
        manifest = make_manifest(self.root)
        findings = lint_manifest(manifest, config=config_with_default("gpt-5.6-sol"))
        self.assertTrue(self.errors(findings), findings)

    def test_lint_errors_when_model_unset(self) -> None:
        manifest = make_manifest(self.root)
        findings = lint_manifest(manifest, config=config_with_default(""))
        self.assertTrue(self.errors(findings), findings)
        self.assertTrue(self.errors(lint_manifest(manifest)), "no config at all")

    def test_lint_accepts_sol_by_task_or_default(self) -> None:
        by_task = make_manifest(self.root, model="gpt-6.1-sol")
        self.assertFalse(self.errors(lint_manifest(by_task, config=config_with_default(""))))
        self.assertFalse(self.errors(lint_manifest(by_task)))
        by_default = make_manifest(self.root)
        self.assertFalse(
            self.errors(lint_manifest(by_default, config=config_with_default("gpt-6.1-sol")))
        )

    def test_task_model_beats_default(self) -> None:
        manifest = make_manifest(self.root, model="gpt-5.5")
        findings = lint_manifest(manifest, config=config_with_default("gpt-6.1-sol"))
        self.assertTrue(self.errors(findings), findings)

    def test_non_codex_engine_unaffected(self) -> None:
        manifest = make_manifest(self.root, engine="opencode", model="openrouter/x/y")
        self.assertFalse(self.errors(lint_manifest(manifest)))

    def test_run_validation_refuses_like_missing_opencode_model(self) -> None:
        for default, task_model in (("gpt-5.6-sol", ""), ("", ""), ("gpt-6.1-sol", "gpt-5.5")):
            config = config_with_default(default)
            fields = {"model": task_model} if task_model else {}
            manifest = make_manifest(self.root, **fields)
            with self.assertRaisesRegex(ValueError, "gpt-6.1-sol"):
                validate_manifest_engines(manifest, config)

    def test_run_validation_accepts_sol(self) -> None:
        validate_manifest_engines(
            make_manifest(self.root), config_with_default("gpt-6.1-sol")
        )
        validate_manifest_engines(
            make_manifest(self.root, model="gpt-6.1-sol"), config_with_default("")
        )


    def test_ask_rejects_other_codex_model(self) -> None:
        packet = ringer.ContextPacket(
            text="q", packet_bytes=1, source_bytes=0, selected=(), skipped=()
        )

        def build(model: str | None) -> Manifest:
            return ringer.one_request_manifest(
                packet=packet,
                workdir=Path(self.root),
                engine="codex",
                timeout_s=60,
                reasoning_effort="low",
                model=model,
                redact=False,
            )

        with self.assertRaisesRegex(ValueError, "gpt-6.1-sol"):
            build("gpt-5.5")
        self.assertEqual("gpt-6.1-sol", build("gpt-6.1-sol").tasks[0].model)
        self.assertEqual("", build(None).tasks[0].model)


if __name__ == "__main__":
    unittest.main()
