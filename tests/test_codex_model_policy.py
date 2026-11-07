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
from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EngineConfig,
    EvalConfig,
    Manifest,
    lint_manifest,
    validate_manifest_engines,
)

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
    engine = codex_like(model_default=model_default)
    return synthetic_config(tempfile.gettempdir(), {"codex": engine})


def synthetic_config(root: str, engines: dict[str, EngineConfig]) -> AppConfig:
    base = Path(root)
    return AppConfig(
        path=None,
        identity_default=None,
        state_dir=base / "state",
        dashboard_port_base=8787,
        hud_port=8700,
        hud_app_path=None,
        allow_full_access=False,
        eval=EvalConfig(backend="jsonl", jsonl_path=base / "runs.jsonl"),
        engines=engines,
        artifact=ArtifactConfig(
            enabled=False,
            out_template=str(base / "live.html"),
            report_template=str(base / "report.html"),
            index_out=base / "index.html",
        ),
    )


CODEX_TEMPLATE = (
    "exec", "--skip-git-repo-check", "{access_args}", "{model_args}",
    "{engine_args}", "-C", "{taskdir}", "{spec}",
)


def codex_like(name: str = "codex", template: tuple[str, ...] = CODEX_TEMPLATE, model_default: str = "gpt-6.1-sol") -> EngineConfig:
    return EngineConfig(
        name=name,
        bin="/usr/local/bin/codex",
        args_template=template,
        full_access_args=("--dangerously-bypass-approvals-and-sandbox",),
        sandbox_args=("--sandbox", "workspace-write"),
        model_default=model_default,
    )


class CodexCommandPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def check(self, engine: EngineConfig, **task_fields: object) -> None:
        fields = {"engine": engine.name, **task_fields}
        validate_manifest_engines(
            make_manifest(self.root, **fields), synthetic_config(self.root, {engine.name: engine})
        )

    def test_standard_template_with_sol_passes(self) -> None:
        self.check(codex_like(), model="gpt-6.1-sol")

    def test_template_without_model_args_is_rejected(self) -> None:
        template = tuple(item for item in CODEX_TEMPLATE if item != "{model_args}")
        with self.assertRaisesRegex(ValueError, "no -m model"):
            self.check(codex_like(template=template), model="gpt-6.1-sol")

    def test_hardcoded_model_in_template_is_rejected(self) -> None:
        template = ("-m", "gpt-5.5") + tuple(i for i in CODEX_TEMPLATE if i != "{model_args}")
        with self.assertRaisesRegex(ValueError, "gpt-5.5"):
            self.check(codex_like(template=template), model="gpt-6.1-sol")

    def test_hardcoded_model_beside_model_args_is_rejected(self) -> None:
        template = ("-m", "gpt-5.5") + CODEX_TEMPLATE
        with self.assertRaisesRegex(ValueError, "gpt-5.5"):
            self.check(codex_like(template=template), model="gpt-6.1-sol")

    def test_model_flags_in_engine_args_are_rejected(self) -> None:
        for args in (["-m", "gpt-5.5"], ["--model", "gpt-5.5"], ["--model=gpt-5.5"], ["-c", "model=gpt-5.5"]):
            with self.assertRaisesRegex(ValueError, "engine_args may not set the model"):
                self.check(codex_like(), model="gpt-6.1-sol", engine_args=args)

    def test_attached_and_equals_model_forms_in_engine_args_are_rejected(self) -> None:
        cases = (
            ["-m", "gpt-6.1-sol", "-mgpt-5.5"],
            ["-mgpt-5.5"],
            ["-cmodel=gpt-5.5"],
            ["-c", " model = gpt-5.5"],
            ["--config=model=gpt-5.5"],
            ["--config= model =gpt-5.5"],
            ["--config", "model=gpt-5.5"],
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "engine_args may not set the model"):
                    self.check(codex_like(), model="gpt-6.1-sol", engine_args=args)

    def test_other_config_and_routing_flags_in_engine_args_are_rejected(self) -> None:
        cases = (
            ["-c", "model_provider=other"],
            ["--config=model_provider=other"],
            ["-cmodel_provider=other"],
            ["--oss"],
            ["--profile", "x"],
            ["--profile=x"],
            ["-p", "x"],
            ["-px"],
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "engine_args may only"):
                    self.check(codex_like(), model="gpt-6.1-sol", engine_args=args)

    def test_command_scan_catches_attached_model_in_template(self) -> None:
        template = ("-mgpt-5.5",) + CODEX_TEMPLATE
        with self.assertRaisesRegex(ValueError, "gpt-5.5"):
            self.check(codex_like(template=template), model="gpt-6.1-sol")

    def test_config_model_never_satisfies_the_model_flag_presence_check(self) -> None:
        base = tuple(i for i in CODEX_TEMPLATE if i != "{model_args}")
        for prefix in (("-c", "model={model}"), ("--config", "model={model}"), ("-cmodel={model}",)):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(ValueError, "no -m model"):
                    self.check(codex_like(template=prefix + base), model="gpt-6.1-sol")

    def test_command_scan_catches_config_model_spellings_in_template(self) -> None:
        for prefix in (("-cmodel=gpt-5.5",), ("-c=model=gpt-5.5",), ("--config=model=gpt-5.5",)):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(ValueError, "gpt-5.5"):
                    self.check(codex_like(template=prefix + CODEX_TEMPLATE), model="gpt-6.1-sol")

    def test_reasoning_effort_must_be_an_enum_value(self) -> None:
        for value in ("low", "medium", "high", "xhigh", '"high"', "'low'"):
            with self.subTest(value=value):
                self.check(codex_like(), model="gpt-6.1-sol",
                           engine_args=["-c", f"model_reasoning_effort={value}"])
        for value in ("low\nmodel=gpt-5.5", "high extra", "turbo", "", "low\n"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "engine_args may only"):
                    self.check(codex_like(), model="gpt-6.1-sol",
                               engine_args=["-c", f"model_reasoning_effort={value}"])

    def test_writable_roots_engine_args_still_allowed(self) -> None:
        self.check(
            codex_like(),
            model="gpt-6.1-sol",
            engine_args=["-c", 'sandbox_workspace_write.writable_roots=["/tmp/x"]'],
        )

    def test_reasoning_effort_engine_args_still_allowed(self) -> None:
        self.check(codex_like(), model="gpt-6.1-sol", engine_args=["-c", "model_reasoning_effort=low"])

    def test_policy_applies_to_any_engine_running_the_codex_binary(self) -> None:
        engine = codex_like(name="codex-fast", model_default="")
        with self.assertRaisesRegex(ValueError, "gpt-6.1-sol"):
            self.check(engine, model="gpt-5.5")
        self.check(engine, model="gpt-6.1-sol")
        findings = lint_manifest(
            make_manifest(self.root, engine="codex-fast", model="gpt-5.5"),
            config=synthetic_config(self.root, {"codex-fast": engine}),
        )
        self.assertTrue(any(f.startswith("ERROR:") and "gpt-6.1-sol" in f for f in findings), findings)


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
