#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EngineConfig,
    EvalConfig,
    Manifest,
    RESERVED_FIXTURE_MODELS,
    RingerRunner,
    VerifyResult,
    WorkerResult,
    aggregate_model_log_rows,
    aggregate_model_scoreboard_rows,
    model_scoreboard_tier,
    model_scoreboard_status,
    wilson_interval,
    evidence_is_stale,
    order_model_scoreboard_rows,
    build_models_api_payload,
    create_read_model_schema,
    enrich_model_groups_with_identity,
    load_model_identity_registry,
    run_models_command,
)


def row(
    run_id: str,
    *,
    model: str,
    engine: str = "codex",
    effort: str | None = None,
    task_type: str = "code-feature",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "task_key": "task",
        "worker_engine": engine,
        "model": model,
        "reasoning_effort": effort,
        "task_type": task_type,
        "retry": False,
        "verdict": "PASS",
        "duration_ms": 10,
        "worker_tokens": 20,
        "logged_at": f"2026-10-0{min(len(run_id), 9)}T10:00:00+00:00",
    }


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_env = os.environ.copy()
        self.addCleanup(self.restore_env)
        os.environ["HOME"] = str(self.root / "home")
        os.environ["RINGER_HOME"] = str(self.root / "ringer-home")
        self.registry_path = self.root / "model-identity.toml"
        self.registry_path.write_text(
            """
[engines.codex]
harness = "Codex CLI"
access = "OAuth plan"
default_model_key = "gpt-5.5"

[engines.codex.models."gpt-5.5"]
display = "GPT-5.5"
lab = "OpenAI"
confidence = "verified"
source = "fixture"

[engines.demo]
harness = "Demo CLI"
access = "OAuth plan"

[engines.demo.models."demo-alias"]
display = "demo-alias (DemoLab, alias)"
lab = "DemoLab"
alias = true
confidence = "verified"
source = "fixture"

[engines.opencode]
harness = "OpenCode"
access = "OpenRouter API"

[engines.opencode.models."openrouter/z-ai/glm-5.2"]
display = "GLM 5.2"
lab = "Z.ai (Zhipu AI)"
confidence = "verified"
source = "fixture"
""",
            encoding="utf-8",
        )

    def restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)

    def config(self, *, log_path: Path | None = None, engines: dict[str, EngineConfig] | None = None) -> AppConfig:
        return AppConfig(
            path=None,
            identity_default=None,
            state_dir=self.root / "state",
            dashboard_port_base=8787,
            hud_port=8700,
            hud_app_path=None,
            allow_full_access=False,
            eval=EvalConfig(backend="jsonl", jsonl_path=log_path or self.root / "default.jsonl"),
            engines=engines or {},
            artifact=ArtifactConfig(
                enabled=False,
                out_template=str(self.root / "live.html"),
                report_template=str(self.root / "report.html"),
                index_out=self.root / "index.html",
            ),
        )

    def write_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    def model_args(self, path: Path, *, html: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            log=path,
            db=None,
            task_type=None,
            model=None,
            engine=None,
            since=None,
            explore=False,
            catalog_file=self.root / "missing-catalog.json",
            notes_file=self.root / "missing-notes.md",
            registry=self.registry_path,
            html=html,
            open=False,
            json=html is None,
        )

    def test_reserved_fixture_names_are_absent_from_aggregation_and_json(self) -> None:
        rows = [row("real", model="gpt-5.5")]
        rows.extend(row(name, model=name) for name in sorted(RESERVED_FIXTURE_MODELS))
        self.assertEqual(["gpt-5.5"], [item["model"] for item in aggregate_model_log_rows(rows)])
        self.assertEqual(["gpt-5.5"], [item["model"] for item in aggregate_model_scoreboard_rows(rows)])

        log_path = self.root / "fixture.jsonl"
        self.write_rows(log_path, rows)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, run_models_command(self.config(), self.model_args(log_path)))
        payload = json.loads(out.getvalue())
        rendered = json.dumps(payload)
        for name in RESERVED_FIXTURE_MODELS:
            self.assertNotIn(name, rendered)

    def test_blank_model_is_quarantined_and_not_credited_to_engine_default(self) -> None:
        log_path = self.root / "legacy.jsonl"
        self.write_rows(
            log_path,
            [row("known", model="gpt-5.5"), row("legacy-a", model=""), row("legacy-b", model="   ")],
        )
        payload = build_models_api_payload(
            log_path=log_path,
            default_log_path=self.root / "different-default.jsonl",
            registry_path=self.registry_path,
            catalog_path=self.root / "missing-catalog.json",
        )
        known = next(item for item in payload["rollup"] if not item["unattributed"])
        legacy = next(item for item in payload["rollup"] if item["unattributed"])
        self.assertEqual(("GPT-5.5", 1, "probation"), (known["model_display"], known["tasks"], known["tier"]))
        self.assertEqual("(unattributed legacy rows)", legacy["model_display"])
        self.assertEqual("", legacy["model"])
        self.assertEqual("codex", legacy["engine"])
        self.assertEqual(2, legacy["tasks"])
        self.assertEqual("unranked", legacy["tier"])
        self.assertIs(payload["rollup"][-1], legacy)
        self.assertTrue(payload["groups"][-1]["unattributed"])

    def test_registry_threads_lab_alias_and_unknown_fallback(self) -> None:
        registry = load_model_identity_registry(self.registry_path)
        self.assertEqual("OpenAI", registry.resolve("codex", "gpt-5.5").lab)
        alias = registry.resolve("demo", "demo-alias")
        self.assertEqual(("DemoLab", True), (alias.lab, alias.alias))
        self.assertEqual("vendor?", registry.resolve("opencode", "openrouter/vendor/model").lab)
        self.assertEqual("(unverified)", registry.resolve("custom", "model").lab)

    def test_attempt_logging_records_explicit_effort_and_null_when_absent(self) -> None:
        log_path = self.root / "attempts.jsonl"
        engine = EngineConfig(
            name="codex",
            bin="codex",
            args_template=("exec", "{engine_args}", "{spec}"),
            full_access_args=(),
            sandbox_args=(),
            model_default="gpt-5.5",
        )
        manifest = Manifest.from_obj(
            {
                "run_name": "taxonomy-log",
                "workdir": str(self.root / "work"),
                "tasks": [
                    {
                        "key": "task",
                        "spec": "Create output.txt with the requested value and keep the change scoped.",
                        "check": "test -s output.txt || { echo 'output.txt is missing'; exit 1; }",
                        "expect_files": ["output.txt"],
                        "verified": "output.txt exists",
                        "engine_args": ["-c", "model_reasoning_effort=high"],
                    }
                ],
            }
        )
        runner = RingerRunner(
            manifest,
            config=self.config(log_path=log_path, engines={"codex": engine}),
            identity="tester",
            dashboard_enabled=False,
        )
        runtime = runner.runtimes[0]
        runtime.last_worker_command = ["codex", "exec", "-c", "model_reasoning_effort=high"]
        worker = WorkerResult(returncode=0, timed_out=False, tokens=1)
        verify = VerifyResult(ok=True, check_returncode=0, check_timed_out=False, raw_output_excerpt="ok")
        runner._log_attempt(runtime, runtime.task.spec, False, worker, verify, "PASS", 1)
        runtime.last_worker_command = ["codex", "exec"]
        runner._log_attempt(runtime, runtime.task.spec, False, worker, verify, "PASS", 1)

        logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual("high", logged[0]["reasoning_effort"])
        self.assertIsNone(logged[1]["reasoning_effort"])

    def test_effort_splits_model_buckets_and_unrecorded_identity(self) -> None:
        rows = [
            row("high", model="gpt-5.5", effort="high"),
            row("medium", model="gpt-5.5", effort="medium"),
            row("old", model="gpt-5.5"),
            row("open", model="openrouter/z-ai/glm-5.2", engine="opencode"),
        ]
        registry = load_model_identity_registry(self.registry_path)
        groups = enrich_model_groups_with_identity(
            aggregate_model_scoreboard_rows(rows), rows, registry, include_task_type=False
        )
        displays = {item["model_display"] for item in groups}
        self.assertIn("GPT-5.5 · high", displays)
        self.assertIn("GPT-5.5 · medium", displays)
        self.assertIn("GPT-5.5 · (effort unrecorded)", displays)
        self.assertIn("GLM 5.2", displays)
        self.assertNotIn("GLM 5.2 · (effort unrecorded)", displays)

    def test_alias_marker_and_lab_render_in_html(self) -> None:
        log_path = self.root / "alias.jsonl"
        self.write_rows(log_path, [row("alias", model="demo-alias", engine="demo")])
        html_path = self.root / "scoreboard.html"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                0,
                run_models_command(
                    self.config(), self.model_args(log_path, html=str(html_path))
                ),
            )
        html = html_path.read_text(encoding="utf-8")
        self.assertIn(">Lab<", html)
        self.assertIn("demo-alias (DemoLab, alias)", html)
        self.assertIn(">DemoLab<", html)

    def test_existing_database_migrates_reasoning_effort_without_data_loss(self) -> None:
        db_path = self.root / "ringer.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                PRAGMA user_version = 1;
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version(version) VALUES (1);
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY, run_id TEXT, task_key TEXT, logged_at TEXT,
                    engine TEXT, model TEXT, task_type TEXT, retry INTEGER, verdict TEXT,
                    duration_ms INTEGER, worker_tokens INTEGER, orchestrator TEXT
                );
                INSERT INTO attempts(model, verdict) VALUES ('gpt-5.5', 'PASS');
                """
            )
            create_read_model_schema(conn)
            columns = {item[1] for item in conn.execute("PRAGMA table_info(attempts)")}
            self.assertIn("reasoning_effort", columns)
            self.assertEqual(("gpt-5.5", "PASS", None), conn.execute(
                "SELECT model, verdict, reasoning_effort FROM attempts"
            ).fetchone())
            self.assertEqual(3, conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(3, conn.execute("SELECT version FROM schema_version").fetchone()[0])


class WilsonTierTests(unittest.TestCase):
    def test_status_thresholds(self) -> None:
        cases = [
            (3, 2 / 3, "UNTESTED"),
            (7, 1.0, "UNTESTED"),
            (8, 6 / 8, "SUPPORTED"),
            (10, 7 / 10, "SUPPORTED"),
            (8, 5 / 8, "PROVISIONAL"),
            (10, 6 / 10, "PROVISIONAL"),
            (10, 0.0, "CONTRADICTED"),
            (20, 3 / 20, "CONTRADICTED"),
        ]
        for tasks, rate, want in cases:
            self.assertEqual(want, model_scoreboard_status(tasks, rate), (tasks, rate))

    def test_legacy_tier_names_map_from_status(self) -> None:
        self.assertEqual("proven", model_scoreboard_tier(8, 6 / 8))
        self.assertEqual("probation", model_scoreboard_tier(3, 2 / 3))
        self.assertEqual("probation", model_scoreboard_tier(8, 5 / 8))
        self.assertEqual("probation", model_scoreboard_tier(10, 0.0))

    def test_wilson_interval_is_80_percent(self) -> None:
        lo, hi = wilson_interval(6, 8)
        self.assertAlmostEqual(0.5, lo, delta=0.06)
        self.assertAlmostEqual(0.89, hi, delta=0.02)
        self.assertEqual((0.0, 1.0), wilson_interval(0, 0))

    def test_stale_flag_after_90_days(self) -> None:
        now = "2026-11-07T00:00:00+00:00"
        self.assertTrue(evidence_is_stale("2026-01-01T00:00:00+00:00", now=now))
        self.assertFalse(evidence_is_stale("2026-10-20T00:00:00+00:00", now=now))
        # A missing or unparseable date counts as stale, never fresh.
        self.assertTrue(evidence_is_stale("", now=now))
        self.assertTrue(evidence_is_stale("not-a-date", now=now))


class EvidenceWindowAndPerTypeTests(unittest.TestCase):
    def typed_rows(self, model: str, task_type: str, verdicts: list[str], *, logged_at: str | None = "2026-10-10T10:00:00+00:00") -> list[dict[str, object]]:
        out = []
        for i, verdict in enumerate(verdicts):
            item = row(f"{model}-{task_type}-{i}", model=model, engine="opencode", task_type=task_type)
            item["task_key"] = f"{task_type}-{i}"
            item["verdict"] = verdict
            if logged_at is None:
                item.pop("logged_at")
            else:
                item["logged_at"] = logged_at
            out.append(item)
        return out

    def test_old_passes_do_not_support_a_model_after_a_recent_fail(self) -> None:
        old = self.typed_rows("openrouter/a/aging", "code-fix", ["PASS"] * 10, logged_at="2026-08-01T10:00:00+00:00")
        recent = self.typed_rows("openrouter/a/aging", "code-fix", ["FAIL"], logged_at="2026-11-01T10:00:00+00:00")
        recent[0]["task_key"] = "recent"
        item = aggregate_model_scoreboard_rows(old + recent)[0]
        self.assertEqual(11, item["tasks"])
        self.assertEqual(1, item["evidence_tasks"])
        self.assertEqual("UNTESTED", item["evidence_status"])
        self.assertEqual("probation", item["tier"])

    def test_undated_rows_are_outside_the_window(self) -> None:
        undated = self.typed_rows("openrouter/a/undated", "code-fix", ["PASS"] * 10, logged_at=None)
        item = aggregate_model_scoreboard_rows(undated)[0]
        self.assertEqual(0, item["evidence_tasks"])
        self.assertEqual("UNTESTED", item["evidence_status"])
        self.assertTrue(item["evidence_stale"])

    def test_pooled_row_does_not_carry_one_types_support_to_the_others(self) -> None:
        model = "openrouter/a/mixed"
        rows = self.typed_rows(model, "code-fix", ["PASS"] * 40) + self.typed_rows(model, "docs", ["FAIL"] * 12)
        item = aggregate_model_scoreboard_rows(rows)[0]
        by_type = {t["task_type"]: t for t in item["task_types"]}
        self.assertEqual("SUPPORTED", by_type["code-fix"]["evidence_status"])
        self.assertEqual("CONTRADICTED", by_type["docs"]["evidence_status"])
        self.assertGreater(by_type["code-fix"]["evidence_lower"], 0.45)
        self.assertLess(by_type["docs"]["evidence_upper"], 0.45)
        self.assertEqual("CONTRADICTED", item["evidence_status"])
        self.assertEqual("probation", item["tier"])

    def test_pooled_row_is_supported_only_if_every_tested_type_is(self) -> None:
        model = "openrouter/a/solid"
        rows = self.typed_rows(model, "code-fix", ["PASS"] * 12) + self.typed_rows(model, "docs", ["PASS"] * 12)
        self.assertEqual("SUPPORTED", aggregate_model_scoreboard_rows(rows)[0]["evidence_status"])
        # A type below n=8 is ignored; pooled n alone (no type at n>=8) is only PROVISIONAL.
        thin = self.typed_rows("openrouter/a/thin", "code-fix", ["PASS"] * 5) + self.typed_rows("openrouter/a/thin", "docs", ["PASS"] * 5)
        item = aggregate_model_scoreboard_rows(thin)[0]
        self.assertEqual("PROVISIONAL", item["evidence_status"])
        self.assertEqual("probation", item["tier"])
        filtered = aggregate_model_scoreboard_rows(rows, task_type="code-fix")[0]
        self.assertEqual("SUPPORTED", filtered["evidence_status"])


class MeasuredCostTests(unittest.TestCase):
    def rows(self, engine: str, model: str, costs: list[float | None], passes: list[bool]) -> list[dict[str, object]]:
        out = []
        for i, (cost, ok) in enumerate(zip(costs, passes)):
            item = row(f"{model}-{i}", model=model, engine=engine)
            item["task_key"] = f"t{i}"
            item["verdict"] = "PASS" if ok else "FAIL"
            if cost is not None:
                item["cost_usd"] = cost
            out.append(item)
        return out

    def test_measured_cost_per_task_and_per_verified_pass(self) -> None:
        rows = self.rows("opencode", "openrouter/a/cheap", [0.10, 0.20, 0.30, 0.40], [True, True, False, False])
        item = aggregate_model_scoreboard_rows(rows)[0]
        self.assertAlmostEqual(0.25, item["measured_cost_per_task"])
        self.assertAlmostEqual(0.50, item["expected_cost_per_verified_pass"])

    def test_cost_is_per_attempt_not_summed_running_totals(self) -> None:
        # cost_usd on each row is the task's RUNNING total (_refresh_cost), so
        # attempt 1 = 0.10 and attempt 2 = 0.30 means the task cost 0.30, not 0.40.
        first = row("t-r", model="openrouter/a/retry", engine="opencode")
        first.update(task_key="t0", verdict="FAIL", cost_usd=0.10, logged_at="2026-10-01T10:00:00+00:00")
        second = row("t-r", model="openrouter/a/retry", engine="opencode")
        second.update(task_key="t0", verdict="PASS", retry=True, cost_usd=0.30, logged_at="2026-10-01T10:05:00+00:00")
        third = row("t-s", model="openrouter/a/retry", engine="opencode")
        third.update(task_key="t1", verdict="PASS", cost_usd=0.10, logged_at="2026-10-01T11:00:00+00:00")
        item = aggregate_model_scoreboard_rows([first, second, third])[0]
        self.assertAlmostEqual(0.20, item["measured_cost_per_task"])  # (0.30 + 0.10) / 2
        # c per attempt = 0.40 / 3 attempts; p = 1/2 first-try; no second division by p.
        self.assertAlmostEqual((0.40 / 3) / 0.5, item["expected_cost_per_verified_pass"])

    def test_no_measured_cost_or_plan_billed_engine_has_none(self) -> None:
        plain = aggregate_model_scoreboard_rows(self.rows("opencode", "openrouter/a/none", [None, None], [True, True]))[0]
        self.assertIsNone(plain["measured_cost_per_task"])
        self.assertIsNone(plain["expected_cost_per_verified_pass"])
        for engine in ("codex", "grok"):
            plan = aggregate_model_scoreboard_rows(self.rows(engine, "plan-model", [0.5, 0.5], [True, True]))[0]
            self.assertIsNone(plan["expected_cost_per_verified_pass"], engine)

    def test_zero_first_try_passes_gives_no_expected_cost(self) -> None:
        item = aggregate_model_scoreboard_rows(self.rows("opencode", "openrouter/a/bad", [0.1, 0.1], [False, False]))[0]
        self.assertIsNone(item["expected_cost_per_verified_pass"])

    def test_sort_prefers_measured_expected_cost_over_estimator(self) -> None:
        base = {"tier": "probation", "first_try_pass_rate": 0.5, "pass_rate": 0.5, "engine": "opencode"}
        cheap = dict(base, model="m-cheap", median_tokens=10**9, expected_cost_per_verified_pass=0.2)
        dear = dict(base, model="m-dear", median_tokens=1, expected_cost_per_verified_pass=2.0)
        catalog = {
            "m-cheap": {"prompt_per_m": 1000, "completion_per_m": 1000},
            "m-dear": {"prompt_per_m": 1, "completion_per_m": 1},
        }
        ordered = order_model_scoreboard_rows([dear, cheap], catalog)
        self.assertEqual(["m-cheap", "m-dear"], [r["model"] for r in ordered])


if __name__ == "__main__":
    unittest.main(verbosity=2)
