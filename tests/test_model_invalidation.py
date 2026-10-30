#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ringer  # noqa: E402
from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EvalConfig,
    aggregate_model_scoreboard_rows,
    build_models_api_payload,
    db_attempt_rows,
    read_model_log_rows,
    rebuild_read_model_db,
    run_models_command,
    sync_read_model_db,
    validate_since_date,
)


def attempt(
    run_id: str,
    task_key: str,
    *,
    model: str = "openrouter/acme/live",
    verdict: str = "PASS",
    task_type: str = "code-feature",
    logged_at: str = "2026-07-06T10:00:00+00:00",
    invalidated: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id,
        "task_key": task_key,
        "worker_engine": "opencode",
        "model": model,
        "task_type": task_type,
        "verdict": verdict,
        "retry": False,
        "duration_ms": 100,
        "worker_tokens": 200,
        "logged_at": logged_at,
        "orchestrator": "tester",
    }
    if invalidated:
        row.update(
            {
                "evidence_excluded": True,
                "invalidated_at": "2026-07-01T00:00:00+00:00",
                "invalidation_reason": "old reason",
            }
        )
    return row


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ModelInvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_env = os.environ.copy()
        self.addCleanup(self.restore_env)
        os.environ["HOME"] = str(self.root / "home")
        os.environ["RINGER_HOME"] = str(self.root / "ringer-home")
        self.log_path = self.root / "models.jsonl"
        self.db_path = self.root / "models.db"
        self.config = AppConfig(
            path=None,
            identity_default=None,
            state_dir=self.root / "state",
            dashboard_port_base=8787,
            hud_port=8700,
            hud_app_path=None,
            allow_full_access=False,
            eval=EvalConfig(backend="jsonl", jsonl_path=self.log_path),
            engines={},
            artifact=ArtifactConfig(
                enabled=False,
                out_template=str(self.root / "live.html"),
                report_template=str(self.root / "report.html"),
                index_out=self.root / "index.html",
            ),
        )

    def restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)

    def model_args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "log": self.log_path,
            "db": None,
            "task_type": None,
            "model": None,
            "engine": None,
            "since": None,
            "explore": False,
            "catalog_file": self.root / "missing-catalog.json",
            "notes_file": self.root / "missing-notes.md",
            "registry": self.root / "missing-registry.toml",
            "html": None,
            "open": False,
            "json": False,
            "invalidate": False,
            "invalidate_run_id": None,
            "invalidate_task_key": None,
            "reason": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def amendments(self) -> list[dict[str, object]]:
        path = self.root / "amendments.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def invalidate(self, **overrides: object) -> tuple[int, str, str]:
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            rc = run_models_command(self.config, self.model_args(invalidate=True, **overrides))
        return rc, out.getvalue(), err.getvalue()

    def test_invalidate_alias_appends_amendments_and_never_rewrites_log(self) -> None:
        write_jsonl(
            self.log_path,
            [
                attempt("run-1", "a", verdict="FAIL"),
                attempt("run-1", "b", verdict="FAIL"),
                attempt("run-2", "a", verdict="FAIL"),
            ],
        )
        before = self.log_path.read_bytes()

        rc, out, _err = self.invalidate(invalidate_run_id="run-1", reason="bad harness login")

        self.assertEqual(0, rc)
        self.assertIn("amended 2 task(s) as infra", out)
        self.assertEqual(before, self.log_path.read_bytes())
        entries = self.amendments()
        self.assertEqual([("run-1", "a"), ("run-1", "b")], [(e["run_id"], e["task_key"]) for e in entries])
        self.assertTrue(all(e["cause"] == "infra" and e["attempt"] is None for e in entries))
        self.assertEqual("bad harness login", entries[0]["note"])

    def test_invalidate_alias_task_selector_and_cause(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a", verdict="FAIL"), attempt("run-1", "b", verdict="FAIL")])

        rc, _out, _err = self.invalidate(
            invalidate_run_id="run-1", invalidate_task_key="b", invalidate_cause="check", reason="check was wrong"
        )

        self.assertEqual(0, rc)
        self.assertEqual([("b", "check")], [(e["task_key"], e["cause"]) for e in self.amendments()])

    def test_invalidate_alias_excludes_whole_tasks_including_passes(self) -> None:
        # A vacuous check made every task "pass", and an infra-broken run
        # passed only on retry: invalidating either must remove the run from
        # the scoreboard, not keep the passes and inflate the model's rates.
        write_jsonl(
            self.log_path,
            [
                attempt("good", "a", model="openrouter/acme/live", verdict="FAIL"),
                attempt("vacuous", "a", model="openrouter/acme/live"),
                attempt("vacuous", "b", model="openrouter/acme/live"),
                attempt("flaky", "a", model="openrouter/acme/live", verdict="FAIL",
                        logged_at="2026-07-06T10:00:00+00:00"),
                attempt("flaky", "a", model="openrouter/acme/live",
                        logged_at="2026-07-06T10:05:00+00:00"),
            ],
        )

        for run in ("vacuous", "flaky"):
            rc, _out, _err = self.invalidate(invalidate_run_id=run, invalidate_cause="check", reason="check was vacuous")
            self.assertEqual(0, rc)

        self.assertTrue(all(e.get("include_pass") is True for e in self.amendments()))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, run_models_command(self.config, self.model_args(json=True)))
        groups = json.loads(out.getvalue())
        self.assertEqual(1, sum(group["tasks"] for group in groups))
        self.assertEqual(0.0, groups[0]["pass_rate"])

    def test_amend_cli_still_refuses_pass_rows(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a")])
        entry = {"run_id": "run-1", "task_key": "a", "attempt": None, "cause": "check", "note": "check was vacuous"}
        error, _numbers = ringer.validate_amendment_entry(entry, ringer.attempts_by_task_from_log(self.log_path))
        self.assertIn("every attempt", error or "")
        entry.update(include_pass=True, cause="worker")
        error, _numbers = ringer.validate_amendment_entry(entry, ringer.attempts_by_task_from_log(self.log_path))
        self.assertIn("cannot be combined", error or "")

    def test_amend_from_file_rejects_include_pass(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a")])
        batch = self.root / "batch.jsonl"
        batch.write_text(json.dumps({
            "run_id": "run-1", "task_key": "a", "attempt": None, "cause": "check",
            "note": "copied from amendments.jsonl", "include_pass": True,
        }) + "\n", encoding="utf-8")
        args = argparse.Namespace(
            log=self.log_path, list=False, run=None, identity=None, from_file=batch,
            run_id=None, task_key=None, cause=None, note=None, attempt=None,
        )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(2, ringer.run_amend_command(self.config, args))
        self.assertIn("reserved for models --invalidate", out.getvalue())
        self.assertEqual([], self.amendments())

    def test_cli_rejects_missing_inputs_and_reports_no_match(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a", verdict="FAIL")])

        rc, _out, err = self.invalidate(reason="bad harness login")
        self.assertEqual(2, rc)
        self.assertIn("requires --run", err)

        rc, _out, err = self.invalidate(invalidate_run_id="run-1")
        self.assertEqual(2, rc)
        self.assertIn("non-empty --reason", err)

        rc, _out, err = self.invalidate(invalidate_run_id="run-1", reason="short")
        self.assertEqual(2, rc)
        self.assertIn("at least 10 characters", err)

        rc, _out, err = self.invalidate(invalidate_run_id="missing", reason="bad harness login")
        self.assertEqual(1, rc)
        self.assertIn("No matching model log rows", err)
        self.assertEqual([], self.amendments())

    def test_aggregates_exclude_invalidated_rows_but_keep_audit_counts_visible(self) -> None:
        rows = [
            attempt("run-1", "a", model="openrouter/acme/live", verdict="PASS"),
            attempt("run-2", "a", model="openrouter/acme/live", verdict="FAIL", invalidated=True),
            attempt("run-3", "a", model="openrouter/acme/invalid-only", verdict="PASS", invalidated=True),
        ]
        write_jsonl(self.log_path, rows)

        rollup = aggregate_model_scoreboard_rows(rows)
        by_model = {row["model"]: row for row in rollup}
        self.assertEqual(1, by_model["openrouter/acme/live"]["tasks"])
        self.assertEqual(1, by_model["openrouter/acme/live"]["attempts"])
        self.assertEqual(1, by_model["openrouter/acme/live"]["invalidated_rows"])
        self.assertEqual(1.0, by_model["openrouter/acme/live"]["pass_rate"])
        self.assertEqual(0, by_model["openrouter/acme/invalid-only"]["tasks"])
        self.assertEqual(0, by_model["openrouter/acme/invalid-only"]["attempts"])
        self.assertEqual(1, by_model["openrouter/acme/invalid-only"]["invalidated_rows"])
        self.assertEqual("unranked", by_model["openrouter/acme/invalid-only"]["tier"])

        payload = build_models_api_payload(
            log_path=self.log_path,
            default_log_path=self.root / "other.jsonl",
            catalog_path=self.root / "missing-catalog.json",
            registry_path=self.root / "missing-registry.toml",
            notes_path=self.root / "missing-notes.md",
        )
        api_by_model = {row["model"]: row for row in payload["rollup"]}
        self.assertEqual(1, api_by_model["openrouter/acme/live"]["invalidated_rows"])
        self.assertEqual(0, api_by_model["openrouter/acme/invalid-only"]["attempts"])

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, run_models_command(self.config, self.model_args(json=True)))
        cli_groups = json.loads(out.getvalue())
        self.assertTrue(any(group["invalidated_rows"] == 1 for group in cli_groups))

    def test_invalidate_alias_is_applied_on_the_db_read_path(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a"), attempt("run-2", "a", verdict="FAIL")])
        rebuild_read_model_db(self.db_path, self.log_path)

        rc, _out, _err = self.invalidate(db=self.db_path, invalidate_run_id="run-2", reason="operator correction")
        self.assertEqual(0, rc)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, run_models_command(self.config, self.model_args(db=self.db_path, json=True)))
        groups = json.loads(out.getvalue())
        self.assertEqual(1, sum(group["attempts"] for group in groups))
        self.assertEqual(1.0, groups[0]["pass_rate"])

    def test_unchanged_sync_skips_prefix_hash_and_metadata_change_validates(self) -> None:
        write_jsonl(self.log_path, [attempt("run-1", "a")])
        rebuild_read_model_db(self.db_path, self.log_path)

        with mock.patch("ringer.hash_file_prefix", side_effect=AssertionError("should not hash")):
            unchanged = sync_read_model_db(self.db_path, self.log_path)
        self.assertFalse(unchanged.rebuilt)
        self.assertEqual(0, unchanged.attempts_inserted)

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(attempt("run-2", "a")) + "\n")
        with mock.patch("ringer.hash_file_prefix", wraps=ringer.hash_file_prefix) as hashed:
            changed = sync_read_model_db(self.db_path, self.log_path)
        self.assertFalse(changed.rebuilt)
        self.assertGreaterEqual(hashed.call_count, 1)

    def test_since_keeps_malformed_missing_invalidated_rows_in_raw_path(self) -> None:
        since = validate_since_date("2026-07-10")
        rows = [
            attempt("run-1", "a", logged_at="2026-07-05T00:00:00+00:00"),
            attempt(
                "run-2",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="2026-07-15T00:00:00+00:00",
            ),
            attempt(
                "run-3",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="not-a-date",
            ),
            attempt(
                "run-4",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="",
            ),
        ]
        write_jsonl(self.log_path, rows)

        selected, skipped = read_model_log_rows(self.log_path, since=since)
        by_run = {row["run_id"]: row for row in selected}

        # run-1 group ends before --since, so it is excluded entirely.
        self.assertNotIn("run-1", by_run)
        # Valid invalidated row at/after --since is retained.
        self.assertIn("run-2", by_run)
        # Malformed and missing logged_at invalidated rows must not raise and
        # must be treated as not matching the --since filter (safely excluded).
        self.assertNotIn("run-3", by_run)
        self.assertNotIn("run-4", by_run)

    def test_since_keeps_malformed_missing_invalidated_rows_in_db_path(self) -> None:
        since = validate_since_date("2026-07-10")
        rows = [
            attempt("run-1", "a", logged_at="2026-07-05T00:00:00+00:00"),
            attempt(
                "run-2",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="2026-07-15T00:00:00+00:00",
            ),
            attempt(
                "run-3",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="not-a-date",
            ),
            attempt(
                "run-4",
                "a",
                verdict="FAIL",
                invalidated=True,
                logged_at="",
            ),
        ]
        write_jsonl(self.log_path, rows)
        rebuild_read_model_db(self.db_path, self.log_path)

        selected, _registry = db_attempt_rows(self.db_path, since=since)
        by_run = {row["run_id"]: row for row in selected}

        self.assertNotIn("run-1", by_run)
        self.assertIn("run-2", by_run)
        self.assertNotIn("run-3", by_run)
        self.assertNotIn("run-4", by_run)

    def test_same_size_prefix_metadata_change_forces_rebuild(self) -> None:
        base = attempt("run-1", "a", model="openrouter/acme/live", verdict="PASS")
        write_jsonl(self.log_path, [base])
        rebuild_read_model_db(self.db_path, self.log_path)

        # Simulate a same-size committed-prefix mutation: identical byte length
        # but changed metadata (verdict PASS -> FAIL) within the synced prefix.
        mutated = dict(base)
        mutated["verdict"] = "FAIL"
        payload = json.dumps(mutated) + "\n"
        # Same byte length as the original committed prefix line.
        self.assertEqual(len(payload), len(json.dumps(base) + "\n"))
        self.log_path.write_bytes(payload.encode("utf-8"))

        with mock.patch("ringer.hash_file_prefix", wraps=ringer.hash_file_prefix) as hashed:
            result = sync_read_model_db(self.db_path, self.log_path)
        self.assertGreaterEqual(hashed.call_count, 1)
        self.assertTrue(result.rebuilt)
        db_rows, _registry = db_attempt_rows(self.db_path)
        by_model = {row["model"]: row for row in aggregate_model_scoreboard_rows(db_rows)}
        # The prefix byte content changed (verdict metadata), so the rebuild
        # must reflect the corrected FAIL verdict instead of the stale PASS.
        self.assertIn("openrouter/acme/live", by_model)
        self.assertEqual(0.0, by_model["openrouter/acme/live"]["pass_rate"])

    def test_truncation_forces_rebuild(self) -> None:
        write_jsonl(
            self.log_path,
            [
                attempt("run-1", "a"),
                attempt("run-2", "a", verdict="FAIL"),
            ],
        )
        rebuild_read_model_db(self.db_path, self.log_path)

        # Truncate the log below the stored offset to force a rebuild.
        self.log_path.write_bytes((json.dumps(attempt("run-1", "a")) + "\n").encode("utf-8"))

        result = sync_read_model_db(self.db_path, self.log_path)
        self.assertTrue(result.rebuilt)
        db_rows, _registry = db_attempt_rows(self.db_path)
        self.assertEqual(1, len(db_rows))
        self.assertEqual("run-1", db_rows[0]["run_id"])


if __name__ == "__main__":
    unittest.main()
