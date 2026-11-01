#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ringer  # noqa: E402
from ringer import (  # noqa: E402
    WorktreeExpiryConfig,
    expire_worktrees,
    load_worktree_expiry_config,
    start_worktree_expiry_auto,
)

DAY = 86400.0


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )
    return result.stdout.strip()


def age_tree(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp))


class WorktreeExpiryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(os.path.realpath(self.tmp.name))
        self.base = base
        self.old_env = os.environ.copy()
        self.addCleanup(self._restore_env)
        os.environ["RINGER_HOME"] = str(base / "ringer-home")
        self.state_dir = base / "state"
        (self.state_dir / "runs").mkdir(parents=True)
        origin = base / "origin.git"
        git("init", "-q", "--bare", "-b", "main", str(origin))
        self.repo = base / "repo"
        git("clone", "-q", str(origin), str(self.repo))
        (self.repo / ".gitignore").write_text(".venv/\n*.log\nnotes/\n")
        (self.repo / "tracked.txt").write_text("one\n")
        (self.repo / "staged.txt").write_text("s\n")
        git("add", ".", cwd=self.repo)
        git("commit", "-q", "-m", "base", cwd=self.repo)
        git("push", "-q", "origin", "HEAD:main", cwd=self.repo)
        self.scheduler_dir = base / "launch-agents"
        self.scheduler_dir.mkdir()
        self.run_seq = 0

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)

    def add_worktree(self, path: Path, *, record: bool = True, branch: str | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if branch:
            git("-C", str(self.repo), "worktree", "add", "-q", "-b", branch, str(path), "HEAD")
        else:
            git("-C", str(self.repo), "worktree", "add", "-q", str(path), "HEAD")
        if record:
            self.run_seq += 1
            run_id = f"job-2026T{self.run_seq:04d}"
            (self.state_dir / "runs" / f"{run_id}.json").write_text(
                json.dumps({"run_id": run_id, "tasks": [{"key": path.name, "taskdir": str(path), "status": "fail"}]})
            )
        return path

    def expire(self, *, dry_run: bool = False, config: WorktreeExpiryConfig | None = None, **kwargs):
        lines: list[str] = []
        results = expire_worktrees(
            self.state_dir,
            config or WorktreeExpiryConfig(),
            dry_run=dry_run,
            scheduler_paths=[self.scheduler_dir],
            crontab_text="",
            log=lines.append,
            **{"open_prs": lambda repo: (set(), set()), **kwargs},
        )
        return {r["path"]: r for r in results}, lines

    def adopt(self) -> WorktreeExpiryConfig:
        return WorktreeExpiryConfig(repos=(self.repo,), roots=(self.base / "wts",))

    def refs(self) -> str:
        return git("-C", str(self.repo), "for-each-ref", "refs/archive")

    def make_dirty(self, wt: Path) -> None:
        (wt / "tracked.txt").write_text("two\n")
        (wt / "staged.txt").write_text("staged change\n")
        git("add", "staged.txt", cwd=wt)
        (wt / "new.txt").write_text("untracked\n")
        (wt / "notes").mkdir()
        (wt / "notes" / "keep.md").write_text("ignored but valuable\n")
        (wt / "run.log").write_text("ignored log\n")
        (wt / ".venv" / "lib").mkdir(parents=True)
        (wt / ".venv" / "lib" / "big.py").write_text("cache\n")

    def archive_dir(self, result: dict) -> Path:
        return Path(result["archive"])

    def test_dry_run_reports_and_changes_nothing(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-a")
        self.make_dirty(wt)
        age_tree(wt, 10 * DAY)
        results, lines = self.expire(dry_run=True)
        self.assertEqual(results[str(wt)]["action"], "would-expire")
        self.assertTrue(wt.exists())
        self.assertFalse((self.state_dir / "patches").exists())
        self.assertEqual(git("-C", str(self.repo), "for-each-ref", "refs/archive"), "")
        self.assertTrue(any("would-expire" in line for line in lines))

    def test_expire_archives_everything_then_removes(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-b", branch="feature/keep-me", record=False)
        self.make_dirty(wt)
        # Display-oriented diff settings must not leak into the archived patch.
        git("config", "diff.external", "/bin/echo", cwd=self.repo)
        git("config", "color.ui", "always", cwd=self.repo)
        git("config", "diff.noprefix", "true", cwd=self.repo)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire(config=self.adopt())
        result = results[str(wt)]
        self.assertEqual(result["action"], "expired", result)
        self.assertFalse(wt.exists())
        # Branches are never deleted.
        self.assertIn("feature/keep-me", git("-C", str(self.repo), "branch", "--list", "feature/keep-me"))
        archive = self.archive_dir(result)
        self.assertTrue(archive.name.startswith("t-"))
        self.assertIn(b"two", (archive / "tracked.patch").read_bytes())
        self.assertIn(b"staged change", (archive / "staged.patch").read_bytes())
        with tarfile.open(archive / "untracked.tar.gz") as tf:
            self.assertEqual(tf.getnames(), ["new.txt"])
        with tarfile.open(archive / "ignored.tar.gz") as tf:
            names = tf.getnames()
        self.assertIn("notes/keep.md", names)
        self.assertIn("run.log", names)
        self.assertFalse(any(".venv" in name for name in names))
        meta = json.loads((archive / "meta.json").read_text())
        self.assertEqual(meta["path"], str(wt))
        self.assertEqual(meta["branch"], "feature/keep-me")
        self.assertEqual(meta["head"], git("-C", str(self.repo), "rev-parse", "feature/keep-me"))
        self.assertEqual(meta["source"], "adopted")
        # The tracked patch restores the change onto a fresh checkout.
        restore = self.base / "restore"
        git("-C", str(self.repo), "worktree", "add", "-q", str(restore), meta["head"])
        git("apply", str(archive / "tracked.patch"), cwd=restore)
        self.assertEqual((restore / "tracked.txt").read_text(), "two\n")
        self.assertEqual((restore / "staged.txt").read_text(), "staged change\n")
        self.assertIn(str(wt), (archive.parent / "INDEX.md").read_text())

    def test_reflog_only_commits_are_pinned_and_bundled(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-c")
        (wt / "tracked.txt").write_text("lost\n")
        git("commit", "-q", "-am", "reflog only", cwd=wt)
        lost = git("rev-parse", "HEAD", cwd=wt)
        git("reset", "-q", "--hard", "HEAD~1", cwd=wt)
        (wt / "tracked.txt").write_text("head\n")
        git("commit", "-q", "-am", "unpushed head", cwd=wt)
        head = git("rev-parse", "HEAD", cwd=wt)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        result = results[str(wt)]
        self.assertEqual(result["action"], "expired", result)
        refs = git("-C", str(self.repo), "for-each-ref", "--format=%(objectname) %(refname)", "refs/archive")
        pinned = {line.split()[0] for line in refs.splitlines()}
        self.assertIn(lost, pinned)
        self.assertIn(head, pinned)
        for line in refs.splitlines():
            for component in line.split()[1].split("/"):
                self.assertFalse(component.startswith("."), line)
        bundle = self.archive_dir(result) / "commits.bundle"
        git("-C", str(self.repo), "bundle", "verify", str(bundle))
        self.assertIn(lost, git("bundle", "list-heads", str(bundle)))
        meta = json.loads((self.archive_dir(result) / "meta.json").read_text())
        self.assertEqual(sorted(meta["pinned_commits"].values()), sorted([lost, head]))

    def test_per_worktree_ref_commits_are_pinned(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-wref")
        orphan = git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "only on refs/worktree", cwd=wt)
        git("update-ref", "refs/worktree/keep", orphan, cwd=wt)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "expired", results[str(wt)])
        self.assertIn(orphan, self.refs())

    def test_dot_prefixed_path(self) -> None:
        wt = self.add_worktree(self.base / ".ringer" / "jobs" / ".hidden-job" / "task")
        (wt / "tracked.txt").write_text("dot\n")
        git("commit", "-q", "-am", "dot commit", cwd=wt)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        result = results[str(wt)]
        self.assertEqual(result["action"], "expired", result)
        self.assertTrue(self.archive_dir(result).name.startswith("t-"))
        meta = json.loads((self.archive_dir(result) / "meta.json").read_text())
        self.assertEqual(meta["task"], "task")
        self.assertTrue(meta["run_id"].startswith("job-2026T"))
        refs = git("-C", str(self.repo), "for-each-ref", "--format=%(refname)", "refs/archive")
        self.assertTrue(refs)
        for ref in refs.splitlines():
            subprocess.run(["git", "check-ref-format", ref], check=True)
            self.assertTrue(all(not part.startswith(".") for part in ref.split("/")), ref)

    def test_live_cwd_tree_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-d")
        age_tree(wt, 10 * DAY)
        proc = subprocess.Popen(["sleep", "60"], cwd=wt)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        time.sleep(0.2)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("open", results[str(wt)]["reason"])
        self.assertTrue(wt.exists())

    def test_locked_tree_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-e")
        git("-C", str(self.repo), "worktree", "lock", str(wt))
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("locked", results[str(wt)]["reason"])
        self.assertTrue(wt.exists())

    def test_fresh_tree_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-f")
        age_tree(wt, 2 * DAY)
        (wt / "new.txt").write_text("recent write\n")
        age_tree(wt, 10 * DAY)
        os.utime(wt / "new.txt", None)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("idle", results[str(wt)]["reason"])

    def test_cache_writes_do_not_count_as_activity(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-g")
        (wt / "__pycache__").mkdir()
        (wt / "__pycache__" / "x.pyc").write_bytes(b"x")
        age_tree(wt, 10 * DAY)
        os.utime(wt / "__pycache__" / "x.pyc", None)
        results, _ = self.expire(dry_run=True)
        self.assertEqual(results[str(wt)]["action"], "would-expire")

    def test_unrecorded_tree_needs_adoption(self) -> None:
        wt = self.add_worktree(self.base / "orchestrator" / "control", record=False)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire(dry_run=True)
        self.assertNotIn(str(wt), results)
        adopt = WorktreeExpiryConfig(repos=(self.repo,), roots=(self.base / "orchestrator",))
        results, _ = self.expire(dry_run=True, config=adopt)
        self.assertEqual(results[str(wt)]["action"], "would-expire")
        self.assertNotIn(str(self.repo), results)

    def test_active_run_workdir_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-h")
        age_tree(wt, 10 * DAY)
        ringer.register_active_run("live-run", "codex", "job", wt.parent, pid=os.getpid())
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("active run", results[str(wt)]["reason"])

    def test_scheduler_reference_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-i")
        age_tree(wt, 10 * DAY)
        (self.scheduler_dir / "com.example.job.plist").write_text(
            f"<plist><string>{wt}/scripts/run.sh</string></plist>"
        )
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("scheduler", results[str(wt)]["reason"])
        # A sibling whose name merely starts with the same text is not referenced.
        other = self.add_worktree(self.base / "wts" / "job-i2")
        age_tree(other, 10 * DAY)
        (self.scheduler_dir / "com.example.job.plist").write_text(f"<string>{wt}</string>")
        results, _ = self.expire(dry_run=True)
        self.assertEqual(results[str(other)]["action"], "would-expire")

    def test_branch_checkout_at_recorded_path_is_not_ringers(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "hand-made", branch="topic/hand")
        age_tree(wt, 10 * DAY)
        results, _ = self.expire(dry_run=True)
        self.assertNotIn(str(wt), results)

    def test_staged_then_deleted_file_expires(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-ad")
        (wt / "gone.txt").write_text("x\n")
        git("add", "gone.txt", cwd=wt)
        (wt / "gone.txt").unlink()
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "expired", results[str(wt)])

    def test_skip_worktree_edit_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-sw")
        git("update-index", "--skip-worktree", "tracked.txt", cwd=wt)
        (wt / "tracked.txt").write_text("hidden edit\n")
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("skip-worktree", results[str(wt)]["reason"])

    def test_submodules_and_operations_in_progress_are_kept(self) -> None:
        sub = self.add_worktree(self.base / "wts" / "job-sub")
        merging = self.add_worktree(self.base / "wts" / "job-merge")
        for wt, name in ((sub, "modules"), (merging, "MERGE_HEAD")):
            git_dir = Path(git("rev-parse", "--absolute-git-dir", cwd=wt))
            if name == "modules":
                (git_dir / name).mkdir()
            else:
                (git_dir / name).write_text(git("rev-parse", "HEAD", cwd=wt) + "\n")
            age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(sub)]["action"], "keep")
        self.assertIn("submodule", results[str(sub)]["reason"])
        self.assertEqual(results[str(merging)]["action"], "keep")
        self.assertIn("MERGE_HEAD", results[str(merging)]["reason"])

    def test_big_ignored_file_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-big")
        (wt / "out.log").write_text("x" * 100)
        age_tree(wt, 10 * DAY)
        with mock.patch.object(ringer, "WORKTREE_EXPIRY_BIG_IGNORED_BYTES", 50):
            results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("out.log", results[str(wt)]["reason"])

    def test_open_pr_branch_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-pr", branch="feature/pr-open", record=False)
        plain = self.add_worktree(self.base / "wts" / "job-nopr", branch="feature/merged", record=False)
        age_tree(wt, 10 * DAY)
        age_tree(plain, 10 * DAY)
        results, _ = self.expire(config=self.adopt(), open_prs=lambda repo: ({"feature/pr-open"}, set()))
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("open PR", results[str(wt)]["reason"])
        self.assertEqual(results[str(plain)]["action"], "expired")
        # Unknown PR state keeps branch trees; detached trees don't need it.
        detached = self.add_worktree(self.base / "wts" / "job-det")
        age_tree(detached, 10 * DAY)
        results, _ = self.expire(config=self.adopt(), open_prs=lambda repo: None)
        self.assertIn("cannot check open PRs", results[str(wt)]["reason"])
        self.assertEqual(results[str(detached)]["action"], "expired")

    def test_pr_matched_by_upstream_name_or_head_sha(self) -> None:
        tracking = self.add_worktree(self.base / "wts" / "job-up", branch="local-name", record=False)
        git("branch", "--set-upstream-to=origin/main", cwd=tracking)
        detached = self.add_worktree(self.base / "wts" / "job-sha")
        (detached / "tracked.txt").write_text("pr head\n")
        git("commit", "-q", "-am", "pr head", cwd=detached)
        head = git("rev-parse", "HEAD", cwd=detached)
        age_tree(tracking, 10 * DAY)
        age_tree(detached, 10 * DAY)
        results, _ = self.expire(dry_run=True, config=self.adopt(), open_prs=lambda repo: ({"main"}, {head}))
        self.assertIn("open PR", results[str(tracking)]["reason"])
        self.assertIn("open PR", results[str(detached)]["reason"])

    def test_github_remotes_parsed(self) -> None:
        git("remote", "add", "fork", "git@github.com:me/thing.git", cwd=self.repo)
        git("remote", "add", "up", "https://github.com/Org/thing", cwd=self.repo)
        self.assertEqual(ringer._github_remotes(str(self.repo)), {"me/thing", "Org/thing"})

    def test_inherited_git_dir_does_not_misdirect(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-env")
        (wt / "tracked.txt").write_text("unique\n")
        git("commit", "-q", "-am", "unique", cwd=wt)
        unique = git("rev-parse", "HEAD", cwd=wt)
        age_tree(wt, 10 * DAY)
        os.environ["GIT_DIR"] = str(self.repo / ".git")
        os.environ["GIT_INDEX_FILE"] = str(self.repo / ".git" / "index")
        try:
            results, _ = self.expire()
        finally:
            os.environ.pop("GIT_DIR")
            os.environ.pop("GIT_INDEX_FILE")
        self.assertEqual(results[str(wt)]["action"], "expired", results[str(wt)])
        self.assertIn(unique, self.refs())

    def test_intent_to_add_expires(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-ita")
        (wt / "later.txt").write_text("not staged yet\n")
        git("add", "-N", "later.txt", cwd=wt)
        age_tree(wt, 10 * DAY)
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "expired", results[str(wt)])

    def test_unmerged_and_sequencer_are_kept(self) -> None:
        seq = self.add_worktree(self.base / "wts" / "job-seq")
        (Path(git("rev-parse", "--absolute-git-dir", cwd=seq)) / "sequencer").mkdir()
        conflict = self.add_worktree(self.base / "wts" / "job-conflict")
        blob = git("hash-object", "-w", "tracked.txt", cwd=conflict)
        subprocess.run(
            ["git", "update-index", "--index-info"], cwd=conflict, check=True,
            input=f"0 {'0' * 40}\ttracked.txt\n100644 {blob} 1\ttracked.txt\n100644 {blob} 2\ttracked.txt\n".encode(),
        )
        age_tree(seq, 10 * DAY)
        age_tree(conflict, 10 * DAY)
        results, _ = self.expire()
        self.assertIn("sequencer", results[str(seq)]["reason"])
        self.assertIn("unmerged", results[str(conflict)]["reason"])

    def test_symlinked_scheduler_dir_into_tree_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-sdir")
        (wt / "tasks" / "job").mkdir(parents=True)
        (self.scheduler_dir / "job").symlink_to(wt / "tasks" / "job")
        age_tree(wt, 10 * DAY)
        results, _ = self.expire(dry_run=True)
        self.assertIn("scheduler", results[str(wt)]["reason"])

    def test_scheduler_file_inside_tree_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-link")
        (wt / "agent.plist").write_text("<plist/>")
        (self.scheduler_dir / "com.example.agent.plist").symlink_to(wt / "agent.plist")
        age_tree(wt, 10 * DAY)
        results, _ = self.expire(dry_run=True)
        self.assertEqual(results[str(wt)]["action"], "keep")
        self.assertIn("scheduler", results[str(wt)]["reason"])

    def test_reused_path_keeps_earlier_archive(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-re")
        age_tree(wt, 10 * DAY)
        first, _ = self.expire()
        first_archive = Path(first[str(wt)]["archive"])
        self.add_worktree(wt)
        age_tree(wt, 10 * DAY)
        second, _ = self.expire()
        second_archive = Path(second[str(wt)]["archive"])
        self.assertEqual(second[str(wt)]["action"], "expired")
        self.assertNotEqual(first_archive, second_archive)
        self.assertTrue((first_archive / "meta.json").exists())

    def test_dry_run_does_not_rewrite_active_runs(self) -> None:
        path = ringer.active_runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dead": {"pid": 999999, "workdir": "/x"}}))
        before = path.read_text()
        self.expire(dry_run=True)
        self.assertEqual(path.read_text(), before)

    def test_unreadable_scheduler_file(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-o")
        age_tree(wt, 10 * DAY)
        secret = self.scheduler_dir / "com.vendor.daemon.plist"
        secret.write_text("<plist/>")
        secret.chmod(0)
        self.addCleanup(secret.chmod, 0o644)
        # Our own unreadable file fails the pass closed.
        results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "fail")
        self.assertIn("scheduler file", results[str(wt)]["reason"])
        self.assertTrue(wt.exists())
        # Another user's (root's vendor daemons) is skipped with a note.
        real_uid = os.getuid()
        with mock.patch.object(ringer.os, "getuid", return_value=real_uid + 1):
            results, lines = self.expire(dry_run=True)
        self.assertEqual(results[str(wt)]["action"], "would-expire")
        self.assertTrue(any("skipped unreadable" in line for line in lines))

    def test_crontab_reference_is_kept(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-j")
        age_tree(wt, 10 * DAY)
        results = expire_worktrees(
            self.state_dir, WorktreeExpiryConfig(), dry_run=False,
            scheduler_paths=[], crontab_text=f"0 * * * * cd {wt} && ./x\n", log=lambda _: None,
        )
        self.assertEqual({r["path"]: r for r in results}[str(wt)]["action"], "keep")
        self.assertTrue(wt.exists())

    def test_fails_closed_when_tar_fails(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-k")
        self.make_dirty(wt)
        age_tree(wt, 10 * DAY)
        real = ringer._expiry_sh

        def broken(args, **kwargs):
            if args[0] == "tar" and "-czf" in args:
                return 1, b"", b"boom"
            return real(args, **kwargs)

        with mock.patch.object(ringer, "_expiry_sh", side_effect=broken):
            results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "fail")
        self.assertTrue(wt.exists())
        # A failed attempt leaves no archive or pins behind to pile up daily.
        self.assertEqual(list((self.state_dir / "patches").rglob("*.patch")), [])
        self.assertEqual(self.refs(), "")

    def test_fails_closed_when_bundle_verify_fails(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-l")
        (wt / "tracked.txt").write_text("x\n")
        git("commit", "-q", "-am", "unpushed", cwd=wt)
        age_tree(wt, 10 * DAY)
        real = ringer._expiry_sh

        def broken(args, **kwargs):
            if "bundle" in args and "verify" in args:
                return 1, b"", b"bad bundle"
            return real(args, **kwargs)

        with mock.patch.object(ringer, "_expiry_sh", side_effect=broken):
            results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "fail")
        self.assertIn("bundle", results[str(wt)]["reason"])
        self.assertTrue(wt.exists())
        self.assertEqual(self.refs(), "")

    def test_tree_written_during_archive_is_not_removed(self) -> None:
        wt = self.add_worktree(self.base / "wts" / "job-m")
        age_tree(wt, 10 * DAY)
        real = ringer._expiry_sh

        def touching(args, **kwargs):
            if args[:2] == ["git", "-C"] and "diff-index" in args:
                (wt / "late.txt").write_text("written mid-archive\n")
            return real(args, **kwargs)

        with mock.patch.object(ringer, "_expiry_sh", side_effect=touching):
            results, _ = self.expire()
        self.assertEqual(results[str(wt)]["action"], "fail")
        self.assertTrue(wt.exists())

    def test_nested_worktree_parent_is_kept(self) -> None:
        outer = self.add_worktree(self.base / "wts" / "job-n")
        self.add_worktree(outer / "inner", record=False)
        age_tree(outer, 10 * DAY)
        results, _ = self.expire(dry_run=True)
        self.assertEqual(results[str(outer)]["action"], "keep")
        self.assertIn("nested", results[str(outer)]["reason"])


class WorktreeExpiryConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = load_worktree_expiry_config(None)
        self.assertEqual(config.idle_days, 7.0)
        self.assertEqual(config.repos, ())
        self.assertTrue(config.auto)

    def test_parses_tables(self) -> None:
        config = load_worktree_expiry_config(
            {"idle_days": 3, "repos": ["~/r"], "roots": ["~/w"], "auto": False}
        )
        self.assertEqual(config.idle_days, 3.0)
        self.assertEqual(config.repos, (Path("~/r").expanduser(),))
        self.assertEqual(config.roots, (Path("~/w").expanduser(),))
        self.assertFalse(config.auto)

    def test_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            load_worktree_expiry_config({"idle_days": 0})
        with self.assertRaises(ValueError):
            load_worktree_expiry_config({"repos": "~/r"})
        with self.assertRaises(ValueError):
            load_worktree_expiry_config({"idle_days": float("nan")})
        with self.assertRaises(ValueError):
            load_worktree_expiry_config({"auto": "false"})


class WorktreeExpiryAutoStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_dir = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old_env)))
        os.environ.pop("RINGER_NO_WORKTREE_EXPIRY", None)

    def test_spawns_detached_once_per_interval(self) -> None:
        with mock.patch.object(ringer.subprocess, "Popen") as popen:
            self.assertIsNotNone(start_worktree_expiry_auto(self.state_dir, WorktreeExpiryConfig()))
            self.assertIsNone(start_worktree_expiry_auto(self.state_dir, WorktreeExpiryConfig()))
        self.assertEqual(popen.call_count, 1)
        args, kwargs = popen.call_args
        self.assertIn("expire-worktrees", args[0])
        self.assertTrue(kwargs.get("start_new_session"))

    def test_opt_outs(self) -> None:
        with mock.patch.object(ringer.subprocess, "Popen") as popen:
            self.assertIsNone(start_worktree_expiry_auto(self.state_dir, WorktreeExpiryConfig(auto=False)))
            os.environ["RINGER_NO_WORKTREE_EXPIRY"] = "1"
            self.assertIsNone(start_worktree_expiry_auto(self.state_dir, WorktreeExpiryConfig()))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
