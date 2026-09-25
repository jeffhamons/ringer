"""End-to-end Seatbelt coverage for engines/opencode-sandboxed.sh.

These tests drive the real wrapper with a real git linked worktree and a fake
opencode executable, so the sandboxed child runs real git commands. Every path
lives under pytest's tmp_path; no user repository, git config, credentials, or
home data is touched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by the repo's unittest runner
    class _PytestStub:
        """Minimal stand-in so `python3 -m unittest discover -s tests` can import.

        The repo's canonical runner has no pytest; these pytest-style tests are
        executed only by the Seatbelt verification's pytest command.
        """

        class mark:
            @staticmethod
            def skipif(*_args: object, **_kwargs: object):
                def decorate(func):
                    return func

                return decorate

        @staticmethod
        def fixture(*args: object, **_kwargs: object):
            if len(args) == 1 and callable(args[0]):
                return args[0]

            def decorate(func):
                return func

            return decorate

    pytest = _PytestStub()  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "engines" / "opencode-sandboxed.sh"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

pytestmark = pytest.mark.skipif(
    not SANDBOX_EXEC.exists(),
    reason="requires the macOS sandbox-exec binary for the Seatbelt integration test",
)

FAKE_OPENCODE = """#!/bin/bash
set -u
mode="$1"; shift
case "$mode" in
  write)
    printf 'sentinel\\n' > "$1" || exit 1
    echo "WROTE $1"
    ;;
  git-add)
    cd "$1" || exit 1
    git add "$2" || exit 1
    echo "STAGED $2"
    ;;
  try-write)
    if printf 'mutated\\n' > "$1" 2>/dev/null; then
      echo "ALLOWED $1"
    else
      echo "DENIED $1"
    fi
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
"""


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    ):
        env.pop(name, None)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Fixture Author",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture Committer",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        }
    )
    return env


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=_git_env(),
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    opencode = bindir / "opencode"
    opencode.write_text(FAKE_OPENCODE)
    opencode.chmod(0o755)
    return bindir


@pytest.fixture
def git_worktree(tmp_path: Path) -> SimpleNamespace:
    origin = tmp_path / "origin.git"
    clone = tmp_path / "clone"
    worktree = tmp_path / "worktree"

    _git(["init", "--bare", "-q", str(origin)], tmp_path)
    _git(["clone", "-q", str(origin), str(clone)], tmp_path)
    (clone / "tracked.txt").write_text("fixture\n")
    _git(["add", "tracked.txt"], clone)
    _git(["commit", "-q", "-m", "initial"], clone)
    _git(["worktree", "add", "-q", "-b", "fixture-branch", str(worktree)], clone)
    _git(["pack-refs", "--all"], clone)

    packed_refs = clone / ".git" / "packed-refs"
    if not packed_refs.exists():
        packed_refs.write_text("# pack-refs with: peeled fully-peeled sorted\n")

    return SimpleNamespace(origin=origin, clone=clone, worktree=worktree)


def _run_wrapper(
    taskdir: Path, bindir: Path, tmp_path: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    scratch = tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)

    env = _git_env()
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(home)
    env["TMPDIR"] = str(scratch)

    return subprocess.run(
        ["/bin/bash", str(WRAPPER), str(taskdir), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plain_task_directory_can_write_sentinel(tmp_path: Path, fake_bin: Path) -> None:
    taskdir = tmp_path / "plain-task"
    taskdir.mkdir()
    sentinel = taskdir / "sentinel"

    result = _run_wrapper(taskdir, fake_bin, tmp_path, "write", str(sentinel))

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text() == "sentinel\n"


def test_linked_worktree_git_add_succeeds_under_sandbox(
    tmp_path: Path, fake_bin: Path, git_worktree: SimpleNamespace
) -> None:
    worktree = git_worktree.worktree
    marker = worktree / "marker"
    marker.write_text("marker\n")

    result = _run_wrapper(
        worktree, fake_bin, tmp_path, "git-add", str(worktree), "marker"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGED marker" in result.stdout

    staged = _git(["diff", "--cached", "--name-only"], worktree)
    assert staged.stdout.split() == ["marker"]

    blob = _git(["rev-parse", ":marker"], worktree).stdout.strip()
    assert blob
    _git(["cat-file", "-e", blob], worktree)


def test_common_git_metadata_writes_are_denied(
    tmp_path: Path, fake_bin: Path, git_worktree: SimpleNamespace
) -> None:
    common = git_worktree.clone / ".git"
    worktree = git_worktree.worktree

    existing = {
        "hook": common / "hooks" / "fixture-hook",
        "info": common / "info" / "fixture-info",
        "ref": common / "refs" / "fixture-ref",
        "config": common / "config",
        "packed-refs": common / "packed-refs",
    }
    for path in existing.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("fixture\n")
    snapshot = {name: path.read_bytes() for name, path in existing.items()}

    new_paths = {
        "new-hook": common / "hooks" / "fixture-new-hook",
        "new-info": common / "info" / "fixture-new-info",
        "new-ref": common / "refs" / "fixture-new-ref",
    }

    # Negative control: these paths are writable outside the sandbox, so a
    # sandboxed denial can only come from Seatbelt.
    control = common / "hooks" / "control-sentinel"
    control.write_text("control\n")
    assert control.read_text() == "control\n"
    control.unlink()
    for name, path in existing.items():
        assert os.access(path, os.W_OK), name
    for name, path in new_paths.items():
        assert os.access(path.parent, os.W_OK), name

    for name, path in existing.items():
        result = _run_wrapper(worktree, fake_bin, tmp_path, "try-write", str(path))
        assert result.returncode == 0, result.stderr
        assert f"DENIED {path}" in result.stdout, (name, result.stdout)

    for name, path in new_paths.items():
        result = _run_wrapper(worktree, fake_bin, tmp_path, "try-write", str(path))
        assert result.returncode == 0, result.stderr
        assert f"DENIED {path}" in result.stdout, (name, result.stdout)
        assert not path.exists(), name

    for name, path in existing.items():
        assert path.read_bytes() == snapshot[name], name


def test_forged_gitdir_cannot_open_other_object_store(
    tmp_path: Path, fake_bin: Path, git_worktree: SimpleNamespace
) -> None:
    # The task controls its .git file. It must not be able to claim an
    # unrelated common directory's object store as its own.
    common = git_worktree.clone / ".git"
    taskdir = tmp_path / "forged-task"
    taskdir.mkdir()
    synthetic = taskdir / "synthetic-gitdir"
    synthetic.mkdir()
    (taskdir / ".git").write_text(f"gitdir: {synthetic}\n")
    (synthetic / "HEAD").write_text("ref: refs/heads/fixture-branch\n")
    (synthetic / "commondir").write_text(f"{common}\n")
    (synthetic / "gitdir").write_text(f"{taskdir / '.git'}\n")

    probe = _git(["rev-parse", "--git-common-dir"], taskdir)
    assert Path(probe.stdout.strip()).resolve() == common.resolve()
    assert _git(["rev-parse", "--is-inside-work-tree"], taskdir).stdout.strip() == "true"

    target = common / "objects" / "forged-marker"
    result = _run_wrapper(taskdir, fake_bin, tmp_path, "try-write", str(target))

    assert result.returncode == 0, result.stderr
    assert f"DENIED {target}" in result.stdout, result.stdout
    assert not target.exists()


def test_symlinked_object_store_does_not_gain_write_allowance(
    tmp_path: Path, fake_bin: Path, git_worktree: SimpleNamespace
) -> None:
    common = git_worktree.clone / ".git"
    objects = common / "objects"
    target_dir = common / "objects-target"
    objects.rename(target_dir)
    objects.symlink_to(target_dir, target_is_directory=True)

    target = target_dir / "symlink-marker"
    result = _run_wrapper(git_worktree.worktree, fake_bin, tmp_path, "try-write", str(target))

    assert result.returncode == 0, result.stderr
    assert f"DENIED {target}" in result.stdout, result.stdout
    assert not target.exists()
