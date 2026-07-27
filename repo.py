"""Git-based repo state management bound to an :class:`Executor`.

Replaces the original harness's rsync-snapshot + ASV-commit machinery with a
single, clean git-only model:

* ``ensure_repo`` makes a baseline commit capturing the initial working tree.
* ``commit_candidate`` snapshots the agent's edits as a new commit (a
  *candidate hash*).
* ``restore_to`` hard-resets the working tree back to any candidate / the
  baseline.
* ``diff`` / ``numstat`` feed the patch capture and the policy checker.

Every candidate is a real git commit, so a branch can be resurrected for a
FULL re-evaluation with ``git reset --hard <hash>``.
"""

from __future__ import annotations

import logging
import re
import shlex

from mokopt.executor.base import Executor

logger = logging.getLogger(__name__)

# Paths excluded from candidate commits so transient build / cache artifacts
# don't inflate patch_lines or leak into the exported diff.
_EXCLUDE_PATHSPECS = (
    ":(exclude)**/__pycache__",
    ":(exclude)*.pyc",
    ":(exclude).mokopt",
    ":(exclude)**/.pytest_cache",
)

_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


class GitRepo:
    def __init__(self, executor: Executor):
        self._ex = executor
        self._repo = executor.repo_path

    def _git(self, args: str, timeout_s: int | None = None):
        return self._ex.run(f"cd {self._repo} && git {args}", timeout_s=timeout_s)

    def ensure_repo(self) -> str | None:
        """Initialise git (if needed), set identity, commit the baseline.

        Returns the baseline commit SHA, or None on failure.
        """
        setup = (
            f"cd {self._repo} && "
            "git config --global --add safe.directory '*' >/dev/null 2>&1 || true; "
            "if [ ! -d .git ]; then git init -q; fi; "
            "git config user.email mokopt@local >/dev/null 2>&1 || true; "
            "git config user.name mokopt >/dev/null 2>&1 || true; "
            "rm -f .git/index.lock 2>/dev/null || true; "
            "git add -A "
            + " ".join(shlex.quote(p) for p in _EXCLUDE_PATHSPECS)
            + " >/dev/null 2>&1 || true; "
            "git commit --allow-empty -q -m 'mokopt baseline' >/dev/null 2>&1 || true; "
            "git rev-parse HEAD"
        )
        res = self._ex.run(setup)
        if not res.ok:
            logger.warning("ensure_repo failed: %s", res.output[-500:])
            return None
        return self._parse_sha(res.output)

    def commit_candidate(self, label: str) -> str | None:
        """Commit the current working tree as a candidate. Returns SHA / None."""
        msg = f"mokopt candidate: {label}"
        excludes = " ".join(shlex.quote(p) for p in _EXCLUDE_PATHSPECS)
        cmd = (
            f"cd {self._repo} && "
            "rm -f .git/index.lock 2>/dev/null || true; "
            f"git add -A {excludes} >/dev/null && "
            f"git commit --allow-empty -q -m {shlex.quote(msg)} >/dev/null && "
            "git rev-parse HEAD"
        )
        res = self._ex.run(cmd)
        if not res.ok:
            logger.warning("commit_candidate(%s) failed: %s", label, res.output[-500:])
            return None
        return self._parse_sha(res.output)

    def restore_to(self, sha: str) -> bool:
        """Hard-reset the working tree to ``sha`` (keeps ignored build files)."""
        cmd = (
            f"cd {self._repo} && "
            "git checkout -- . 2>/dev/null && "
            f"git reset --hard {shlex.quote(sha)} >/dev/null 2>&1 && "
            "git clean -fd >/dev/null 2>&1"
        )
        res = self._ex.run(cmd)
        return res.ok

    def head(self) -> str | None:
        res = self._git("rev-parse HEAD")
        return self._parse_sha(res.output) if res.ok else None

    def diff(self, parent: str, cand: str) -> str:
        res = self._git(f"diff {shlex.quote(parent)} {shlex.quote(cand)}")
        return res.output if res.ok else ""

    def numstat(self, parent: str, cand: str):
        """Return ``(exit_code, raw_numstat_output)`` for the policy checker."""
        res = self._git(f"diff --numstat {shlex.quote(parent)} {shlex.quote(cand)}")
        return res.exit_code, res.output

    @staticmethod
    def _parse_sha(output: str) -> str | None:
        for line in reversed(output.strip().splitlines()):
            tok = line.strip()
            if _SHA_RE.fullmatch(tok):
                return tok
        return None
