"""Policy checker: enforce anti-cheat / safety rules at the harness layer.

A patch is rejected (PRUNE) when it touches any path under ``risk_paths`` (e.g.
``tests/``, ``benchmarks/``, ``setup.py``) or exceeds ``patch_max_lines``. The
checker reads the diff against the *parent* commit rather than the original
baseline so deltas across iterations stay attributable.
"""

from __future__ import annotations

from dataclasses import dataclass

from mokopt.repo import GitRepo
from mokopt.state import BetaConfig


@dataclass
class PolicyVerdict:
    ok: bool
    patch_lines: int
    touched_files: list[str]
    reason: str | None = None


class PolicyChecker:
    """Inspect a candidate commit against a parent and return a PolicyVerdict."""

    def __init__(self, beta: BetaConfig, repo: GitRepo):
        self._beta = beta
        self._repo = repo

    def check(self, parent_hash: str, candidate_hash: str) -> PolicyVerdict:
        exit_code, raw = self._repo.numstat(parent_hash, candidate_hash)
        if exit_code != 0:
            return PolicyVerdict(
                ok=False,
                patch_lines=0,
                touched_files=[],
                reason=f"git diff failed (exit={exit_code})",
            )
        added = removed = 0
        files: list[str] = []
        for line in raw.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            a, r, path = parts
            try:
                added += int(a)
                removed += int(r)
            except ValueError:
                # binary diff: count as 1 line each
                added += 1
                removed += 1
            files.append(path)
        patch_lines = added + removed

        for path in files:
            for risky in self._beta.risk_paths:
                norm = risky.rstrip("/")
                is_directory = risky.endswith("/")
                if path == norm or (is_directory and path.startswith(risky)):
                    return PolicyVerdict(
                        ok=False,
                        patch_lines=patch_lines,
                        touched_files=files,
                        reason=f"touched forbidden path: {path}",
                    )

        if patch_lines > self._beta.patch_max_lines:
            return PolicyVerdict(
                ok=False,
                patch_lines=patch_lines,
                touched_files=files,
                reason=(
                    f"patch too large: {patch_lines} > "
                    f"{self._beta.patch_max_lines} lines"
                ),
            )

        return PolicyVerdict(ok=True, patch_lines=patch_lines, touched_files=files)
