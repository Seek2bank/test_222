"""Anti-cheat guards for MoKOpt.

Two complementary layers sit on top of the soft :class:`mokopt.policy.PolicyChecker`:

``ReadOnlyGuard``
    *Hard* gate. Marks the benchmark / test paths immutable with
    ``chattr +i`` (falling back to ``chmod a-w`` on filesystems that don't
    support extended attributes) so even the in-container root user can't
    edit them. Temporarily released around snapshot/restore.

``EnvGuard``
    Detects the agent mutating the runtime environment (e.g. ``pip install``)
    by diffing the output of a configurable *env-check command* (default
    ``pip freeze``) before and after each branch. Opt-in via
    ``BetaConfig.env_check_cmd``.
"""

from __future__ import annotations

import logging
import shlex
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from mokopt.executor.base import Executor

logger = logging.getLogger(__name__)

# Paths that must remain writable for the controller (git lives here).
_ALWAYS_WRITABLE: frozenset[str] = frozenset({".git", ".git/"})


@dataclass
class ReadOnlyGuard:
    repo_path: str
    paths: tuple[str, ...]
    locked: set[str] = field(default_factory=set)
    supported: bool | None = None

    def __post_init__(self) -> None:
        self.paths = tuple(
            p for p in self.paths if p.strip("/").lower() not in _ALWAYS_WRITABLE
        )

    # ---- capability probe ---------------------------------------------

    def probe_support(self, ex: Executor) -> bool:
        if self.supported is not None:
            return self.supported
        cmd = (
            "command -v chattr >/dev/null 2>&1 || { echo NO_CHATTR; exit 0; }; "
            'tmp="$(mktemp /tmp/.ro_probe.XXXXXX)" || { echo NO_TMP; exit 0; }; '
            'if chattr +i "$tmp" 2>/dev/null && chattr -i "$tmp" 2>/dev/null; then '
            '  rm -f "$tmp"; echo OK; '
            'else rm -f "$tmp" 2>/dev/null; echo UNSUPPORTED; fi'
        )
        r = ex.run(cmd)
        token = (r.output.strip().splitlines() or [""])[-1]
        self.supported = token == "OK" and r.ok
        if not self.supported:
            logger.warning(
                "ReadOnlyGuard: chattr +i unavailable (probe=%r). Falling back "
                "to chmod a-w; PolicyChecker remains the authoritative gate.",
                token,
            )
        return self.supported

    # ---- lock / unlock ------------------------------------------------

    def _resolve_existing(self, ex: Executor) -> list[str]:
        candidates = [f"{self.repo_path}/{p.rstrip('/')}" for p in self.paths]
        if not candidates:
            return []
        quoted = " ".join(shlex.quote(c) for c in candidates)
        r = ex.run(f'for p in {quoted}; do [ -e "$p" ] && echo "$p"; done')
        return [ln.strip() for ln in r.output.splitlines() if ln.strip()]

    def lock(self, ex: Executor) -> int:
        supported = self.probe_support(ex)
        newly = 0
        for p in self._resolve_existing(ex):
            if p in self.locked:
                continue
            ex.run(f"chmod -R a-w {shlex.quote(p)}")
            if supported:
                r = ex.run(f"chattr -R +i {shlex.quote(p)}")
                if not r.ok:
                    logger.warning("ReadOnlyGuard: chattr +i failed on %s", p)
            self.locked.add(p)
            newly += 1
        if newly:
            logger.info(
                "ReadOnlyGuard: locked %d path(s) (%s)",
                newly,
                "immutable" if supported else "chmod-only",
            )
        return newly

    def unlock(self, ex: Executor) -> int:
        if not self.locked:
            return 0
        released = 0
        for p in list(self.locked):
            ex.run(
                f"chattr -R -i {shlex.quote(p)} 2>/dev/null; "
                f"chmod -R u+w {shlex.quote(p)} 2>/dev/null || true"
            )
            self.locked.discard(p)
            released += 1
        return released

    @contextmanager
    def unlocked(self, ex: Executor) -> Iterator[None]:
        prev = sorted(self.locked)
        self.unlock(ex)
        try:
            yield
        finally:
            if prev:
                self.lock(ex)


def make_default_guard(repo_path: str, risk_paths: Iterable[str]) -> ReadOnlyGuard:
    return ReadOnlyGuard(repo_path=repo_path, paths=tuple(risk_paths))


# ---------------------------------------------------------------------------
# Env tampering guard
# ---------------------------------------------------------------------------


@dataclass
class EnvDiff:
    added: list[str]
    removed: list[str]
    changed: list[tuple[str, str, str]]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def summary(self, limit: int = 8) -> str:
        bits: list[str] = []
        if self.added:
            head = ", ".join(self.added[:limit])
            bits.append(f"added [{head}{'' if len(self.added) <= limit else ' ...'}]")
        if self.removed:
            head = ", ".join(self.removed[:limit])
            bits.append(
                f"removed [{head}{'' if len(self.removed) <= limit else ' ...'}]"
            )
        if self.changed:
            head = ", ".join(f"{n}: {a}->{b}" for n, a, b in self.changed[:limit])
            bits.append(
                f"changed [{head}{'' if len(self.changed) <= limit else ' ...'}]"
            )
        return "; ".join(bits) or "no changes"


class EnvGuard:
    """Snapshot + diff the output of an env-check command (default pip freeze)."""

    def __init__(self, env_check_cmd: str | None):
        self._cmd = env_check_cmd or None
        self._baseline: dict[str, str] | None = None

    @property
    def enabled(self) -> bool:
        return self._cmd is not None

    @property
    def baseline_size(self) -> int:
        return len(self._baseline) if self._baseline else 0

    def capture_baseline(self, ex: Executor) -> int:
        if not self.enabled:
            return 0
        self._baseline = self._snapshot(ex)
        return len(self._baseline)

    def diff_against_baseline(self, ex: Executor) -> EnvDiff:
        if not self.enabled or self._baseline is None:
            return EnvDiff(added=[], removed=[], changed=[])
        current = self._snapshot(ex)
        base = self._baseline
        added = sorted(set(current) - set(base))
        removed = sorted(set(base) - set(current))
        changed = sorted(
            (n, base[n], current[n])
            for n in (set(current) & set(base))
            if base[n] != current[n]
        )
        return EnvDiff(added=added, removed=removed, changed=changed)

    def _snapshot(self, ex: Executor) -> dict[str, str]:
        if not self._cmd:
            return {}
        res = ex.run(self._cmd)
        if not res.ok:
            return {}
        out: dict[str, str] = {}
        for line in res.output.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if " @ " in line:
                name = line.split("==")[0].split(" @ ")[0].strip()
                if name:
                    out[name.lower()] = line
                continue
            if "==" in line:
                name, _, ver = line.partition("==")
                out[name.strip().lower()] = ver.strip()
            else:
                out[line.lower()] = ""
        return out
