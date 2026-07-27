"""Dataclasses for MoKOpt: TaskSpec, State, Action, Outcome, Budget, BetaConfig.

The controller's behaviour is driven entirely by these objects so the loop
itself stays free of magic numbers. ``BetaConfig`` exposes every knob; defaults
live in ``default_beta.json`` next to this file.

This module is harness-independent — it has no Docker / executor / backend
imports, so it can be reused and unit-tested in isolation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------


class ActionKind(str, Enum):
    PROBE_BASELINE = "probe_baseline"
    PROBE_PROFILE = "probe_profile"
    BRANCH = "branch"
    EVAL_FAST = "eval_fast"
    EVAL_FULL = "eval_full"
    PRUNE = "prune"
    STOP = "stop"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Structured task input. MoKOpt only relies on ``repo_path`` +
    ``instruction`` (the natural-language task description).
    """

    task_id: str
    repo_path: str = "/workspace/repo"
    instruction: str = ""


# ---------------------------------------------------------------------------
# State / Outcome
# ---------------------------------------------------------------------------


@dataclass
class BranchRecord:
    """Per-branch bookkeeping kept inside ``MoKOptState``."""

    branch_id: str
    skill: str
    iteration: int
    parent_hash: str | None = None
    snapshot_hash: str | None = None
    fast_speedup: float | None = None
    full_speedup: float | None = None
    tests_pass: bool | None = None
    pruned: bool = False
    prune_reason: str | None = None
    patch_lines: int = 0
    touched_files: list[str] = field(default_factory=list)
    patch_diff: str = ""


@dataclass
class MoKOptState:
    """Controller-visible state. Only depends on observable evidence."""

    baseline_hash: str | None = None
    baseline_seconds: float | None = None
    history: list[BranchRecord] = field(default_factory=list)
    best_branch: BranchRecord | None = None
    rounds_without_improvement: int = 0


@dataclass
class Outcome:
    """What an action returns. ``cost`` is wall-time in seconds when known."""

    speedup: float | None = None
    tests_pass: bool | None = None
    cost_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Mutable budget tracked per task."""

    rounds: int = 3
    fast_evals: int = 24
    full_evals: int = 6
    profile_runs: int = 2
    wall_time_s: float = 7200.0
    spent_wall_s: float = 0.0

    def can_round(self) -> bool:
        return self.rounds > 0 and self.spent_wall_s < self.wall_time_s

    def can_fast(self) -> bool:
        return self.fast_evals > 0 and self.spent_wall_s < self.wall_time_s

    def can_full(self) -> bool:
        return self.full_evals > 0 and self.spent_wall_s < self.wall_time_s

    def can_profile(self) -> bool:
        return self.profile_runs > 0 and self.spent_wall_s < self.wall_time_s

    def charge(self, kind: ActionKind, dt_s: float) -> None:
        self.spent_wall_s += max(0.0, dt_s)
        if kind is ActionKind.EVAL_FAST:
            self.fast_evals = max(0, self.fast_evals - 1)
        elif kind is ActionKind.EVAL_FULL:
            self.full_evals = max(0, self.full_evals - 1)
        elif kind is ActionKind.PROBE_PROFILE:
            self.profile_runs = max(0, self.profile_runs - 1)

    def end_round(self) -> None:
        self.rounds = max(0, self.rounds - 1)


# ---------------------------------------------------------------------------
# β-Config (controller knobs)
# ---------------------------------------------------------------------------


@dataclass
class BetaConfig:
    """Every knob the controller respects. Loaded from ``default_beta.json``
    and overridable via ``overlay()``.
    """

    branch_k: int = 3
    max_rounds: int = 3
    sig_threshold: float = 1.02
    full_promotion_threshold: float = 1.03
    patch_max_lines: int = 400
    risk_paths: list[str] = field(
        default_factory=lambda: [
            "tests/",
            "test/",
            "benchmarks/",
            "setup.py",
            "pyproject.toml",
            ".git/",
        ]
    )
    eval_fast_repeats: int = 3
    eval_full_repeats: int = 6
    stop_no_improvement: int = 2
    fast_test_timeout_s: int = 180
    full_test_timeout_s: int = 600
    # Top-N FULL evaluation: when multiple FAST-survivors have similar
    # speedups, FAST noise can pick the wrong "best" candidate. Instead of
    # FULL-ing only the top-1 by fast_speedup, FULL up to ``full_top_n``
    # contenders whose fast_speedup is within ``full_tie_ratio`` of the
    # leader (i.e. leader/cand <= ratio), then promote the best by
    # full_speedup. Setting full_top_n=1 preserves legacy behaviour.
    full_top_n: int = 2
    full_tie_ratio: float = 1.03
    # Branch isolation mode: "single" = one session/prompt per branch (default),
    # "phased" = 3 sessions per branch (profile -> localize -> optimize).
    isolation: str = "single"
    # Physically prevent the agent from modifying benchmark/test paths via
    # chattr +i (with chmod a-w fallback). PolicyChecker remains the
    # second-line soft gate.
    readonly_guard: bool = True
    # Ralph loop: inject a summary of previous rounds' branch outcomes into
    # each branch prompt so the agent can avoid repeating failed approaches
    # and build on what worked. Disabled by default (opt-in).
    ralph_history: bool = True
    # Inject-all-skills mode: instead of fanning out into ``branch_k``
    # per-skill branches, emit a SINGLE candidate per round whose prompt
    # references every skill card at once.
    inject_all_skills: bool = False
    # Dynamic routing: use the LLM-as-router to select top-k skills based on
    # profile context instead of round-robin rotation.
    dynamic_routing: bool = False
    # UCB1 bandit: maintain per-skill success/failure stats across branches.
    use_bandit: bool = False
    bandit_stats_path: str | None = None
    # Disable skill injection entirely: branches are still emitted but the
    # prompt carries NO skill block.
    disable_skills: bool = False
    # Optional anti-cheat: snapshot the environment package set and reject any
    # branch that mutates it. Off by default (set ``env_check_cmd`` to enable).
    env_check_cmd: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "BetaConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        # Ignore unknown keys to keep the config forward-compatible.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def overlay(self, overrides: dict[str, Any] | None) -> "BetaConfig":
        if not overrides:
            return self
        merged = asdict(self)
        for k, v in overrides.items():
            if k in merged and v is not None:
                merged[k] = v
        return BetaConfig(**merged)
