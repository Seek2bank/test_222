"""UCB1 bandit over BRANCH skills.

Opt-in via ``BetaConfig.use_bandit``. Persists per-skill statistics (pulls,
successes, mean speedup) to a JSON file so learning carries across tasks.
Concurrent writes from parallel
experiments are serialised with ``fcntl.flock``; reads tolerate a missing
or corrupt file by falling back to zero-stats.

Reward model
------------
Each BRANCH that produced a usable FAST eval feeds one update:
* ``tests_pass=False`` or ``speedup is None``  → reward 0.0
* ``tests_pass=True`` and speedup s            → reward max(0.0, s - 1.0)

Pruned-before-eval branches (env tampering, policy, SDK error) do NOT
update stats — they say nothing about the skill's merit.
"""

from __future__ import annotations

import fcntl
import json
import math
import random
import time
import typing
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SkillStat:
    pulls: int = 0
    successes: int = 0  # branches with tests_pass AND speedup > 1
    reward_sum: float = 0.0  # cumulative max(0, speedup - 1)
    last_update_ts: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


@dataclass
class _Store:
    skills: dict[str, SkillStat] = field(default_factory=dict)

    def to_json(self) -> dict[str, typing.Any]:
        return {
            "skills": {name: asdict(stat) for name, stat in self.skills.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, typing.Any]) -> "_Store":
        skills_raw = data.get("skills", {}) if isinstance(data, dict) else {}
        skills: dict[str, SkillStat] = {}
        if isinstance(skills_raw, dict):
            for name, raw in skills_raw.items():
                if not isinstance(raw, dict):
                    continue
                skills[name] = SkillStat(
                    pulls=int(raw.get("pulls", 0) or 0),
                    successes=int(raw.get("successes", 0) or 0),
                    reward_sum=float(raw.get("reward_sum", 0.0) or 0.0),
                    last_update_ts=float(raw.get("last_update_ts", 0.0) or 0.0),
                )
        return cls(skills=skills)


@contextmanager
def _locked(path: Path, mode: str):
    """Open ``path`` with an exclusive flock held for the duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, mode)
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


class SkillBandit:
    """UCB1 bandit with optional JSON-file persistence.

    ``stats_path=None`` keeps everything in-memory (useful for tests and
    for single-task runs that should not pollute the global state file).
    """

    def __init__(
        self,
        stats_path: Path | None,
        exploration_c: float = 1.4,
        rng: random.Random | None = None,
    ):
        self._stats_path = Path(stats_path) if stats_path else None
        self._c = exploration_c
        self._rng = rng or random.Random()
        self._store = self._load()

    # ----- persistence -----

    def _load(self) -> _Store:
        if self._stats_path is None or not self._stats_path.exists():
            return _Store()
        try:
            with _locked(self._stats_path, "r") as f:
                data = json.load(f)
            return _Store.from_json(data)
        except (OSError, json.JSONDecodeError):
            return _Store()

    def _persist(self) -> None:
        if self._stats_path is None:
            return
        tmp = self._stats_path.with_suffix(self._stats_path.suffix + ".tmp")
        payload = json.dumps(self._store.to_json(), indent=2, sort_keys=True)
        try:
            with _locked(self._stats_path, "a+") as _:
                tmp.write_text(payload)
                tmp.replace(self._stats_path)
        except OSError:
            # Persistence is best-effort; never raise into the controller.
            pass

    # ----- core API -----

    def stat(self, skill: str) -> SkillStat:
        return self._store.skills.get(skill, SkillStat())

    def score(self, skill: str, total_pulls: int) -> float:
        """UCB1 score. Unseen skills get +inf so they are tried first."""
        s = self.stat(skill)
        if s.pulls == 0:
            return float("inf")
        if total_pulls <= 0:
            return s.mean_reward
        exploration = self._c * math.sqrt(math.log(max(1, total_pulls)) / s.pulls)
        return s.mean_reward + exploration

    def select_k(self, k: int, candidates: list[str]) -> list[str]:
        """Pick ``k`` distinct skills by descending UCB1, shuffling ties."""
        if k <= 0 or not candidates:
            return []
        total = sum(self.stat(name).pulls for name in candidates)
        scored = [
            (self.score(name, total), self._rng.random(), name) for name in candidates
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [name for _, _, name in scored[:k]]

    def tiebreak_order(self, candidates: list[str]) -> list[str]:
        """Stable order for breaking router ties: higher mean_reward first."""
        return sorted(
            candidates,
            key=lambda name: (
                -self.stat(name).mean_reward,
                -self.stat(name).pulls,
                name,
            ),
        )

    def update(
        self,
        skill: str,
        speedup: float | None,
        tests_pass: bool | None,
    ) -> None:
        """Record one BRANCH outcome. Persists after each update."""
        stat = self._store.skills.setdefault(skill, SkillStat())
        stat.pulls += 1
        stat.last_update_ts = time.time()
        if tests_pass and speedup is not None and speedup > 1.0:
            stat.successes += 1
            stat.reward_sum += speedup - 1.0
        self._persist()

    def snapshot(self) -> dict[str, dict[str, typing.Any]]:
        """Read-only view for logging / trajectory."""
        return {name: asdict(stat) for name, stat in self._store.skills.items()}
