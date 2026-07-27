"""BRANCH(K): skill-routed candidate generator.

For each round the controller asks ``BranchPlanner.plan(state, k)`` for K
``BranchPlan`` instances. Each plan binds an ``Optimization Skill`` (a short
markdown card mounted at ``/skills/<name>.md`` in the container) to a candidate
prompt.

Skill selection strategy (controlled by ``BetaConfig`` flags):
  1. ``inject_all_skills`` — single branch per round, all skills referenced.
  2. ``dynamic_routing``   — LLM-as-router selects top-k skills from profile
                             context (falls back to bandit or round-robin).
  3. ``use_bandit``        — UCB1 bandit selects top-k skills (falls back to
                             round-robin).
  4. (default)             — iteration-aware round-robin over DEFAULT_ROTATION.
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from pathlib import Path

from mokopt.state import (
    BetaConfig,
    MoKOptState,
)

if typing.TYPE_CHECKING:
    from mokopt.backends.base import (
        BackendSession,
    )
    from mokopt.bandit import SkillBandit

logger = logging.getLogger(__name__)

SKILL_FILES = {
    # Existing (cover the Replace/Reorder branches of the Park-Sujin
    # OSDI'25 taxonomy).
    "overhead_reduction": "skills/overhead_reduction.md",
    "vectorization": "skills/vectorization.md",
    "caching": "skills/caching.md",
    "algorithmic": "skills/algorithmic.md",
    "structural": "skills/structural.md",
    "redundant_work": "skills/redundant_work.md",
    "fast_path": "skills/fast_path.md",
    "precomputation": "skills/precomputation.md",
    "deferral": "skills/deferral.md",
}

# Ordered so that the empirically most frequent / highest-leverage skills
# land first, giving a mix of Park-Sujin principles (Remove / Replace /
# Reorder) across the first few branches when ``branch_k`` < len(rotation).
DEFAULT_ROTATION = [
    "redundant_work",  # Remove   — task removal
    "fast_path",  # Replace  — contextualization
    "vectorization",  # Reorder  — HW specialization
    "caching",  # Replace  — caching
    "algorithmic",  # Reorder  — layering
    "precomputation",  # Replace  — precomputing
    "overhead_reduction",  # Replace  — contextualization (dispatch)
    "structural",  # Remove   — batching (I/O)
    "deferral",  # Remove   — deferring (lazy)
]

# In-container mount point for the skill cards (see MoKOptRunner._push_skills).
CONTAINER_SKILLS_DIR = "/skills"

# Sentinel ``skill`` value used by inject-all-skills mode (a single candidate
# whose prompt references every card rather than one round-robin skill).
ALL_SKILLS = "all"

# Sentinel ``skill`` value used by ``BetaConfig.disable_skills`` mode: the
# branch prompt carries NO skill block at all. Branching/isolation/eval
# behave exactly as in the default round-robin path; only the skill
# reference is suppressed.
NO_SKILL = "none"


def all_skills_reference_block(
    skills_mount: str = CONTAINER_SKILLS_DIR,
) -> str:
    """Render the prompt block that lists every optimization skill card.

    Used by inject-all-skills mode (``BetaConfig.inject_all_skills``) in place
    of the single-skill block, for both the single-session prompt and the
    phased OPTIMIZE prompt.
    """
    lines = [
        "OPTIMIZATION SKILLS — all skill cards are mounted under "
        f"{skills_mount}/ in the environment. Pick whichever best fits "
        "the hot spot you find and make ONE focused change:",
    ]
    for name in DEFAULT_ROTATION:
        card = Path(SKILL_FILES[name]).name
        lines.append(f"  * {name} — {skills_mount}/{card}")
    lines.append(
        "  * Read the relevant card(s) before editing if you are unsure "
        "which idiom applies.\n"
    )
    return "\n".join(lines) + "\n"


@dataclass
class BranchPlan:
    branch_id: str
    iteration: int
    skill: str
    skill_file: str  # in-container path
    prompt: str


class BranchPlanner:
    """Picks K skills + builds the BRANCH prompt for each candidate.

    Skill selection follows a fallback chain:
      dynamic_routing (LLM router) → use_bandit (UCB1) → round-robin.
    Each level is opt-in via BetaConfig flags; all three are off by default,
    which preserves the original round-robin behaviour.
    """

    def __init__(
        self,
        beta: BetaConfig,
        skills_dir: Path,
        bandit: "SkillBandit | None" = None,
        skills_mount: str = CONTAINER_SKILLS_DIR,
    ):
        self._beta = beta
        self._skills_dir = skills_dir
        self._bandit = bandit
        self._skills_mount = skills_mount
        self._env_name: str | None = None
        # Perf-aware prompt fragments. Set by the controller via
        # set_perf_fragments() (which reads them off a PromptProfile).
        # Default is empty; the PromptProfile supplies the real text.
        self._perf_reference_text: str = ""
        # Callables: (env_name=...) -> str. Set by the controller.
        self._perf_env_block_fn = None
        self._perf_policy_block_fn = None  # (patch_max_lines=..., env_name=...) -> str
        self._perf_self_profile_block_fn = None  # (env_name=...) -> str
        self._perf_branch_task_block_fn = None  # () -> str

    def set_env_name(self, env_name: str | None) -> None:
        self._env_name = env_name or None

    def set_perf_reference_text(self, text: str) -> None:
        if text:
            self._perf_reference_text = text

    def set_perf_fragments(self, profile) -> None:
        """Wire prompt fragments from a ``PromptProfile`` into the planner.

        Called by ``MoKOptController`` after constructing the planner. Reads
        each ``perf_*`` method off the profile so the benchmark / test command
        context is threaded into every prompt.
        """
        if profile is None:
            return
        self.set_perf_reference_text(profile.PERF_REFERENCE_TEXT)
        self._perf_env_block_fn = profile.perf_env_block
        self._perf_policy_block_fn = profile.perf_policy_block
        self._perf_self_profile_block_fn = profile.perf_self_profile_block
        self._perf_branch_task_block_fn = profile.perf_branch_task_block

    async def plan(
        self,
        state: MoKOptState,
        iteration: int,
        instruction: str,
        sdk_session: "BackendSession | None" = None,
    ) -> list[BranchPlan]:
        if self._beta.disable_skills:
            k = self._beta.branch_k
            plans: list[BranchPlan] = []
            for i in range(k):
                branch_id = f"r{iteration}-b{i}"
                plans.append(
                    BranchPlan(
                        branch_id=branch_id,
                        iteration=iteration,
                        skill=NO_SKILL,
                        skill_file=self._skills_mount,
                        prompt=self._build_prompt(
                            NO_SKILL, instruction, state, branch_id, iteration
                        ),
                    )
                )
            return plans

        if self._beta.inject_all_skills:
            branch_id = f"r{iteration}-all"
            return [
                BranchPlan(
                    branch_id=branch_id,
                    iteration=iteration,
                    skill=ALL_SKILLS,
                    skill_file=self._skills_mount,
                    prompt=self._build_prompt(
                        ALL_SKILLS, instruction, state, branch_id, iteration
                    ),
                )
            ]

        k = self._beta.branch_k
        chosen = await self._select_skills(k, iteration, instruction, sdk_session)

        plans: list[BranchPlan] = []
        for i, skill in enumerate(chosen):
            branch_id = f"r{iteration}-b{i}"
            plans.append(
                BranchPlan(
                    branch_id=branch_id,
                    iteration=iteration,
                    skill=skill,
                    skill_file=(
                        f"{self._skills_mount}/{Path(SKILL_FILES[skill]).name}"
                    ),
                    prompt=self._build_prompt(
                        skill, instruction, state, branch_id, iteration
                    ),
                )
            )
        return plans

    # ----- skill selection fallback chain -----

    async def _select_skills(
        self,
        k: int,
        iteration: int,
        instruction: str,
        sdk_session: "BackendSession | None",
    ) -> list[str]:
        """Select k skills via the fallback chain: router → bandit → round-robin."""
        candidates = list(DEFAULT_ROTATION)

        # 1. Try LLM router
        if self._beta.dynamic_routing and sdk_session is not None:
            routed = await self._try_route(k, instruction, sdk_session)
            if routed is not None and len(routed) >= k:
                logger.info("Skill selection: router returned %s", routed[:k])
                return routed[:k]
            # Router returned partial results — pad with bandit or round-robin.
            if routed:
                logger.info(
                    "Skill selection: router returned %d/%d, padding",
                    len(routed),
                    k,
                )
                return self._pad_skills(routed, k, candidates, iteration)

        # 2. Try bandit
        if self._beta.use_bandit and self._bandit is not None:
            selected = self._bandit.select_k(k, candidates)
            if selected:
                logger.info("Skill selection: bandit returned %s", selected)
                return selected

        # 3. Fallback: round-robin
        return self._round_robin(k, iteration, candidates)

    async def _try_route(
        self,
        k: int,
        instruction: str,
        sdk_session: "BackendSession",
    ) -> list[str] | None:
        """Call the LLM router; return None on any failure."""
        try:
            from mokopt.router import (
                route_skills,
            )

            return await route_skills(
                sdk_session=sdk_session,
                instruction=instruction,
                k=k,
                candidates=list(DEFAULT_ROTATION),
            )
        except Exception as exc:
            logger.warning("Skill router failed: %s", exc)
            return None

    def _pad_skills(
        self,
        partial: list[str],
        k: int,
        candidates: list[str],
        iteration: int,
    ) -> list[str]:
        """Pad a partial router result to length k using bandit or round-robin."""
        seen = set(partial)
        remaining = [s for s in candidates if s not in seen]
        if self._beta.use_bandit and self._bandit is not None:
            extras = self._bandit.select_k(k - len(partial), remaining)
        else:
            extras = self._round_robin(k - len(partial), iteration, remaining)
        return partial + extras

    @staticmethod
    def _round_robin(k: int, iteration: int, candidates: list[str]) -> list[str]:
        n = len(candidates)
        if n == 0:
            return []
        start = (iteration * k) % n
        return [candidates[(start + j) % n] for j in range(k)]

    def _build_prompt(
        self,
        skill: str,
        instruction: str,
        state: MoKOptState,
        branch_id: str,
        iteration: int = 0,
    ) -> str:
        env_name = self._env_name or "$ENV_NAME"

        if self._perf_env_block_fn is not None:
            env_block = self._perf_env_block_fn(env_name=env_name)
        else:
            # Fallback (only hit if set_perf_fragments was not called).
            env_block = ""

        if self._perf_self_profile_block_fn is not None:
            self_profile_block = self._perf_self_profile_block_fn(env_name=env_name)
        else:
            self_profile_block = ""

        if self._perf_policy_block_fn is not None:
            risk_block = self._perf_policy_block_fn(
                patch_max_lines=self._beta.patch_max_lines,
                env_name=env_name,
            )
        else:
            risk_block = ""

        skill_block = self._skill_block(skill)

        history_block = (
            self._build_history_block(state, iteration)
            if self._beta.ralph_history and iteration > 0
            else ""
        )

        perf_ref = self._perf_reference_text
        task_block = (
            self._perf_branch_task_block_fn()
            if self._perf_branch_task_block_fn is not None
            else "TASK:\nApply ONE optimization following the skill(s) above.\n\n"
        )

        return (
            f"=== MOKOPT BRANCH {branch_id} — skill={skill} ===\n\n"
            + history_block
            + env_block
            + self_profile_block
            + perf_ref
            + skill_block
            + risk_block
            + task_block
            + "--- ORIGINAL TASK INSTRUCTIONS ---\n"
            + instruction
        )

    def _skill_block(self, skill: str) -> str:
        """Render the OPTIMIZATION SKILL block for a branch prompt.

        ``skill == ALL_SKILLS`` (inject-all-skills mode) lists every card and
        lets the agent choose; ``skill == NO_SKILL`` (disable-skills mode)
        suppresses the block entirely; otherwise a single skill card is
        referenced.
        """
        if skill == NO_SKILL:
            return ""
        if skill == ALL_SKILLS:
            return all_skills_reference_block(self._skills_mount)
        return (
            f"OPTIMIZATION SKILL: {skill}\n"
            f"  * The full skill card is mounted at "
            f"{self._skills_mount}/{Path(SKILL_FILES[skill]).name}.\n"
            "  * Read it before editing if you are unsure which idiom applies.\n\n"
        )

    def _build_history_block(self, state: MoKOptState, current_iteration: int) -> str:
        """Build a RALPH LOOP HISTORY block summarising all branches from
        previous rounds. Only included when ``ralph_history=True`` and at
        least one prior round has completed.
        """
        prior = [r for r in state.history if r.iteration < current_iteration]
        if not prior:
            return ""

        lines: list[str] = [
            "=== RALPH LOOP HISTORY (previous rounds) ===",
        ]
        # Group by round (iteration).
        by_round: dict[int, list] = {}
        for rec in prior:
            by_round.setdefault(rec.iteration, []).append(rec)

        for it in sorted(by_round):
            lines.append(f"Round {it}:")
            for rec in by_round[it]:
                if rec.pruned:
                    reason = rec.prune_reason or "pruned"
                    lines.append(
                        f"  • {rec.skill:<22} [{rec.branch_id}]: FAILED — {reason}"
                    )
                else:
                    speedup_str = (
                        f"{rec.fast_speedup:.3f}x"
                        if rec.fast_speedup is not None
                        else "?"
                    )
                    full_str = (
                        f", full={rec.full_speedup:.3f}x"
                        if rec.full_speedup is not None
                        else ""
                    )
                    patch_hint = (
                        f" (patch: /tmp/mokopt/patches/{rec.branch_id}.diff)"
                    )
                    lines.append(
                        f"  • {rec.skill:<22} [{rec.branch_id}]: survived — "
                        f"fast={speedup_str}{full_str}{patch_hint}"
                    )

        if state.best_branch is not None:
            b = state.best_branch
            best_speedup = b.full_speedup or b.fast_speedup
            speedup_str = f"{best_speedup:.3f}x" if best_speedup is not None else "?"
            lines.append(
                f"Current best: {b.skill} [{b.branch_id}] — speedup={speedup_str}"
            )

        lines += [
            "Use this history to:",
            "  * AVOID directions that already failed; the same root cause is "
            "unlikely to succeed.",
            "  * BUILD ON survivors; if a prior round found a hot spot, go deeper.",
            "  * Try a DIFFERENT code path or abstraction layer when prior "
            "attempts targeted the same area.",
            "==============================================\n",
        ]
        return "\n".join(lines) + "\n"
