"""Harness-independent MoKOpt search controller."""

from __future__ import annotations

import json
import logging
import time
import typing
from pathlib import Path

from mokopt.bandit import SkillBandit
from mokopt.branch import ALL_SKILLS, NO_SKILL, BranchPlan, BranchPlanner
from mokopt.evals import EvalResult, Evaluator
from mokopt.guards import EnvGuard, ReadOnlyGuard, make_default_guard
from mokopt.policy import PolicyChecker
from mokopt.prompts import PromptProfile
from mokopt.repo import GitRepo
from mokopt.state import (
    ActionKind,
    BetaConfig,
    BranchRecord,
    Budget,
    MoKOptState,
    Outcome,
    TaskSpec,
)
from mokopt.trajectory import TrajectoryWriter

if typing.TYPE_CHECKING:
    from mokopt.backends.base import BackendSession
    from mokopt.executor.base import Executor

logger = logging.getLogger(__name__)


class MoKOptController:
    """Run profile, branch, prune, and evaluation rounds for one task."""

    ARTIFACT_DIR = "/tmp/mokopt"

    def __init__(
        self,
        *,
        executor: "Executor",
        beta: BetaConfig,
        budget: Budget,
        benchmark_cmd: str,
        test_cmd: str | None,
        metric_regex: str | None,
        prepare_cmd: str | None,
        skills_dir: Path,
        trajectory: TrajectoryWriter,
        logging_dir: Path | None = None,
        skills_mount: str = "/skills",
    ):
        self._ex = executor
        self._beta = beta
        self._budget = budget
        self._repo = GitRepo(executor)
        self._eval = Evaluator(
            executor,
            beta,
            benchmark_cmd,
            test_cmd,
            metric_regex,
            prepare_cmd,
        )
        self._policy = PolicyChecker(beta, self._repo)
        self._profile = PromptProfile(benchmark_cmd, test_cmd)
        self._trajectory = trajectory
        self._logging_dir = logging_dir
        bandit = None
        if beta.use_bandit:
            stats_path = (
                Path(beta.bandit_stats_path) if beta.bandit_stats_path else None
            )
            bandit = SkillBandit(stats_path)
        self._bandit = bandit
        self._planner = BranchPlanner(
            beta,
            skills_dir,
            bandit=bandit,
            skills_mount=skills_mount,
        )
        self._planner.set_perf_fragments(self._profile)

    async def run(
        self,
        sdk_session: "BackendSession",
        task: TaskSpec,
        *,
        session_factory: typing.Callable[[], typing.Awaitable["BackendSession"]]
        | None = None,
    ) -> MoKOptState:
        state = MoKOptState()
        baseline_hash = self._repo.ensure_repo()
        if baseline_hash is None:
            raise RuntimeError("could not create the git baseline")
        state.baseline_hash = baseline_hash

        baseline_seconds = self._eval.measure_baseline(
            self._beta.eval_full_repeats,
            self._beta.full_test_timeout_s,
        )
        if baseline_seconds is None:
            raise RuntimeError("baseline benchmark command produced no metric")
        state.baseline_seconds = baseline_seconds
        self._trajectory.log(
            ActionKind.PROBE_BASELINE.value,
            Outcome(notes={"seconds": baseline_seconds}),
            {"baseline_hash": baseline_hash},
        )

        if self._budget.can_profile():
            started = time.time()
            prompt = self._profile.perf_probe_profile_preamble(
                baseline_hash=baseline_hash
            ) + self._profile.perf_probe_profile_body(baseline_hash=baseline_hash)
            try:
                await sdk_session.send(prompt, label="probe_profile")
            finally:
                self._repo.restore_to(baseline_hash)
            elapsed = time.time() - started
            self._budget.charge(ActionKind.PROBE_PROFILE, elapsed)
            self._trajectory.log(
                ActionKind.PROBE_PROFILE.value,
                Outcome(cost_s=elapsed),
            )

        env_guard = EnvGuard(self._beta.env_check_cmd)
        env_guard.capture_baseline(self._ex)
        readonly: ReadOnlyGuard | None = None
        if self._beta.readonly_guard:
            readonly = make_default_guard(self._ex.repo_path, self._beta.risk_paths)
            readonly.lock(self._ex)

        running_best_hash = baseline_hash
        try:
            for iteration in range(self._beta.max_rounds):
                if not self._budget.can_round():
                    break
                plans = await self._planner.plan(
                    state,
                    iteration,
                    task.instruction,
                    sdk_session=sdk_session,
                )
                survivors: list[tuple[BranchRecord, str]] = []
                for plan in plans:
                    if not self._repo.restore_to(running_best_hash):
                        raise RuntimeError(
                            f"could not restore parent commit {running_best_hash}"
                        )
                    record = BranchRecord(
                        branch_id=plan.branch_id,
                        skill=plan.skill,
                        iteration=iteration,
                        parent_hash=running_best_hash,
                    )
                    state.history.append(record)
                    self._trajectory.log(
                        ActionKind.BRANCH.value,
                        extra={
                            "branch_id": plan.branch_id,
                            "skill": plan.skill,
                            "isolation": self._beta.isolation,
                        },
                    )
                    branch_started = time.time()
                    try:
                        await self._run_branch(
                            plan,
                            sdk_session,
                            session_factory,
                        )
                    except Exception as exc:
                        self._budget.charge(
                            ActionKind.BRANCH, time.time() - branch_started
                        )
                        self._prune(record, f"agent error: {exc}")
                        continue
                    self._budget.charge(
                        ActionKind.BRANCH, time.time() - branch_started
                    )

                    candidate_hash = self._repo.commit_candidate(plan.branch_id)
                    if candidate_hash is None:
                        self._prune(record, "could not commit candidate")
                        continue
                    record.snapshot_hash = candidate_hash
                    record.patch_diff = self._repo.diff(
                        running_best_hash, candidate_hash
                    )
                    self._save_patch(record)
                    if not record.patch_diff.strip():
                        self._prune(record, "agent produced no source changes")
                        continue

                    env_diff = env_guard.diff_against_baseline(self._ex)
                    if not env_diff.is_empty:
                        self._prune(
                            record, f"environment changed: {env_diff.summary()}"
                        )
                        continue

                    verdict = self._policy.check(running_best_hash, candidate_hash)
                    record.patch_lines = verdict.patch_lines
                    record.touched_files = verdict.touched_files
                    if not verdict.ok:
                        self._prune(record, verdict.reason or "policy rejected patch")
                        continue
                    if not self._budget.can_fast():
                        self._prune(record, "FAST evaluation budget exhausted")
                        continue

                    result = self._eval.fast(baseline_seconds, plan.branch_id)
                    self._budget.charge(ActionKind.EVAL_FAST, result.outcome.cost_s)
                    record.tests_pass = result.outcome.tests_pass
                    record.fast_speedup = result.outcome.speedup
                    self._trajectory.log(
                        ActionKind.EVAL_FAST.value,
                        result.outcome,
                        {
                            "branch_id": plan.branch_id,
                            "branch_cost_s": time.time() - branch_started,
                        },
                    )
                    if self._bandit is not None and record.skill not in {
                        ALL_SKILLS,
                        NO_SKILL,
                    }:
                        self._bandit.update(
                            record.skill,
                            result.outcome.speedup,
                            result.outcome.tests_pass,
                        )
                    if not result.outcome.tests_pass:
                        self._prune(record, "correctness check failed")
                    elif result.outcome.speedup is None:
                        self._prune(record, "benchmark produced no speedup")
                    elif result.outcome.speedup < self._beta.sig_threshold:
                        self._prune(
                            record,
                            f"speedup {result.outcome.speedup:.4f} is below "
                            f"{self._beta.sig_threshold:.4f}",
                        )
                    else:
                        survivors.append((record, candidate_hash))

                improved = await self._promote(
                    survivors,
                    baseline_seconds,
                    state,
                )
                if improved and state.best_branch is not None:
                    running_best_hash = (
                        state.best_branch.snapshot_hash or running_best_hash
                    )
                    state.rounds_without_improvement = 0
                else:
                    state.rounds_without_improvement += 1
                self._budget.end_round()
                if (
                    self._beta.stop_no_improvement > 0
                    and state.rounds_without_improvement
                    >= self._beta.stop_no_improvement
                ):
                    break
        finally:
            if readonly is not None:
                readonly.unlock(self._ex)

        final_hash = (
            state.best_branch.snapshot_hash
            if state.best_branch and state.best_branch.snapshot_hash
            else baseline_hash
        )
        if not self._repo.restore_to(final_hash):
            raise RuntimeError(f"could not materialize final commit {final_hash}")
        self._write_summary(state)
        self._trajectory.log(
            ActionKind.STOP.value,
            extra={
                "best_branch": (
                    state.best_branch.branch_id if state.best_branch else None
                ),
                "final_hash": final_hash,
            },
        )
        return state

    async def _run_branch(
        self,
        plan: BranchPlan,
        shared_session: "BackendSession",
        session_factory: typing.Callable[[], typing.Awaitable["BackendSession"]] | None,
    ) -> None:
        if self._beta.isolation != "phased":
            session = await session_factory() if session_factory else shared_session
            try:
                await session.send(plan.prompt, label=plan.branch_id)
            finally:
                if session is not shared_session:
                    await session.close()
            return

        artifact_dir = f"{self.ARTIFACT_DIR}/{plan.branch_id}"
        self._ex.run(f"rm -rf {artifact_dir} && mkdir -p {artifact_dir}")
        phases = ("profile", "localize", "optimize")
        for phase in phases:
            session = await session_factory() if session_factory else shared_session
            prompt = self._phase_prompt(plan, phase, artifact_dir)
            try:
                await session.send(prompt, label=f"{plan.branch_id}/{phase}")
            finally:
                if session is not shared_session:
                    await session.close()

    def _phase_prompt(self, plan: BranchPlan, phase: str, artifact_dir: str) -> str:
        original = plan.prompt.split("--- ORIGINAL TASK INSTRUCTIONS ---\n")[-1]
        profile_path = f"{artifact_dir}/profile.md"
        localization_path = f"{artifact_dir}/localization.md"
        if phase == "profile":
            directive = self._profile.perf_profile_phase_directive(
                artifact_path=artifact_dir
            )
        elif phase == "localize":
            directive = (
                "You are in the LOCALIZE phase. Do not edit source code. "
                f"Read {profile_path}, rank the exact bottlenecks, and write file, "
                f"symbol, line range, root cause, and strategy to {localization_path}."
            )
        else:
            skill = ""
            if plan.skill == ALL_SKILLS:
                skill = "Choose the most relevant mounted skill card under /skills."
            elif plan.skill != NO_SKILL:
                skill = f"Read and apply the {plan.skill} skill at {plan.skill_file}."
            directive = (
                "You are in the OPTIMIZE phase. "
                f"Read {profile_path} and {localization_path}. {skill} "
                "Implement one focused source-code optimization now. "
                "Do not modify tests or benchmarks.\n\n"
                + self._profile.perf_policy_block(
                    patch_max_lines=self._beta.patch_max_lines
                )
            )
        return (
            f"=== MOKOPT {plan.branch_id} / {phase.upper()} ===\n\n"
            f"{directive}\n\n--- ORIGINAL TASK INSTRUCTIONS ---\n{original}"
        )

    async def _promote(
        self,
        survivors: list[tuple[BranchRecord, str]],
        baseline_seconds: float,
        state: MoKOptState,
    ) -> bool:
        if not survivors or not self._budget.can_full():
            return False
        ordered = sorted(
            survivors,
            key=lambda item: item[0].fast_speedup or 0.0,
            reverse=True,
        )
        leader = ordered[0][0].fast_speedup or 0.0
        contenders = [
            item
            for item in ordered
            if item[0].fast_speedup
            and leader / (item[0].fast_speedup or leader) <= self._beta.full_tie_ratio
        ][: max(1, self._beta.full_top_n)]

        results: list[tuple[BranchRecord, str, EvalResult]] = []
        for record, candidate_hash in contenders:
            if not self._budget.can_full():
                break
            if not self._repo.restore_to(candidate_hash):
                continue
            result = self._eval.full(baseline_seconds, record.branch_id)
            self._budget.charge(ActionKind.EVAL_FULL, result.outcome.cost_s)
            record.full_speedup = result.outcome.speedup
            self._trajectory.log(
                ActionKind.EVAL_FULL.value,
                result.outcome,
                {"branch_id": record.branch_id},
            )
            if result.outcome.tests_pass and result.outcome.speedup is not None:
                results.append((record, candidate_hash, result))
        if not results:
            return False
        best_record, best_hash, best_result = max(
            results, key=lambda item: item[2].outcome.speedup or 0.0
        )
        prior = (
            state.best_branch.full_speedup if state.best_branch is not None else 1.0
        ) or 1.0
        threshold = max(self._beta.full_promotion_threshold, prior)
        if (best_result.outcome.speedup or 0.0) < threshold:
            return False
        state.best_branch = best_record
        if not self._repo.restore_to(best_hash):
            raise RuntimeError(
                f"could not restore promoted branch {best_record.branch_id}"
            )
        return True

    def _prune(self, record: BranchRecord, reason: str) -> None:
        record.pruned = True
        record.prune_reason = reason
        self._trajectory.log(
            ActionKind.PRUNE.value,
            extra={"branch_id": record.branch_id, "reason": reason},
        )

    def _save_patch(self, record: BranchRecord) -> None:
        if record.patch_diff:
            self._ex.write_file(
                f"{self.ARTIFACT_DIR}/patches/{record.branch_id}.diff",
                record.patch_diff,
            )
        if not self._logging_dir or not record.patch_diff:
            return
        patch_dir = self._logging_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        (patch_dir / f"{record.branch_id}.diff").write_text(
            record.patch_diff, encoding="utf-8"
        )

    def _write_summary(self, state: MoKOptState) -> None:
        if self._logging_dir is None:
            return
        path = self._logging_dir / "branches.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in state.history:
                stream.write(
                    json.dumps(
                        {
                            "branch_id": record.branch_id,
                            "skill": record.skill,
                            "iteration": record.iteration,
                            "pruned": record.pruned,
                            "prune_reason": record.prune_reason,
                            "fast_speedup": record.fast_speedup,
                            "full_speedup": record.full_speedup,
                            "tests_pass": record.tests_pass,
                            "patch_lines": record.patch_lines,
                            "touched_files": record.touched_files,
                            "snapshot_hash": record.snapshot_hash,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
