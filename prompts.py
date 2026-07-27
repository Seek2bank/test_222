"""Prompt fragments for MoKOpt branch / probe / phase prompts.

The original AutoOpt agent baked ASV / micromamba-specific guidance into its
prompt fragments. MoKOpt is benchmark-command-agnostic: a ``PromptProfile``
carries the *configured* benchmark + test commands and renders generic,
self-contained guidance around them. The controller and ``BranchPlanner``
consume a ``PromptProfile`` exactly where the old code consumed ``self._agent``,
so the ported control flow is unchanged.

All ``perf_*`` methods accept an ``env_name`` keyword for signature symmetry
with the planner/controller call sites; MoKOpt runs commands directly (no conda
env), so the argument is accepted and ignored.
"""

from __future__ import annotations


class PromptProfile:
    """Renders the perf-aware prompt blocks from the configured commands."""

    #: Optional reference text injected before skill blocks. Empty by default.
    PERF_REFERENCE_TEXT: str = ""

    def __init__(self, benchmark_cmd: str, test_cmd: str | None = None):
        self._bench = benchmark_cmd
        self._test = test_cmd

    # ------------------------------------------------------------------
    # Shared context block
    # ------------------------------------------------------------------

    def _measurement_block(self) -> str:
        test_line = (
            f"  * Correctness is verified by running: `{self._test}`\n"
            if self._test
            else "  * No automated test command is configured for this task.\n"
        )
        return (
            "HOW YOUR WORK IS SCORED:\n"
            f"  * Performance is measured by timing this command (lower wall "
            f"time is better): `{self._bench}`\n"
            f"{test_line}"
            "  * The harness times the benchmark command before and after your "
            "patch and computes the speedup. You do NOT need to run it for "
            "scoring — just make the code faster while keeping it correct.\n\n"
        )

    # ------------------------------------------------------------------
    # Non-phased BRANCH prompt fragments
    # ------------------------------------------------------------------

    def perf_env_block(self, *, env_name: str | None = None) -> str:
        return self._measurement_block()

    def perf_policy_block(
        self, *, patch_max_lines: int, env_name: str | None = None
    ) -> str:
        return (
            "POLICY (the harness will REJECT any patch violating these):\n"
            "  * Do NOT modify benchmark, test, setup.py, pyproject.toml, "
            "or .git files.\n"
            "  * Do NOT install, uninstall, upgrade, or reinstall any package, "
            "and do NOT edit the environment. Optimise the source only.\n"
            f"  * Keep the patch under {patch_max_lines} added+removed lines.\n"
            "  * Make a SINGLE focused change targeting the hot spot.\n\n"
        )

    def perf_self_profile_block(self, *, env_name: str | None = None) -> str:
        return (
            "SELF-PROFILE BEFORE editing so you see the exact hot spots your "
            "change will impact:\n"
            "  1. Run the benchmark command under a profiler, e.g.\n"
            "       python -m cProfile -s cumtime -m <module>   "
            "(or wrap the benchmark in a small repro script).\n"
            "  2. Read the top cumtime frames — those are your hot spots.\n"
            "  3. Apply the optimisation skill below to the slowest hot spot "
            "you can address within the patch-size policy.\n"
            "Save any scratch output OUTSIDE the repo (e.g. under /tmp) so the "
            "harness's git operations leave it alone.\n\n"
        )

    def perf_branch_task_block(self) -> str:
        return (
            "TASK:\n"
            "Apply ONE optimization following the skill(s) above. Focus on the "
            "code path exercised by the benchmark command. Do not edit tests "
            "or benchmarks — the harness measures and tests your patch for "
            "you.\n\n"
        )

    # ------------------------------------------------------------------
    # PROBE_PROFILE prompt fragments
    # ------------------------------------------------------------------

    def perf_probe_profile_preamble(self, *, baseline_hash: str | None) -> str:
        ref = f" at commit {baseline_hash}" if baseline_hash else ""
        return (
            f"MOKOPT PROBE_PROFILE: the harness has established a baseline{ref}. "
            "Before the optimisation rounds begin, profile this baseline "
            "yourself so the call trees are fresh in your context.\n\n"
        )

    def perf_probe_profile_body(
        self,
        *,
        bench_regex: str | None = None,
        baseline_hash: str | None = None,
        env_name: str | None = None,
    ) -> str:
        return (
            self._measurement_block()
            + "Do at least the following (READ-ONLY — make no source edits):\n"
            "  1. Identify the module / function the benchmark command "
            "exercises.\n"
            "  2. Profile it, e.g. `python -m cProfile -s cumtime <repro>` or "
            "by wrapping the benchmark in a timing harness.\n"
            "  3. Note the top cumtime frames — those are the hot spots the "
            "upcoming BRANCH prompts will ask you to optimise.\n\n"
            "DO NOT modify any source code in this step. The harness "
            "snapshots/restores the tree before the optimisation loop, so any "
            "edits here are discarded."
        )

    def perf_fallback_single_shot_intro(self, *, reason: str) -> str:
        return (
            f"MOKOPT FALLBACK (reason: {reason}): treat the task below as a "
            "single-shot optimization.\n"
            + self._measurement_block()
            + "Profile first, apply a targeted source-only optimization, and "
            "keep the project's tests passing.\n\n"
            "--- TASK ---\n"
        )

    # ------------------------------------------------------------------
    # Phased BRANCH prompt fragments (isolation == "phased")
    # ------------------------------------------------------------------

    def perf_profile_phase_summary(self) -> str:
        return (
            "profiled the benchmark command to identify performance hot spots "
            "(no source edits). Results are at"
        )

    def perf_profile_phase_directive(
        self,
        *,
        bench_regex: str | None = None,
        artifact_path: str,
        env_name: str | None = None,
    ) -> str:
        return (
            "You are in the PROFILE phase.\n"
            "Your task: profile the benchmark command to identify performance "
            "hot spots.\n"
            "CRITICAL: Do NOT modify any source code in this phase. Only run "
            "profiling commands.\n\n" + self._measurement_block() + "Steps:\n"
            "  1. Identify what the benchmark command runs.\n"
            "  2. Profile it (`python -m cProfile -s cumtime <repro>` or "
            "similar).\n"
            f"  3. Write the slowest frames / your findings to {artifact_path}/"
            "profile.md so the LOCALIZE phase can read them.\n"
        )
