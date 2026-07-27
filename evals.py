"""EVAL_FAST and EVAL_FULL for MoKOpt.

The original AutoOpt evaluator ran ASV benchmark suites. MoKOpt instead times a
*configurable benchmark command* and derives a speedup from the wall-clock
ratio against the baseline:

    speedup = baseline_seconds / candidate_seconds      (higher is better)

FAST is the cheap filter run on every branch right after BRANCH:
  * the configured test command (timeout-bounded) as a correctness gate;
  * ``eval_fast_repeats`` timed runs of the benchmark command, median.

FULL is the expensive confirmation run only on FAST-survivors:
  * the test command with a longer timeout;
  * ``eval_full_repeats`` timed runs, median.

Timing is done *inside* the container with ``date +%s.%N`` so the docker-exec
round-trip is excluded. Alternatively, set ``metric_regex`` to parse a number
the benchmark prints (still interpreted as "lower is better").
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from dataclasses import dataclass

from mokopt.executor.base import Executor
from mokopt.state import BetaConfig, Outcome

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"MOKOPT_TIME=([0-9]+(?:\.[0-9]+)?)")


@dataclass
class EvalResult:
    outcome: Outcome
    candidate_hash: str | None
    seconds: float | None


class Evaluator:
    """FAST/FULL evaluators bound to an :class:`Executor`."""

    def __init__(
        self,
        executor: Executor,
        beta: BetaConfig,
        benchmark_cmd: str,
        test_cmd: str | None = None,
        metric_regex: str | None = None,
        prepare_cmd: str | None = None,
    ):
        self._ex = executor
        self._beta = beta
        self._repo = executor.repo_path
        self._bench = benchmark_cmd
        self._test = test_cmd or None
        self._metric_re = re.compile(metric_regex) if metric_regex else None
        self._prepare = prepare_cmd or None

    def prepare_candidate(self, timeout_s: int) -> tuple[bool, str]:
        if not self._prepare:
            return True, ""
        res = self._ex.run(f"cd {self._repo} && {self._prepare}", timeout_s=timeout_s)
        return res.ok, res.output[-1500:]

    # -- correctness gate ---------------------------------------------------

    def check_tests(self, timeout_s: int) -> tuple[bool, str]:
        if not self._test:
            return True, "no test command configured; skipping correctness gate"
        res = self._ex.run(f"cd {self._repo} && {self._test}", timeout_s=timeout_s)
        tail = res.output[-1500:]
        if res.exit_code == 0:
            return True, tail[-500:]
        if res.exit_code == 124:
            return False, "tests timed out"
        return False, tail

    # -- timing -------------------------------------------------------------

    def _time_once(self, timeout_s: int) -> float | None:
        """Run the benchmark command once; return elapsed seconds or None."""
        script = (
            "__t0=$(date +%s.%N); "
            f"( {self._bench} ) > /tmp/.mokopt_bench_out 2>&1; __rc=$?; "
            "__t1=$(date +%s.%N); "
            'awk -v a="$__t0" -v b="$__t1" '
            "'BEGIN{printf \"MOKOPT_TIME=%.6f\\n\", b-a}'; "
            "cat /tmp/.mokopt_bench_out; "
            "exit $__rc"
        )
        res = self._ex.run(f"cd {self._repo} && {script}", timeout_s=timeout_s)
        if res.exit_code != 0:
            logger.warning(
                "benchmark command exit=%d: %s", res.exit_code, res.output[-400:]
            )
            return None
        if self._metric_re is not None:
            matches = self._metric_re.findall(res.output)
            if matches:
                try:
                    return float(matches[-1])
                except (TypeError, ValueError):
                    pass
            logger.warning("metric_regex matched nothing in benchmark output")
            return None
        m = _TIME_RE.search(res.output)
        if not m:
            logger.warning("could not parse MOKOPT_TIME from benchmark output")
            return None
        return float(m.group(1))

    def _measure(self, repeats: int, timeout_s: int) -> float | None:
        """Median of ``repeats`` timed runs (drops failed runs)."""
        times: list[float] = []
        for _ in range(max(1, repeats)):
            t = self._time_once(timeout_s)
            if t is not None and t >= 0:
                times.append(t)
        if not times:
            return None
        return statistics.median(times)

    @staticmethod
    def _speedup(baseline_s: float | None, cand_s: float | None) -> float | None:
        if not baseline_s or not cand_s or cand_s <= 0:
            return None
        return baseline_s / cand_s

    # -- public ------------------------------------------------------------

    def measure_baseline(self, repeats: int, timeout_s: int) -> float | None:
        return self._measure(repeats, timeout_s)

    def fast(self, baseline_seconds: float | None, branch_id: str) -> EvalResult:
        """Cheap filter: tests + ``eval_fast_repeats`` timed runs."""
        t0 = time.time()
        prepared, tail = self.prepare_candidate(self._beta.fast_test_timeout_s)
        if not prepared:
            return EvalResult(
                outcome=Outcome(
                    tests_pass=False,
                    cost_s=time.time() - t0,
                    notes={"prepare_tail": tail},
                ),
                candidate_hash=None,
                seconds=None,
            )
        tests_pass, tail = self.check_tests(self._beta.fast_test_timeout_s)
        if not tests_pass:
            return EvalResult(
                outcome=Outcome(
                    speedup=None,
                    tests_pass=False,
                    cost_s=time.time() - t0,
                    notes={"test_tail": tail[-800:]},
                ),
                candidate_hash=None,
                seconds=None,
            )
        cand_s = self._measure(
            self._beta.eval_fast_repeats, self._beta.fast_test_timeout_s
        )
        speedup = self._speedup(baseline_seconds, cand_s)
        return EvalResult(
            outcome=Outcome(
                speedup=speedup,
                tests_pass=True,
                cost_s=time.time() - t0,
                notes={
                    "candidate_seconds": cand_s,
                    "baseline_seconds": baseline_seconds,
                    "n_repeats": self._beta.eval_fast_repeats,
                },
            ),
            candidate_hash=None,
            seconds=cand_s,
        )

    def full(self, baseline_seconds: float | None, branch_id: str) -> EvalResult:
        """High-fidelity: tests + ``eval_full_repeats`` timed runs, median."""
        t0 = time.time()
        prepared, tail = self.prepare_candidate(self._beta.full_test_timeout_s)
        if not prepared:
            return EvalResult(
                outcome=Outcome(
                    tests_pass=False,
                    cost_s=time.time() - t0,
                    notes={"prepare_tail": tail},
                ),
                candidate_hash=None,
                seconds=None,
            )
        tests_pass, tail = self.check_tests(self._beta.full_test_timeout_s)
        if not tests_pass:
            return EvalResult(
                outcome=Outcome(
                    speedup=None,
                    tests_pass=False,
                    cost_s=time.time() - t0,
                    notes={"test_tail": tail[-800:]},
                ),
                candidate_hash=None,
                seconds=None,
            )
        cand_s = self._measure(
            self._beta.eval_full_repeats, self._beta.full_test_timeout_s
        )
        speedup = self._speedup(baseline_seconds, cand_s)
        return EvalResult(
            outcome=Outcome(
                speedup=speedup,
                tests_pass=True,
                cost_s=time.time() - t0,
                notes={
                    "candidate_seconds": cand_s,
                    "baseline_seconds": baseline_seconds,
                    "n_repeats": self._beta.eval_full_repeats,
                },
            ),
            candidate_hash=None,
            seconds=cand_s,
        )
