"""Mock / dry-run backend for MoKOpt.

Lets the whole controller (PROBE -> BRANCH(K) -> EVAL_FAST -> PRUNE ->
EVAL_FULL -> STOP) run end-to-end **without any API key or LLM**. Instead of
spawning a CLI, the mock session edits the repo *directly through the
executor*:

* PROBE_PROFILE / router / phased non-optimize prompts -> no-op (canned reply).
* A branch's "apply" prompt -> runs the next configured *solution* (a shell
  command executed in the repo), simulating the patch an LLM would have made.

With no solutions configured every branch makes no change and is pruned for
"no improvement" — still exercising the full control flow. Supplying one or
more solutions (e.g. for the bundled demo repo) lets a branch show a real
speedup and get promoted, so the demo is end-to-end meaningful.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from mokopt.backends.base import Backend, BackendSession

if typing.TYPE_CHECKING:
    from mokopt.executor.base import Executor

logger = logging.getLogger(__name__)


def _is_apply_prompt(label: str) -> bool:
    """True for prompts where an LLM would actually edit code.

    Single isolation: the branch label is the branch id, e.g. ``r0-b1``.
    Phased isolation: only the ``.../optimize`` phase edits code.
    """
    if not label:
        return False
    if "/" in label:
        return label.endswith("/optimize")
    return bool(label) and label[0] == "r" and "-" in label


class _MockSession(BackendSession):
    def __init__(self, backend: "MockBackend", transcript_path: Path | None):
        self._backend = backend
        self._transcript = transcript_path
        self._closed = False

    async def send(self, prompt: str, *, label: str = "") -> str:
        reply: str
        if _is_apply_prompt(label):
            reply = self._backend.apply_next_solution(label)
        else:
            reply = f"[mock] acknowledged ({label or 'prompt'}); no edits made."
        if self._transcript is not None:
            with self._transcript.open("a") as f:
                f.write(f"\n--- [{label}] mock assistant ---\n{reply}\n")
        return reply

    async def ping(self) -> None:
        return

    async def close(self) -> None:
        self._closed = True


class MockBackend(Backend):
    name = "mock"

    def __init__(
        self,
        *,
        model: str | None = None,
        solutions: list[str] | None = None,
    ):
        self._model = model
        self._solutions = list(solutions or [])
        self._cursor = 0
        self._executor: "Executor | None" = None

    # ---- executor binding ---------------------------------------------

    def bind_executor(self, executor: "Executor") -> None:
        self._executor = executor

    def apply_next_solution(self, label: str) -> str:
        """Run the next configured solution command in the repo (if any)."""
        if self._executor is None:
            return "[mock] no executor bound; cannot apply edits."
        if self._cursor >= len(self._solutions):
            return f"[mock] branch {label}: no solution left; made no change."
        sol = self._solutions[self._cursor]
        self._cursor += 1
        res = self._executor.run(f"cd {self._executor.repo_path} && {sol}")
        status = "ok" if res.ok else f"exit={res.exit_code}"
        logger.info(
            "mock applied solution #%d for %s (%s)", self._cursor, label, status
        )
        return (
            f"[mock] branch {label}: applied solution #{self._cursor} "
            f"({status}).\n{res.output[-400:]}"
        )

    # ---- Install-time (nothing to install) ----------------------------

    def host_binary_path(self) -> Path | None:
        return None

    def container_binary_path(self) -> str:
        return "/bin/true"

    def setup_script_path(self) -> Path | None:
        return None

    def env_vars(self) -> dict[str, str]:
        return {}

    # ---- Run-time ------------------------------------------------------

    async def open_session(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> BackendSession:
        return _MockSession(self, transcript_path)

    async def new_session_factory(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> typing.Callable[[], typing.Awaitable[BackendSession]]:
        async def _factory() -> BackendSession:
            return _MockSession(self, transcript_path)

        return _factory

    # ---- Post-run ------------------------------------------------------

    def copy_logs(self, executor: "Executor", logging_dir: Path | None) -> None:
        return

    def parse_metrics(self, logging_dir: Path | None) -> tuple[int, int, float]:
        return (0, 0, 0.0)
