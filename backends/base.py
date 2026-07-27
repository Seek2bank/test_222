"""Backend abstraction for MoKOpt.

A ``Backend`` packages everything that differs between LLM CLIs / SDKs:
  * which binary to look for on the host and where to install it in the
    container,
  * which setup script to run at install time,
  * which env vars to inject into the container,
  * how to open an SDK ``BackendSession`` the controller talks to,
  * how to copy per-run logs out and extract token / cost metrics.

The controller and the rest of MoKOpt only see this interface; the
Copilot / Claude / Codex / Mock specifics live in sibling modules.

The in-container CLI is driven through a host-side *docker-exec wrapper*
(a shell script the runner generates from the container id). SDK backends
spawn that wrapper as their CLI subprocess; it ``docker exec``s into the
container so every tool call the model makes runs against the repo.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from pathlib import Path

if typing.TYPE_CHECKING:
    from mokopt.executor.base import Executor


class BackendSession(ABC):
    """A live multi-turn conversation with the in-container CLI agent."""

    @abstractmethod
    async def send(self, prompt: str, *, label: str = "") -> str:
        """Send one user prompt; wait for the assistant to go idle; return
        the concatenated assistant text. ``label`` is for transcript logging."""

    @abstractmethod
    async def ping(self) -> None:
        """Keepalive (no-op for SDKs that don't need it)."""

    @abstractmethod
    async def close(self) -> None:
        """Tear the session down. Idempotent."""

    async def disconnect(self) -> None:
        """Alias for :meth:`close` — the controller closes branch sessions via
        ``disconnect()``; concrete sessions may override but the default routes
        to ``close`` so isolation always tears down cleanly."""
        await self.close()


class Backend(ABC):
    """Configuration + factory for a particular LLM CLI integration."""

    name: str  # "copilot" | "claude" | "codex" | "mock"

    # ---- Install-time --------------------------------------------------

    @abstractmethod
    def host_binary_path(self) -> Path | None:
        """Path to the CLI binary on the host, or None if not needed (e.g.
        when the setup script installs it inside the container). If non-None,
        the runner copies it into the container at ``container_binary_path()``.
        """

    @abstractmethod
    def container_binary_path(self) -> str:
        """Absolute path of the CLI inside the container — the docker-exec
        wrapper invokes this and the SDK talks stdio with it."""

    @abstractmethod
    def setup_script_path(self) -> Path | None:
        """Host-side path to the install script run inside the container, or
        None for backends (e.g. mock) that need no install."""

    @abstractmethod
    def env_vars(self) -> dict[str, str]:
        """Env vars to inject into the container during install AND into the
        docker-exec wrapper (so the CLI sees API keys etc.)."""

    def passthrough_env_keys(self) -> tuple[str, ...]:
        """Extra env var names the docker-exec wrapper forwards from the host
        shell at call time (not baked in at write time)."""
        return ()

    def bind_executor(self, executor: "Executor") -> None:
        """Optional hook: give the backend a handle to the live executor.

        SDK backends drive the container through the docker-exec wrapper and
        ignore this; the mock backend uses it to apply edits directly. Called
        by the runner once the container is up. Default no-op."""
        return

    # ---- Run-time ------------------------------------------------------

    @abstractmethod
    async def open_session(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> BackendSession:
        """Open a multi-turn session whose CLI subprocess is launched via
        ``wrapper_path`` (a host shell script that docker-execs into the
        container)."""

    @abstractmethod
    async def new_session_factory(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> typing.Callable[[], typing.Awaitable[BackendSession]]:
        """Return an async factory that mints a fresh ``BackendSession`` per
        branch in the controller's BRANCH/EVAL loop, for proper isolation."""

    async def shutdown(self) -> None:
        """Optional: release any heavy client objects. Default no-op."""
        return

    # ---- Post-run ------------------------------------------------------

    @abstractmethod
    def copy_logs(self, executor: "Executor", logging_dir: Path | None) -> None:
        """Copy any agent-internal logs out of the container into
        ``logging_dir``. No-op if logging_dir is None."""

    @abstractmethod
    def parse_metrics(self, logging_dir: Path | None) -> tuple[int, int, float]:
        """Return ``(input_tokens, output_tokens, total_cost_usd)``; zeros if
        metrics aren't recoverable."""
