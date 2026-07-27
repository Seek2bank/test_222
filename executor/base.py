"""Executor abstraction — the command-execution / isolation layer.

MoKOpt drives every command (agent edits, snapshot/restore, benchmark
timing, tests) through an :class:`Executor`. This decouples the controller
from any particular sandbox: the reference implementation
(:class:`mokopt.executor.docker_executor.DockerExecutor`) runs everything
inside a Docker container, but the interface is small enough that an SSH /
remote / local-subprocess backend could be slotted in.

The interface intentionally mirrors the handful of operations the original
FormulaCode harness exposed on its ``TmuxSession`` (``container.exec_run``,
``copy_to_container``, ``get_archive``) so the ported controller logic stays
recognisable.
"""

from __future__ import annotations

import shlex
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    """Result of a single command execution."""

    exit_code: int
    output: str  # decoded stdout+stderr (errors="replace")

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Executor(ABC):
    """A live handle to an isolated environment containing the repo."""

    repo_path: str

    # ---- command execution --------------------------------------------

    @abstractmethod
    def run(
        self,
        cmd: str,
        *,
        timeout_s: int | None = None,
        user: str = "root",
        workdir: str | None = None,
    ) -> ExecResult:
        """Run ``cmd`` through ``bash -lc`` inside the environment.

        ``timeout_s`` wraps the command with ``timeout`` (best-effort).
        ``workdir`` defaults to :attr:`repo_path` when None.
        """

    # ---- filesystem ----------------------------------------------------

    @abstractmethod
    def copy_in(
        self, host_path: Path, container_dir: str, filename: str | None = None
    ) -> None:
        """Copy a single host file into ``container_dir`` (created if absent)."""

    @abstractmethod
    def copy_dir_in(self, host_dir: Path, container_dir: str) -> None:
        """Copy the *contents* of a host directory into ``container_dir``."""

    @abstractmethod
    def get_archive(self, container_src: str, dest_tar: Path) -> bool:
        """Stream a container path out to a host tar file. Returns success."""

    def write_file(self, container_path: str, content: str) -> ExecResult:
        """Write ``content`` to a file inside the container."""
        parent = str(Path(container_path).parent)
        # Use a heredoc so arbitrary content is written verbatim.
        delimiter = f"__MOKOPT_EOF_{uuid.uuid4().hex}__"
        cmd = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"cat > {shlex.quote(container_path)} <<'{delimiter}'\n"
            f"{content}\n{delimiter}\n"
        )
        return self.run(cmd)

    def read_file(self, container_path: str) -> str | None:
        res = self.run(f"cat {shlex.quote(container_path)}")
        return res.output if res.ok else None

    def path_exists(self, container_path: str) -> bool:
        return self.run(f"test -e {shlex.quote(container_path)}").ok

    # ---- identity / lifecycle -----------------------------------------

    @property
    @abstractmethod
    def container_id(self) -> str | None:
        """Backend container id, or None for non-container executors."""

    @abstractmethod
    def close(self) -> None:
        """Tear the environment down. Idempotent."""
