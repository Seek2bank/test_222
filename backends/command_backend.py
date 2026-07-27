"""Command-line agent backend.

Each prompt starts a fresh non-interactive CLI process inside the executor.
This gives branch isolation without coupling MoKOpt to any vendor SDK.
"""

from __future__ import annotations

import shlex
import shutil
import typing
import uuid
from pathlib import Path

from mokopt.backends.base import Backend, BackendSession

if typing.TYPE_CHECKING:
    from mokopt.executor.base import Executor


DEFAULT_COMMANDS = {
    "copilot": ('/usr/local/bin/copilot -p "$(cat {prompt_file})" --yolo{model_flag}'),
    "codex": (
        "codex exec --sandbox danger-full-access --skip-git-repo-check"
        '{model_flag} -- "$(cat {prompt_file})"'
    ),
    "claude": (
        'claude -p "$(cat {prompt_file})" --dangerously-skip-permissions{model_flag}'
    ),
}


class _CommandSession(BackendSession):
    def __init__(
        self,
        executor: "Executor",
        command_template: str,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
        timeout_s: int,
    ):
        self._executor = executor
        self._template = command_template
        self._model = model
        self._workdir = working_directory
        self._transcript = transcript_path
        self._timeout_s = timeout_s

    async def send(self, prompt: str, *, label: str = "") -> str:
        prompt_file = f"/tmp/mokopt-prompt-{uuid.uuid4().hex}.txt"
        written = self._executor.write_file(prompt_file, prompt)
        if not written.ok:
            raise RuntimeError(f"could not write agent prompt: {written.output}")
        model_flag = f" --model {shlex.quote(self._model)}" if self._model else ""
        command = self._template.format(
            prompt_file=shlex.quote(prompt_file),
            model=shlex.quote(self._model or ""),
            model_flag=model_flag,
            workdir=shlex.quote(self._workdir),
        )
        result = self._executor.run(
            command,
            timeout_s=self._timeout_s,
            workdir=self._workdir,
        )
        self._executor.run(f"rm -f {shlex.quote(prompt_file)}")
        text = result.output
        if self._transcript is not None:
            with self._transcript.open("a", encoding="utf-8") as stream:
                stream.write(f"\n--- [{label}] agent (exit={result.exit_code}) ---\n")
                stream.write(text)
                stream.write("\n")
        if not result.ok:
            raise RuntimeError(
                f"agent command failed with exit {result.exit_code}: {text[-1000:]}"
            )
        return text

    async def ping(self) -> None:
        return

    async def close(self) -> None:
        return


class CommandBackend(Backend):
    """Run a non-interactive agent CLI for each branch."""

    def __init__(
        self,
        *,
        name: str = "command",
        model: str | None = None,
        command: str | None = None,
        setup_command: str | None = None,
        timeout_s: int = 3600,
    ):
        self.name = name
        self._model = model
        self._command = command or DEFAULT_COMMANDS.get(name)
        if not self._command:
            raise ValueError("a command template is required for the command backend")
        default_setup = {
            "codex": "command -v codex >/dev/null || npm install -g @openai/codex",
            "claude": (
                "command -v claude >/dev/null || "
                "npm install -g @anthropic-ai/claude-code"
            ),
        }
        self._setup_command = setup_command or default_setup.get(name)
        self._timeout_s = timeout_s
        self._executor: "Executor | None" = None

    def bind_executor(self, executor: "Executor") -> None:
        self._executor = executor
        if self.name == "copilot" and executor.container_id is None:
            host_binary = self.host_binary_path()
            if host_binary is not None:
                self._command = self._command.replace(
                    "/usr/local/bin/copilot",
                    shlex.quote(str(host_binary)),
                    1,
                )
        if self._setup_command:
            result = executor.run(self._setup_command, timeout_s=self._timeout_s)
            if not result.ok:
                raise RuntimeError(
                    f"agent setup failed with exit {result.exit_code}: "
                    f"{result.output[-1000:]}"
                )

    def host_binary_path(self) -> Path | None:
        if self.name == "copilot":
            binary = shutil.which("copilot")
            if binary:
                return Path(binary)
            candidate = Path.home() / ".local/bin/copilot"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(
                "Copilot CLI was not found on the host; install it or use "
                "--agent-command with another in-container agent"
            )
        return None

    def container_binary_path(self) -> str:
        return "/usr/local/bin/copilot" if self.name == "copilot" else ""

    def setup_script_path(self) -> Path | None:
        return None

    def env_vars(self) -> dict[str, str]:
        return {}

    async def open_session(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> BackendSession:
        del wrapper_path
        if self._executor is None:
            raise RuntimeError("backend is not bound to an executor")
        return _CommandSession(
            self._executor,
            self._command,
            model or self._model,
            working_directory,
            transcript_path,
            self._timeout_s,
        )

    async def new_session_factory(
        self,
        wrapper_path: Path,
        *,
        model: str | None,
        working_directory: str,
        transcript_path: Path | None,
    ) -> typing.Callable[[], typing.Awaitable[BackendSession]]:
        async def factory() -> BackendSession:
            return await self.open_session(
                wrapper_path,
                model=model,
                working_directory=working_directory,
                transcript_path=transcript_path,
            )

        return factory

    def copy_logs(self, executor: "Executor", logging_dir: Path | None) -> None:
        return

    def parse_metrics(self, logging_dir: Path | None) -> tuple[int, int, float]:
        return (0, 0, 0.0)
