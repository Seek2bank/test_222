"""High-level MoKOpt runner and persistent result contract."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from mokopt.backends.base import Backend
from mokopt.controller import MoKOptController
from mokopt.executor.base import Executor
from mokopt.repo import GitRepo
from mokopt.state import BetaConfig, Budget, TaskSpec
from mokopt.trajectory import TrajectoryWriter


@dataclass
class RunResult:
    task_id: str
    baseline_hash: str
    final_hash: str
    baseline_seconds: float
    final_seconds: float
    speedup: float
    tests_pass: bool
    best_branch: str | None
    patch_path: str
    trajectory_path: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class MoKOptRunner:
    def __init__(
        self,
        *,
        executor: Executor,
        backend: Backend,
        task: TaskSpec,
        benchmark_cmd: str,
        output_dir: Path,
        test_cmd: str | None = None,
        metric_regex: str | None = None,
        prepare_cmd: str | None = None,
        setup_cmd: str | None = None,
        model: str | None = None,
        beta: BetaConfig | None = None,
        budget: Budget | None = None,
        skills_dir: Path | None = None,
        injected_files: dict[str, str] | None = None,
    ):
        self._ex = executor
        self._backend = backend
        self._task = task
        self._benchmark_cmd = benchmark_cmd
        self._test_cmd = test_cmd
        self._metric_regex = metric_regex
        self._prepare_cmd = prepare_cmd
        self._setup_cmd = setup_cmd
        self._model = model
        self._beta = beta or BetaConfig()
        self._budget = budget or Budget(rounds=self._beta.max_rounds)
        self._skills_dir = skills_dir or Path(__file__).parent / "skills"
        self._output = output_dir
        self._files = injected_files or {}

    def run(self) -> RunResult:
        return asyncio.run(self._run())

    async def _run(self) -> RunResult:
        if self._ex.container_id is None:
            repo = Path(self._ex.repo_path).resolve()
            output = self._output.resolve()
            if output == repo or output.is_relative_to(repo):
                raise ValueError(
                    "local output must be outside the repository because git "
                    "snapshot cleanup would remove it"
                )
        self._output.mkdir(parents=True, exist_ok=True)
        for path, content in self._files.items():
            result = self._ex.write_file(path, content)
            if not result.ok:
                raise RuntimeError(f"could not inject {path}: {result.output}")
        skills_mount = (
            "/skills"
            if self._ex.container_id is not None
            else str((self._output / "skills").resolve())
        )
        self._ex.copy_dir_in(self._skills_dir, skills_mount)
        if self._setup_cmd:
            setup = self._ex.run(
                self._setup_cmd,
                timeout_s=self._beta.full_test_timeout_s,
            )
            if not setup.ok:
                raise RuntimeError(
                    f"task setup failed with exit {setup.exit_code}: "
                    f"{setup.output[-1500:]}"
                )

        binary = self._backend.host_binary_path()
        if binary is not None and self._ex.container_id is not None:
            container_binary = self._backend.container_binary_path()
            if not container_binary:
                raise RuntimeError("backend returned a binary without a target path")
            self._ex.copy_in(
                binary,
                str(Path(container_binary).parent),
                Path(container_binary).name,
            )
            copied = self._ex.run(f"chmod +x {container_binary}")
            if not copied.ok:
                raise RuntimeError(f"could not install agent binary: {copied.output}")
            preflight = self._ex.run(f"{container_binary} --version", timeout_s=30)
            if not preflight.ok:
                raise RuntimeError(
                    "the copied agent binary cannot run in the benchmark "
                    f"environment: {preflight.output[-1000:]}"
                )
        if self._backend.name == "copilot" and self._ex.container_id is not None:
            self._copy_copilot_auth()
        self._backend.bind_executor(self._ex)
        transcript = self._output / "transcript.log"
        session = await self._backend.open_session(
            Path("/bin/true"),
            model=self._model,
            working_directory=self._ex.repo_path,
            transcript_path=transcript,
        )
        factory = await self._backend.new_session_factory(
            Path("/bin/true"),
            model=self._model,
            working_directory=self._ex.repo_path,
            transcript_path=transcript,
        )
        trajectory_path = self._output / "trajectory.jsonl"
        trajectory = TrajectoryWriter(trajectory_path, self._task.task_id)
        controller = MoKOptController(
            executor=self._ex,
            beta=self._beta,
            budget=self._budget,
            benchmark_cmd=self._benchmark_cmd,
            test_cmd=self._test_cmd,
            metric_regex=self._metric_regex,
            prepare_cmd=self._prepare_cmd,
            skills_dir=self._skills_dir,
            trajectory=trajectory,
            logging_dir=self._output,
            skills_mount=skills_mount,
        )
        try:
            state = await controller.run(
                session,
                self._task,
                session_factory=factory,
            )
        finally:
            await session.close()
            await self._backend.shutdown()
            self._backend.copy_logs(self._ex, self._output)

        repo = GitRepo(self._ex)
        baseline_hash = state.baseline_hash
        if baseline_hash is None or state.baseline_seconds is None:
            raise RuntimeError("controller returned an incomplete baseline")
        final_hash = repo.head() or baseline_hash
        patch = repo.diff(baseline_hash, final_hash)
        patch_path = self._output / "final.patch"
        patch_path.write_text(patch, encoding="utf-8")
        final_seconds = state.baseline_seconds
        speedup = 1.0
        tests_pass = True
        if state.best_branch is not None:
            speedup = (
                state.best_branch.full_speedup or state.best_branch.fast_speedup or 1.0
            )
            final_seconds = state.baseline_seconds / speedup
            tests_pass = state.best_branch.tests_pass is True
        input_tokens, output_tokens, cost = self._backend.parse_metrics(self._output)
        result = RunResult(
            task_id=self._task.task_id,
            baseline_hash=baseline_hash,
            final_hash=final_hash,
            baseline_seconds=state.baseline_seconds,
            final_seconds=final_seconds,
            speedup=speedup,
            tests_pass=tests_pass,
            best_branch=(state.best_branch.branch_id if state.best_branch else None),
            patch_path=str(patch_path),
            trajectory_path=str(trajectory_path),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        (self._output / "result.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    def _copy_copilot_auth(self) -> None:
        """Copy host OAuth configuration when no token environment is available."""
        token_keys = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
        if any(os.environ.get(key) for key in token_keys):
            return
        config_dir = Path.home() / ".copilot"
        config_file = config_dir / "config.json"
        if not config_file.is_file():
            raise RuntimeError(
                "Copilot is not authenticated. Run `copilot /login` on the "
                "host, or export a valid Copilot token."
            )
        created = self._ex.run("mkdir -p /root/.copilot && chmod 700 /root/.copilot")
        if not created.ok:
            raise RuntimeError(
                f"could not create Copilot config directory: {created.output}"
            )
        for name in ("config.json", "settings.json"):
            source = config_dir / name
            if source.is_file():
                self._ex.copy_in(source, "/root/.copilot", name)
        secured = self._ex.run(
            "chmod 600 /root/.copilot/config.json "
            "/root/.copilot/settings.json 2>/dev/null || true"
        )
        if not secured.ok:
            raise RuntimeError(f"could not secure Copilot config: {secured.output}")
