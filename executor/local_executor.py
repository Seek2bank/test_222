"""Local subprocess executor.

This executor is primarily useful for development and CI. Production benchmark
runs should prefer :class:`DockerExecutor` so an agent cannot modify the host
checkout or environment.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

from mokopt.executor.base import ExecResult, Executor


class LocalExecutor(Executor):
    def __init__(self, repo_path: Path):
        self.repo_path = str(repo_path.resolve())

    def run(
        self,
        cmd: str,
        *,
        timeout_s: int | None = None,
        user: str = "root",
        workdir: str | None = None,
    ) -> ExecResult:
        del user
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=workdir or self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return ExecResult(exit_code=124, output=output)
        except OSError as exc:
            return ExecResult(exit_code=1, output=str(exc))
        return ExecResult(
            exit_code=result.returncode,
            output=result.stdout + result.stderr,
        )

    def copy_in(
        self, host_path: Path, container_dir: str, filename: str | None = None
    ) -> None:
        destination = Path(container_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host_path, destination / (filename or host_path.name))

    def copy_dir_in(self, host_dir: Path, container_dir: str) -> None:
        destination = Path(container_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for child in host_dir.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    def get_archive(self, container_src: str, dest_tar: Path) -> bool:
        source = Path(container_src)
        if not source.exists():
            return False
        dest_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest_tar, "w") as archive:
            archive.add(source, arcname=source.name)
        return True

    @property
    def container_id(self) -> str | None:
        return None

    def close(self) -> None:
        return
