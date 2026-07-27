"""Docker-backed :class:`Executor`.

Every branch runs inside an isolated container. The host repo is *copied*
(not bind-mounted) into the container at :attr:`repo_path`, so the host
working tree is never touched — MoKOpt commits, resets and edits freely
inside the container and exports only the final diff.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

from mokopt.executor.base import ExecResult, Executor

logger = logging.getLogger(__name__)


class DockerExecutor(Executor):
    def __init__(self, container, repo_path: str = "/workspace/repo"):
        self._container = container
        self.repo_path = repo_path

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        image: str,
        repo_host_path: Path | None = None,
        repo_path: str = "/workspace/repo",
        env: dict[str, str] | None = None,
        name: str | None = None,
        network: str | None = None,
        extra_hosts: dict[str, str] | None = None,
        pull: bool = False,
    ) -> "DockerExecutor":
        """Start a fresh container from ``image`` with the repo copied in."""
        import docker

        client = docker.from_env()
        if pull:
            logger.info("Pulling image %s ...", image)
            client.images.pull(image)

        # ``host.docker.internal`` lets an in-container CLI reach a gateway
        # running on the host (used by some LLM backends).
        eh = {"host.docker.internal": "host-gateway"}
        if extra_hosts:
            eh.update(extra_hosts)

        logger.info("Starting container from %s ...", image)
        container = client.containers.run(
            image,
            command=["sleep", "infinity"],
            detach=True,
            environment=env or {},
            working_dir=repo_path,
            name=name,
            network=network,
            extra_hosts=eh,
            tty=False,
        )
        ex = cls(container, repo_path=repo_path)
        if repo_host_path is not None:
            ex.run(f"mkdir -p {repo_path}")
            ex.copy_dir_in(repo_host_path, repo_path)
        return ex

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self,
        cmd: str,
        *,
        timeout_s: int | None = None,
        user: str = "root",
        workdir: str | None = None,
    ) -> ExecResult:
        argv = ["bash", "-lc", cmd]
        if timeout_s is not None:
            argv = ["timeout", str(timeout_s), *argv]
        try:
            res = self._container.exec_run(
                argv, user=user, workdir=workdir or self.repo_path, demux=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("exec_run failed: %s", exc)
            return ExecResult(exit_code=1, output=str(exc))
        out = res.output
        if isinstance(out, (bytes, bytearray)):
            text = bytes(out).decode(errors="replace")
        elif out is None:
            text = ""
        else:
            text = str(out)
        return ExecResult(exit_code=res.exit_code or 0, output=text)

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    @staticmethod
    def _tar_bytes(items: list[tuple[Path, str]]) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for host_path, arcname in items:
                tf.add(str(host_path), arcname=arcname, recursive=True)
        buf.seek(0)
        return buf.getvalue()

    def copy_in(
        self, host_path: Path, container_dir: str, filename: str | None = None
    ) -> None:
        self.run(f"mkdir -p {container_dir}")
        arc = filename or host_path.name
        data = self._tar_bytes([(host_path, arc)])
        self._container.put_archive(container_dir, data)

    def copy_dir_in(self, host_dir: Path, container_dir: str) -> None:
        self.run(f"mkdir -p {container_dir}")
        items = [(child, child.name) for child in sorted(Path(host_dir).iterdir())]
        if not items:
            return
        data = self._tar_bytes(items)
        self._container.put_archive(container_dir, data)

    def get_archive(self, container_src: str, dest_tar: Path) -> bool:
        try:
            bits, _ = self._container.get_archive(container_src)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_archive(%s) failed: %s", container_src, exc)
            return False
        dest_tar.parent.mkdir(parents=True, exist_ok=True)
        with dest_tar.open("wb") as f:
            for chunk in bits:
                f.write(chunk)
        return True

    # ------------------------------------------------------------------
    # Identity / lifecycle
    # ------------------------------------------------------------------

    @property
    def container(self):
        return self._container

    @property
    def container_id(self) -> str | None:
        return getattr(self._container, "id", None)

    def close(self) -> None:
        try:
            self._container.stop(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
