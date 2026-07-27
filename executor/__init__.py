"""Execution / isolation layer for MoKOpt."""

from mokopt.executor.base import ExecResult, Executor
from mokopt.executor.local_executor import LocalExecutor

__all__ = ["ExecResult", "Executor", "LocalExecutor", "DockerExecutor"]


def __getattr__(name: str):
    # Lazy import so ``import mokopt.executor`` doesn't require docker-py
    # unless the Docker executor is actually used.
    if name == "DockerExecutor":
        from mokopt.executor.docker_executor import DockerExecutor

        return DockerExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
