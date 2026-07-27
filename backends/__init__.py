"""Backend factory for MoKOpt."""

from __future__ import annotations

from typing import Any

from mokopt.backends.base import Backend, BackendSession
from mokopt.backends.command_backend import CommandBackend


def build_backend(name: str, *, model: str | None = None, **kwargs: Any) -> Backend:
    """Construct a backend by short name.

    Supported: ``mock`` (default, no API key), ``claude``, ``copilot``,
    ``codex``. Extra kwargs are forwarded to the backend constructor (e.g.
    ``solutions=`` for the mock backend).
    """
    name = (name or "mock").lower()
    if name == "mock":
        from mokopt.backends.mock_backend import MockBackend

        return MockBackend(model=model, **kwargs)
    if name in {"command", "claude", "copilot", "codex"}:
        return CommandBackend(name=name, model=model, **kwargs)
    raise ValueError(
        f"Unknown MoKOpt backend {name!r}; supported: mock, claude, copilot, codex"
    )


__all__ = ["Backend", "BackendSession", "CommandBackend", "build_backend"]
