"""Trajectory writer.

Each row is one ``(state, action, outcome)`` event in JSON-Lines format. The
file is written to ``{logging_dir}/mokopt_trajectory.jsonl`` so downstream
offline search / analysis can replay the controller's decisions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return repr(obj)


class TrajectoryWriter:
    def __init__(self, path: Path | None, task_id: str):
        self._path = path
        self._task_id = task_id
        self._step = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: str,
        outcome: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return
        self._step += 1
        row = {
            "task_id": self._task_id,
            "step": self._step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "outcome": _to_jsonable(outcome) if outcome is not None else None,
        }
        if extra:
            row.update(_to_jsonable(extra))
        with self._path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
