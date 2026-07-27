"""LLM-as-Router for dynamic BRANCH skill selection.

Opt-in via ``BetaConfig.dynamic_routing``. The router takes the nine skill-card
summaries and the agent's just-collected profile context, then asks the session to
rank skills for *this* task, and returns an ordered ``list[str]`` of skill
names. On any failure it returns ``None`` so the caller can fall back to
the bandit / round-robin.
"""

from __future__ import annotations

import json
import re
import typing

if typing.TYPE_CHECKING:
    from mokopt.backends.base import (
        BackendSession,
    )


# Hand-curated one-line summaries of each skill card. Kept inline so the
# router doesn't have to read the markdown files at runtime (the skill
# cards are mounted inside the container, not on the controller host).
SKILL_SUMMARIES: dict[str, str] = {
    "overhead_reduction": (
        "Dispatch/lookup/type-check machinery dominates "
        "(getattr, isinstance, decorator wrappers) on tiny functions "
        "called millions of times."
    ),
    "vectorization": (
        "Explicit Python for-loops over array-shaped data in a repo that "
        "already uses NumPy/pandas/xarray/scipy."
    ),
    "caching": (
        "Same expensive deterministic computation (parse/compile/load/"
        "re.compile/JSON-decode) repeated with identical inputs."
    ),
    "algorithmic": (
        "Runtime grows worse than necessary with input size: nested loops, "
        "repeated linear scans, O(n^2) joins, find/index/count in hot paths."
    ),
    "structural": (
        "Per-row or per-call IO / serialisation / allocation / coordinate "
        "transforms (read/write/open/close, HDF5/Parquet/CSV, astype/_concat)."
    ),
    "redundant_work": (
        "Same expensive call invoked twice with identical or "
        "mutually-derivable inputs, or a mathematical no-op still executed."
    ),
    "fast_path": (
        "Generic worst-case algorithm running on inputs that hit a trivial "
        "special case the great majority of the time."
    ),
    "precomputation": (
        "Same constant-valued setup work performed on every call of a hot "
        "function — can be hoisted to import or constructor time."
    ),
    "deferral": (
        "Expensive eager work the benchmark almost never consumes: "
        "top-of-module heavy imports, eager validation, unused feature init."
    ),
}


_JSON_BLOCK_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)+\]", re.DOTALL)


def _build_router_prompt(
    instruction: str,
    k: int,
    candidates: list[str],
    profile_hint: str | None,
) -> str:
    catalog_lines = [
        f"  - {name}: {SKILL_SUMMARIES[name]}"
        for name in candidates
        if name in SKILL_SUMMARIES
    ]
    catalog = "\n".join(catalog_lines)
    profile_block = (
        "PROFILE CONTEXT (from your earlier PROBE_PROFILE step — top hot "
        "spots, slowest benchmarks, dominant call frames):\n"
        f"{profile_hint.strip()}\n\n"
        if profile_hint
        else (
            "PROFILE CONTEXT: rely on the profile you collected in the "
            "PROBE_PROFILE step earlier in this conversation.\n\n"
        )
    )
    return (
        "AUTO-OPT ROUTE: based on the profile you just collected, rank the "
        f"top {k} optimisation skills most likely to yield a speedup on "
        "THIS particular benchmark suite. You are choosing the directions "
        "the harness will branch on next.\n\n"
        + profile_block
        + "SKILL CATALOG (one-line summary each):\n"
        + catalog
        + "\n\n"
        "ORIGINAL TASK INSTRUCTIONS (for reference, do not re-read the "
        "whole repo):\n" + instruction.strip()[:2000] + "\n\n"
        "Reply with a SINGLE JSON array, no prose, no markdown fences, "
        "in priority order. Example shape:\n"
        '  [{"skill": "vectorization", "confidence": 0.8, '
        '"rationale": "top frame is a python for-loop over a numpy array"}, '
        '{"skill": "caching", "confidence": 0.5, "rationale": "..."}]\n'
        f"Include exactly {k} entries. Each `skill` MUST be one of: "
        + ", ".join(candidates)
        + "."
    )


def _parse_router_reply(reply: str, candidates: list[str]) -> list[str] | None:
    """Extract an ordered list of valid skill names from the agent's reply.

    Tolerant of: markdown fences, leading/trailing prose, duplicate
    entries, unknown skill names (silently dropped). Returns ``None`` if
    no JSON array can be found at all.
    """
    if not reply:
        return None
    text = reply.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    parsed: typing.Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    valid = set(candidates)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("skill") or item.get("name")
        else:
            continue
        if not isinstance(name, str):
            continue
        if name in valid and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered or None


async def route_skills(
    sdk_session: "BackendSession",
    instruction: str,
    k: int,
    candidates: list[str],
    profile_hint: str | None = None,
    label: str = "probe_route",
) -> list[str] | None:
    """Ask the SDK to rank the top ``k`` skills for this task.

    Returns the ordered list (length ``<= k``, may be shorter if the
    model emitted fewer valid names) or ``None`` on any failure. The
    caller is responsible for padding / falling back to the bandit or
    round-robin when this returns ``None`` or a too-short list.
    """
    if k <= 0 or not candidates:
        return None
    prompt = _build_router_prompt(instruction, k, candidates, profile_hint)
    try:
        reply = await sdk_session.send(prompt, label=label)
    except Exception:
        return None
    return _parse_router_reply(reply or "", candidates)
