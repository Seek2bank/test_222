# Skill: Precomputation
*Park-Sujin principle: **Replace** — Precomputing (move work earlier).*

Use when the profile shows **the same constant-valued setup work
performed on every call** of a hot function. Unlike caching (which
memoises results of *user-input* calls), precomputation hoists work that
depends only on *constants* (or on long-lived configuration) to module
import or instance construction time, so the hot path pays zero cost.

## Trigger

* `cProfile` shows `re.compile`, `np.tri`, `np.arange`, `np.empty(...)`
  type-table builds, dtype/unit lookups, `tuple(...)` over a literal
  list, or constructor calls for objects that never change after
  initialisation.
* `ncalls` ≈ 1 but `cumtime` is large for a frame called *during* hot
  loop set-up (often inside a list comprehension that builds the same
  array on every call).
* `numpy`/`pandas` operations whose arguments are pure constants
  (e.g. `np.triu_indices(n)` for a fixed `n`).
* Many calls to `getattr(<module>, '<constant>')` or
  `importlib.import_module(...)` where the resolved target never changes.

## Recipe

1. **Pre-compile regex at module level**:
   ```python
   # before
   def parse(s):
       return re.findall(r"\d+(?:\.\d+)?", s)

   # after
   _NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
   def parse(s):
       return _NUMBER_RE.findall(s)
   ```

2. **Materialise constant masks/index tables at import time**:
   ```python
   # before — rebuilds the lower-triangular mask on every call
   def is_pareto_front(loss):
       n = loss.shape[0]
       mask = np.tri(n - 1, dtype=bool).T
       ...

   # after — precompute for all sizes the hot path actually sees
   _PARETO_SMALL_N = 64
   _TRI_MASKS = tuple(
       np.tri(n - 1, dtype=bool).T if n >= 2 else np.empty((0, 0), bool)
       for n in range(_PARETO_SMALL_N + 1)
   )

   def is_pareto_front(loss):
       n = loss.shape[0]
       if n <= _PARETO_SMALL_N:
           mask = _TRI_MASKS[n]
           ...
   ```

3. **Bind expensive method resolutions at `__init__`**:
   ```python
   class TPESampler:
       def __init__(self, ...):
           # Resolve the implementation once; the dispatch cost dominates
           # the actual compute on small problems.
           self._compute = self._select_compute()
       def _select_compute(self):
           return _vectorised if _have_numba else _python_loop
   ```

4. **Cache the result of expensive type-introspection once per class**:
   ```python
   # before
   def get_param_type_hints(cls):
       return [(f, t) for f, t in get_type_hints(cls).items() if _is_param(t)]

   # after — `cls` is hashable, the result is fixed for a class
   @lru_cache(maxsize=512)
   def _cached_param_type_hints(cls):
       return tuple((f, t) for f, t in get_type_hints(cls).items()
                    if _is_param(t))
   def get_param_type_hints(cls):
       return list(_cached_param_type_hints(cls))
   ```

5. **Materialise a `tuple` of `re.Pattern` lexer rules lazily but
   exactly once**:
   ```python
   @lru_cache(maxsize=1)
   def _lexer_rules():
       return tuple(
           (re.compile(p), k)
           for p, k in _LEXER_PATTERNS
       )
   ```

## Real-world examples

| Repo                | Issue / branch       | Function                          | Effect |
|---------------------|----------------------|-----------------------------------|--------|
| numpy/numpy         | #12575               | `find_duplicate`                  | builds Counter once vs nested scan |
| optuna/optuna       | 6223 (r0-b1)         | `_is_pareto_front_nd`             | precomputed lower-tri masks |
| xdsl/xdsl           | 3197 auto-opt (r1-b2)| `_lexer_rules`                    | regex compile hoisted |
| xdsl/xdsl           | 3197 auto-opt (r2-b2)| `irdl_param_attr_get_param_type_hints` | `lru_cache` on class |
| matplotlib/matplotlib| #18018              | `findfont._findfont_cached`       | cached lookup table |

## Watch-outs

* Module-level precomputation **runs at import** — make sure the
  `import` cost itself isn't measured by the benchmark (some ASV suites
  do `time_import`).
* Constants computed at import are shared across threads/processes;
  if any caller mutates them you get a non-local bug. Mark them with
  a leading underscore and use immutable types (`tuple`, frozenset).
* Class-level `lru_cache` on a method that takes `cls` as the only key
  is fine, but on methods that take `self` it pins instances in memory
  (memory leak). Use `WeakKeyDictionary` instead, or move the cache to
  `__init__`.
* `lru_cache(maxsize=None)` is unbounded and leaks. Always cap.
* Some "constants" are actually configurable (e.g. `mpl.rcParams`).
  Re-check that no code path mutates them after import; if it does,
  precompute lazily on first access and invalidate on `rcParams` writes.
