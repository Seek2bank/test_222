# Skill: Deferral / Lazy Work
*Park-Sujin principle: **Remove** — Deferring (postpone until needed).*

Use when the profile shows **expensive work performed eagerly that the
benchmark almost never consumes**. Common forms: top-of-module imports
that drag in heavyweight optional dependencies, eager validation of
configuration that is rarely changed, or feature initialisation that
fires for every entry point even though most calls use a small subset.

This is the dual of precomputation: precomputation moves *needed* work
earlier; deferral moves *rarely-needed* work later (or skips it
entirely).

## Trigger

* `python -X importtime` (or `asv profile` of `import_<package>`) shows
  large `cumulative` time inside top-of-module statements:
  `from xdsl.parser import Parser`, `import argparse`, heavy plugin
  registration loops.
* `cProfile` of the first benchmark iteration includes one-shot
  `__init__`/`register`/`build` frames that don't recur in subsequent
  iterations and are not actually exercised by the benchmarked code path.
* `ncalls = 1` frames with `cumtime` ≫ per-iter cost — those are
  setup-time costs masquerading as steady-state work.
* The hot path runs only one of several optional code paths
  (e.g. only the `mlir` frontend, not the others) but all paths get
  registered up front.

## Recipe

1. **Move heavy imports inside the function that actually needs them**.
   Combine with `from __future__ import annotations` + `TYPE_CHECKING`
   to keep static type hints free:
   ```python
   # before
   import argparse
   import sys
   from xdsl.parser import Parser

   class CLI:
       def parse(self, src):
           return Parser(src)

   # after
   from __future__ import annotations
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:                       # static type checker only
       import argparse
       from xdsl.parser import Parser

   class CLI:
       def parse(self, src):
           from xdsl.parser import Parser  # paid only on first parse()
           return Parser(src)
   ```

2. **Cache the lazy import result** with `functools.lru_cache(maxsize=1)`
   when the import itself is expensive *and* the function is hot:
   ```python
   @lru_cache(maxsize=1)
   def _get_argparse():
       import argparse
       return argparse
   ```

3. **Defer plugin / dialect registration to first use** via a sentinel:
   ```python
   class Registry:
       _dialects = None
       def register_all(self):
           if self._dialects is not None:
               return
           # heavy work
           self._dialects = {n: f() for n, f in _all_dialect_factories.items()}
   ```

4. **Lazily initialise IO/IERS-style globals** behind a property:
   ```python
   class TimeContext:
       _iers = None
       @property
       def iers(self):
           if self._iers is None:
               from astropy.utils.iers import IERS_Auto
               self._iers = IERS_Auto.open()
           return self._iers
   ```

5. **Drop deep validation in hot constructors** when callers pre-validate:
   ```python
   def __init__(self, value, *, validated=False):
       if not validated:
           value = self._validate(value)   # expensive
       self._value = value
   ```

## Real-world examples

| Repo                | Issue / branch          | Change                              |
|---------------------|-------------------------|-------------------------------------|
| xdsl/xdsl           | 3197 auto-opt (r2-b0, r3-b0) | `command_line_tool.py`, `xdsl_opt_main.py` — moved 8+ imports to `TYPE_CHECKING` + function-local |
| xdsl/xdsl           | 3197 auto-opt (r1-b0)   | `xdsl_opt_main.py` — `importlib.metadata.version` deferred to `register_all_arguments` |
| xdsl/xdsl           | 3197 auto-opt (r0-b0)   | `dialect_loader.py` — `DialectStubGenerator` import moved into the one method that needs it |
| astropy/astropy     | #17461                  | `ConfigNamespace` — removed metaclass scan; defer item naming to first access |
| matplotlib/matplotlib| #18756                 | `dates.get_epoch` — defer parsing the epoch string |

## Watch-outs

* Function-local imports are **paid on every call** unless the import is
  cached by `sys.modules` (which it is for normal `import`). They're
  free after the first call, but make sure the *first* call isn't on a
  latency-critical request path.
* `TYPE_CHECKING`-guarded imports are invisible to runtime introspection:
  `inspect.get_annotations(func)` may fail at runtime. Pair with
  `from __future__ import annotations` (PEP 563) so annotations stay
  string-form.
* Lazy registration changes ordering semantics: code that relied on
  `entry_points` being populated at import time may break. Audit
  consumers that iterate over plugin registries.
* Deferring validation moves bugs to runtime. Ensure the deferred
  validation eventually runs — at least in the test suite.
* Some "lazy import" wrappers (e.g. wrapping the module in a class) are
  slower than the original eager import because of the extra
  attribute lookup; prefer plain function-local `import` over custom
  proxy types.
