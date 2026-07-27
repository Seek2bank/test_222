# Skill: Overhead Reduction

Use when the profile shows hot spots in **dispatch / lookup / type-check
machinery** rather than "real" compute. Symptoms:

* Top frames are `getattr`, `__getattribute__`, `isinstance`, `hasattr`,
  `_dispatch`, `lookup`, `_resolve`, decorator wrappers, `functools.wraps`.
* The hot function does very little arithmetic but is called millions of times.
* `cProfile` shows high `tottime` but low per-call cost — death by 1000 cuts.

## Recipe

1. **Cache the resolution result** at module import or instance construction
   instead of resolving on every call:
   ```python
   class X:
       def __init__(self):
           self._compute = self._select_compute()  # bind once
       def _select_compute(self):
           return _fast if has_numba else _slow
   ```
2. **Replace polymorphic dispatch with a fast path** for the dominant case,
   falling back to the generic version:
   ```python
   def step(x):
       if type(x) is np.ndarray:        # exact-type check is faster than isinstance
           return _fast(x)
       return _generic(x)
   ```
3. **Hoist invariants out of the hot loop** — `len()`, attribute reads,
   method binds:
   ```python
   # before
   for i in range(len(self.items)):
       self.items[i] = self.transform(self.items[i])
   # after
   items = self.items
   transform = self.transform
   for i in range(len(items)):
       items[i] = transform(items[i])
   ```
4. **Skip optional checks at runtime** if they were already validated at
   construction (gate behind `if __debug__:` or a class-level flag).
5. **Use `__slots__`** on small per-row objects that get created millions of
   times — kills `__dict__` overhead.

## What NOT to do

* Don't introduce C extensions or numba unless the existing repo already uses
  them — that's structural change, not overhead reduction.
* Don't replace `dict` with `__slots__` if the class is rarely instantiated —
  cost > benefit.
