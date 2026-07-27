# Skill: Caching / Memoization

Use when the profile shows **the same expensive computation repeated** with
the same inputs. Symptoms:

* Top frames include `parse`, `compile`, `load`, `open`, `re.compile`, JSON
  decoding, schema validation, file-existence checks.
* Same function called with the same args many times within one benchmark
  invocation (visible from `ncalls` in `cProfile`).

## Recipe

1. **`functools.lru_cache`** on small, hashable-arg helper functions:
   ```python
   from functools import lru_cache
   @lru_cache(maxsize=512)
   def _parse_spec(spec_str: str) -> Spec:
       ...
   ```
2. **Module-level constant tables** instead of building dicts on each call.
3. **Compile regexes once** at module import:
   ```python
   _RE_FOO = re.compile(r"...")
   def find(s): return _RE_FOO.findall(s)
   ```
4. **Cache derived attributes** on objects, invalidate on mutation:
   ```python
   class X:
       @property
       def shape_signature(self):
           if self._sig is None:
               self._sig = self._compute_sig()
           return self._sig
       def _mutate(self):
           self._sig = None
   ```
5. **Memoize per-instance, not globally** when arguments include large
   objects — use `WeakKeyDictionary`.

## Watch outs

* `lru_cache` arguments must be hashable. NumPy arrays / dicts won't work.
* Make sure the cache is **invalidated** if the input data is mutated;
  otherwise correctness regressions appear in pytest.
* `lru_cache(maxsize=None)` (unbounded) leaks memory across calls — pick a
  finite cap.
