# Skill: Redundant-Work Removal
*Park-Sujin principle: **Remove** — Task removal.*

Use when the profile shows **the same expensive call invoked twice with
identical (or mutually-derivable) inputs**, or an op that is mathematically
a no-op for the actual data shape but is still being executed.

The dominant Park-Sujin pattern in real perf bug reports: the work didn't
need caching, deferring, or vectorising — it shouldn't have happened at
all.

## Trigger

* `cProfile` shows `ncalls > 1` for an expensive frame whose arguments are
  determined by the same upstream parameter (e.g. `epv00` called once via
  `get_body_barycentric_posvel('earth')` and again via
  `get_body_barycentric('sun')` inside the same caller).
* Two callees of the hot function read different but **derivable** halves
  of the same internal product (position vs. velocity from one ERFA call,
  numerator vs. denominator from one decomposition).
* A loop calls a transform like `astype(...)`, `np.array(x)`, `tuple(x)`,
  `x[..., None][..., 0]` whose effect is identity when the input already
  has the target shape/dtype — but the unconditional branch still runs.
* The `tottime` of a function is dominated by `list.append` or
  `dict.update` of items that have already been inserted in an earlier
  iteration (visible from the `ncalls`/argument trace).
* Defensive `deepcopy(None)` / `deepcopy(default)` on hot paths where the
  default itself is immutable.

## Recipe

1. **Coalesce two derived calls into one upstream call**:
   ```python
   # before
   earth_p, earth_v = get_body_barycentric_posvel('earth', time)  # calls epv00
   sun              = get_body_barycentric('sun', time)            # calls epv00 again
   earth_h = (earth_p - sun).get_xyz(xyz_axis=-1).to_value(u.au)

   # after — use the joint primitive that returns both halves
   jd1, jd2 = get_jd12(time, 'tdb')
   earth_pv_heliocentric, earth_pv = erfa.epv00(jd1, jd2)
   earth_h = earth_pv_heliocentric['p']
   ```
   Look for the "joint" primitive a level below the two convenience wrappers.

2. **Skip identity transforms** with an early-return guard:
   ```python
   # before
   def atleast_nd(x, ndim):
       x = asanyarray(x)
       diff = max(ndim - x.ndim, 0)
       return x[(None,) * diff + (Ellipsis,)]       # rewraps even when diff == 0

   # after
   def atleast_nd(x, ndim):
       x = asanyarray(x)
       diff = max(ndim - x.ndim, 0)
       if diff == 0:
           return x
       return x[(None,) * diff + (Ellipsis,)]
   ```

3. **Dedupe iterable mutations** when the inputs are guaranteed-shared:
   ```python
   # before — same sub-dict appears multiple times in `d.dicts.values()`
   result = {}
   for dd in d.dicts.values():
       result.update(dd)

   # after — skip already-merged dicts by identity
   result = {}
   seen = set()
   for dd in d.dicts.values():
       if id(dd) not in seen:
           result.update(dd)
           seen.add(id(dd))
   ```

4. **Avoid defensive copies** when the call site does not mutate the
   argument:
   ```python
   # before — copies the Angle even though the wrap only reads it
   self._wrap_angle = Angle(value)

   # after
   self._wrap_angle = Angle(value, copy=False)
   ```

5. **Replace `mask=None` (which triggers expensive broadcast inside
   `MaskedArray`) with `mask=False`** when the array is genuinely
   un-masked.

## Real-world examples

| Repo                | Issue           | Function                       | Speed-up |
|---------------------|-----------------|--------------------------------|----------|
| astropy/astropy     | #10814          | `prepare_earth_position_vel`   | +50 %    |
| dask/dask           | #5501           | `ensure_dict`                  | sharp tail|
| dask/dask           | #5884           | `block.atleast_nd`             | small but ubiquitous |
| astropy/astropy     | #7616           | `Longitude.wrap_angle.setter`  | +40 %    |
| astropy/astropy     | #7422           | `MaskedColumn.__new__`         | 10× (large arrays)|
| numpy/numpy         | #12575          | `find_duplicate`               | O(n²)→O(n)|

## Watch-outs

* Coalescing two calls into one shared primitive **changes argument
  shape** for downstream callers — re-verify dtype/units of the merged
  output. `pav2pv` vs. `epv00` structured dtype in #10814 is the
  canonical gotcha.
* Identity-fast-paths must not change ndim/dtype semantics: callers
  that expected the wrapped view (read-only) get the original (writable)
  back. Inspect `__array_finalize__` consumers.
* `id()`-based dedup only works for **identity** of the container, not
  for structural equality. Don't reuse this trick for value-equal but
  separately constructed dicts — that's a caching skill, not removal.
* `copy=False` is only safe when no downstream callsite mutates the
  shared buffer in-place. Audit `__iadd__`, `out=` parameters, and
  `setattr`-style writes.
