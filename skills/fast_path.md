# Skill: Contextual Fast-Path
*Park-Sujin principle: **Replace** — Contextualization.*

Use when the profile shows **a generic algorithm running on inputs that hit
a trivial special case** the great majority of the time. The cost lives in
machinery designed for the worst case, but the actual workload almost
never needs that machinery.

## Trigger

* The hot function takes a size/shape/`len` parameter that is overwhelmingly
  small (`n == 1`, `n == 2`, `len(inputs) == 1`, `degree <= 1`) inside the
  benchmark.
* `cProfile` shows time inside a polymorphic dispatcher
  (`isinstance` ladder, `singledispatch`) where one branch covers >95 % of
  calls; specialising it removes the dispatch overhead.
* The benchmark name encodes a degenerate shape: `time_loc_simple`,
  `time_single_input`, `time_empty`, `time_scalar`.
* A wrapper passes its input through unmodified before delegating to a
  costly generic implementation (e.g. wrapping a `SkyCoord` in another
  `SkyCoord`).

## Recipe

1. **Add an early-return for the trivial size at the *top* of the
   function**, before any setup:
   ```python
   # before
   def axis_aligned_extrema(self):
       n = self.degree
       Cj = self.polynomial_coefficients
       dCj = np.arange(1, n+1)[:, None] * Cj[1:]
       if len(dCj) == 0:
           return np.array([]), np.array([])
       ...

   # after
   def axis_aligned_extrema(self):
       n = self.degree
       if n <= 1:
           return np.array([]), np.array([])
       Cj = self.polynomial_coefficients
       dCj = np.arange(1, n+1)[:, None] * Cj[1:]
       ...
   ```

2. **Type-specialise the dominant case** so the generic constructor
   isn't re-invoked:
   ```python
   # before
   if value is None:
       return None, False
   elif isinstance(value, self._frame):
       return value, False
   else:
       return SkyCoord(value, frame=self._frame).frame, True   # heavy

   # after — recognise the wrapped-twice case explicitly
   if value is None:
       return None, False
   elif isinstance(value, SkyCoord) and isinstance(value.frame, self._frame):
       return value.frame, True            # cheap pass-through
   elif isinstance(value, self._frame):
       return value, False
   else:
       return SkyCoord(value, frame=self._frame).frame, True
   ```

3. **Bypass a graph/rewriter when only one input is present**:
   ```python
   def rewrite_blockwise(inputs):
       if len(inputs) == 1:
           # Fast path: nothing to fuse.
           return inputs[0]
       ...
   ```

4. **Replace a slow default sentinel** that triggers an expensive code
   path inside a library:
   ```python
   # before — passing mask=None makes ma.MaskedArray broadcast a None
   # through ~3 s of work for 1e7 elements.
   if mask is None and hasattr(data, 'mask'):
       mask = data.mask

   # after — None becomes a single-bool False, which broadcasts trivially.
   if mask is None:
       mask = data.mask if hasattr(data, 'mask') else False
   ```

5. **Branch on `loss_values.shape[0]`** for very small `n` in pairwise
   pareto/dominance loops:
   ```python
   if n_trials == 1:
       return np.array([True])
   if n_trials == 2:
       return np.array([True, (loss_values[1] < loss_values[0]).any()])
   ```

## Real-world examples

| Repo                | Issue            | Function                       | Speed-up |
|---------------------|------------------|--------------------------------|----------|
| dask/dask           | #5890            | `rewrite_blockwise`            | single-input fast path |
| matplotlib/matplotlib| #17994          | `BezierSegment.axis_aligned_extrema` | n<=1 short-circuit |
| astropy/astropy     | #13471           | `attributes.convert_input`     | SkyCoord pass-through |
| astropy/astropy     | #7422            | `MaskedColumn.__new__`         | 10× on `mask=None` |
| optuna/optuna       | #6223 (r1-b1)    | `_is_pareto_front_nd`          | n==1, n==2 |

## Watch-outs

* Fast paths must not change return *types*: a path that returns a list
  where the slow path returns an ndarray will break callers that index
  with multi-dim slices. Mirror the slow path's container exactly.
* If the trigger condition is approximate (e.g. `if n <= 1` for a
  benchmark whose distribution is `n in [0, 1, 2]`), profile both
  branches; sometimes the trivial case is also rare and the if-check
  itself becomes the dominant cost.
* Type-specialised paths can mask bugs in the generic path because the
  generic path stops being exercised by the benchmark suite. Make sure
  there is a non-benchmark test that still hits the generic branch.
* Fast paths multiply: adding a 4th branch to a 3-branch `isinstance`
  ladder grows linear dispatch time. Profile post-change to confirm the
  ladder didn't regress for the *other* cases.
