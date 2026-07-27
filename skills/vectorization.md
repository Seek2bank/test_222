# Skill: Vectorization

Use when the profile shows **explicit Python loops over array-shaped data**.
Symptoms:

* Top frames are `for_`, `range`, list `append`, element-wise indexing, or
  user-defined per-row functions called from within `for`.
* Repository already uses NumPy / pandas / xarray / scipy — vectorization is
  idiomatic.

## Recipe

1. **Replace per-element loops with NumPy ops** (broadcasting, fancy indexing):
   ```python
   # before
   out = []
   for x in arr:
       out.append(x * 2 + 1)
   # after
   out = arr * 2 + 1
   ```
2. **Use `np.where`, `np.select`, or boolean masks** for branchy loops.
3. **Pre-allocate** the output buffer instead of growing a list:
   ```python
   out = np.empty(n, dtype=arr.dtype)
   out[:] = arr * 2 + 1
   ```
4. **Batch operations across axes** with `axis=` instead of looping over rows
   in Python.
5. For pandas: prefer `df[col].map(...)` or vectorised `Series` arithmetic
   over `df.apply` row-by-row; use `Categorical` for repeated string keys.

## Watch outs

* `np.vectorize` is **not** vectorisation — it's a Python loop in disguise.
  Use it only as a stub for clarity, never for speed.
* Beware of dtype promotion (`float32 → float64`) bloating memory and slowing
  the run; pin dtypes explicitly.
* If the loop has an early break, vectorising can do unnecessary work — keep
  the loop unless the early break is rarely taken.
