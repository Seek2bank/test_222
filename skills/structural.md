# Skill: Structural / Data-Layout / IO Batching

Use when the profile shows **expensive per-row or per-call IO / serialisation
/ allocation / coordinate transforms**. Symptoms:

* Top frames are `read`, `write`, `flush`, `seek`, `open`, `close`, HDF5
  / netCDF / Parquet / CSV, `_make_ndarray`, `astype`, `_concat`.
* `open()` / `close()` called many times inside the hot loop.
* Repeated `pd.concat` / `np.append` / list-of-lists → eventual flat array.

## Recipe

1. **Batch the IO**:
   ```python
   # before
   for row in rows:
       with open(path, "a") as f:
           f.write(serialise(row))
   # after
   buf = io.StringIO()
   for row in rows:
       buf.write(serialise(row))
   with open(path, "a") as f:
       f.write(buf.getvalue())
   ```
2. **Read once, slice many** instead of re-opening files per row.
3. **Avoid repeated `pd.concat`** in a loop (`O(n²)`); collect to a list and
   concat once at the end, or pre-allocate a NumPy array.
4. **Pin dtypes** at construction so downstream ops don't re-allocate to
   widen them.
5. **Stream + chunk** for very large datasets — process `N` rows at a time
   instead of holding everything in memory.
6. **Prefer `numpy`/`pandas` Cython kernels** (e.g. `df.groupby(...).agg`)
   over Python loops over rows.

## Watch outs

* Batched IO can change error-recovery semantics — make sure the test suite
  still passes (especially failure-mode tests).
* Pinning dtypes can break code that expected a widened dtype downstream;
  scan callers.
