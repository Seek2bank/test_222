# Skill: Algorithmic

Use when the profile shows **time growing with input size in a way that
suggests a worse-than-necessary asymptotic class**. Symptoms:

* Top frames are `find`, `index`, `count`, `sort`, `unique`, `dedup`, graph
  traversals, repeated linear scans, nested loops, `O(n²)` joins.
* Benchmark name has a parameter like `n=1000, n=10000` and runtime grows
  super-linearly.

## Recipe

1. **Replace `list.index()` / `in list` with a `set`/`dict` lookup**
   (`O(n)` → `O(1)`).
2. **Hash-join instead of nested loop**:
   ```python
   # before: O(n*m)
   for x in xs:
       for y in ys:
           if y.key == x.key: ...
   # after: O(n+m)
   index = {y.key: y for y in ys}
   for x in xs:
       y = index.get(x.key)
   ```
3. **Sorted-arg + `bisect` for ordered lookups** (`O(log n)`).
4. **`set` for membership / dedup** instead of `list` + `if x not in list`.
5. **Use `collections.Counter` / `defaultdict(int)`** instead of building
   counts in a loop with `if k in d: d[k] += 1 else: d[k] = 1`.
6. **Replace recursion with iteration / DP** when the same subproblem repeats
   (memoize, see also caching skill).

## Watch outs

* Constant factors matter: a "worse" big-O may win on the small inputs the
  benchmark actually exercises. Check the benchmark's parameter range first
  (`asv_bench/benchmarks/<file>.py`).
* Switching data structure may break tests that assume ordering — check the
  pytest output carefully.
