# Numba nightly benchmarks

Lightweight **NumPy-vs-Numba** benchmarks with a **nightly regression gate**.
No ASV, no conda — just `numpy`, `numba`, and a couple of plain-Python scripts.

## Layout

```
bench/run.py       run the benchmarks, emit results.json
bench/compare.py   compare a run against the baseline, exit non-zero on regression
.github/workflows/nightly-benchmarks.yml   the nightly job
```

## Run locally

```bash
pip install numpy numba          # for numba main: pip install git+https://github.com/numba/numba.git@main
python bench/run.py --out results.json
python bench/compare.py --current results.json --baseline baseline.json
```

`bench/run.py` times a curated set of operations twice — once with NumPy and once
with an `@njit` kernel — takes the **min** of several samples, and records
`numpy_s`, `numba_s`, and their ratio per operation. `bench/compare.py` flags any
operation whose *numba* time grew by more than the threshold (default 15%).

## Nightly gate

`.github/workflows/nightly-benchmarks.yml` runs **once a night** (06:00 UTC), not
on merges to `main`. It:

1. builds and installs **numba `main`** from source,
2. runs the benchmarks (`bench/run.py`),
3. compares against the previous run stored on the **`bench-baseline` branch**
   (`bench/compare.py`), and
4. opens/updates an issue titled **"Nightly benchmark regression"** in this repo
   when an operation regresses.

The first run seeds the baseline (no comparison). Thresholds live in
`bench/compare.py` (`--threshold`). Add an operation by adding a
`(name, numpy_fn, numba_fn)` tuple in `bench/run.py:build_pairs()`.
