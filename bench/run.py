#!/usr/bin/env python3
"""Lightweight NumPy-vs-Numba benchmarks (no ASV, no conda).

Times a curated set of NumPy operations against their ``@njit`` equivalents and
writes a JSON document (stdout and/or --out). The nightly workflow compares it
against the previous run stored on the ``bench-baseline`` branch.

Usage:
    python bench/run.py [--out results.json] [--samples 7] [--target 0.02]
"""

import argparse
import json
import platform
import time

import numpy as np


N = 1_000_000       # 1-D element count for reductions / scan / misc
N_SORT = 200_000    # element count for sort / search


def build_pairs():
    """Return a list of (name, numpy_fn, numba_fn). Inputs are built once here
    (seeded, so runs are comparable) and the JIT is warmed by the callers."""
    from numba import njit

    rnd = np.random.RandomState(0)
    f = rnd.rand(N)                                    # float64
    i = rnd.randint(0, 1000, N).astype(np.int64)       # int64
    a = rnd.rand(N)
    b = rnd.rand(N)
    cond = a > 0.5
    idx = rnd.randint(0, 256, N).astype(np.int64)
    ca = rnd.rand(N)                                   # for cumsum
    sr = rnd.rand(N_SORT)                              # for sort/argsort
    sorted_a = np.sort(rnd.rand(N_SORT))               # for searchsorted
    v = rnd.rand(N_SORT)
    m1 = rnd.rand(256, 256)
    m2 = rnd.rand(256, 256)

    @njit
    def nb_sum(x):        return np.sum(x)

    @njit
    def nb_mean(x):       return np.mean(x)

    @njit
    def nb_std(x):        return np.std(x)

    @njit
    def nb_min(x):        return np.min(x)

    @njit
    def nb_max(x):        return np.max(x)

    @njit
    def nb_cumsum(x):     return np.cumsum(x)

    @njit
    def nb_sort(x):       return np.sort(x)

    @njit
    def nb_argsort(x):    return np.argsort(x)

    @njit
    def nb_searchsorted(x, y):
        return np.searchsorted(x, y)

    @njit
    def nb_dot(x, y):     return np.dot(x, y)

    @njit
    def nb_bincount(x):   return np.bincount(x)

    @njit
    def nb_where(c, x, y):
        return np.where(c, x, y)

    @njit
    def nb_histogram(x, bins):
        return np.histogram(x, bins)

    return [
        ("sum_f8",       lambda: np.sum(f),                lambda: nb_sum(f)),
        ("mean_f8",      lambda: np.mean(f),               lambda: nb_mean(f)),
        ("std_f8",       lambda: np.std(f),                lambda: nb_std(f)),
        ("min_f8",       lambda: np.min(f),                lambda: nb_min(f)),
        ("max_f8",       lambda: np.max(f),                lambda: nb_max(f)),
        ("sum_i8",       lambda: np.sum(i),                lambda: nb_sum(i)),
        ("cumsum",       lambda: np.cumsum(ca),            lambda: nb_cumsum(ca)),
        ("sort",         lambda: np.sort(sr),              lambda: nb_sort(sr)),
        ("argsort",      lambda: np.argsort(sr),           lambda: nb_argsort(sr)),
        ("searchsorted", lambda: np.searchsorted(sorted_a, v),
                         lambda: nb_searchsorted(sorted_a, v)),
        ("dot_256",      lambda: np.dot(m1, m2),           lambda: nb_dot(m1, m2)),
        ("bincount",     lambda: np.bincount(idx),         lambda: nb_bincount(idx)),
        ("where",        lambda: np.where(cond, a, b),     lambda: nb_where(cond, a, b)),
        ("histogram",    lambda: np.histogram(a, 100),     lambda: nb_histogram(a, 100)),
    ]


def time_min(fn, samples, target_s):
    """Warm up, auto-size an inner loop to ~target_s, and return the min of
    `samples` per-call times (min is the most stable microbenchmark statistic)."""
    fn()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    number = max(1, int(target_s / max(dt, 1e-9)))
    best = float("inf")
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write the JSON document here as well")
    ap.add_argument("--samples", type=int, default=7,
                    help="timed samples per benchmark (default 7)")
    ap.add_argument("--target", type=float, default=0.02,
                    help="target seconds per sample (default 0.02)")
    args = ap.parse_args()

    pairs = build_pairs()
    results = {}
    for name, npfn, nbfn in pairs:
        np_s = time_min(npfn, args.samples, args.target)
        nb_s = time_min(nbfn, args.samples, args.target)
        results[name] = {"numpy_s": np_s, "numba_s": nb_s,
                         "ratio": nb_s / np_s}

    import numba
    doc = {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "machine": platform.platform(),
        "samples": args.samples,
        "target_s": args.target,
        "results": results,
    }
    text = json.dumps(doc, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
