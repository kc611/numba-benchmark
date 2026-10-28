#!/usr/bin/env python3
"""Compare a `bench/run.py` result against a stored baseline.

Flags a regression when a benchmark's *numba* time grows by more than the
threshold relative to the baseline (default 15%). Missing baseline => seed run
(success). Prints a Markdown report and exits 1 on any regression.

Usage:
    python bench/compare.py --current results.json --baseline baseline.json [--out regressions.md]
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.current) as fh:
        cur = json.load(fh)

    base = None
    if args.baseline:
        try:
            with open(args.baseline) as fh:
                base = json.load(fh)
        except FileNotFoundError:
            base = None

    if not base or not base.get("results"):
        report = ("# Nightly benchmark comparison\n\n"
                  "No baseline yet (seed run). This run becomes the baseline.\n")
        if args.out:
            open(args.out, "w").write(report)
        print(report)
        return 0

    lines = [
        "# Nightly benchmark comparison",
        "",
        (f"baseline numba `{base.get('numba', '?')}` -> "
         f"current numba `{cur.get('numba', '?')}`; "
         f"threshold {args.threshold:.0%} on numba time"),
        "",
        "| benchmark | numba base | numba cur | ratio | numpy cur | verdict |",
        "|---|---|---|---|---|---|",
    ]
    regressed = []
    for name in sorted(cur.get("results", {})):
        c = cur["results"][name]
        b = (base.get("results") or {}).get(name)
        np_us = c["numpy_s"] * 1e6
        if not b:
            lines.append(f"| {name} | - | {c['numba_s'] * 1e6:.1f}us | - | "
                         f"{np_us:.1f}us | new |")
            continue
        ratio = c["numba_s"] / b["numba_s"]
        if ratio > 1 + args.threshold:
            verdict = "REGRESSED"
            regressed.append(name)
        else:
            verdict = "ok"
        lines.append(f"| {name} | {b['numba_s'] * 1e6:.1f}us | "
                     f"{c['numba_s'] * 1e6:.1f}us | {ratio:.3f} | "
                     f"{np_us:.1f}us | {verdict} |")

    report = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(report)
    print(report)

    if regressed:
        print(f"{len(regressed)} regression(s): {', '.join(regressed)}")
        return 1
    print("No regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
