"""Full benchmark sweep for the report -> results/metrics.csv

Runs the grid sizes and configurations from the synopsis's evaluation table
in benchmark mode (fixed step count) and records time, speedup, efficiency
and overheads for every run.

    ~/.venvs/pcap/bin/python benchmark.py                 # full sweep
    ~/.venvs/pcap/bin/python benchmark.py --sizes 500 1000 --steps 100 --repeats 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ui"))
import runner as R  # noqa: E402

PLAN = [
    ("serial", R.ModeConfig()),
    *[("openmp", R.ModeConfig(threads=t)) for t in (1, 2, 4, 8)],
    *[("mpi", R.ModeConfig(procs=p)) for p in (1, 2, 4, 8)],
    *[("hybrid", R.ModeConfig(procs=p, threads=t)) for p, t in ((2, 2), (2, 4), (4, 2))],
    ("cuda", R.ModeConfig(block=16)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    available = R.available_modes()
    plan = [(m, c) for m, c in PLAN if m in available]
    skipped = sorted({m for m, _ in PLAN} - set(available))
    if skipped:
        print(f"skipping (not built): {', '.join(skipped)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["grid", "steps", "mode", "config", "pes", "median_time_s", "speedup", "efficiency",
              "cell_updates_per_s", "comm_s", "kernel_s", "h2d_s", "d2h_s", "checksum", "matches_serial"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for size in args.sizes:
            params = R.SimParams(size=size, steps=args.steps, fixed=True)
            serial = None
            for mode, cfg in plan:
                r = R.run_median(mode, params, cfg, args.repeats)
                if mode == "serial":
                    serial = r
                speedup = serial["median_time_s"] / r["median_time_s"]
                row = {
                    "grid": size, "steps": r["steps_run"], "mode": mode, "config": r["config"],
                    "pes": r["pes"] or "", "median_time_s": f"{r['median_time_s']:.6f}",
                    "speedup": f"{speedup:.3f}",
                    "efficiency": f"{speedup / r['pes']:.3f}" if r["pes"] else "",
                    "cell_updates_per_s": f"{r['cell_updates_per_s']:.4e}",
                    "comm_s": r.get("comm_s", ""), "kernel_s": r.get("kernel_s", ""),
                    "h2d_s": r.get("h2d_s", ""), "d2h_s": r.get("d2h_s", ""),
                    "checksum": r["checksum"], "matches_serial": r["checksum"] == serial["checksum"],
                }
                w.writerow(row)
                f.flush()
                print(f"{size:>5}  {R.LABELS[mode]:<18} {r['config']:<22} "
                      f"{r['median_time_s'] * 1000:10.2f} ms  {speedup:6.2f}x  "
                      f"{'ok' if row['matches_serial'] else 'MISMATCH'}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
