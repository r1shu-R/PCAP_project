"""Check that every built mode produces exactly the serial result.

    ~/.venvs/pcap/bin/python tests/check_equivalence.py

Runs several parameter sets (wind directions, odd grid sizes that don't
divide evenly across ranks, early-stop and fixed-step modes) and compares
checksum, steps run and the burning-cells history against the serial run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))
import runner as R  # noqa: E402

CASES = [
    R.SimParams(size=200, steps=400),
    R.SimParams(size=257, steps=150, wind="NE", intensity=0.9, seed=7),
    R.SimParams(size=301, steps=120, wind="W", intensity=0.4, fixed=True, seed=3),
    R.SimParams(size=500, steps=60, density=0.65, p0=0.35, humidity=0.3),   # fire dies early
]
CONFIGS = {
    "openmp": [R.ModeConfig(threads=1), R.ModeConfig(threads=3, schedule="dynamic"), R.ModeConfig(threads=8)],
    "mpi": [R.ModeConfig(procs=1), R.ModeConfig(procs=3), R.ModeConfig(procs=7)],
    "hybrid": [R.ModeConfig(procs=2, threads=3), R.ModeConfig(procs=3, threads=2)],
    "cuda": [R.ModeConfig(block=8), R.ModeConfig(block=16), R.ModeConfig(block=32)],
}

failures = 0
for p in CASES:
    ref = R.run("serial", p)
    print(f"size={p.size} wind={p.wind} fixed={p.fixed}: serial {ref['steps_run']} steps, "
          f"checksum {ref['checksum']}")
    for mode in R.available_modes():
        for cfg in CONFIGS.get(mode, []):
            r = R.run(mode, p, cfg)
            ok = (r["checksum"] == ref["checksum"] and r["steps_run"] == ref["steps_run"]
                  and r["burning_history"] == ref["burning_history"])
            failures += not ok
            print(f"   {'PASS' if ok else 'FAIL'}  {R.LABELS[mode]:<18} {cfg.describe(mode)}")

print("\nall modes match serial" if not failures else f"\n{failures} FAILURES")
sys.exit(1 if failures else 0)
