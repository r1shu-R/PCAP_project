# Project context (for resuming with Claude)

> **To resume:** open a Claude Code session in this folder and say
> "read context.md and continue". Last updated: 2026-10-22.

## What this is

PCAP (Parallel Computing Architecture & Programming) **lab end-sem mini project** at MIT
Manipal. Team: Myisha Siddiqui (58), Manan Bhasin (53), Rishabh Raj (47).
The synopsis PDF is at `C:\Users\risha\Downloads\PCAP_Forest_Fire_Synopsis.pdf`.

Topic: grid-based forest-fire cellular automaton. Same model implemented as **Serial, OpenMP,
MPI, Hybrid MPI+OpenMP, CUDA**, plus a **Streamlit UI** to switch modes, run, animate, and
compare performance. The user wanted it to run on their own laptop with a basic UI.

## Status: working end to end ✅

- All 5 binaries build and run (`make` inside WSL).
- `tests/check_equivalence.py`: **44/44 PASS**. Every mode gives a bit-identical final grid to
  serial (checksum + step count + burning history), across odd sizes, uneven MPI splits, wind,
  and fixed/early-stop modes.
- UI tested in the browser: Run tab (MPI, CUDA), Compare tab all work. The Scaling tab is
  built but was **not clicked through in the browser** yet. Test it first next time.
- `benchmark.py` works and writes `results/metrics.csv`.

## Environment (decided and installed)

- Laptop: i5-13420H (8 cores / 12 threads, 4P+4E cores), RTX 4050 Laptop GPU, 16 GB RAM.
- **Everything runs in WSL2 Ubuntu 24.04**, not native Windows (Windows has no MSVC for nvcc
  and no MPI).
  - gcc 13.3, OpenMPI 4.1.6 (were already present)
  - CUDA toolkit **13.3** installed via NVIDIA's wsl-ubuntu apt repo → `/usr/local/cuda`
  - Python venv at `~/.venvs/pcap` (streamlit 1.66, numpy, pandas, pillow, altair)
  - WSL user has passwordless sudo.
- Project lives at `C:\Users\risha\OneDrive\Desktop\PCAP_project`
  (= `/mnt/c/Users/risha/OneDrive/Desktop/PCAP_project` in WSL). Not a git repo yet.

## How to run

- Double-click `start_ui.bat` → builds, starts Streamlit, opens http://localhost:8501
- Or in WSL: `./run_ui.sh`
- Tests: `~/.venvs/pcap/bin/python tests/check_equivalence.py`
- Benchmarks: `~/.venvs/pcap/bin/python benchmark.py [--sizes 500 1000 --steps 100 --repeats 3]`

## Architecture (key design decisions)

- `src/fire_core.h`: **the only copy of the model** (`fire_cell`, `fire_hash`). It's
  `__host__ __device__` under nvcc, so the CPU and GPU share the same code.
- `src/fire_common.h`: CLI parsing, grid init, frames file, checksum, `RESULT {json}` output.
- `src/fire_mpi.c` is compiled **twice**: without `-fopenmp` → `fire_mpi`, with it →
  `fire_hybrid`.
- UI (`ui/app.py`) does not link the C code. It launches the binaries through `ui/runner.py`
  (mpirun for MPI/hybrid) and parses the `RESULT` JSON line. Frames go to a binary file in /tmp
  and are turned into a GIF.
- Determinism: counter-based RNG `hash(seed, step, row, col)` + **16.16 fixed-point**
  probabilities (q[8] table built on the host) → CPU and GPU produce bit-identical results.
- MPI: horizontal row blocks, 1 halo row above and below, two `MPI_Sendrecv` per step,
  `MPI_Allreduce` only in early-stop mode. Each rank generates its own slice from the hash (no
  scatter).
- CUDA: 1 thread per cell, 2D BxB blocks, `__syncthreads_count` + one `atomicAdd` per block
  for burning counts, pinned host memory, one copy in and one copy out, CUDA events for timing.
- Wind: dot-product weighting, so diagonals get a partial effect. 8 directions + none.
  "Wind E" = fire pushed east.
- `--fixed` (UI: "Benchmark mode") runs all steps → equal work for fair speedups. Scaling
  sweeps always force it.
- mpirun flags: `--oversubscribe --bind-to none` (without `--bind-to none`, hybrid threads all
  get pinned to one core), plus env `OMPI_MCA_btl_vader_single_copy_mechanism=none` (WSL needs
  it).

## Findings worth putting in the report / viva

1. **The synopsis baseline params make the fire die out.** p0=0.35, H=0.30, density 0.65 →
   effective spread 0.245 → burns out in 5–80 steps for every seed. **Defaults changed to
   p0 0.65, H 0.20, density 0.70** (~66% burned, ~400 steps on 500×500). There's a sharp
   percolation threshold at effective spread ≈ 0.45 (data table in README).
   → **The synopsis table still needs updating to these values.**
2. MPI on a 500×500 grid: ~44% of the run time is communication (halo exchange + per-step
   Allreduce).
3. CUDA in early-stop mode: kernels 32 ms vs ~63 ms of overhead (the per-step 4-byte
   device→host "still burning?" check forces a sync). Benchmark mode removes it.
4. Benchmark (100 steps, **1 repeat only, noisy**). Speedup vs serial:

   | Grid | OpenMP 8T | MPI 8P | Hybrid 2×4 | CUDA |
   |---|---|---|---|---|
   | 1000² | 2.4× | 3.9× | 0.67× (outlier) | 44× |
   | 2000² | 4.0× | 5.4× | 6.4× | 94× |
   | 5000² | 3.0× | 4.7× | 5.0× | 91× |

   Serial at 5000² ≈ 10.7 s. CPU scaling flattens past ~4–8 workers (P+E core mix). Rerun
   with `--repeats 3+`, plugged in, for the final numbers.

## Possible next steps (not started)

- [ ] Click through the Scaling sweep tab in the browser and check it works.
- [ ] Run the full `benchmark.py` (default sizes 500–5000, 300 steps, 3 repeats) for report data.
- [ ] Update the synopsis parameter table and baseline values.
- [ ] Optional extras for an "optimized vs naive" comparison: CUDA shared-memory tiling,
      non-blocking MPI (`MPI_Isend/Irecv`) to overlap halo exchange with interior compute,
      2D MPI decomposition, comparing OpenMP `dynamic` vs `static` on P+E cores.
- [ ] Optional: generate report graphs from `results/metrics.csv`, `git init` the folder.
- [ ] Write the report and presentation.

## User preferences noticed

- Casual tone and wants direct opinions. Builds on their own laptop. Likes things explained
  for the viva (which concept lives where in the code).
