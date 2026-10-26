# Parallel Forest Fire Spread Simulation — OpenMP, MPI, CUDA

PCAP mini project. A cellular-automaton forest-fire model implemented five
ways, with a browser UI to switch between them and compare performance:

| Mode | Binary | Source |
|---|---|---|
| Serial (baseline) | `build/fire_serial` | `src/fire_serial.c` |
| OpenMP | `build/fire_omp` | `src/fire_omp.c` |
| MPI | `build/fire_mpi` | `src/fire_mpi.c` |
| Hybrid MPI + OpenMP | `build/fire_hybrid` | `src/fire_mpi.c` built with `-fopenmp` |
| CUDA | `build/fire_cuda` | `src/fire_cuda.cu` |

The model itself (update rule and RNG) lives in **one** file,
`src/fire_core.h`, which every version includes, including the CUDA kernel.

## Running it

Everything runs inside **WSL Ubuntu** (gcc, OpenMPI, CUDA toolkit, Python).

**Easiest:** double-click `start_ui.bat`. It builds everything, starts the
UI, and opens http://localhost:8501.

**From a WSL terminal:**

```bash
cd /mnt/c/Users/risha/OneDrive/Desktop/PCAP_project
./run_ui.sh                 # build + launch UI
make                        # just build
~/.venvs/pcap/bin/python tests/check_equivalence.py   # verify all modes match serial
~/.venvs/pcap/bin/python benchmark.py   # full sweep -> results/metrics.csv
```

**Running a binary by hand:**

```bash
./build/fire_serial --size 2000 --steps 300 --fixed
OMP_NUM_THREADS=8 ./build/fire_omp --size 2000 --steps 300 --fixed
mpirun --oversubscribe --bind-to none -np 4 ./build/fire_mpi --size 2000 --steps 300 --fixed
mpirun --oversubscribe --bind-to none -np 2 -x OMP_NUM_THREADS=4 ./build/fire_hybrid --size 2000 --steps 300 --fixed
./build/fire_cuda --size 2000 --steps 300 --fixed --block 16
./build/fire_serial --help    # all options
```

Each program prints one `RESULT {...}` JSON line (time, steps, cell counts,
checksum, per-mode overheads). The UI and `benchmark.py` parse this line.

> If `mpirun` prints `Read -1, expected ..., errno = 1`, set
> `export OMPI_MCA_btl_vader_single_copy_mechanism=none` (WSL blocks the
> memory-copy mechanism OpenMPI uses by default). The UI sets it for you.

## The UI

- **Run one mode:** pick a mode and its settings (threads, processes,
  schedule, CUDA block size), run it, and watch the fire spread. Shows time,
  cell updates/s, the burning-cells curve, MPI per-rank load balance and
  communication share, and the CUDA kernel vs transfer breakdown.
- **Compare all modes:** runs every mode on the same forest (median of N
  repeats) and shows time, speedup, efficiency, and whether each final grid
  is **identical** to the serial one.
- **Scaling sweep:** OpenMP threads 1→12, MPI processes 1→8, or grid size
  across all modes. Plots speedup against ideal linear scaling and
  efficiency. Each table can be downloaded as CSV.

## Model

States: `0 empty`, `1 vegetation`, `2 burning`, `3 burned`. Moore
neighbourhood (8 neighbours). Burning → burned after one step. A vegetation
cell with burning neighbours ignites with

```
p_j  = clamp(p0 · (1 − H) · W_j, 0, 1)
P    = 1 − Π (1 − p_j)          over burning neighbours j
W_j  = 1 + I·cos θ   if cos θ ≥ 0      (θ = angle between spread direction and wind)
       1 + (I/2)·cos θ otherwise         → aligned 1+I, opposite 1−I/2, side 1
```

Diagonal neighbours get partial wind effect through `cos θ`. Wind direction
means *the direction the fire is pushed* (wind `E` → fire runs east).

**Defaults:** density 0.70, p0 0.65, humidity 0.20. These differ from the
synopsis baseline (0.65 / 0.35 / 0.30) on purpose: with the synopsis
values the effective spread chance per neighbour is only 0.245, and the
fire burns out within 5–80 steps for every seed tested. The model shows a
sharp **percolation threshold** at an effective spread chance of about 0.45:

| density | p0 | H | result (500×500, 5 seeds) |
|---|---|---|---|
| 0.65 | 0.35 | 0.30 | dies in 5–9 steps, 0% burned |
| 0.65 | 0.55 | 0.20 | mostly dies; 0–5% burned |
| 0.65 | 0.55 | 0.10 | 50–54% burned |
| 0.70 | 0.65 | 0.20 | 65–66% burned, ~400 steps |

This makes a good point for the report.

## Correctness: why all five modes give the *identical* grid

1. **Counter-based RNG.** The random number for a cell is
   `hash(seed, step, row, col)` (SplitMix64), not the next value from a
   shared generator. Thread order and process layout cannot change it. MPI
   ranks pass the **global** row index.
2. **Fixed-point probabilities.** p0, H and wind are folded on the host into
   a table `q[8]` of 16.16 fixed-point "survival" probabilities. The per-cell
   work is pure integer math, so the GPU (which fuses multiply-adds
   differently) gets bit-identical results.
3. **Double buffering.** Each step reads `cur` and writes `nxt`, so no cell
   ever sees a half-updated neighbourhood.
4. **Deterministic initial forest.** Each cell's starting state is also a
   hash of `(seed, row, col)`, so each MPI rank generates its own slice with
   no scatter, and it matches the serial grid.

`tests/check_equivalence.py` checks all of this: checksum, step count and the
full burning-cells history must match serial for every mode and config,
including grid sizes that don't divide evenly across ranks.

## Viva notes: where each concept is in the code

| Concept | Where |
|---|---|
| Loop parallelism, reduction, `schedule(runtime)` | `fire_omp.c`: `#pragma omp parallel for` |
| Implicit barrier between steps | end of that parallel-for |
| Domain decomposition (row blocks, uneven remainder) | `fire_mpi.c`: `base`, `rem`, `nr`, `r0` |
| Halo / ghost rows, `MPI_PROC_NULL` at edges | `fire_mpi.c`: two `MPI_Sendrecv` calls |
| Collective for global termination | `MPI_Allreduce` (skipped in `--fixed` mode) |
| Gathering results | `MPI_Gatherv`, `MPI_Reduce` (MAX time, SUM history) |
| `MPI_THREAD_FUNNELED` | `fire_mpi.c` hybrid build |
| One thread per cell, 2D grid of blocks | `fire_cuda.cu`: `fire_step_kernel` |
| Block-level reduction | `__syncthreads_count` + one `atomicAdd` per block |
| Pinned host memory, single H2D/D2H copy | `cudaMallocHost`, copies outside the loop |
| Kernel vs transfer timing | CUDA events |
| Same code on CPU and GPU | `FIRE_HD` macro → `__host__ __device__` |

## Measurement notes

- **Benchmark mode (`--fixed`)** runs every step even after the fire is out,
  so every mode does identical work. Use it for speedup numbers. Without it,
  MPI needs an `Allreduce` and CUDA a 4-byte device→host read every step to
  decide when to stop. That cost is real and worth discussing.
- **Timing excludes** grid generation and frame dumping. CUDA time is
  end-to-end (H2D + kernels + D2H). MPI time is the slowest rank.
- **This laptop** (i5-13420H) has 4 performance cores (8 threads) and 4
  efficiency cores, 12 hardware threads total. Expect OpenMP and MPI scaling to
  flatten past ~4–8 workers. `dynamic` scheduling can help when slow and
  fast cores are mixed.
- **MPI on one machine** communicates through shared memory, so halo
  exchange is much cheaper than on a real cluster. Say so in the report.
- **Small grids** (≤500×500) can be *slower* in parallel: barriers,
  messages and kernel launches cost more than the work they split.

## Layout

```
src/        fire_core.h  fire_common.h  fire_serial.c  fire_omp.c  fire_mpi.c  fire_cuda.cu
ui/         app.py (Streamlit UI)   runner.py (launches binaries, parses results)
tests/      check_equivalence.py
benchmark.py  Makefile  run_ui.sh  start_ui.bat  requirements.txt
```
