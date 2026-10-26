"""Launch the compiled simulators and parse their results.

Shared by the Streamlit UI (app.py) and the command-line sweep (benchmark.py).
Each simulator prints one line `RESULT {json}`; this module builds the right
command line for each mode (mpirun for MPI/hybrid, env vars for OpenMP),
runs it, and returns the parsed dict.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

MODES = ["serial", "openmp", "mpi", "hybrid", "cuda"]
LABELS = {
    "serial": "Serial",
    "openmp": "OpenMP",
    "mpi": "MPI",
    "hybrid": "Hybrid MPI+OpenMP",
    "cuda": "CUDA",
}
BINARIES = {
    "serial": "fire_serial",
    "openmp": "fire_omp",
    "mpi": "fire_mpi",
    "hybrid": "fire_hybrid",
    "cuda": "fire_cuda",
}

# --oversubscribe: allow more ranks than OpenMPI thinks we have cores.
# --bind-to none: otherwise each rank is pinned to one core and a hybrid
#                 rank's OpenMP threads all fight over that single core.
MPI_FLAGS = ["--oversubscribe", "--bind-to", "none"]


@dataclass
class SimParams:
    size: int = 500
    steps: int = 500
    seed: int = 42
    density: float = 0.70
    p0: float = 0.65
    humidity: float = 0.20
    wind: str = "none"
    intensity: float = 0.5
    fixed: bool = False

    def args(self) -> list[str]:
        a = [
            "--size", str(self.size), "--steps", str(self.steps), "--seed", str(self.seed),
            "--density", f"{self.density:g}", "--p0", f"{self.p0:g}",
            "--humidity", f"{self.humidity:g}", "--wind", self.wind,
            "--intensity", f"{self.intensity:g}",
        ]
        if self.fixed:
            a.append("--fixed")
        return a


@dataclass
class ModeConfig:
    threads: int = 1        # OpenMP threads (openmp, hybrid)
    procs: int = 1          # MPI processes (mpi, hybrid)
    block: int = 16         # CUDA block edge
    schedule: str = "static"

    def describe(self, mode: str) -> str:
        if mode == "openmp":
            return f"{self.threads} threads ({self.schedule})"
        if mode == "mpi":
            return f"{self.procs} processes"
        if mode == "hybrid":
            return f"{self.procs} procs × {self.threads} threads"
        if mode == "cuda":
            return f"{self.block}×{self.block} blocks"
        return "1 thread"

    def processing_elements(self, mode: str) -> int | None:
        """CPU cores used, for efficiency = speedup / PEs (None for GPU)."""
        return {
            "serial": 1, "openmp": self.threads, "mpi": self.procs,
            "hybrid": self.procs * self.threads, "cuda": None,
        }[mode]


class RunError(RuntimeError):
    pass


def available_modes() -> list[str]:
    return [m for m in MODES if (BUILD / BINARIES[m]).is_file()]


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_command(mode: str, params: SimParams, cfg: ModeConfig,
                  frames: str | None = None, every: int = 0) -> tuple[list[str], dict]:
    exe = str(BUILD / BINARIES[mode])
    sim = params.args()
    if frames:
        sim += ["--frames", frames, "--every", str(every)]

    env = dict(os.environ)
    # WSL blocks the cross-process memory copy OpenMPI uses by default.
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    env["OMP_SCHEDULE"] = cfg.schedule

    if mode == "serial":
        cmd = [exe] + sim
    elif mode == "openmp":
        env["OMP_NUM_THREADS"] = str(cfg.threads)
        cmd = [exe, "--threads", str(cfg.threads)] + sim
    elif mode == "mpi":
        env["OMP_NUM_THREADS"] = "1"
        cmd = ["mpirun", *MPI_FLAGS, "-np", str(cfg.procs), exe] + sim
    elif mode == "hybrid":
        env["OMP_NUM_THREADS"] = str(cfg.threads)
        cmd = ["mpirun", *MPI_FLAGS, "-x", "OMP_NUM_THREADS", "-x", "OMP_SCHEDULE",
               "-np", str(cfg.procs), exe, "--threads", str(cfg.threads)] + sim
    elif mode == "cuda":
        cmd = [exe, "--block", str(cfg.block)] + sim
    else:
        raise ValueError(f"unknown mode {mode}")
    return cmd, env


def run(mode: str, params: SimParams, cfg: ModeConfig | None = None,
        frames: str | None = None, every: int = 0, timeout: float = 900) -> dict:
    """Run one simulation and return its parsed RESULT (plus `command`)."""
    cfg = cfg or ModeConfig()
    cmd, env = build_command(mode, params, cfg, frames, every)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=ROOT)
    except FileNotFoundError as e:
        raise RunError(f"Could not start {cmd[0]}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise RunError(f"{LABELS[mode]} timed out after {timeout:.0f}s") from e

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
            result["command"] = " ".join(cmd)
            result["config"] = cfg.describe(mode)
            result["pes"] = cfg.processing_elements(mode)
            return result
    raise RunError(
        f"{LABELS[mode]} failed (exit {proc.returncode}).\n"
        f"command: {' '.join(cmd)}\n{proc.stderr.strip() or proc.stdout.strip()}"
    )


def run_median(mode: str, params: SimParams, cfg: ModeConfig, repeats: int) -> dict:
    """Run `repeats` times and return the run with the median time."""
    results = [run(mode, params, cfg) for _ in range(max(1, repeats))]
    results.sort(key=lambda r: r["time_s"])
    best = results[len(results) // 2]
    best["all_times"] = [r["time_s"] for r in results]
    best["median_time_s"] = statistics.median(best["all_times"])
    return best


# ---------------------------------------------------------------- frames ---

# Display priority when shrinking a big grid: a burning cell must survive
# downsampling or a thin fire front would vanish.
_PRIORITY = np.array([0, 1, 3, 2], dtype=np.uint8)   # state -> priority
_FROM_PRIORITY = np.array([0, 1, 3, 2], dtype=np.uint8)  # priority -> state

# empty, vegetation, burning, burned
PALETTE = np.array([
    [196, 178, 140],   # empty ground - sand
    [46, 125, 50],     # vegetation - forest green
    [255, 111, 0],     # burning - orange
    [48, 42, 40],      # burned - charcoal
], dtype=np.uint8)


def downsample(grid: np.ndarray, max_px: int) -> np.ndarray:
    h, w = grid.shape
    k = max(1, int(np.ceil(max(h, w) / max_px)))
    if k == 1:
        return grid
    pri = _PRIORITY[grid]
    H, W = -(-h // k) * k, -(-w // k) * k
    padded = np.zeros((H, W), dtype=np.uint8)
    padded[:h, :w] = pri
    pooled = padded.reshape(H // k, k, W // k, k).max(axis=(1, 3))
    return _FROM_PRIORITY[pooled]


def read_frames(path: str, max_px: int = 500) -> tuple[list[int], list[np.ndarray]]:
    """Read a frame file written by --frames, downsampling each frame."""
    steps, frames = [], []
    with open(path, "rb") as f:
        if f.read(4) != b"FFR1":
            raise RunError("bad frame file")
        rows, cols = struct.unpack("<ii", f.read(8))
        n = rows * cols
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (step,) = struct.unpack("<i", head)
            data = f.read(n)
            if len(data) < n:
                break
            grid = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
            steps.append(step)
            frames.append(downsample(grid, max_px))
    return steps, frames


def to_rgb(grid: np.ndarray, min_px: int = 400) -> np.ndarray:
    rgb = PALETTE[grid]
    scale = max(1, min_px // max(grid.shape))
    if scale > 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    return rgb


def frames_to_gif(frames: list[np.ndarray], ms_per_frame: int = 70) -> bytes:
    from PIL import Image

    images = [Image.fromarray(to_rgb(f)) for f in frames]
    buf = io.BytesIO()
    durations = [ms_per_frame] * len(images)
    durations[-1] = 1500   # hold the final state
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0)
    return buf.getvalue()


def frame_png(grid: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(to_rgb(grid)).save(buf, format="PNG")
    return buf.getvalue()
