"""Streamlit front-end for the parallel forest-fire simulator.

Run with ./run_ui.sh (or start_ui.bat from Windows). Every button here just
launches one of the compiled programs in build/ via runner.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R  # noqa: E402

st.set_page_config(page_title="Parallel Forest Fire", page_icon="🔥", layout="wide")

CPU_THREADS = os.cpu_count() or 4
# Fixed colour per mode (never re-ordered when a mode is filtered out).
MODE_COLORS = {"serial": "#2a78d6", "openmp": "#eb6834", "mpi": "#1baf7a",
               "hybrid": "#eda100", "cuda": "#e87ba4"}
COLOR_SCALE = alt.Scale(domain=[R.LABELS[m] for m in R.MODES],
                        range=[MODE_COLORS[m] for m in R.MODES])
WINDS = ["none", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@st.cache_data(show_spinner=False)
def gpu_name():
    return R.gpu_name()


def label(mode: str) -> str:
    return R.LABELS[mode]


def fmt_time(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.3f} s"


def show_error(e: Exception):
    st.error("Run failed")
    st.code(str(e))


# ------------------------------------------------------------- sidebar ----
st.title("🔥 Parallel Forest Fire Simulator")
st.caption("One cellular-automaton model, five execution modes: Serial · OpenMP · MPI · "
           "Hybrid MPI+OpenMP · CUDA. Same seed, so every mode should produce the same final forest.")

available = R.available_modes()
missing = [m for m in R.MODES if m not in available]
if not available:
    st.error("No simulators are built yet. Run `make` in the project folder.")
    st.stop()
if missing:
    st.warning(f"Not built: {', '.join(label(m) for m in missing)}. Run `make` to build them.")

with st.sidebar:
    st.header("Forest")
    size = st.select_slider("Grid size (N × N)", options=[100, 200, 500, 1000, 2000, 5000], value=500)
    steps = st.number_input("Max time steps", min_value=50, max_value=5000, value=500, step=50)
    fixed = st.checkbox(
        "Benchmark mode: always run all steps", value=False,
        help="Off: stop as soon as the fire burns out. On: run every step even after the fire is "
             "out, so each run does the same amount of work. Use On when measuring speedup.")
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42)
    density = st.slider("Vegetation density", 0.0, 1.0, 0.70, 0.01)

    st.header("Fire")
    p0 = st.slider("Base ignition probability p0", 0.0, 1.0, 0.65, 0.01)
    humidity = st.slider("Humidity H", 0.0, 1.0, 0.20, 0.01)
    wind = st.selectbox("Wind (direction fire is pushed)", WINDS, index=0)
    intensity = st.slider("Wind intensity", 0.0, 1.0, 0.5, 0.05, disabled=wind == "none")
    eff = p0 * (1 - humidity)
    st.caption(f"Effective spread chance per burning neighbour: **{eff:.2f}** "
               + ("(fire usually dies out below ≈0.45)" if eff < 0.45 else ""))

    st.header("This machine")
    st.caption(f"CPU threads: **{CPU_THREADS}**  \nGPU: **{gpu_name() or 'not detected'}**")

params = R.SimParams(size=int(size), steps=int(steps), seed=int(seed), density=density, p0=p0,
                     humidity=humidity, wind=wind, intensity=intensity, fixed=fixed)


def mode_controls(mode: str, key: str) -> R.ModeConfig:
    """Widgets for one mode's parallel configuration."""
    cfg = R.ModeConfig()
    if mode == "openmp":
        c1, c2 = st.columns(2)
        cfg.threads = c1.slider("OpenMP threads", 1, CPU_THREADS, min(8, CPU_THREADS), key=f"{key}_t")
        cfg.schedule = c2.selectbox("Loop schedule", ["static", "dynamic", "guided"], key=f"{key}_s")
    elif mode == "mpi":
        cfg.procs = st.slider("MPI processes", 1, CPU_THREADS, min(4, CPU_THREADS), key=f"{key}_p")
    elif mode == "hybrid":
        c1, c2 = st.columns(2)
        cfg.procs = c1.slider("MPI processes", 1, 8, 2, key=f"{key}_p")
        cfg.threads = c2.slider("OpenMP threads per process", 1, CPU_THREADS, 4, key=f"{key}_t")
        if cfg.procs * cfg.threads > CPU_THREADS:
            st.caption(f"⚠️ {cfg.procs * cfg.threads} threads total on {CPU_THREADS} hardware threads: oversubscribed.")
    elif mode == "cuda":
        b = st.select_slider("Thread block", options=[8, 16, 32], value=16, key=f"{key}_b",
                             format_func=lambda v: f"{v}×{v} = {v * v} threads")
        cfg.block = b
    return cfg


def mode_details(res: dict):
    """Mode-specific breakdown below the headline metrics."""
    m = res["mode"]
    if m in ("mpi", "hybrid"):
        comm = res["comm_s"]
        frac = comm / res["time_s"] if res["time_s"] else 0
        st.markdown(f"**Communication** (halo exchange + collectives): {fmt_time(comm)} "
                    f"= **{frac:.0%}** of run time")
        df = pd.DataFrame({"Rank": [f"rank {i}" for i in range(len(res["rank_compute_s"]))],
                           "Compute time (ms)": [t * 1000 for t in res["rank_compute_s"]]})
        chart = alt.Chart(df).mark_bar(color=MODE_COLORS[m], cornerRadiusEnd=4).encode(
            x=alt.X("Compute time (ms):Q"), y=alt.Y("Rank:N", sort=None, title=None),
            tooltip=["Rank", alt.Tooltip("Compute time (ms):Q", format=".2f")],
        ).properties(height=alt.Step(30), title="Load balance: compute time per rank")
        st.altair_chart(chart, width="stretch")
    elif m == "cuda":
        k, h2d, d2h = res["kernel_s"], res["h2d_s"], res["d2h_s"]
        other = max(0.0, res["time_s"] - k - h2d - d2h)
        df = pd.DataFrame({"Part": ["Kernels", "Host→Device copy", "Device→Host copy", "Launch/sync overhead"],
                           "ms": [k * 1000, h2d * 1000, d2h * 1000, other * 1000]})
        st.markdown(f"**GPU:** {res['gpu']} · block {res['block']}×{res['block']}")
        chart = alt.Chart(df).mark_bar(color=MODE_COLORS["cuda"], cornerRadiusEnd=4).encode(
            x=alt.X("ms:Q", title="Time (ms)"), y=alt.Y("Part:N", sort=None, title=None),
            tooltip=["Part", alt.Tooltip("ms:Q", format=".3f")],
        ).properties(height=alt.Step(34), title="Where the GPU time goes")
        st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------- tabs ----
tab_run, tab_cmp, tab_scale, tab_about = st.tabs(
    ["▶ Run one mode", "⚖ Compare all modes", "📈 Scaling sweep", "ℹ How it works"])

# ---- Run one mode ---------------------------------------------------------
with tab_run:
    mode = st.radio("Execution mode", available, format_func=label, horizontal=True)
    cfg = mode_controls(mode, "run")

    can_animate = params.size <= 1000
    c1, c2 = st.columns(2)
    animate = c1.checkbox("Record animation", value=can_animate, disabled=not can_animate,
                          help="Only for grids up to 1000×1000. Larger grids show the final frame.")
    every = c2.slider("Snapshot every k steps", 1, 25, max(1, int(steps) // 120), disabled=not animate)

    if st.button(f"Run {label(mode)}", type="primary"):
        frames_path = os.path.join(tempfile.gettempdir(), f"fire_frames_{os.getpid()}.bin")
        try:
            with st.spinner(f"Running {label(mode)}…"):
                res = R.run(mode, params, cfg, frames=frames_path, every=every if animate else 0)
                frame_steps, frames = R.read_frames(frames_path)
                media = R.frames_to_gif(frames) if animate and len(frames) > 2 else R.frame_png(frames[-1])
            st.session_state["run"] = {"res": res, "media": media, "animated": animate and len(frames) > 2}
        except R.RunError as e:
            show_error(e)
        finally:
            if os.path.exists(frames_path):
                os.remove(frames_path)

    if "run" in st.session_state:
        res = st.session_state["run"]["res"]
        cells = res["cells"]
        total = res["rows"] * res["cols"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{label(res['mode'])} time", fmt_time(res["time_s"]), help=res["config"])
        m2.metric("Steps run", f"{res['steps_run']} / {res['max_steps']}")
        m3.metric("Cell updates / s", f"{res['cell_updates_per_s'] / 1e6:,.0f} M")
        m4.metric("Forest burned", f"{cells['burned'] / total:.1%}")

        left, right = st.columns([1, 1])
        with left:
            st.image(st.session_state["run"]["media"],
                     caption=("Fire spread animation" if st.session_state["run"]["animated"] else "Final forest")
                     + " · sand = empty · green = vegetation · orange = burning · charcoal = burned",
                     width="stretch")
        with right:
            hist = pd.DataFrame({"Step": range(len(res["burning_history"])),
                                 "Burning cells": res["burning_history"]})
            chart = alt.Chart(hist).mark_line(color="#eb6834", strokeWidth=2).encode(
                x="Step:Q", y="Burning cells:Q", tooltip=["Step", "Burning cells"],
            ).properties(height=260, title="Active fire front over time")
            st.altair_chart(chart, width="stretch")
            mode_details(res)

        with st.expander("Command and raw result"):
            st.code(res["command"], language="bash")
            st.markdown(f"Checksum of final grid: `{res['checksum']}`")
            st.json({k: v for k, v in res.items() if k not in ("burning_history", "command")})

# ---- Compare all modes ----------------------------------------------------
with tab_cmp:
    st.markdown("Runs every selected mode on **the same forest** and compares time, speedup, and "
                "whether the final grid matches the serial run exactly.")
    chosen = st.multiselect("Modes", available, default=available, format_func=label)
    configs = {}
    cols = st.columns(3)
    for i, m in enumerate([m for m in chosen if m != "serial"]):
        with cols[i % 3]:
            st.markdown(f"**{label(m)}**")
            configs[m] = mode_controls(m, f"cmp_{m}")
    repeats = st.slider("Repeats per mode (median is reported)", 1, 7, 3)
    if not fixed:
        st.info("Tip: turn on **Benchmark mode** in the sidebar so every mode does the same number of steps.")

    if st.button("Run comparison", type="primary", disabled="serial" not in available):
        modes = ["serial"] + [m for m in chosen if m != "serial"]
        rows = []
        progress = st.progress(0.0)
        try:
            for i, m in enumerate(modes):
                progress.progress(i / len(modes), text=f"Running {label(m)}…")
                rows.append(R.run_median(m, params, configs.get(m, R.ModeConfig()), repeats))
            progress.progress(1.0, text="Done")
            st.session_state["cmp"] = rows
        except R.RunError as e:
            progress.empty()
            show_error(e)

    if "cmp" in st.session_state:
        rows = st.session_state["cmp"]
        base = rows[0]
        table = []
        for r in rows:
            speedup = base["median_time_s"] / r["median_time_s"]
            table.append({
                "Mode": label(r["mode"]),
                "Config": r["config"],
                "Median time (ms)": r["median_time_s"] * 1000,
                "Speedup": speedup,
                "Efficiency": 100 * speedup / r["pes"] if r["pes"] else None,
                "Steps": r["steps_run"],
                "Matches serial": "✅ identical" if r["checksum"] == base["checksum"] else "❌ differs",
            })
        df = pd.DataFrame(table)
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Median time (ms)": st.column_config.NumberColumn(format="%.2f"),
            "Speedup": st.column_config.NumberColumn(format="%.2f×"),
            "Efficiency": st.column_config.NumberColumn(format="%.0f%%", help="Speedup ÷ CPU cores used"),
        })

        c1, c2 = st.columns(2)
        bar = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Mode:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Mode:N", scale=COLOR_SCALE, legend=None),
            tooltip=["Mode", "Config", alt.Tooltip("Median time (ms):Q", format=".2f"),
                     alt.Tooltip("Speedup:Q", format=".2f")],
        )
        c1.altair_chart(bar.encode(x=alt.X("Median time (ms):Q")).properties(
            title="Execution time (lower is better)", height=alt.Step(36)), width="stretch")
        rule = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
        c2.altair_chart((bar.encode(x=alt.X("Speedup:Q", title="Speedup vs serial (×)")) + rule).properties(
            title="Speedup (higher is better; dashed = serial)", height=alt.Step(36)),
            width="stretch")
        if base["rows"] <= 500:
            st.caption("On small grids the parallel overheads (thread barriers, MPI messages, GPU "
                       "launches and copies) can outweigh the work. Try 2000×2000 or larger.")
        st.download_button("Download CSV", df.to_csv(index=False), "compare.csv", "text/csv")

# ---- Scaling sweep --------------------------------------------------------
with tab_scale:
    kind = st.radio("Sweep", ["OpenMP threads", "MPI processes", "Grid size (all modes)"], horizontal=True)
    repeats_s = st.slider("Repeats per point", 1, 5, 3, key="sweep_rep")

    if kind == "OpenMP threads":
        counts = st.multiselect("Thread counts", [1, 2, 4, 6, 8, 10, 12], default=[1, 2, 4, 8])
        sweep_mode = "openmp"
    elif kind == "MPI processes":
        counts = st.multiselect("Process counts", [1, 2, 4, 6, 8], default=[1, 2, 4, 8])
        sweep_mode = "mpi"
    else:
        sizes = st.multiselect("Grid sizes", [250, 500, 1000, 2000, 3000, 5000], default=[500, 1000, 2000])
        st.caption("Uses the configurations chosen on the Compare tab (or defaults).")
    st.caption("Sweeps always use benchmark mode (fixed step count) so every point does equal work.")

    if st.button("Run sweep", type="primary", disabled="serial" not in available):
        sweep_params = R.SimParams(**{**params.__dict__, "fixed": True})
        out = []
        progress = st.progress(0.0)
        try:
            if kind != "Grid size (all modes)":
                if sweep_mode not in available:
                    raise R.RunError(f"{label(sweep_mode)} is not built")
                serial = R.run_median("serial", sweep_params, R.ModeConfig(), repeats_s)
                for i, n in enumerate(sorted(counts)):
                    progress.progress(i / len(counts), text=f"{label(sweep_mode)} with {n}…")
                    cfg = R.ModeConfig(threads=n) if sweep_mode == "openmp" else R.ModeConfig(procs=n)
                    r = R.run_median(sweep_mode, sweep_params, cfg, repeats_s)
                    sp = serial["median_time_s"] / r["median_time_s"]
                    out.append({"Workers": n, "Time (ms)": r["median_time_s"] * 1000, "Speedup": sp,
                                "Efficiency (%)": 100 * sp / n,
                                "Matches serial": r["checksum"] == serial["checksum"]})
            else:
                modes = [m for m in R.MODES if m in available]
                total = len(sizes) * len(modes)
                k = 0
                for s in sorted(sizes):
                    p = R.SimParams(**{**sweep_params.__dict__, "size": s})
                    serial_t = None
                    for m in modes:
                        progress.progress(k / total, text=f"{label(m)} at {s}×{s}…")
                        k += 1
                        cfg = configs.get(m, R.ModeConfig(threads=min(8, CPU_THREADS), procs=4)
                                          if m != "hybrid" else R.ModeConfig(threads=4, procs=2))
                        r = R.run_median(m, p, cfg, repeats_s)
                        if m == "serial":
                            serial_t = r["median_time_s"]
                        out.append({"Grid": s, "Mode": label(m), "Config": r["config"],
                                    "Time (ms)": r["median_time_s"] * 1000,
                                    "Speedup": serial_t / r["median_time_s"]})
            progress.progress(1.0, text="Done")
            st.session_state["sweep"] = (kind, pd.DataFrame(out))
        except R.RunError as e:
            progress.empty()
            show_error(e)

    if "sweep" in st.session_state:
        skind, df = st.session_state["sweep"]
        st.dataframe(df, hide_index=True, width="stretch")
        if skind != "Grid size (all modes)":
            mode_key = "openmp" if skind == "OpenMP threads" else "mpi"
            ideal = pd.DataFrame({"Workers": df["Workers"], "Speedup": df["Workers"], "Series": "Ideal (linear)"})
            meas = df[["Workers", "Speedup"]].assign(Series=f"Measured {label(mode_key)}")
            both = pd.concat([meas, ideal])
            line = alt.Chart(both).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
                x=alt.X("Workers:Q", title="Threads" if mode_key == "openmp" else "Processes"),
                y=alt.Y("Speedup:Q", title="Speedup vs serial (×)"),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=[f"Measured {label(mode_key)}", "Ideal (linear)"],
                    range=[MODE_COLORS[mode_key], "#9a9890"]), legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
                    domain=[f"Measured {label(mode_key)}", "Ideal (linear)"], range=[[1, 0], [5, 4]]), legend=None),
                tooltip=["Series", "Workers", alt.Tooltip("Speedup:Q", format=".2f")],
            ).properties(height=320, title="Strong scaling")
            eff = alt.Chart(df).mark_bar(color=MODE_COLORS[mode_key], cornerRadiusEnd=4).encode(
                x=alt.X("Workers:O", title="Threads" if mode_key == "openmp" else "Processes"),
                y=alt.Y("Efficiency (%):Q"),
                tooltip=["Workers", alt.Tooltip("Efficiency (%):Q", format=".0f")],
            ).properties(height=320, title="Parallel efficiency")
            c1, c2 = st.columns(2)
            c1.altair_chart(line, width="stretch")
            c2.altair_chart(eff, width="stretch")
        else:
            base = alt.Chart(df).encode(
                x=alt.X("Grid:Q", title="Grid edge N (N×N cells)"),
                color=alt.Color("Mode:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)),
                tooltip=["Mode", "Config", "Grid", alt.Tooltip("Time (ms):Q", format=".2f"),
                         alt.Tooltip("Speedup:Q", format=".2f")],
            )
            c1, c2 = st.columns(2)
            c1.altair_chart(base.mark_line(point=True, strokeWidth=2).encode(
                y=alt.Y("Time (ms):Q", scale=alt.Scale(type="log"))).properties(
                height=320, title="Execution time vs grid size (log scale)"), width="stretch")
            c2.altair_chart(base.mark_line(point=True, strokeWidth=2).encode(
                y=alt.Y("Speedup:Q", title="Speedup vs serial (×)")).properties(
                height=320, title="Speedup vs grid size"), width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False), "sweep.csv", "text/csv")

# ---- About ----------------------------------------------------------------
with tab_about:
    st.markdown("""
**Model.** Each cell is empty, vegetation, burning or burned. A burning cell burns out after one
step. A vegetation cell with burning neighbours ignites with probability
`P = 1 − Π(1 − p_j)`, where `p_j = p0 · (1 − H) · W_j` and `W_j` boosts spread along the wind.

**Why every mode gives the same answer.** Random numbers come from a hash of
`(seed, step, row, col)` rather than a shared generator, and probabilities are 16-bit fixed-point
integers, so CPU and GPU do bit-identical arithmetic. The *Matches serial* column checks this with
a checksum of the final grid.

| Mode | What runs in parallel | Synchronization each step |
|---|---|---|
| Serial | nothing (baseline) | none |
| OpenMP | rows split across threads sharing one grid | barrier at end of the parallel-for |
| MPI | grid cut into row blocks, one per process | halo-row exchange (`MPI_Sendrecv`) + `MPI_Allreduce` |
| Hybrid | row blocks per process, threads inside each | halo exchange + thread barrier |
| CUDA | one GPU thread per cell | kernel boundary; per-block `__syncthreads_count` + `atomicAdd` |

**Reading the results.** Speedup = T_serial / T_parallel. Efficiency = speedup / cores used.
This laptop's CPU has 4 performance cores (2 threads each) and 4 slower efficiency cores, so
scaling usually bends after ~4–8 threads. MPI on one laptop talks through shared memory, so its
communication cost is far lower than on a real cluster.
""")
