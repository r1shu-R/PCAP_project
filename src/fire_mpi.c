/*
 * fire_mpi.c - distributed-memory version (MPI), and the hybrid version.
 *
 * Built twice by the Makefile:
 *   fire_mpi    : mpicc            -> pure MPI, one process = one core
 *   fire_hybrid : mpicc -fopenmp   -> MPI between processes + OpenMP threads
 *                                     inside each process (the `omp` pragma
 *                                     below only activates in this build)
 *
 * Domain decomposition: the grid is cut into horizontal row blocks, one per
 * rank. Each rank stores its rows plus one halo (ghost) row above and below:
 *
 *     local row 0          halo  <- copy of the last row of rank-1
 *     local rows 1..nr     rows this rank owns and updates
 *     local row nr+1       halo  <- copy of the first row of rank+1
 *
 * Each step: exchange halos with the neighbours (MPI_Sendrecv), update the
 * owned rows, swap buffers. At the top/bottom edge of the forest the halo
 * stays EMPTY, which behaves exactly like "no neighbour" in the serial code.
 *
 * Every rank generates its own slice of the initial forest from the
 * deterministic hash, so no initial scatter is needed.
 */
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fire_common.h"

int main(int argc, char **argv)
{
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  /* only the main thread calls MPI */
    const char *mode = "hybrid";
#else
    MPI_Init(&argc, &argv);
    const char *mode = "mpi";
#endif
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    FireParams P;
    fire_parse_args(argc, argv, &P);
#ifdef _OPENMP
    if (P.threads > 0) omp_set_num_threads(P.threads);
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_static, 0);
#endif
    if (nprocs > P.rows) {
        if (rank == 0) fprintf(stderr, "more MPI processes (%d) than grid rows (%d)\n", nprocs, P.rows);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const int rows = P.rows, cols = P.cols;
    const size_t n = (size_t)rows * cols;

    /* Row-block decomposition; the first `rem` ranks get one extra row. */
    const int base = rows / nprocs, rem = rows % nprocs;
    const int nr = base + (rank < rem);
    const int r0 = rank * base + (rank < rem ? rank : rem);
    const int up_rank = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int dn_rank = rank < nprocs - 1 ? rank + 1 : MPI_PROC_NULL;

    /* calloc -> halo rows start EMPTY, which is what the forest edges need. */
    const size_t local_n = (size_t)(nr + 2) * cols;
    uint8_t *cur = (uint8_t *)calloc(local_n, 1);
    uint8_t *nxt = (uint8_t *)calloc(local_n, 1);
    uint32_t *local_hist = (uint32_t *)calloc((size_t)P.max_steps + 1, sizeof *local_hist);
    if (!cur || !nxt || !local_hist) { fprintf(stderr, "rank %d: out of memory\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    fire_init_rows(cur + cols, r0, nr, &P);
    local_hist[0] = fire_count_state(cur + cols, (size_t)nr * cols, CELL_BURNING);

    /* Rank 0 needs counts/offsets to gather the full grid (frames + final checksum). */
    int *counts = NULL, *displs = NULL;
    uint8_t *full = NULL;
    if (rank == 0) {
        counts = (int *)malloc(sizeof(int) * nprocs);
        displs = (int *)malloc(sizeof(int) * nprocs);
        for (int p = 0, off = 0; p < nprocs; p++) {
            counts[p] = (base + (p < rem)) * cols;
            displs[p] = off;
            off += counts[p];
        }
        full = (uint8_t *)malloc(n);
        if (!full) { fprintf(stderr, "rank 0: out of memory\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
#define GATHER_GRID(src) \
    MPI_Gatherv((src) + cols, nr * cols, MPI_UINT8_T, full, counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD)

    FireFrames *frames = NULL;
    if (P.frames_path) {
        GATHER_GRID(cur);
        if (rank == 0) { frames = fire_frames_open(&P); fire_frames_write(frames, 0, full, n); }
    }

    double comm_time = 0.0, io_time = 0.0;
    int steps = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < P.max_steps; s++) {
        /* 1. Halo exchange. Send my first owned row up and receive the row
              below me into the bottom halo; then the mirror image. */
        double a = MPI_Wtime();
        MPI_Sendrecv(cur + cols, cols, MPI_UINT8_T, up_rank, 0,
                     cur + (size_t)(nr + 1) * cols, cols, MPI_UINT8_T, dn_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(cur + (size_t)nr * cols, cols, MPI_UINT8_T, dn_rank, 1,
                     cur, cols, MPI_UINT8_T, up_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        comm_time += MPI_Wtime() - a;

        /* 2. Update owned rows (threaded in the hybrid build). */
        uint32_t burning = 0;
        #pragma omp parallel for schedule(runtime) reduction(+:burning)
        for (int i = 1; i <= nr; i++) {
            const uint8_t *mid = cur + (size_t)i * cols;
            uint8_t *out = nxt + (size_t)i * cols;
            const uint32_t grow = (uint32_t)(r0 + i - 1);   /* global row for the RNG */
            for (int c = 0; c < cols; c++) {
                uint8_t v = fire_cell(mid - cols, mid, mid + cols, c, cols, grow, (uint32_t)s, &P.rule);
                out[c] = v;
                burning += (v == CELL_BURNING);
            }
        }

        /* 3. Swap. nxt's halo rows are stale, but they are overwritten by
              the next exchange before anyone reads them. */
        uint8_t *tmp = cur; cur = nxt; nxt = tmp;
        steps = s + 1;
        local_hist[steps] = burning;

        /* 4. Early stop needs a global count every step (a collective). */
        uint32_t global_burning = 1;
        if (!P.fixed_steps) {
            a = MPI_Wtime();
            MPI_Allreduce(&burning, &global_burning, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
            comm_time += MPI_Wtime() - a;
        }

        if (P.frames_path && fire_frame_due(&P, steps)) {
            a = MPI_Wtime();
            GATHER_GRID(cur);
            if (rank == 0) fire_frames_write(frames, steps, full, n);
            io_time += MPI_Wtime() - a;
        }
        if (global_burning == 0) break;
    }

    double elapsed = MPI_Wtime() - t0 - io_time;
    double compute = elapsed - comm_time;

    /* Gather results outside the timed region. */
    double max_elapsed, max_comm;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double *rank_compute = rank == 0 ? (double *)malloc(sizeof(double) * nprocs) : NULL;
    MPI_Gather(&compute, 1, MPI_DOUBLE, rank_compute, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    uint32_t *history = rank == 0 ? (uint32_t *)calloc((size_t)steps + 1, sizeof *history) : NULL;
    MPI_Reduce(local_hist, history, steps + 1, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
    GATHER_GRID(cur);

    if (rank == 0) {
        fire_frames_close(frames, steps, full, n);
#ifdef _OPENMP
        int threads = omp_get_max_threads();
#else
        int threads = 1;
#endif
        char *extra = (char *)malloc(256 + 24 * (size_t)nprocs);
        int len = sprintf(extra, "\"threads\":%d,\"procs\":%d,\"comm_s\":%.6f,\"rank_compute_s\":[",
                          threads, nprocs, max_comm);
        for (int p = 0; p < nprocs; p++) len += sprintf(extra + len, p ? ",%.6f" : "%.6f", rank_compute[p]);
        sprintf(extra + len, "]");
        fire_print_result(mode, &P, steps, full, max_elapsed, extra, history);
        free(extra); free(rank_compute); free(history);
        free(full); free(counts); free(displs);
    }

    free(cur); free(nxt); free(local_hist);
    MPI_Finalize();
    return 0;
}
