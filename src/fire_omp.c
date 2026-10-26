/*
 * fire_omp.c - shared-memory version with OpenMP.
 *
 * All threads share the `cur` and `nxt` grids. Each time step, the row loop
 * is split across threads; every thread reads `cur` (read-only during the
 * step) and writes its own rows of `nxt`, so there are no write conflicts.
 * The implicit barrier at the end of the parallel-for is the only
 * synchronization per step. Burning cells are summed with a reduction.
 *
 * The loop uses schedule(runtime) so the UI can switch between
 * static / dynamic / guided through OMP_SCHEDULE without recompiling.
 */
#include <omp.h>
#include "fire_common.h"

int main(int argc, char **argv)
{
    FireParams P;
    fire_parse_args(argc, argv, &P);

    if (P.threads > 0) omp_set_num_threads(P.threads);
    const char *sched_env = getenv("OMP_SCHEDULE");
    if (!sched_env) omp_set_schedule(omp_sched_static, 0);   /* libgomp's default would be dynamic,1 */

    const int rows = P.rows, cols = P.cols;
    const size_t n = (size_t)rows * cols;
    uint8_t *cur = (uint8_t *)malloc(n);
    uint8_t *nxt = (uint8_t *)malloc(n);
    uint32_t *history = (uint32_t *)calloc((size_t)P.max_steps + 1, sizeof *history);
    if (!cur || !nxt || !history) { fprintf(stderr, "out of memory\n"); return 1; }

    fire_init_rows(cur, 0, rows, &P);
    history[0] = fire_count_state(cur, n, CELL_BURNING);

    FireFrames *frames = fire_frames_open(&P);
    if (frames) fire_frames_write(frames, 0, cur, n);

    double io_time = 0.0;
    int steps = 0;
    double t0 = omp_get_wtime();

    for (int s = 0; s < P.max_steps; s++) {
        uint32_t burning = 0;

        #pragma omp parallel for schedule(runtime) reduction(+:burning)
        for (int r = 0; r < rows; r++) {
            const uint8_t *mid = cur + (size_t)r * cols;
            const uint8_t *up = r > 0 ? mid - cols : NULL;
            const uint8_t *dn = r < rows - 1 ? mid + cols : NULL;
            uint8_t *out = nxt + (size_t)r * cols;
            for (int c = 0; c < cols; c++) {
                uint8_t v = fire_cell(up, mid, dn, c, cols, (uint32_t)r, (uint32_t)s, &P.rule);
                out[c] = v;
                burning += (v == CELL_BURNING);
            }
        }   /* implicit barrier: all of nxt is written before we swap */

        uint8_t *tmp = cur; cur = nxt; nxt = tmp;
        steps = s + 1;
        history[steps] = burning;

        if (frames && fire_frame_due(&P, steps)) {
            double a = omp_get_wtime();
            fire_frames_write(frames, steps, cur, n);
            io_time += omp_get_wtime() - a;
        }
        if (!P.fixed_steps && burning == 0) break;
    }

    double elapsed = omp_get_wtime() - t0 - io_time;
    fire_frames_close(frames, steps, cur, n);

    char extra[160];
    snprintf(extra, sizeof extra, "\"threads\":%d,\"procs\":1,\"schedule\":\"%s\"",
             omp_get_max_threads(), sched_env ? sched_env : "static");
    fire_print_result("openmp", &P, steps, cur, elapsed, extra, history);

    free(cur); free(nxt); free(history);
    return 0;
}
