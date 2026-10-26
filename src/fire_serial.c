/*
 * fire_serial.c - single-threaded baseline. Its output is the reference
 * every parallel version must match, and its time is T1 for speedup.
 */
#include "fire_common.h"

int main(int argc, char **argv)
{
    FireParams P;
    fire_parse_args(argc, argv, &P);

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
    double t0 = fire_now();

    for (int s = 0; s < P.max_steps; s++) {
        uint32_t burning = 0;
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
        }
        uint8_t *tmp = cur; cur = nxt; nxt = tmp;   /* double buffering */
        steps = s + 1;
        history[steps] = burning;

        if (frames && fire_frame_due(&P, steps)) {
            double a = fire_now();
            fire_frames_write(frames, steps, cur, n);
            io_time += fire_now() - a;
        }
        if (!P.fixed_steps && burning == 0) break;
    }

    double elapsed = fire_now() - t0 - io_time;
    fire_frames_close(frames, steps, cur, n);
    fire_print_result("serial", &P, steps, cur, elapsed, "\"threads\":1,\"procs\":1", history);

    free(cur); free(nxt); free(history);
    return 0;
}
