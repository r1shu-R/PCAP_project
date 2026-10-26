/*
 * fire_common.h - host-side helpers shared by every version:
 * command-line parsing, synthetic grid generation, frame output, timing,
 * checksums and the RESULT line the UI parses.
 *
 * Header-only and C/C++-compatible so nvcc can include it too.
 */
#ifndef FIRE_COMMON_H
#define FIRE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include "fire_core.h"

typedef struct {
    int rows, cols;
    int max_steps;
    uint32_t seed;
    double density;     /* probability a cell starts as vegetation */
    double p0;          /* base spread probability per burning neighbour */
    double humidity;    /* 0..1, scales spread down by (1 - H) */
    char wind[8];       /* none, N, NE, E, SE, S, SW, W, NW (direction fire is pushed toward) */
    double intensity;   /* 0..1 */
    int fixed_steps;    /* 1 = always run max_steps (benchmarking); 0 = stop when fire dies */
    const char *frames_path;
    int frame_every;    /* 0 = only initial + final frame */
    int threads;        /* OpenMP threads (0 = OMP default / OMP_NUM_THREADS) */
    int block;          /* CUDA block edge (block = block x block threads) */

    /* derived */
    FireRule rule;
    uint64_t veg_threshold;  /* hash < threshold -> vegetation */
} FireParams;

/* Neighbour offsets in the same order as FireRule.q[] (see fire_core.h). */
static const int FIRE_NB_DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int FIRE_NB_DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

static inline double fire_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline void fire_usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --size N            square grid N x N (default 500)\n"
        "  --rows N --cols N   rectangular grid\n"
        "  --steps N           maximum time steps (default 300)\n"
        "  --fixed             always run all steps (benchmark mode)\n"
        "  --seed N            RNG seed (default 42)\n"
        "  --density F         vegetation density 0..1 (default 0.70)\n"
        "  --p0 F              base ignition probability 0..1 (default 0.65)\n"
        "  --humidity F        humidity 0..1 (default 0.20)\n"
        "  --wind DIR          none|N|NE|E|SE|S|SW|W|NW (default none)\n"
        "  --intensity F       wind intensity 0..1 (default 0.5)\n"
        "  --threads N         OpenMP threads (openmp/hybrid)\n"
        "  --block N           CUDA block edge 8|16|32 (cuda, default 16)\n"
        "  --frames FILE       write grid snapshots to FILE\n"
        "  --every K           snapshot every K steps (0 = first+last only)\n",
        prog);
}

static inline void fire_die(const char *prog, const char *msg, const char *arg)
{
    fprintf(stderr, "%s: %s%s%s\n", prog, msg, arg ? ": " : "", arg ? arg : "");
    exit(2);
}

static inline double fire_parse_unit(const char *prog, const char *name, const char *v)
{
    char *end;
    double x = strtod(v, &end);
    if (*end != '\0' || x < 0.0 || x > 1.0) fire_die(prog, "value must be in [0,1] for", name);
    return x;
}

static inline int fire_parse_int(const char *prog, const char *name, const char *v, int lo, int hi)
{
    char *end;
    long x = strtol(v, &end, 10);
    if (*end != '\0' || x < lo || x > hi) fire_die(prog, "integer out of range for", name);
    return (int)x;
}

/* Unit wind vector (row, col) for a direction name; rows grow southward. */
static inline int fire_wind_vector(const char *w, double *wr, double *wc)
{
    static const struct { const char *name; int r, c; } dirs[] = {
        {"none", 0, 0}, {"N", -1, 0}, {"NE", -1, 1}, {"E", 0, 1}, {"SE", 1, 1},
        {"S", 1, 0}, {"SW", 1, -1}, {"W", 0, -1}, {"NW", -1, -1},
    };
    for (size_t i = 0; i < sizeof dirs / sizeof dirs[0]; i++) {
        if (strcasecmp(w, dirs[i].name) == 0) {
            double len = sqrt((double)(dirs[i].r * dirs[i].r + dirs[i].c * dirs[i].c));
            *wr = len > 0 ? dirs[i].r / len : 0.0;
            *wc = len > 0 ? dirs[i].c / len : 0.0;
            return 1;
        }
    }
    return 0;
}

/*
 * Fold p0, humidity and wind into the fixed-point survival table q[8].
 * For a burning neighbour at offset (dr, dc), fire travels in direction
 * (-dr, -dc). With cos = alignment between that direction and the wind:
 *   W = 1 + I*cos      when cos >= 0  (aligned: 1 + I)
 *   W = 1 + (I/2)*cos  when cos <  0  (opposite: 1 - I/2)
 *   p = clamp(p0 * (1 - H) * W, 0, 1)
 * Diagonal neighbours get partial wind effect through cos.
 */
static inline void fire_build_rule(FireParams *P)
{
    double wr, wc;
    fire_wind_vector(P->wind, &wr, &wc);
    P->rule.seed = P->seed;
    for (int d = 0; d < 8; d++) {
        double sr = -FIRE_NB_DR[d], sc = -FIRE_NB_DC[d];
        double cosv = (sr * wr + sc * wc) / sqrt(sr * sr + sc * sc);
        double W = cosv >= 0 ? 1.0 + P->intensity * cosv : 1.0 + 0.5 * P->intensity * cosv;
        double p = P->p0 * (1.0 - P->humidity) * W;
        if (p < 0) p = 0;
        if (p > 1) p = 1;
        long q = lround((1.0 - p) * FIRE_ONE);
        if (q > (long)FIRE_ONE) q = FIRE_ONE;
        P->rule.q[d] = (uint32_t)q;
    }
    P->veg_threshold = (uint64_t)llround(P->density * 4294967296.0);
}

static inline void fire_parse_args(int argc, char **argv, FireParams *P)
{
    const char *prog = argv[0];
    memset(P, 0, sizeof *P);
    P->rows = P->cols = 500;
    P->max_steps = 300;
    P->seed = 42;
    P->density = 0.70;
    P->p0 = 0.65;
    P->humidity = 0.20;
    strcpy(P->wind, "none");
    P->intensity = 0.5;
    P->block = 16;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDV() do { if (!v) fire_die(prog, "missing value for", a); i++; } while (0)
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { fire_usage(prog); exit(0); }
        else if (!strcmp(a, "--fixed")) P->fixed_steps = 1;
        else if (!strcmp(a, "--size"))      { NEEDV(); P->rows = P->cols = fire_parse_int(prog, a, v, 3, 50000); }
        else if (!strcmp(a, "--rows"))      { NEEDV(); P->rows = fire_parse_int(prog, a, v, 3, 50000); }
        else if (!strcmp(a, "--cols"))      { NEEDV(); P->cols = fire_parse_int(prog, a, v, 3, 50000); }
        else if (!strcmp(a, "--steps"))     { NEEDV(); P->max_steps = fire_parse_int(prog, a, v, 1, 1000000); }
        else if (!strcmp(a, "--seed"))      { NEEDV(); P->seed = (uint32_t)strtoul(v, NULL, 10); }
        else if (!strcmp(a, "--density"))   { NEEDV(); P->density = fire_parse_unit(prog, a, v); }
        else if (!strcmp(a, "--p0"))        { NEEDV(); P->p0 = fire_parse_unit(prog, a, v); }
        else if (!strcmp(a, "--humidity"))  { NEEDV(); P->humidity = fire_parse_unit(prog, a, v); }
        else if (!strcmp(a, "--intensity")) { NEEDV(); P->intensity = fire_parse_unit(prog, a, v); }
        else if (!strcmp(a, "--wind")) {
            NEEDV();
            double wr, wc;
            if (strlen(v) >= sizeof P->wind || !fire_wind_vector(v, &wr, &wc))
                fire_die(prog, "unknown wind direction", v);
            strcpy(P->wind, v);
        }
        else if (!strcmp(a, "--threads"))   { NEEDV(); P->threads = fire_parse_int(prog, a, v, 1, 1024); }
        else if (!strcmp(a, "--block"))     { NEEDV(); P->block = fire_parse_int(prog, a, v, 1, 32); }
        else if (!strcmp(a, "--frames"))    { NEEDV(); P->frames_path = v; }
        else if (!strcmp(a, "--every"))     { NEEDV(); P->frame_every = fire_parse_int(prog, a, v, 0, 1000000); }
        else { fire_usage(prog); fire_die(prog, "unknown option", a); }
#undef NEEDV
    }
    fire_build_rule(P);
}

/* Is (r, c) inside the 3x3 ignition block at the centre of the forest? */
static inline int fire_is_ignition(int r, int c, const FireParams *P)
{
    int cr = P->rows / 2, cc = P->cols / 2;
    return r >= cr - 1 && r <= cr + 1 && c >= cc - 1 && c <= cc + 1;
}

/*
 * Generate global rows [r0, r0 + nrows) of the synthetic forest into dst.
 * Each cell depends only on (seed, row, col), so any process can generate
 * its own slice without communication and still match the serial grid.
 */
static inline void fire_init_rows(uint8_t *dst, int r0, int nrows, const FireParams *P)
{
    for (int i = 0; i < nrows; i++) {
        int r = r0 + i;
        for (int c = 0; c < P->cols; c++) {
            uint8_t v = (uint64_t)fire_hash(P->seed, 0, (uint32_t)r, (uint32_t)c) < P->veg_threshold
                            ? CELL_VEG : CELL_EMPTY;
            if (fire_is_ignition(r, c, P)) v = CELL_BURNING;
            dst[(size_t)i * P->cols + c] = v;
        }
    }
}

static inline uint32_t fire_count_state(const uint8_t *g, size_t n, uint8_t state)
{
    uint32_t k = 0;
    for (size_t i = 0; i < n; i++) k += (g[i] == state);
    return k;
}

/* FNV-1a 64-bit over the final grid: equal checksum == identical grids. */
static inline uint64_t fire_checksum(const uint8_t *g, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= g[i]; h *= 0x100000001b3ULL; }
    return h;
}

/*
 * Frame file: "FFR1", int32 rows, int32 cols, then repeated
 * [int32 step][rows*cols bytes]. The UI turns this into the animation.
 */
typedef struct {
    FILE *f;
    int last_step;
} FireFrames;

static inline FireFrames *fire_frames_open(const FireParams *P)
{
    if (!P->frames_path) return NULL;
    FireFrames *fr = (FireFrames *)calloc(1, sizeof *fr);
    fr->f = fopen(P->frames_path, "wb");
    if (!fr->f) { perror(P->frames_path); exit(1); }
    int32_t hdr[2] = {P->rows, P->cols};
    fwrite("FFR1", 1, 4, fr->f);
    fwrite(hdr, sizeof hdr[0], 2, fr->f);
    fr->last_step = -1;
    return fr;
}

static inline void fire_frames_write(FireFrames *fr, int step, const uint8_t *g, size_t n)
{
    int32_t s = step;
    fwrite(&s, sizeof s, 1, fr->f);
    fwrite(g, 1, n, fr->f);
    fr->last_step = step;
}

static inline int fire_frame_due(const FireParams *P, int step)
{
    return P->frame_every > 0 && step % P->frame_every == 0;
}

/* Write the final frame (unless it was just written) and close. */
static inline void fire_frames_close(FireFrames *fr, int step, const uint8_t *g, size_t n)
{
    if (!fr) return;
    if (fr->last_step != step) fire_frames_write(fr, step, g, n);
    fclose(fr->f);
    free(fr);
}

/*
 * Print the single machine-readable result line:
 *   RESULT {json}
 * `extra` holds mode-specific JSON members without surrounding braces.
 * history[0..steps_run] = burning cells after each step (index 0 = initial).
 */
static inline void fire_print_result(const char *mode, const FireParams *P, int steps_run,
                              const uint8_t *grid, double time_s, const char *extra,
                              const uint32_t *history)
{
    size_t n = (size_t)P->rows * P->cols;
    double updates = (double)n * steps_run;
    printf("RESULT {\"mode\":\"%s\",\"rows\":%d,\"cols\":%d,\"max_steps\":%d,"
           "\"steps_run\":%d,\"fixed_steps\":%s,\"time_s\":%.6f,"
           "\"cell_updates_per_s\":%.6e,"
           "\"cells\":{\"empty\":%u,\"vegetation\":%u,\"burning\":%u,\"burned\":%u},"
           "\"checksum\":\"%016llx\",%s%s\"burning_history\":[",
           mode, P->rows, P->cols, P->max_steps, steps_run,
           P->fixed_steps ? "true" : "false", time_s,
           time_s > 0 ? updates / time_s : 0.0,
           fire_count_state(grid, n, CELL_EMPTY), fire_count_state(grid, n, CELL_VEG),
           fire_count_state(grid, n, CELL_BURNING), fire_count_state(grid, n, CELL_BURNED),
           (unsigned long long)fire_checksum(grid, n),
           extra, extra[0] ? "," : "");
    for (int s = 0; s <= steps_run; s++) printf(s ? ",%u" : "%u", history[s]);
    printf("]}\n");
    fflush(stdout);
}

#endif /* FIRE_COMMON_H */
