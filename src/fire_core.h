/*
 * fire_core.h - the forest-fire cellular automaton rule.
 *
 * This header is the single source of truth for the model. Every version
 * (serial, OpenMP, MPI, hybrid, CUDA) calls fire_cell() for each cell, so any
 * difference in output between versions is a parallelization bug, not a
 * model difference.
 *
 * It compiles as C (gcc/mpicc) and as CUDA (nvcc): under nvcc the functions
 * become __host__ __device__ so the GPU kernel runs exactly the same code.
 */
#ifndef FIRE_CORE_H
#define FIRE_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#define FIRE_HD static __host__ __device__ inline
#else
#define FIRE_HD static inline
#endif

/* Cell states (stored as one byte per cell). */
#define CELL_EMPTY   0
#define CELL_VEG     1
#define CELL_BURNING 2
#define CELL_BURNED  3

/* Fixed-point scale for probabilities: 1.0 == 65536. */
#define FIRE_ONE 65536u

/*
 * Precomputed per-run rule. q[d] is the probability that a burning neighbour
 * in direction d does NOT ignite this cell, in 16.16 fixed point. Wind,
 * humidity and p0 are folded into q[] on the host (see fire_common.h), so the
 * per-cell work is pure integer math and gives bit-identical results on CPU
 * and GPU.
 *
 * Neighbour order d (row offset, col offset):
 *   0:(-1,-1) 1:(-1,0) 2:(-1,+1)
 *   3:( 0,-1)          4:( 0,+1)
 *   5:(+1,-1) 6:(+1,0) 7:(+1,+1)
 */
typedef struct {
    uint32_t seed;
    uint32_t q[8];
} FireRule;

/*
 * Counter-based RNG: a random value is a pure function of
 * (seed, step, row, col). No shared generator state means thread/process
 * scheduling cannot change which random number a cell gets.
 * (SplitMix64 finalizer.)
 */
FIRE_HD uint32_t fire_hash(uint32_t seed, uint32_t step, uint32_t row, uint32_t col)
{
    uint64_t x = ((uint64_t)row << 32) | col;
    x ^= (uint64_t)seed * 0x9E3779B97F4A7C15ULL;
    x += (uint64_t)step * 0xD1B54A32D192ED03ULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (uint32_t)(x >> 32);
}

/* One burning neighbour multiplies the survival probability by q[d]. */
#define FIRE_NEIGHBOUR(rowp, cc, d)                                            \
    if ((rowp) && (cc) >= 0 && (cc) < cols && (rowp)[cc] == CELL_BURNING) {    \
        surv = (uint32_t)(((uint64_t)surv * rule->q[d]) >> 16);                \
        any = 1;                                                               \
    }

/*
 * Next state of cell (grow, c) at time step `step`.
 *   up / mid / dn : the row above, the cell's own row, the row below
 *                   (up or dn is NULL at the top/bottom edge of the forest)
 *   grow          : the cell's GLOBAL row index (used by the RNG, so MPI
 *                   ranks agree with the serial run)
 *
 * Rule: empty and burned cells never change; a burning cell burns out after
 * one step; a vegetation cell with burning neighbours ignites with
 * P = 1 - prod(1 - p_j) over its burning neighbours j.
 */
FIRE_HD uint8_t fire_cell(const uint8_t *up, const uint8_t *mid, const uint8_t *dn,
                          int c, int cols, uint32_t grow, uint32_t step,
                          const FireRule *rule)
{
    uint8_t s = mid[c];
    if (s == CELL_BURNING) return CELL_BURNED;
    if (s != CELL_VEG) return s;

    uint32_t surv = FIRE_ONE;
    int any = 0;
    FIRE_NEIGHBOUR(up,  c - 1, 0)
    FIRE_NEIGHBOUR(up,  c,     1)
    FIRE_NEIGHBOUR(up,  c + 1, 2)
    FIRE_NEIGHBOUR(mid, c - 1, 3)
    FIRE_NEIGHBOUR(mid, c + 1, 4)
    FIRE_NEIGHBOUR(dn,  c - 1, 5)
    FIRE_NEIGHBOUR(dn,  c,     6)
    FIRE_NEIGHBOUR(dn,  c + 1, 7)
    if (!any) return CELL_VEG;

    /* step + 1: step 0 of the hash stream is reserved for grid generation. */
    uint32_t u = fire_hash(rule->seed, step + 1, grow, (uint32_t)c) >> 16;
    return (u < FIRE_ONE - surv) ? CELL_BURNING : CELL_VEG;
}

#endif /* FIRE_CORE_H */
