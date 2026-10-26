/*
 * fire_cuda.cu - GPU version.
 *
 * One CUDA thread per cell, launched as a 2D grid of BxB thread blocks.
 * Two device buffers (cur / nxt) give the same double buffering as the CPU
 * versions; after each kernel launch only the pointers are swapped.
 *
 * The initial grid is generated on the host (same code as every other
 * version), copied to the GPU once, and the final grid is copied back once.
 *
 * Burning cells per step are counted on the GPU: __syncthreads_count()
 * reduces inside a block, then one atomicAdd per block into counts[step].
 * In benchmark mode (--fixed) the host never waits on the GPU mid-run; with
 * early stopping it must read one 4-byte counter per step.
 */
#include <cuda_runtime.h>
#include "fire_common.h"

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void fire_step_kernel(const uint8_t *cur, uint8_t *nxt, int rows, int cols,
                                 uint32_t step, FireRule rule, uint32_t *burning_count)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int burning = 0;
    if (r < rows && c < cols) {
        const uint8_t *mid = cur + (size_t)r * cols;
        const uint8_t *up = r > 0 ? mid - cols : NULL;
        const uint8_t *dn = r < rows - 1 ? mid + cols : NULL;
        uint8_t v = fire_cell(up, mid, dn, c, cols, (uint32_t)r, step, &rule);
        nxt[(size_t)r * cols + c] = v;
        burning = (v == CELL_BURNING);
    }
    /* Every thread in the block (including out-of-range ones) must reach this. */
    int block_burning = __syncthreads_count(burning);
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_burning)
        atomicAdd(burning_count, (uint32_t)block_burning);
}

int main(int argc, char **argv)
{
    FireParams P;
    fire_parse_args(argc, argv, &P);
    if (P.block * P.block > 1024) { fprintf(stderr, "block %dx%d exceeds 1024 threads\n", P.block, P.block); return 2; }

    const int rows = P.rows, cols = P.cols;
    const size_t n = (size_t)rows * cols;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaFree(0));   /* create the CUDA context now so it is not timed */

    uint8_t *h_grid;
    CK(cudaMallocHost(&h_grid, n));   /* pinned memory -> faster PCIe copies */
    uint32_t *history = (uint32_t *)calloc((size_t)P.max_steps + 1, sizeof *history);
    fire_init_rows(h_grid, 0, rows, &P);
    history[0] = fire_count_state(h_grid, n, CELL_BURNING);

    uint8_t *d_cur, *d_nxt;
    uint32_t *d_counts;
    CK(cudaMalloc(&d_cur, n));
    CK(cudaMalloc(&d_nxt, n));
    CK(cudaMalloc(&d_counts, sizeof(uint32_t) * ((size_t)P.max_steps + 1)));
    CK(cudaMemset(d_counts, 0, sizeof(uint32_t) * ((size_t)P.max_steps + 1)));

    FireFrames *frames = fire_frames_open(&P);
    if (frames) fire_frames_write(frames, 0, h_grid, n);

    dim3 block(P.block, P.block);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cudaEvent_t h2d_a, h2d_b, d2h_a, d2h_b;
    CK(cudaEventCreate(&h2d_a)); CK(cudaEventCreate(&h2d_b));
    CK(cudaEventCreate(&d2h_a)); CK(cudaEventCreate(&d2h_b));
    cudaEvent_t *kev = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2 * (size_t)P.max_steps);
    for (int i = 0; i < 2 * P.max_steps; i++) CK(cudaEventCreate(&kev[i]));

    double io_time = 0.0;
    int steps = 0;
    double t0 = fire_now();

    /* Host -> device, once. */
    CK(cudaEventRecord(h2d_a));
    CK(cudaMemcpy(d_cur, h_grid, n, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(h2d_b));

    for (int s = 0; s < P.max_steps; s++) {
        CK(cudaEventRecord(kev[2 * s]));
        fire_step_kernel<<<grid, block>>>(d_cur, d_nxt, rows, cols, (uint32_t)s, P.rule, d_counts + s + 1);
        CK(cudaEventRecord(kev[2 * s + 1]));
        CK(cudaGetLastError());

        uint8_t *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        steps = s + 1;

        if (frames && fire_frame_due(&P, steps)) {
            CK(cudaDeviceSynchronize());          /* don't charge kernel time to I/O */
            double a = fire_now();
            CK(cudaMemcpy(h_grid, d_cur, n, cudaMemcpyDeviceToHost));
            fire_frames_write(frames, steps, h_grid, n);
            io_time += fire_now() - a;
        }
        if (!P.fixed_steps) {
            uint32_t burning;
            CK(cudaMemcpy(&burning, d_counts + steps, sizeof burning, cudaMemcpyDeviceToHost));
            if (burning == 0) break;
        }
    }

    /* Device -> host, once. */
    CK(cudaEventRecord(d2h_a));
    CK(cudaMemcpy(h_grid, d_cur, n, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(d2h_b));
    CK(cudaDeviceSynchronize());
    double elapsed = fire_now() - t0 - io_time;

    float ms, h2d_ms, d2h_ms, kernel_ms = 0.0f;
    CK(cudaEventElapsedTime(&h2d_ms, h2d_a, h2d_b));
    CK(cudaEventElapsedTime(&d2h_ms, d2h_a, d2h_b));
    for (int s = 0; s < steps; s++) {
        CK(cudaEventElapsedTime(&ms, kev[2 * s], kev[2 * s + 1]));
        kernel_ms += ms;
    }
    CK(cudaMemcpy(history + 1, d_counts + 1, sizeof(uint32_t) * steps, cudaMemcpyDeviceToHost));

    fire_frames_close(frames, steps, h_grid, n);

    char extra[400];
    snprintf(extra, sizeof extra,
             "\"threads\":1,\"procs\":1,\"block\":%d,\"gpu\":\"%s\","
             "\"kernel_s\":%.6f,\"h2d_s\":%.6f,\"d2h_s\":%.6f",
             P.block, prop.name, kernel_ms / 1e3, h2d_ms / 1e3, d2h_ms / 1e3);
    fire_print_result("cuda", &P, steps, h_grid, elapsed, extra, history);

    for (int i = 0; i < 2 * P.max_steps; i++) cudaEventDestroy(kev[i]);
    free(kev);
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_counts);
    cudaFreeHost(h_grid);
    free(history);
    return 0;
}
