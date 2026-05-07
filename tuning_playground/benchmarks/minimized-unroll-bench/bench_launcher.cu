// Benchmark launcher for fused_apply_small/large backward kernels.
#ifndef BENCH_CUDA_PATH
#error "Define BENCH_CUDA_PATH to the generated .cu file path"
#endif
#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)
#include STRINGIFY(BENCH_CUDA_PATH)

#include <cstdio>
#include <cstring>

#define COMMA ,

static TensorView mkTV(size_t d0, size_t d1) {
    TensorView tv; memset(&tv, 0, sizeof(tv));
    tv.dimensionCount = 2; tv.sizes[0] = d0; tv.sizes[1] = d1;
    tv.strides[1] = sizeof(float); tv.strides[0] = d1 * sizeof(float);
    cudaMalloc(&tv.data, d0 * d1 * sizeof(float));
    cudaMemset(tv.data, 0, d0 * d1 * sizeof(float));
    return tv;
}

static DiffTensorView_0 mkDiff(size_t d0, size_t d1) {
    DiffTensorView_0 d;
    d.primal_1 = mkTV(d0, d1);
    d.diff_1.diff_0 = mkTV(d0, d1);
    return d;
}

#define BENCH(name, kernel_call, W, I) do { \
    for (int _i = 0; _i < (W); _i++) { kernel_call; } \
    cudaDeviceSynchronize(); \
    cudaEvent_t _start, _stop; \
    cudaEventCreate(&_start); cudaEventCreate(&_stop); \
    cudaEventRecord(_start); \
    for (int _i = 0; _i < (I); _i++) { kernel_call; } \
    cudaEventRecord(_stop); cudaEventSynchronize(_stop); \
    float _ms; cudaEventElapsedTime(&_ms, _start, _stop); \
    cudaEventDestroy(_start); cudaEventDestroy(_stop); \
    printf("%-34s %8.1f us\n", name, _ms * 1000.0f / (I)); \
} while(0)

int main() {
    const int N = 500000;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const int W = 50, I = 1000;

    // Allocate buffers for each task
    auto vec_in  = mkDiff(N, 3),  vec_out  = mkDiff(N, 3);
    auto scl_in  = mkDiff(N, 3),  scl_out  = mkDiff(N, 3);
    auto opa_in  = mkDiff(N, 1),  opa_out  = mkDiff(N, 1);
    auto feat_in = mkDiff(N, 20), feat_out = mkDiff(N, 20);
    auto sig_a   = mkDiff(N, 3),  sig_b    = mkDiff(N, 45), sig_out = mkDiff(N, 48);

    // Task structs matching the Slang kernel signatures
    CopyTask_0 vectors   = { vec_in, vec_out };
    ExpTask_0 scales     = { scl_in, scl_out };
    SigmoidTask_0 opacities = { opa_in, opa_out };
    CopyTask_0 features  = { feat_in, feat_out };
    MergedCopyTask_0 merged = { sig_a, sig_b, sig_out };

    uint row_offset = 0, count = N;
    dim3 grid(GRID), block(BLOCK);

    printf("=== Backward kernels ===\n");
    BENCH("fused_apply_small_bwd:", __kernel__fused_apply_small_bwd_diff<<<grid COMMA block>>>(row_offset COMMA count COMMA vectors COMMA scales COMMA opacities COMMA features), W, I);
    BENCH("fused_apply_large_bwd:", __kernel__fused_apply_large_bwd_diff<<<grid COMMA block>>>(row_offset COMMA count COMMA vectors COMMA scales COMMA opacities COMMA features COMMA merged), W, I);

    printf("\n=== Forward kernels ===\n");
    BENCH("fused_apply_small_fwd:", __kernel__fused_apply_small<<<grid COMMA block>>>(row_offset COMMA count COMMA vectors COMMA scales COMMA opacities COMMA features), W, I);
    BENCH("fused_apply_large_fwd:", __kernel__fused_apply_large<<<grid COMMA block>>>(row_offset COMMA count COMMA vectors COMMA scales COMMA opacities COMMA features COMMA merged), W, I);

    return 0;
}
