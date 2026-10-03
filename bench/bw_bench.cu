// bw_bench.cu -- memory bandwidth ground truth for DGX Spark (GB10, sm_121).
//
// Measures what this unit's LPDDR5x actually delivers to CUDA kernels, which is the
// number every decode target in PLAN.md is derived from (decode is bandwidth-bound).
//
// Tests (all over one buffer, default 8 GiB, so L2 (24 MB) is irrelevant):
//   read      grid-stride read-sum, 16 B loads, 4x unrolled      (the headline number)
//   read-u1   same, no unrolling                                  (one load in flight / thread)
//   read-8B / read-4B   narrower loads                            (vector-width sensitivity)
//   read sweep over grid size                                     (how many blocks saturate)
//   write     fill                                                (write-only)
//   rmw       in-place x += c                                     (read+write, same lines)
//   copy      dst = src (STREAM copy, needs a 2nd buffer)         (read+write, different lines)
//   strided   one 16 B load per thread every S*16 bytes           (partial-sector efficiency)
//
// Every test verifies its result on the host (read-sums must equal the known fill value,
// rmw words must equal exactly what k passes should have produced), because a kernel that
// silently does nothing would otherwise report infinite bandwidth.
//
// Build:  make -C bench            (nvcc -O3 -arch=sm_121a)
// Run:    bench/build/bw_bench [--gib 8] [--iters 10] [--blocks 2048] [--threads 256] [--no-copy]
// GB/s below means 1e9 bytes/s, the same unit as the 273 GB/s spec figure.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <vector>

#define CK(x)                                                                            \
    do {                                                                                 \
        cudaError_t e_ = (x);                                                            \
        if (e_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n", cudaGetErrorString(e_), \
                    __FILE__, __LINE__, #x);                                             \
            exit(2);                                                                     \
        }                                                                                \
    } while (0)

// ---- vector helpers -------------------------------------------------------------------
__device__ __forceinline__ uint32_t hsum(uint32_t v) { return v; }
__device__ __forceinline__ uint32_t hsum(uint2 v) { return v.x + v.y; }
__device__ __forceinline__ uint32_t hsum(uint4 v) { return v.x + v.y + v.z + v.w; }
__device__ __forceinline__ uint32_t splat(uint32_t x, uint32_t*) { return x; }
__device__ __forceinline__ uint2 splat(uint32_t x, uint2*) { return make_uint2(x, x); }
__device__ __forceinline__ uint4 splat(uint32_t x, uint4*) { return make_uint4(x, x, x, x); }
__device__ __forceinline__ void addc(uint32_t& v, uint32_t x) { v += x; }
__device__ __forceinline__ void addc(uint2& v, uint32_t x) { v.x += x; v.y += x; }
__device__ __forceinline__ void addc(uint4& v, uint32_t x) { v.x += x; v.y += x; v.z += x; v.w += x; }

// ---- kernels --------------------------------------------------------------------------
template <typename T, int UNROLL>
__global__ void k_read_sum(const T* __restrict__ p, size_t n, uint32_t* __restrict__ out) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    size_t i = tid;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        T v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc += hsum(v[u]);
    }
    for (; i < n; i += stride) acc += hsum(p[i]);
    out[tid] = acc;
}

template <typename T>
__global__ void k_write_fill(T* __restrict__ p, size_t n, uint32_t x) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const T v = splat(x, (T*)nullptr);
    for (size_t i = tid; i < n; i += stride) p[i] = v;
}

template <typename T>
__global__ void k_rmw_add(T* p, size_t n, uint32_t x) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        T v = p[i];
        addc(v, x);
        p[i] = v;
    }
}

template <typename T>
__global__ void k_copy(const T* __restrict__ s, T* __restrict__ d, size_t n) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) d[i] = s[i];
}

// Thread t reads element t*S, t*S + nthreads*S, ...  Consecutive lanes are 16*S bytes apart,
// so for S >= 2 each 16 B load pulls its own 32 B sector and the rest of the line is wasted.
__global__ void k_read_strided(const uint4* __restrict__ p, size_t n, int S, uint32_t* __restrict__ out) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x * S;
    uint32_t acc = 0;
    for (size_t i = tid * S; i < n; i += stride) acc += hsum(p[i]);
    out[tid] = acc;
}

__global__ void k_gather(const uint32_t* __restrict__ p, const size_t* __restrict__ idx, uint32_t* __restrict__ out, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < m) out[t] = p[idx[t]];
}

// ---- host harness ---------------------------------------------------------------------
struct Result {
    double median_ms, best_ms;
};

static Result time_it(const std::function<void()>& launch, int warm, int iters) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < warm; ++i) launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
        float t;
        CK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    return {ms[ms.size() / 2], ms[0]};
}

static uint32_t host_reduce(const uint32_t* d_out, size_t nthreads, std::vector<uint32_t>& scratch) {
    scratch.resize(nthreads);
    CK(cudaMemcpy(scratch.data(), d_out, nthreads * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t s = 0;
    for (size_t i = 0; i < nthreads; ++i) s += scratch[i];
    return s;
}

static void print_row(const char* name, const char* launch, double bytes, Result r, bool ok, const char* note = "") {
    printf("  %-28s %-10s %7.3f GB  %8.3f  %8.3f  %8.1f  %8.1f  %s%s\n", name, launch, bytes / 1e9, r.median_ms,
           r.best_ms, bytes / r.median_ms / 1e6, bytes / r.best_ms / 1e6, ok ? "ok" : "**MISMATCH**", note);
}

int main(int argc, char** argv) {
    double gib = 8.0;
    int iters = 10, warm = 2, blocks = 2048, threads = 256;
    bool do_copy = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--gib") && i + 1 < argc) gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-copy")) do_copy = false;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    const size_t bytes = ((size_t)(gib * (1ull << 30)) / 4096) * 4096;  // multiple of 4 KiB
    const size_t nwords = bytes / 4, n16 = bytes / 16, n8 = bytes / 8;

    // ---- device info ----
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    int drv = 0, rt = 0, memclk = 0, bus = 0, smclk = 0, l2 = 0;
    CK(cudaDriverGetVersion(&drv));
    CK(cudaRuntimeGetVersion(&rt));
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);      // kHz, may be 0 on iGPU-style parts
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);    // bits
    cudaDeviceGetAttribute(&smclk, cudaDevAttrClockRate, dev);             // kHz
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    char ts[64];
    time_t now = time(nullptr);
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S %Z", localtime(&now));
    bool big_enough = bytes >= 16ull * (size_t)l2;

    printf("bw_bench  %s\n", ts);
    printf("device    %s  cc %d.%d  %d SMs  L2 %d MB  sm clock %.2f GHz  driver API %d.%d  runtime %d.%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, l2 >> 20, smclk / 1e6, drv / 1000,
           (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    if (memclk > 0 && bus > 0)
        // On GB10 cudaDevAttrMemoryClockRate returns the LPDDR5x data rate (8533 MT/s), not the
        // half-rate clock that discrete GDDR parts report, so no x2 here.
        printf("memory    bus %d bit, attr clock %.0f MHz = data rate -> %.1f GB/s theoretical (spec 273)\n", bus,
               memclk / 1e3, (double)memclk * 1e3 * (bus / 8) / 1e9);
    else
        printf("memory    bus/clock attributes not reported (%d bit, %d kHz); spec LPDDR5x 256-bit 8533 MT/s = 273 GB/s\n",
               bus, memclk);
    printf("buffer    %.2f GiB (%zu B)%s   launch %d x %d   %d iters + %d warmup   GB = 1e9 bytes\n\n", bytes / 1073741824.0,
           bytes, big_enough ? "" : "  **WARNING: buffer < 16x L2, numbers will be inflated**", blocks, threads, iters,
           warm);

    // ---- buffers ----
    uint32_t *buf = nullptr, *buf2 = nullptr, *d_out = nullptr;
    const size_t max_threads = 1ull << 22;  // 4M partials = 16 MB
    if ((size_t)blocks * threads > max_threads) { fprintf(stderr, "blocks*threads too large\n"); return 1; }
    CK(cudaMalloc(&buf, bytes));
    if (do_copy) CK(cudaMalloc(&buf2, bytes));
    CK(cudaMalloc(&d_out, max_threads * sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0, max_threads * sizeof(uint32_t)));
    std::vector<uint32_t> scratch;
    const size_t nt = (size_t)blocks * threads;
    const uint32_t FILL = 1u;
    const uint32_t expect_sum = (uint32_t)(nwords * FILL);  // mod 2^32
    int bad = 0;

    printf("  %-28s %-10s %10s  %8s  %8s  %8s  %8s  %s\n", "test", "launch", "bytes/iter", "med ms", "best ms",
           "med GB/s", "best GB/s", "check");
    printf("  %-28s %-10s %10s  %8s  %8s  %8s  %8s  %s\n", "----", "------", "----------", "------", "-------",
           "--------", "---------", "-----");

    // ---- write (fill) ----
    {
        char lab[32];
        snprintf(lab, sizeof lab, "%dx%d", blocks, threads);
        Result r = time_it([&] { k_write_fill<uint4><<<blocks, threads>>>((uint4*)buf, n16, FILL); }, warm, iters);
        // verify via a read-sum
        k_read_sum<uint4, 4><<<blocks, threads>>>((const uint4*)buf, n16, d_out);
        CK(cudaDeviceSynchronize());
        bool ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("write  16B fill", lab, (double)bytes, r, ok);
    }

    // ---- read: headline + variants ----
    {
        char lab[32];
        snprintf(lab, sizeof lab, "%dx%d", blocks, threads);
        Result r = time_it([&] { k_read_sum<uint4, 4><<<blocks, threads>>>((const uint4*)buf, n16, d_out); }, warm, iters);
        bool ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("read   16B x4 unroll", lab, (double)bytes, r, ok, "   <- headline");

        r = time_it([&] { k_read_sum<uint4, 1><<<blocks, threads>>>((const uint4*)buf, n16, d_out); }, warm, iters);
        ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("read   16B no unroll", lab, (double)bytes, r, ok);

        r = time_it([&] { k_read_sum<uint2, 4><<<blocks, threads>>>((const uint2*)buf, n8, d_out); }, warm, iters);
        ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("read    8B x4 unroll", lab, (double)bytes, r, ok);

        r = time_it([&] { k_read_sum<uint32_t, 4><<<blocks, threads>>>((const uint32_t*)buf, nwords, d_out); }, warm, iters);
        ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("read    4B x4 unroll", lab, (double)bytes, r, ok);
    }

    // ---- read: grid-size sweep (16B x4) ----
    {
        const int sweep[] = {48, 96, 192, 384, 768, 1536, 4096, 8192};
        for (int b : sweep) {
            if ((size_t)b * threads > max_threads) continue;
            char lab[32];
            snprintf(lab, sizeof lab, "%dx%d", b, threads);
            Result r = time_it([&] { k_read_sum<uint4, 4><<<b, threads>>>((const uint4*)buf, n16, d_out); }, warm, iters);
            bool ok = host_reduce(d_out, (size_t)b * threads, scratch) == expect_sum;
            bad += !ok;
            print_row("read   16B x4 (grid sweep)", lab, (double)bytes, r, ok);
        }
    }

    // ---- rmw: in-place x += 2, exact per-word verification ----
    {
        char lab[32];
        snprintf(lab, sizeof lab, "%dx%d", blocks, threads);
        CK(cudaMemset(buf, 0, bytes));
        k_write_fill<uint4><<<blocks, threads>>>((uint4*)buf, n16, FILL);
        CK(cudaDeviceSynchronize());
        const uint32_t ADD = 2u;
        Result r = time_it([&] { k_rmw_add<uint4><<<blocks, threads>>>((uint4*)buf, n16, ADD); }, warm, iters);
        const uint32_t expect_word = FILL + ADD * (uint32_t)(warm + iters);
        const int m = 64;
        std::vector<size_t> idx(m);
        for (int i = 0; i < m; ++i) idx[i] = (size_t)((nwords - 1) * (double)i / (m - 1));
        size_t* d_idx;
        uint32_t* d_g;
        CK(cudaMalloc(&d_idx, m * sizeof(size_t)));
        CK(cudaMalloc(&d_g, m * sizeof(uint32_t)));
        CK(cudaMemcpy(d_idx, idx.data(), m * sizeof(size_t), cudaMemcpyHostToDevice));
        k_gather<<<1, m>>>(buf, d_idx, d_g, m);
        std::vector<uint32_t> g(m);
        CK(cudaMemcpy(g.data(), d_g, m * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < m; ++i) ok &= (g[i] == expect_word);
        k_read_sum<uint4, 4><<<blocks, threads>>>((const uint4*)buf, n16, d_out);
        CK(cudaDeviceSynchronize());
        ok &= host_reduce(d_out, nt, scratch) == (uint32_t)(nwords * expect_word);
        bad += !ok;
        print_row("rmw    16B in-place x+=c", lab, 2.0 * bytes, r, ok, "   (read+write)");
        CK(cudaFree(d_idx));
        CK(cudaFree(d_g));
        // restore the FILL pattern for the remaining tests
        k_write_fill<uint4><<<blocks, threads>>>((uint4*)buf, n16, FILL);
        CK(cudaDeviceSynchronize());
    }

    // ---- copy ----
    if (do_copy) {
        char lab[32];
        snprintf(lab, sizeof lab, "%dx%d", blocks, threads);
        CK(cudaMemset(buf2, 0, bytes));
        Result r = time_it([&] { k_copy<uint4><<<blocks, threads>>>((const uint4*)buf, (uint4*)buf2, n16); }, warm, iters);
        k_read_sum<uint4, 4><<<blocks, threads>>>((const uint4*)buf2, n16, d_out);
        CK(cudaDeviceSynchronize());
        bool ok = host_reduce(d_out, nt, scratch) == expect_sum;
        bad += !ok;
        print_row("copy   16B dst=src", lab, 2.0 * bytes, r, ok, "   (read+write)");
        // cudaMemcpy D2D for reference
        r = time_it([&] { CK(cudaMemcpyAsync(buf2, buf, bytes, cudaMemcpyDeviceToDevice)); }, warm, iters);
        print_row("copy   cudaMemcpy D2D", "-", 2.0 * bytes, r, true);
    }

    // ---- strided 16B ----
    {
        char lab[32];
        snprintf(lab, sizeof lab, "%dx%d", blocks, threads);
        const int strides[] = {1, 2, 4, 8};
        for (int S : strides) {
            const size_t nread = (n16 + S - 1) / S;
            Result r = time_it([&] { k_read_strided<<<blocks, threads>>>((const uint4*)buf, n16, S, d_out); }, warm, iters);
            bool ok = host_reduce(d_out, nt, scratch) == (uint32_t)(nread * 4 * FILL);
            bad += !ok;
            char name[48], note[64];
            snprintf(name, sizeof name, "strided 16B every %3d B", 16 * S);
            snprintf(note, sizeof note, "   useful bytes; %.0f%% of lines touched", 100.0 / S);
            print_row(name, lab, (double)nread * 16, r, ok, S == 1 ? "   (= coalesced)" : note);
        }
    }

    printf("\n%s\n", bad ? "RESULT: VERIFICATION FAILED in one or more tests" : "RESULT: all checks passed");
    CK(cudaFree(buf));
    if (buf2) CK(cudaFree(buf2));
    CK(cudaFree(d_out));
    return bad ? 1 : 0;
}
