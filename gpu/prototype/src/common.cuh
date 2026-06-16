// gpu/prototype/src/common.cuh
//
// Shared utilities for the OpenZL-style GPU compression prototype:
//   - CUDA error checking
//   - lightweight GPU/CPU timing
//   - byte-buffer helpers
//
// Header-only; included by every translation unit in the prototype.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define ZL_GPU_CHECK(expr)                                                    \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s -> %s\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_err));          \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// Check the last asynchronous kernel launch (call after a kernel).
#define ZL_GPU_CHECK_KERNEL()                                                 \
    do {                                                                      \
        cudaError_t _e1 = cudaGetLastError();                                 \
        if (_e1 != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA launch error %s:%d: %s\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(_e1));                  \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// ---------------------------------------------------------------------------
// GPU timing (CUDA events). Measures device-side wall time of a region.
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t beg{}, end{};
    GpuTimer() {
        ZL_GPU_CHECK(cudaEventCreate(&beg));
        ZL_GPU_CHECK(cudaEventCreate(&end));
    }
    ~GpuTimer() {
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }
    void start() { ZL_GPU_CHECK(cudaEventRecord(beg)); }
    // Returns elapsed milliseconds since start().
    float stop() {
        ZL_GPU_CHECK(cudaEventRecord(end));
        ZL_GPU_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        ZL_GPU_CHECK(cudaEventElapsedTime(&ms, beg, end));
        return ms;
    }
};

// ---------------------------------------------------------------------------
// CPU timing
// ---------------------------------------------------------------------------
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Throughput helper: GB/s given bytes processed and milliseconds elapsed.
// ---------------------------------------------------------------------------
static inline double gbps(size_t bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return (double)bytes / (ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// Simple RAII device buffer.
// ---------------------------------------------------------------------------
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { alloc(n); }
    ~DeviceBuffer() { free(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr; o.count = 0;
    }
    void alloc(size_t n) {
        free();
        count = n;
        if (n) ZL_GPU_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    void free() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr; count = 0;
    }
    size_t bytes() const { return count * sizeof(T); }
    void from_host(const T* h, size_t n) {
        if (count < n) alloc(n);
        ZL_GPU_CHECK(cudaMemcpy(ptr, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void to_host(T* h, size_t n) const {
        ZL_GPU_CHECK(cudaMemcpy(h, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Round up a / b.
static inline constexpr size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}
