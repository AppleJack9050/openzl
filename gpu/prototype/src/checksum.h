// gpu/prototype/src/checksum.h
//
// Content checksum used for round-trip verification. Plain serial FNV over a
// large buffer is a host bottleneck (~1.5 GB/s), so we use a CHUNKED FNV: each
// 4 KB chunk is hashed independently (parallel on the GPU) and the chunk hashes
// are combined with a final FNV. The host and device produce identical results
// for the same chunking, so CPU-reference and GPU frames stay byte-identical.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>

namespace gpuzl {

static const size_t HASH_CHUNK = 4096;
static const uint64_t FNV_OFFSET = 1469598103934665603ull;
static const uint64_t FNV_PRIME  = 1099511628211ull;

inline uint64_t fnv64_range(const uint8_t* p, size_t len){
    uint64_t h = FNV_OFFSET;
    for(size_t i=0;i<len;++i){ h ^= p[i]; h *= FNV_PRIME; }
    return h;
}

// Combine per-chunk 64-bit hashes (in chunk order) into the final 32-bit digest.
inline uint32_t combine_chunks(const uint64_t* hc, size_t nch){
    uint64_t h = FNV_OFFSET;
    for(size_t c=0;c<nch;++c){
        uint64_t v = hc[c];
        for(int b=0;b<8;++b){ h ^= (uint8_t)(v >> (8*b)); h *= FNV_PRIME; }
    }
    return (uint32_t)(h & 0xffffffffu);
}

// Host reference (used by the CPU pipeline + as the canonical definition).
inline uint32_t content_hash(const uint8_t* p, size_t n){
    if(n==0) return combine_chunks(nullptr, 0);
    size_t nch = (n + HASH_CHUNK - 1) / HASH_CHUNK;
    std::vector<uint64_t> hc(nch);
    for(size_t c=0;c<nch;++c){
        size_t off = c*HASH_CHUNK;
        size_t len = (off+HASH_CHUNK <= n) ? HASH_CHUNK : (n-off);
        hc[c] = fnv64_range(p+off, len);
    }
    return combine_chunks(hc.data(), nch);
}

// GPU launcher (defined in checksum.cu): hashes a DEVICE buffer in parallel and
// returns the same digest as content_hash() over the equivalent host bytes.
uint32_t gpu_content_hash(const uint8_t* d_in, size_t n, cudaStream_t s = 0);

} // namespace gpuzl
