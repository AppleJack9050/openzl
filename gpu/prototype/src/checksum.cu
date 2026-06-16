// gpu/prototype/src/checksum.cu — GPU implementation of the chunked content hash.
#include "checksum.h"
#include "common.cuh"

namespace gpuzl {

// One thread per 4 KB chunk; serial FNV within the chunk, fully parallel across.
__global__ void k_chunk_hash(const uint8_t* in, size_t n, uint64_t* hc, size_t nch){
    for(size_t c = blockIdx.x*(size_t)blockDim.x + threadIdx.x; c < nch;
        c += (size_t)gridDim.x*blockDim.x){
        size_t off = c*HASH_CHUNK;
        size_t len = (off+HASH_CHUNK <= n) ? HASH_CHUNK : (n-off);
        const uint8_t* p = in + off;
        uint64_t h = FNV_OFFSET;
        for(size_t i=0;i<len;++i){ h ^= p[i]; h *= FNV_PRIME; }
        hc[c] = h;
    }
}

uint32_t gpu_content_hash(const uint8_t* d_in, size_t n, cudaStream_t s){
    if(n==0) return combine_chunks(nullptr, 0);
    size_t nch = (n + HASH_CHUNK - 1) / HASH_CHUNK;
    uint64_t* d_hc=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_hc, nch*sizeof(uint64_t), s));
    const int bs=256;
    size_t g = (nch + bs - 1)/bs;
    int grid = (int)(g > 65535 ? 65535 : (g ? g : 1));
    k_chunk_hash<<<grid, bs, 0, s>>>(d_in, n, d_hc, nch);
    ZL_GPU_CHECK_KERNEL();
    std::vector<uint64_t> hc(nch);
    ZL_GPU_CHECK(cudaMemcpyAsync(hc.data(), d_hc, nch*sizeof(uint64_t), cudaMemcpyDeviceToHost, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
    ZL_GPU_CHECK(cudaFreeAsync(d_hc, s));
    return combine_chunks(hc.data(), nch);
}

} // namespace gpuzl
