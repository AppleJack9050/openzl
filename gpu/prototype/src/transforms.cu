// gpu/prototype/src/transforms.cu
//
// GPU kernels + host launchers for the OpenZL-style transform codecs.
// Element math is shared with the CPU reference via codec_core.cuh.
#include "transforms.cuh"
#include "common.cuh"
#include "codec_core.cuh"

#include <cub/cub.cuh>

namespace gpuzl {

static constexpr int BS = 256; // default block size

// ===========================================================================
// ZIGZAG — embarrassingly parallel, one element per thread (grid-stride).
// ===========================================================================
__global__ void k_zigzag_encode(const uint8_t* in, uint8_t* out, size_t n, int W){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        uint64_t v = load_le(in + i*W, W);
        store_le(out + i*W, zigzag_enc_w(v, W), W);
    }
}
__global__ void k_zigzag_decode(const uint8_t* in, uint8_t* out, size_t n, int W){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        uint64_t v = load_le(in + i*W, W);
        store_le(out + i*W, zigzag_dec_w(v, W), W);
    }
}

static inline int grid_for(size_t n){
    size_t g = ceil_div(n, BS);
    return (int)(g > 65535 ? 65535 : (g ? g : 1)); // cap; grid-stride covers rest
}

void zigzag_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    k_zigzag_encode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
}
void zigzag_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    k_zigzag_decode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
}

// ===========================================================================
// DELTA encode — stencil: out[0]=src[0]; out[k]=src[k]-src[k-1] (mod 2^W).
// (We store `first` inline as element 0, the OpenZL legacy/v12- form, so the
//  stage stays length-preserving and decode is a pure inclusive prefix sum.)
// ===========================================================================
__global__ void k_delta_encode(const uint8_t* in, uint8_t* out, size_t n, int W){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        uint64_t cur = load_le(in + i*W, W);
        uint64_t prv = (i==0) ? 0 : load_le(in + (i-1)*W, W);
        store_le(out + i*W, mask_w(cur - prv, W), W);
    }
}
void delta_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    k_delta_encode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
}

// DELTA decode = inclusive modular prefix sum of the W-byte elements.
// Strategy: expand W-byte LE elements -> u64, CUB InclusiveScan with a modular
// add op (mask to width each combine, so any scan tree is width-correct),
// contract back to W-byte LE. CUB gives a single-pass decoupled-look-back scan.
struct ModAddW {
    int W;
    __host__ __device__ uint64_t operator()(uint64_t a, uint64_t b) const {
        return mask_w(a + b, W);
    }
};
__global__ void k_expand(const uint8_t* in, uint64_t* vals, size_t n, int W){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x)
        vals[i] = load_le(in + i*W, W);
}
__global__ void k_contract(const uint64_t* vals, uint8_t* out, size_t n, int W){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x)
        store_le(out + i*W, mask_w(vals[i], W), W);
}
void delta_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    uint64_t* vals = nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&vals, n*sizeof(uint64_t), s));
    k_expand<<<grid_for(n), BS, 0, s>>>(d_in, vals, n, W);
    ZL_GPU_CHECK_KERNEL();

    ModAddW op{W};
    void*  tmp = nullptr; size_t tmpBytes = 0;
    // First call (tmp==null) queries the required temp size; second does the scan.
    cub::DeviceScan::InclusiveScan(tmp, tmpBytes, vals, vals, op, n, s);
    ZL_GPU_CHECK(cudaMallocAsync(&tmp, tmpBytes, s));
    cub::DeviceScan::InclusiveScan(tmp, tmpBytes, vals, vals, op, n, s);

    k_contract<<<grid_for(n), BS, 0, s>>>(vals, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
    ZL_GPU_CHECK(cudaFreeAsync(tmp, s));
    ZL_GPU_CHECK(cudaFreeAsync(vals, s));
}

// ===========================================================================
// TRANSPOSE — byte-plane (SoA) permutation. N elements x W bytes.
//   encode: out[pos*N + elt] = in[elt*W + pos]
//   decode: out[elt*W + pos] = in[pos*N + elt]
// One thread per (elt) copying its W bytes. (W is tiny: 1..8.)
// ===========================================================================
__global__ void k_transpose_encode(const uint8_t* in, uint8_t* out, size_t N, int W){
    for(size_t e = blockIdx.x*(size_t)blockDim.x + threadIdx.x; e < N;
        e += (size_t)gridDim.x*blockDim.x){
        const uint8_t* src = in + e*W;
        for(int p=0;p<W;++p) out[(size_t)p*N + e] = src[p];
    }
}
__global__ void k_transpose_decode(const uint8_t* in, uint8_t* out, size_t N, int W){
    for(size_t e = blockIdx.x*(size_t)blockDim.x + threadIdx.x; e < N;
        e += (size_t)gridDim.x*blockDim.x){
        uint8_t* dst = out + e*W;
        for(int p=0;p<W;++p) dst[p] = in[(size_t)p*N + e];
    }
}
void transpose_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    k_transpose_encode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
}
void transpose_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s){
    if(!n) return;
    k_transpose_decode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W);
    ZL_GPU_CHECK_KERNEL();
}

// ===========================================================================
// BITPACK — LSB-first, little-endian, uniform nbits.
// ===========================================================================
__global__ void k_max_reduce(const uint8_t* in, size_t n, int W, unsigned long long* gmax){
    __shared__ unsigned long long smax;
    if(threadIdx.x==0) smax=0;
    __syncthreads();
    unsigned long long local=0;
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        unsigned long long v = load_le(in + i*W, W);
        if(v>local) local=v;
    }
    atomicMax(&smax, local);
    __syncthreads();
    if(threadIdx.x==0) atomicMax(gmax, smax);
}
int bitpack_compute_nbits(const uint8_t* d_in, size_t n, int W, cudaStream_t s){
    if(!n) return 1;
    unsigned long long* d_max=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_max, sizeof(unsigned long long), s));
    ZL_GPU_CHECK(cudaMemsetAsync(d_max, 0, sizeof(unsigned long long), s));
    k_max_reduce<<<grid_for(n), BS, 0, s>>>(d_in, n, W, d_max);
    ZL_GPU_CHECK_KERNEL();
    unsigned long long h_max=0;
    ZL_GPU_CHECK(cudaMemcpyAsync(&h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
    ZL_GPU_CHECK(cudaFreeAsync(d_max, s));
    if(h_max==0) return 1;
    int hb = 63 - __builtin_clzll(h_max);
    return hb + 1; // 1 + highbit
}

__global__ void k_bitpack_encode(const uint8_t* in, uint32_t* out_words, size_t n, int W, int nbits){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        uint64_t v = mask_bits(load_le(in + i*W, W), nbits);
        size_t bitpos = i*(size_t)nbits;
        size_t word = bitpos >> 5;
        int off = (int)(bitpos & 31);
        atomicOr(&out_words[word], (uint32_t)(v << off));
        if(off + nbits > 32) atomicOr(&out_words[word+1], (uint32_t)(v >> (32-off)));
        if(off + nbits > 64) atomicOr(&out_words[word+2], (uint32_t)(v >> (64-off)));
    }
}
void bitpack_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, int nbits, cudaStream_t s){
    if(!n) return;
    k_bitpack_encode<<<grid_for(n), BS, 0, s>>>(d_in, (uint32_t*)d_out, n, W, nbits);
    ZL_GPU_CHECK_KERNEL();
}

__global__ void k_bitpack_decode(const uint8_t* in, uint8_t* out, size_t n, int W, int nbits){
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
        i += (size_t)gridDim.x*blockDim.x){
        size_t bitpos = i*(size_t)nbits;
        size_t byte = bitpos >> 3;
        int bit = (int)(bitpos & 7);
        uint64_t lo = load_le(in + byte, 8);     // requires >=16B padding past end
        uint64_t v;
        if(nbits + bit <= 64){
            v = (lo >> bit);
        } else {
            uint64_t hi = load_le(in + byte + 8, 8);
            v = (lo >> bit) | (hi << (64 - bit)); // bit>=1 here, so no UB
        }
        v = mask_bits(v, nbits);
        store_le(out + i*W, v, W);
    }
}
void bitpack_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, int nbits, cudaStream_t s){
    if(!n) return;
    k_bitpack_decode<<<grid_for(n), BS, 0, s>>>(d_in, d_out, n, W, nbits);
    ZL_GPU_CHECK_KERNEL();
}

} // namespace gpuzl
