// gpu/prototype/src/rans.cu
//
// GPU kernels + host orchestration for the interleaved (per-segment) rANS coder.
#include "rans.cuh"
#include "common.cuh"
#include "codec_core.cuh"

namespace gpuzl {

static constexpr int BS = 256;

static inline int grid_cap(size_t n, int bs){
    size_t g = ceil_div(n, bs);
    return (int)(g > 65535 ? 65535 : (g ? g : 1));
}

// ---------------------------------------------------------------------------
// Histogram: privatized shared-memory 256-bin, then atomic-add to global u64.
// ---------------------------------------------------------------------------
__global__ void k_histogram(const uint8_t* in, size_t L, unsigned long long* gcounts){
    __shared__ unsigned int local[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) local[i]=0;
    __syncthreads();
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < L;
        i += (size_t)gridDim.x*blockDim.x)
        atomicAdd(&local[in[i]], 1u);
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=blockDim.x)
        if(local[i]) atomicAdd(&gcounts[i], (unsigned long long)local[i]);
}

void rans_histogram(const uint8_t* d_in, size_t L, uint64_t counts[256], cudaStream_t s){
    unsigned long long* d_counts=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_counts, 256*sizeof(unsigned long long), s));
    ZL_GPU_CHECK(cudaMemsetAsync(d_counts, 0, 256*sizeof(unsigned long long), s));
    if(L){
        k_histogram<<<grid_cap(L,BS), BS, 0, s>>>(d_in, L, d_counts);
        ZL_GPU_CHECK_KERNEL();
    }
    ZL_GPU_CHECK(cudaMemcpyAsync(counts, d_counts, 256*sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
    ZL_GPU_CHECK(cudaFreeAsync(d_counts, s));
}

// ---------------------------------------------------------------------------
// Encode: one thread per segment, reverse-encode into a bounded scratch slot.
// ---------------------------------------------------------------------------
__global__ void k_rans_encode(const uint8_t* in, size_t L,
                              const uint16_t* freq, const uint16_t* start,
                              uint8_t* scratch, size_t BOUND,
                              uint32_t SEG, uint32_t nbSeg, uint32_t* segLen){
    for(uint32_t seg = blockIdx.x*blockDim.x + threadIdx.x; seg < nbSeg;
        seg += gridDim.x*blockDim.x){
        size_t base = (size_t)seg * SEG;
        uint32_t len = (base + SEG <= L) ? SEG : (uint32_t)(L - base);
        uint8_t* top = scratch + (size_t)seg*BOUND + BOUND;
        uint8_t* ptr = top;
        uint32_t state; rans_enc_init(&state);
        for(int k=(int)len-1; k>=0; --k){
            uint8_t sym = in[base + (uint32_t)k];
            rans_enc_put(&state, &ptr, start[sym], freq[sym]);
        }
        rans_enc_flush(&state, &ptr);
        segLen[seg] = (uint32_t)(top - ptr);
    }
}

// Compaction: one block per segment, threads copy the blob to its packed offset.
// Offsets are 64-bit so the packed stream may exceed 4 GiB.
__global__ void k_compact(const uint8_t* scratch, size_t BOUND,
                          const uint32_t* segLen, const uint64_t* offset,
                          uint32_t nbSeg, uint8_t* packed){
    for(uint32_t seg = blockIdx.x; seg < nbSeg; seg += gridDim.x){
        uint32_t len = segLen[seg];
        const uint8_t* src = scratch + (size_t)seg*BOUND + (BOUND - len);
        uint8_t* dst = packed + offset[seg];
        for(uint32_t j=threadIdx.x; j<len; j+=blockDim.x) dst[j] = src[j];
    }
}

void rans_encode(const uint8_t* d_in, size_t L, RansEncoded& enc, cudaStream_t s){
    enc.seg_size = RANS_SEG;
    enc.nb_seg   = (uint32_t)ceil_div(L, RANS_SEG);
    enc.total    = 0;
    enc.d_packed = nullptr;
    enc.seg_len.clear();
    if(L==0){ std::memset(enc.table.freq, 0, sizeof(enc.table.freq)); enc.table.valid=false; return; }

    // 1) histogram + normalized table (host).
    uint64_t counts[256];
    rans_histogram(d_in, L, counts, s);
    rans_build_table(counts, enc.table);

    // 2) device copies of freq/start.
    uint16_t *d_freq=nullptr, *d_start=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_freq,  256*sizeof(uint16_t), s));
    ZL_GPU_CHECK(cudaMallocAsync(&d_start, 256*sizeof(uint16_t), s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_freq,  enc.table.freq,  256*sizeof(uint16_t), cudaMemcpyHostToDevice, s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_start, enc.table.start, 256*sizeof(uint16_t), cudaMemcpyHostToDevice, s));

    // 3) per-segment encode into scratch.
    const size_t BOUND = (size_t)RANS_SEG*2 + 64;  // safe upper bound on clen
    uint8_t*  d_scratch=nullptr;
    uint32_t* d_segLen=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_scratch, (size_t)enc.nb_seg*BOUND, s));
    ZL_GPU_CHECK(cudaMallocAsync(&d_segLen,  (size_t)enc.nb_seg*sizeof(uint32_t), s));
    k_rans_encode<<<grid_cap(enc.nb_seg,BS), BS, 0, s>>>(
        d_in, L, d_freq, d_start, d_scratch, BOUND, RANS_SEG, enc.nb_seg, d_segLen);
    ZL_GPU_CHECK_KERNEL();

    // 4) bring seg lengths to host, then compute 64-bit offsets on the host
    //    (each seg_len < BOUND fits in u32, but the running offset and total can
    //    exceed 4 GiB, so they must be 64-bit).
    enc.seg_len.resize(enc.nb_seg);
    ZL_GPU_CHECK(cudaMemcpyAsync(enc.seg_len.data(), d_segLen,
                 (size_t)enc.nb_seg*sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));

    std::vector<uint64_t> offsets(enc.nb_seg);
    size_t total = 0;
    for(uint32_t i=0;i<enc.nb_seg;++i){ offsets[i] = total; total += enc.seg_len[i]; }
    enc.total = total;

    uint64_t* d_offset=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_offset, (size_t)enc.nb_seg*sizeof(uint64_t), s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_offset, offsets.data(),
                 (size_t)enc.nb_seg*sizeof(uint64_t), cudaMemcpyHostToDevice, s));

    // 5) compact into packed buffer.
    ZL_GPU_CHECK(cudaMallocAsync(&enc.d_packed, enc.total ? enc.total : 1, s));
    k_compact<<<grid_cap(enc.nb_seg,1), 128, 0, s>>>(
        d_scratch, BOUND, d_segLen, d_offset, enc.nb_seg, enc.d_packed);
    ZL_GPU_CHECK_KERNEL();
    ZL_GPU_CHECK(cudaStreamSynchronize(s));

    ZL_GPU_CHECK(cudaFreeAsync(d_offset, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_segLen, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_scratch, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_freq, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_start, s));
}

// ---------------------------------------------------------------------------
// Decode: one thread per segment, forward-decode using the slot table.
// ---------------------------------------------------------------------------
__global__ void k_rans_decode(const uint8_t* packed, const uint64_t* offset,
                              const uint8_t* slot2sym, const uint16_t* freq,
                              const uint16_t* start, uint8_t* out, size_t L,
                              uint32_t SEG, uint32_t nbSeg){
    for(uint32_t seg = blockIdx.x*blockDim.x + threadIdx.x; seg < nbSeg;
        seg += gridDim.x*blockDim.x){
        size_t base = (size_t)seg * SEG;
        if(base >= L) continue; // defensive: never underflow len (valid frames satisfy base<L)
        uint32_t len = (base + SEG <= L) ? SEG : (uint32_t)(L - base);
        const uint8_t* ptr = packed + offset[seg];
        uint32_t state; rans_dec_init(&state, &ptr);
        for(uint32_t k=0;k<len;++k){
            uint32_t slot = rans_dec_get(&state);
            uint8_t sym = slot2sym[slot];
            out[base + k] = sym;
            rans_dec_advance(&state, &ptr, start[sym], freq[sym]);
        }
    }
}

void rans_decode(const uint8_t* d_packed, size_t /*total*/, const RansTable& table,
                 const std::vector<uint32_t>& seg_len, uint32_t seg_size,
                 uint32_t nb_seg, size_t L, uint8_t* d_out, cudaStream_t s){
    if(L==0) return;
    // Rebuild 64-bit offsets from seg_len (host prefix sum) and upload.
    std::vector<uint64_t> offset(nb_seg);
    size_t acc=0;
    for(uint32_t i=0;i<nb_seg;++i){ offset[i]=acc; acc+=seg_len[i]; }

    uint64_t* d_offset=nullptr;
    uint16_t *d_freq=nullptr, *d_start=nullptr;
    uint8_t*  d_slot=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&d_offset, (size_t)nb_seg*sizeof(uint64_t), s));
    ZL_GPU_CHECK(cudaMallocAsync(&d_freq,  256*sizeof(uint16_t), s));
    ZL_GPU_CHECK(cudaMallocAsync(&d_start, 256*sizeof(uint16_t), s));
    ZL_GPU_CHECK(cudaMallocAsync(&d_slot,  RANS_PROB_SCALE*sizeof(uint8_t), s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_offset, offset.data(), (size_t)nb_seg*sizeof(uint64_t), cudaMemcpyHostToDevice, s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_freq,  table.freq,  256*sizeof(uint16_t), cudaMemcpyHostToDevice, s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_start, table.start, 256*sizeof(uint16_t), cudaMemcpyHostToDevice, s));
    ZL_GPU_CHECK(cudaMemcpyAsync(d_slot,  table.slot2sym, RANS_PROB_SCALE*sizeof(uint8_t), cudaMemcpyHostToDevice, s));

    k_rans_decode<<<grid_cap(nb_seg,BS), BS, 0, s>>>(
        d_packed, d_offset, d_slot, d_freq, d_start, d_out, L, seg_size, nb_seg);
    ZL_GPU_CHECK_KERNEL();

    ZL_GPU_CHECK(cudaFreeAsync(d_offset, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_freq, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_start, s));
    ZL_GPU_CHECK(cudaFreeAsync(d_slot, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
}

} // namespace gpuzl
