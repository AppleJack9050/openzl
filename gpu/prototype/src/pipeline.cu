// gpu/prototype/src/pipeline.cu
//
// Host orchestration of the GPU compress/decompress pipeline. The control plane
// (stage ordering, frame (de)serialization) runs on the host; the data plane
// (transforms + rANS) runs on the GPU via transforms.cuh / rans.cuh.
#include "pipeline.cuh"
#include "transforms.cuh"
#include "rans.cuh"
#include "checksum.h"
#include "common.cuh"

#include <cassert>

namespace gpuzl {

static inline size_t round_up(size_t a, size_t m){ return ((a + m - 1)/m)*m; }

std::vector<uint8_t> gpu_compress(const uint8_t* input, size_t n, const PipelineConfig& cfg){
    cudaStream_t s; ZL_GPU_CHECK(cudaStreamCreate(&s));
    const size_t cap = n + 64;
    uint8_t *A=nullptr, *B=nullptr;
    // Pooled (stream-ordered) allocation: repeated calls reuse pool memory
    // instead of paying a full cudaMalloc/cudaFree of large buffers each time.
    ZL_GPU_CHECK(cudaMallocAsync(&A, cap, s));
    ZL_GPU_CHECK(cudaMallocAsync(&B, cap, s));
    if(n) ZL_GPU_CHECK(cudaMemcpyAsync(A, input, n, cudaMemcpyHostToDevice, s));
    // Content checksum computed on the GPU over the resident input (parallel),
    // avoiding a ~1.5 GB/s serial host hash of the whole buffer.
    const uint32_t checksum = gpu_content_hash(A, n, s);

    uint8_t* cur = A; uint8_t* other = B;
    size_t cur_bytes = n;
    std::vector<StageMeta> metas;

    for(const auto& st : cfg.stages){
        const int W = st.elt_width;
        assert(cur_bytes % W == 0 && "stage input not a multiple of element width");
        const size_t nelts = cur_bytes / W;
        uint8_t param0 = 0;
        size_t out_bytes = 0;
        switch(st.codec_id){
            case CODEC_DELTA:     delta_encode(cur, other, nelts, W, s);     out_bytes = nelts*W; break;
            case CODEC_ZIGZAG:    zigzag_encode(cur, other, nelts, W, s);    out_bytes = nelts*W; break;
            case CODEC_TRANSPOSE: transpose_encode(cur, other, nelts, W, s); out_bytes = nelts*W; break;
            case CODEC_BITPACK: {
                int nbits = bitpack_compute_nbits(cur, nelts, W, s);
                out_bytes = ceil_div(nelts*(size_t)nbits, 8);
                ZL_GPU_CHECK(cudaMemsetAsync(other, 0, round_up(out_bytes,4), s));
                bitpack_encode(cur, other, nelts, W, nbits, s);
                param0 = (uint8_t)nbits;
                break;
            }
            default: assert(false);
        }
        metas.push_back(StageMeta{st.codec_id, (uint8_t)W, param0, (uint32_t)nelts});
        std::swap(cur, other);
        cur_bytes = out_bytes;
    }

    EntropyPayload ent;
    ent.pre_entropy_nbytes = cur_bytes;
    if(cfg.entropy == ENT_RANS){
        RansEncoded enc;
        rans_encode(cur, cur_bytes, enc, s);
        ent.mode    = ENT_RANS;
        ent.seg_size= enc.seg_size;
        ent.nb_seg  = enc.nb_seg;
        ent.table   = enc.table;
        ent.seg_len = enc.seg_len;
        ent.packed.resize(enc.total);
        if(enc.total)
            ZL_GPU_CHECK(cudaMemcpyAsync(ent.packed.data(), enc.d_packed, enc.total, cudaMemcpyDeviceToHost, s));
        ZL_GPU_CHECK(cudaStreamSynchronize(s));
        if(enc.d_packed) ZL_GPU_CHECK(cudaFree(enc.d_packed));
    } else {
        ent.mode = ENT_STORED;
        ent.stored.resize(cur_bytes);
        if(cur_bytes)
            ZL_GPU_CHECK(cudaMemcpyAsync(ent.stored.data(), cur, cur_bytes, cudaMemcpyDeviceToHost, s));
        ZL_GPU_CHECK(cudaStreamSynchronize(s));
    }

    ZL_GPU_CHECK(cudaFreeAsync(A, s));
    ZL_GPU_CHECK(cudaFreeAsync(B, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
    ZL_GPU_CHECK(cudaStreamDestroy(s));
    return build_frame(n, checksum, metas, ent);
}

std::vector<uint8_t> gpu_decompress(const uint8_t* frame, size_t n, bool* ok, std::string* err){
    ParsedFrame pf = parse_frame(frame, n);
    if(!pf.ok){ if(ok)*ok=false; if(err)*err=pf.err; return {}; }
    const size_t cap = (size_t)pf.orig_nbytes + 64;
    { std::string verr; if(!validate_decodable(pf, cap, verr)){ if(ok)*ok=false; if(err)*err=verr; return {}; } }

    cudaStream_t s; ZL_GPU_CHECK(cudaStreamCreate(&s));
    uint8_t *A=nullptr, *B=nullptr;
    ZL_GPU_CHECK(cudaMallocAsync(&A, cap ? cap : 64, s));
    ZL_GPU_CHECK(cudaMallocAsync(&B, cap ? cap : 64, s));

    const size_t L = pf.pre_entropy_nbytes;

    // Entropy decode -> A (L bytes).
    if(pf.ent.mode == ENT_RANS){
        uint8_t* d_packed=nullptr;
        const size_t pk = pf.ent.packed.size();
        // +16 padding so a corrupt-but-valid-length packed stream cannot drive a
        // renorm read past the end of the allocation (the digest still catches it).
        ZL_GPU_CHECK(cudaMallocAsync(&d_packed, pk + 16, s));
        ZL_GPU_CHECK(cudaMemsetAsync(d_packed, 0, pk + 16, s));
        if(pk) ZL_GPU_CHECK(cudaMemcpyAsync(d_packed, pf.ent.packed.data(), pk, cudaMemcpyHostToDevice, s));
        rans_decode(d_packed, pf.ent.packed.size(), pf.ent.table, pf.ent.seg_len,
                    pf.ent.seg_size, pf.ent.nb_seg, L, A, s);
        ZL_GPU_CHECK(cudaFreeAsync(d_packed, s));
    } else {
        if(L) ZL_GPU_CHECK(cudaMemcpyAsync(A, pf.ent.stored.data(), L, cudaMemcpyHostToDevice, s));
    }

    uint8_t* cur = A; uint8_t* other = B;
    size_t cur_bytes = L;
    for(const auto& st : pf.stages_decode_order){
        const int W = st.elt_width;
        const size_t nelts = st.n_elts;
        size_t out_bytes = nelts*(size_t)W;
        switch(st.codec_id){
            case CODEC_DELTA:     delta_decode(cur, other, nelts, W, s);     break;
            case CODEC_ZIGZAG:    zigzag_decode(cur, other, nelts, W, s);    break;
            case CODEC_TRANSPOSE: transpose_decode(cur, other, nelts, W, s); break;
            case CODEC_BITPACK:   bitpack_decode(cur, other, nelts, W, st.param0, s); break;
            default: if(ok)*ok=false; if(err)*err="bad codec id"; return {};
        }
        std::swap(cur, other);
        cur_bytes = out_bytes;
    }

    // Verify the checksum on the GPU over the resident decoded buffer (parallel).
    uint32_t cs = gpu_content_hash(cur, cur_bytes, s);

    std::vector<uint8_t> out(pf.orig_nbytes);
    if(pf.orig_nbytes)
        ZL_GPU_CHECK(cudaMemcpyAsync(out.data(), cur, pf.orig_nbytes, cudaMemcpyDeviceToHost, s));
    ZL_GPU_CHECK(cudaFreeAsync(A, s));
    ZL_GPU_CHECK(cudaFreeAsync(B, s));
    ZL_GPU_CHECK(cudaStreamSynchronize(s));
    ZL_GPU_CHECK(cudaStreamDestroy(s));

    if(cur_bytes != pf.orig_nbytes){ if(ok)*ok=false; if(err)*err="size mismatch after inverse chain"; return out; }
    if(cs != pf.content_checksum){ if(ok)*ok=false; if(err)*err="checksum mismatch"; return out; }
    if(ok)*ok=true; if(err)err->clear();
    return out;
}

} // namespace gpuzl
