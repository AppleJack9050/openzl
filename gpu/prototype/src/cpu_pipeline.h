// gpu/prototype/src/cpu_pipeline.h
//
// Pure-host reference implementation of the exact same pipeline as the GPU path.
// It uses the SAME shared element math and rANS core (codec_core.cuh) and the
// SAME frame builder (frame.h), so cpu_compress() must produce a frame that is
// byte-for-byte identical to gpu_compress(). This is the strongest correctness
// oracle: any kernel bug shows up as a frame diff, independent of round-trip.
//
// Compiled by nvcc (host side). Not performance-critical — clarity first.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include "common.cuh"      // ceil_div
#include "codec_core.cuh"
#include "rans_table.h"
#include "rans.cuh"        // RansEncoded, RANS_SEG
#include "frame.h"
#include "checksum.h"      // content_hash
#include "pipeline.cuh"

namespace gpuzl {

// --- host transforms (mirror the kernels) ----------------------------------
inline void cpu_zigzag_encode(const uint8_t* in, uint8_t* out, size_t n, int W){
    for(size_t i=0;i<n;++i) store_le(out+i*W, zigzag_enc_w(load_le(in+i*W,W),W), W);
}
inline void cpu_zigzag_decode(const uint8_t* in, uint8_t* out, size_t n, int W){
    for(size_t i=0;i<n;++i) store_le(out+i*W, zigzag_dec_w(load_le(in+i*W,W),W), W);
}
inline void cpu_delta_encode(const uint8_t* in, uint8_t* out, size_t n, int W){
    uint64_t prv=0;
    for(size_t i=0;i<n;++i){ uint64_t cur=load_le(in+i*W,W); store_le(out+i*W, mask_w(cur-prv,W), W); prv=cur; }
}
inline void cpu_delta_decode(const uint8_t* in, uint8_t* out, size_t n, int W){
    uint64_t acc=0;
    for(size_t i=0;i<n;++i){ acc = mask_w(acc + load_le(in+i*W,W), W); store_le(out+i*W, acc, W); }
}
inline void cpu_transpose_encode(const uint8_t* in, uint8_t* out, size_t N, int W){
    for(size_t e=0;e<N;++e) for(int p=0;p<W;++p) out[(size_t)p*N+e] = in[e*W+p];
}
inline void cpu_transpose_decode(const uint8_t* in, uint8_t* out, size_t N, int W){
    for(size_t e=0;e<N;++e) for(int p=0;p<W;++p) out[e*W+p] = in[(size_t)p*N+e];
}
inline int cpu_bitpack_nbits(const uint8_t* in, size_t n, int W){
    uint64_t mx=0; for(size_t i=0;i<n;++i){ uint64_t v=load_le(in+i*W,W); if(v>mx) mx=v; }
    if(mx==0) return 1;
    return 1 + (63 - __builtin_clzll(mx));
}
inline void cpu_bitpack_encode(const uint8_t* in, std::vector<uint8_t>& out, size_t n, int W, int nbits){
    out.assign(ceil_div(n*(size_t)nbits,8), 0);
    for(size_t i=0;i<n;++i){
        uint64_t v = mask_bits(load_le(in+i*W,W), nbits);
        size_t bitpos = i*(size_t)nbits;
        for(int b=0;b<nbits;++b) if((v>>b)&1) out[(bitpos+b)>>3] |= (uint8_t)(1u << ((bitpos+b)&7));
    }
}
inline void cpu_bitpack_decode(const uint8_t* in, uint8_t* out, size_t n, int W, int nbits){
    for(size_t i=0;i<n;++i){
        size_t bitpos = i*(size_t)nbits;
        uint64_t v=0;
        for(int b=0;b<nbits;++b){ size_t g=bitpos+b; if((in[g>>3]>>(g&7))&1) v |= (1ull<<b); }
        store_le(out+i*W, v, W);
    }
}

// --- host rANS (mirror rans.cu, byte-identical output) ----------------------
inline void cpu_rans_encode(const uint8_t* in, size_t L, RansEncoded& enc){
    enc.seg_size = RANS_SEG;
    enc.nb_seg   = (uint32_t)ceil_div(L, RANS_SEG);
    enc.seg_len.clear();
    enc.packed.clear();
    enc.total = 0;
    if(L==0){ std::memset(enc.table.freq, 0, sizeof(enc.table.freq)); enc.table.valid=false; return; }
    uint64_t counts[256]; std::memset(counts,0,sizeof(counts));
    for(size_t i=0;i<L;++i) counts[in[i]]++;
    rans_build_table(counts, enc.table);

    const size_t BOUND = (size_t)RANS_SEG*2 + 64;
    std::vector<uint8_t> tmp(BOUND);
    enc.seg_len.resize(enc.nb_seg);
    for(uint32_t seg=0; seg<enc.nb_seg; ++seg){
        size_t base = (size_t)seg*RANS_SEG;
        uint32_t len = (base+RANS_SEG <= L) ? RANS_SEG : (uint32_t)(L-base);
        uint8_t* top = tmp.data()+BOUND;
        uint8_t* ptr = top;
        uint32_t state; rans_enc_init(&state);
        for(int k=(int)len-1;k>=0;--k){
            uint8_t sym = in[base+(uint32_t)k];
            rans_enc_put(&state,&ptr,enc.table.start[sym],enc.table.freq[sym]);
        }
        rans_enc_flush(&state,&ptr);
        uint32_t clen = (uint32_t)(top-ptr);
        enc.seg_len[seg]=clen;
        enc.packed.insert(enc.packed.end(), ptr, top);
    }
    enc.total = enc.packed.size();
}
inline void cpu_rans_decode(const uint8_t* packed, const RansTable& table,
                            const std::vector<uint32_t>& seg_len, uint32_t seg_size,
                            uint32_t nb_seg, size_t L, uint8_t* out){
    if(L==0) return;
    size_t off=0;
    for(uint32_t seg=0; seg<nb_seg; ++seg){
        size_t base=(size_t)seg*seg_size;
        uint32_t len = (base+seg_size <= L) ? seg_size : (uint32_t)(L-base);
        const uint8_t* ptr = packed + off;
        uint32_t state; rans_dec_init(&state,&ptr);
        for(uint32_t k=0;k<len;++k){
            uint32_t slot = rans_dec_get(&state);
            uint8_t sym = table.slot2sym[slot];
            out[base+k]=sym;
            rans_dec_advance(&state,&ptr,table.start[sym],table.freq[sym]);
        }
        off += seg_len[seg];
    }
}

// --- frame-building compress/decompress (host) ------------------------------
inline std::vector<uint8_t> cpu_compress(const uint8_t* input, size_t n, const PipelineConfig& cfg){
    const uint32_t checksum = content_hash(input, n);
    std::vector<uint8_t> A(input, input+n), B(n+64, 0);
    uint8_t* cur = A.data(); uint8_t* other = B.data();
    size_t cur_bytes = n;
    std::vector<StageMeta> metas;
    std::vector<uint8_t> bp; // bitpack scratch (may resize buffers)

    for(const auto& st : cfg.stages){
        const int W = st.elt_width;
        const size_t nelts = cur_bytes / W;
        uint8_t param0 = 0; size_t out_bytes = 0;
        switch(st.codec_id){
            case CODEC_DELTA:     cpu_delta_encode(cur, other, nelts, W);     out_bytes=nelts*W; break;
            case CODEC_ZIGZAG:    cpu_zigzag_encode(cur, other, nelts, W);    out_bytes=nelts*W; break;
            case CODEC_TRANSPOSE: cpu_transpose_encode(cur, other, nelts, W); out_bytes=nelts*W; break;
            case CODEC_BITPACK: {
                int nbits = cpu_bitpack_nbits(cur, nelts, W);
                cpu_bitpack_encode(cur, bp, nelts, W, nbits);
                out_bytes = bp.size();
                std::memcpy(other, bp.data(), out_bytes);
                param0 = (uint8_t)nbits;
                break;
            }
        }
        metas.push_back(StageMeta{st.codec_id,(uint8_t)W,param0,(uint32_t)nelts});
        std::swap(cur, other); // ping-pong: cur now points at the stage output
        cur_bytes = out_bytes;
    }

    EntropyPayload ent;
    ent.pre_entropy_nbytes = cur_bytes;
    if(cfg.entropy == ENT_RANS){
        RansEncoded enc; cpu_rans_encode(cur, cur_bytes, enc);
        ent.mode=ENT_RANS; ent.seg_size=enc.seg_size; ent.nb_seg=enc.nb_seg;
        ent.table=enc.table; ent.seg_len=enc.seg_len; ent.packed=enc.packed;
    } else {
        ent.mode=ENT_STORED; ent.stored.assign(cur, cur+cur_bytes);
    }
    return build_frame(n, checksum, metas, ent);
}

inline std::vector<uint8_t> cpu_decompress(const uint8_t* frame, size_t n, bool* ok=nullptr, std::string* err=nullptr){
    ParsedFrame pf = parse_frame(frame, n);
    if(!pf.ok){ if(ok)*ok=false; if(err)*err=pf.err; return {}; }
    const size_t cap = pf.orig_nbytes + 64;
    { std::string verr; if(!validate_decodable(pf, cap, verr)){ if(ok)*ok=false; if(err)*err=verr; return {}; } }
    const size_t L = pf.pre_entropy_nbytes;
    std::vector<uint8_t> A(cap, 0), B(cap, 0);
    uint8_t* cur=A.data(); uint8_t* other=B.data();
    if(pf.ent.mode == ENT_RANS)
        cpu_rans_decode(pf.ent.packed.data(), pf.ent.table, pf.ent.seg_len, pf.ent.seg_size, pf.ent.nb_seg, L, cur);
    else
        std::memcpy(cur, pf.ent.stored.data(), L);
    size_t cur_bytes = L;
    for(const auto& st : pf.stages_decode_order){
        const int W=st.elt_width; const size_t nelts=st.n_elts;
        switch(st.codec_id){
            case CODEC_DELTA:     cpu_delta_decode(cur, other, nelts, W);     break;
            case CODEC_ZIGZAG:    cpu_zigzag_decode(cur, other, nelts, W);    break;
            case CODEC_TRANSPOSE: cpu_transpose_decode(cur, other, nelts, W); break;
            case CODEC_BITPACK:   cpu_bitpack_decode(cur, other, nelts, W, st.param0); break;
        }
        std::swap(cur, other);
        cur_bytes = nelts*(size_t)W;
    }
    std::vector<uint8_t> out(cur, cur+pf.orig_nbytes);
    (void)cur_bytes;
    uint32_t cs = content_hash(out.data(), out.size());
    bool good = (cs==pf.content_checksum);
    if(ok)*ok=good; if(err){ if(good) err->clear(); else *err="checksum mismatch"; }
    return out;
}

} // namespace gpuzl
