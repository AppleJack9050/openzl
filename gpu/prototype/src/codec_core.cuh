// gpu/prototype/src/codec_core.cuh
//
// The single source of truth for all per-element codec math and the rANS
// entropy coder. Every function here is marked ZL_HD so the SAME code compiles
// for the host (CPU reference) and the device (GPU kernels). That guarantees
// the CPU and GPU paths are bit-for-bit identical, which the tests rely on.
//
// Semantics mirror OpenZL's codecs (see src/openzl/codecs/{delta,zigzag,
// transpose,bitpack}/spec.md):
//   * zigzag : p>=0 -> 2p,  n<0 -> 2|n|-1   (LSB sign interleave)
//   * delta  : encoded[0]=src[0]; encoded[k]=src[k]-src[k-1] (mod 2^W);
//              decode = inclusive modular prefix sum
//   * bitpack: LSB-first, little-endian; element i at bit [i*nbits,(i+1)*nbits)
// The entropy stage is interleaved byte-wise rANS (Ryg's rans_byte scheme),
// the GPU-appropriate analog of OpenZL's serial FSE/Huffman.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define ZL_HD __host__ __device__ __forceinline__
#else
#define ZL_HD inline
#endif

// ===========================================================================
// Zig-zag (signed <-> unsigned interleave), per element width.
// Encode reads a SIGNED value, returns UNSIGNED. Decode is the inverse.
// We use the explicit sign-mask form (no implementation-defined signed shift).
// ===========================================================================
ZL_HD uint8_t  zigzag_enc8 (uint8_t  u){ int8_t  n=(int8_t)u;  return (uint8_t )((uint8_t )(u<<1) ^ (uint8_t )(0u-(uint8_t )(n<0))); }
ZL_HD uint16_t zigzag_enc16(uint16_t u){ int16_t n=(int16_t)u; return (uint16_t)((uint16_t)(u<<1) ^ (uint16_t)(0u-(uint16_t)(n<0))); }
ZL_HD uint32_t zigzag_enc32(uint32_t u){ int32_t n=(int32_t)u; return (uint32_t)((uint32_t)(u<<1) ^ (uint32_t)(0u-(uint32_t)(n<0))); }
ZL_HD uint64_t zigzag_enc64(uint64_t u){ int64_t n=(int64_t)u; return (uint64_t)((uint64_t)(u<<1) ^ (uint64_t)(0ull-(uint64_t)(n<0))); }

ZL_HD uint8_t  zigzag_dec8 (uint8_t  z){ uint8_t  m=(uint8_t )(0u-(uint8_t )(z&1)); return (uint8_t )((z>>1) ^ m); }
ZL_HD uint16_t zigzag_dec16(uint16_t z){ uint16_t m=(uint16_t)(0u-(uint16_t)(z&1)); return (uint16_t)((z>>1) ^ m); }
ZL_HD uint32_t zigzag_dec32(uint32_t z){ uint32_t m=(uint32_t)(0u-(uint32_t)(z&1)); return (uint32_t)((z>>1) ^ m); }
ZL_HD uint64_t zigzag_dec64(uint64_t z){ uint64_t m=(uint64_t)(0ull-(uint64_t)(z&1)); return (uint64_t)((z>>1) ^ m); }

// ===========================================================================
// Width-generic little-endian load/store of an element of W bytes (1/2/4/8).
// ===========================================================================
ZL_HD uint64_t load_le(const uint8_t* p, int W){
    uint64_t v=0;
    for(int b=0;b<W;++b) v |= (uint64_t)p[b] << (8*b);
    return v;
}
ZL_HD void store_le(uint8_t* p, uint64_t v, int W){
    for(int b=0;b<W;++b) p[b] = (uint8_t)(v >> (8*b));
}
// Mask a value to W bytes (handles W==8 without UB shift-by-64).
ZL_HD uint64_t mask_w(uint64_t v, int W){
    return (W>=8) ? v : (v & ((1ull << (8*W)) - 1));
}
// Mask a value to `nbits` low bits (handles nbits>=64 without UB shift-by-64).
ZL_HD uint64_t mask_bits(uint64_t v, int nbits){
    return (nbits>=64) ? v : (v & ((1ull << nbits) - 1));
}

ZL_HD uint64_t zigzag_enc_w(uint64_t u, int W){
    switch(W){ case 1: return zigzag_enc8 ((uint8_t )u);
               case 2: return zigzag_enc16((uint16_t)u);
               case 4: return zigzag_enc32((uint32_t)u);
               default:return zigzag_enc64(u); }
}
ZL_HD uint64_t zigzag_dec_w(uint64_t z, int W){
    switch(W){ case 1: return zigzag_dec8 ((uint8_t )z);
               case 2: return zigzag_dec16((uint16_t)z);
               case 4: return zigzag_dec32((uint32_t)z);
               default:return zigzag_dec64(z); }
}

// ===========================================================================
// Interleaved byte-wise rANS (Ryg's rans_byte). 32-bit state, 8-bit renorm.
// PROB_BITS frequency precision. Encoder writes bytes BACKWARD (pptr--),
// decoder reads them FORWARD (pptr++). Used per-segment so thousands of
// independent streams run in parallel on the GPU.
// ===========================================================================
static const uint32_t RANS_PROB_BITS  = 12;
static const uint32_t RANS_PROB_SCALE = 1u << RANS_PROB_BITS; // 4096
static const uint32_t RANS_L          = 1u << 23;             // lower bound

ZL_HD void rans_enc_init(uint32_t* state){ *state = RANS_L; }

// Encode one symbol (start=cumfreq, freq=normalized frequency). Emits renorm
// bytes backward through *pptr.
ZL_HD void rans_enc_put(uint32_t* state, uint8_t** pptr, uint32_t start, uint32_t freq){
    uint32_t x = *state;
    const uint32_t x_max = ((RANS_L >> RANS_PROB_BITS) << 8) * freq;
    while (x >= x_max){ *(--(*pptr)) = (uint8_t)(x & 0xff); x >>= 8; }
    *state = ((x / freq) << RANS_PROB_BITS) + (x % freq) + start;
}

// Flush the final 32-bit state (4 bytes, LE) backward.
ZL_HD void rans_enc_flush(uint32_t* state, uint8_t** pptr){
    uint32_t x = *state;
    *pptr -= 4;
    (*pptr)[0] = (uint8_t)(x >> 0);
    (*pptr)[1] = (uint8_t)(x >> 8);
    (*pptr)[2] = (uint8_t)(x >> 16);
    (*pptr)[3] = (uint8_t)(x >> 24);
}

// Initialize decoder state from the 4 leading bytes (LE), advancing forward.
ZL_HD void rans_dec_init(uint32_t* state, const uint8_t** pptr){
    uint32_t x;
    x  = (uint32_t)((*pptr)[0]) << 0;
    x |= (uint32_t)((*pptr)[1]) << 8;
    x |= (uint32_t)((*pptr)[2]) << 16;
    x |= (uint32_t)((*pptr)[3]) << 24;
    *pptr += 4;
    *state = x;
}

// Current decode slot in [0, RANS_PROB_SCALE).
ZL_HD uint32_t rans_dec_get(const uint32_t* state){
    return *state & (RANS_PROB_SCALE - 1);
}

// Advance decoder past one symbol, consuming renorm bytes forward.
ZL_HD void rans_dec_advance(uint32_t* state, const uint8_t** pptr, uint32_t start, uint32_t freq){
    const uint32_t mask = RANS_PROB_SCALE - 1;
    uint32_t x = *state;
    x = freq * (x >> RANS_PROB_BITS) + (x & mask) - start;
    while (x < RANS_L){ x = (x << 8) | (uint32_t)(*((*pptr)++)); }
    *state = x;
}
