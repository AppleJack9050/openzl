// gpu/prototype/src/frame.h
//
// Self-describing container for the prototype. The frame fully describes how to
// invert the pipeline: an ordered transform table (stored in DECODE order) plus
// the entropy section. Shared by the CPU and GPU paths so both produce
// byte-identical frames. All multi-byte fields little-endian.
//
//   [Header 24B] [TransformTable nbT*8B] [EntropySection | StoredPayload] [u32 checksum]
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include "rans_table.h"

namespace gpuzl {

enum CodecId : uint8_t {
    CODEC_DELTA     = 0,
    CODEC_ZIGZAG    = 1,
    CODEC_TRANSPOSE = 2,
    CODEC_BITPACK   = 3,
};
enum EntropyMode : uint8_t { ENT_STORED = 0, ENT_RANS = 1 };

// Per-stage metadata. `n_elts` is the element count of this stage's ENCODE
// input (== its decode output). For bitpack, param0 = nbits.
struct StageMeta {
    uint8_t  codec_id;
    uint8_t  elt_width;
    uint8_t  param0;
    uint32_t n_elts;
};

// Everything needed to (de)serialize the entropy section, host-resident.
struct EntropyPayload {
    EntropyMode           mode = ENT_STORED;
    // stored mode:
    std::vector<uint8_t>  stored;       // raw pre-entropy bytes
    // rANS mode:
    uint32_t              seg_size = 0;
    uint32_t              nb_seg   = 0;
    RansTable             table;
    std::vector<uint32_t> seg_len;
    std::vector<uint8_t>  packed;
    uint64_t              pre_entropy_nbytes = 0; // L (rANS symbol count)
};

// Content checksum lives in checksum.h (chunked FNV, GPU-parallel). The frame
// just stores/reads the 32-bit digest produced there.

// --- little-endian append helpers ------------------------------------------
inline void put_u8 (std::vector<uint8_t>& b, uint8_t  v){ b.push_back(v); }
inline void put_u32(std::vector<uint8_t>& b, uint32_t v){ for(int i=0;i<4;++i) b.push_back((uint8_t)(v>>(8*i))); }
inline void put_u64(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((uint8_t)(v>>(8*i))); }
inline uint8_t  get_u8 (const uint8_t* p){ return p[0]; }
inline uint32_t get_u32(const uint8_t* p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }
inline uint64_t get_u64(const uint8_t* p){ uint64_t v=0; for(int i=0;i<8;++i) v|=(uint64_t)p[i]<<(8*i); return v; }

static const uint8_t  FRAME_MAGIC[4] = {'G','Z','L','1'};
static const uint8_t  FRAME_VERSION  = 1;

// Build a frame from already-computed components (host-resident). `stages` is in
// ENCODE order; it is written reversed (decode order). `content_checksum` is the
// fnv1a32 of the original input.
inline std::vector<uint8_t> build_frame(uint64_t orig_nbytes,
                                        uint32_t content_checksum,
                                        const std::vector<StageMeta>& stages,
                                        const EntropyPayload& ent){
    std::vector<uint8_t> f;
    // Header
    f.insert(f.end(), FRAME_MAGIC, FRAME_MAGIC+4);
    put_u8 (f, FRAME_VERSION);
    put_u8 (f, (uint8_t)stages.size());
    put_u8 (f, (uint8_t)ent.mode);
    put_u8 (f, 0); // reserved
    put_u64(f, orig_nbytes);
    put_u64(f, ent.pre_entropy_nbytes);
    // Transform table in DECODE order (reverse of encode order)
    for(auto it = stages.rbegin(); it != stages.rend(); ++it){
        put_u8 (f, it->codec_id);
        put_u8 (f, it->elt_width);
        put_u8 (f, it->param0);
        put_u8 (f, 0);
        put_u32(f, it->n_elts);
    }
    // Entropy / stored section
    if(ent.mode == ENT_RANS){
        // seg_size_log2
        uint8_t log2=0; while((1u<<log2) < ent.seg_size) ++log2;
        put_u8 (f, log2);
        put_u8 (f, 0); put_u8(f, 0); put_u8(f, 0);
        put_u32(f, ent.nb_seg);
        // table (512B)
        uint8_t tbl[RANS_TABLE_BYTES];
        rans_table_serialize(ent.table, tbl);
        f.insert(f.end(), tbl, tbl+RANS_TABLE_BYTES);
        // seg_len array
        for(uint32_t i=0;i<ent.nb_seg;++i) put_u32(f, ent.seg_len[i]);
        // packed bytes
        f.insert(f.end(), ent.packed.begin(), ent.packed.end());
    } else {
        f.insert(f.end(), ent.stored.begin(), ent.stored.end());
    }
    // Trailer checksum
    put_u32(f, content_checksum);
    return f;
}

// Parsed view of a frame.
struct ParsedFrame {
    uint64_t orig_nbytes = 0;
    uint64_t pre_entropy_nbytes = 0;
    uint32_t content_checksum = 0;
    std::vector<StageMeta> stages_decode_order; // as stored
    EntropyPayload ent;
    bool ok = false;
    std::string err;
};

inline ParsedFrame parse_frame(const uint8_t* f, size_t n){
    ParsedFrame pf;
    auto fail = [&](const char* m){ pf.ok=false; pf.err=m; return pf; };
    if(n < 24) return fail("frame too small");
    if(std::memcmp(f, FRAME_MAGIC, 4)!=0) return fail("bad magic");
    if(f[4] != FRAME_VERSION) return fail("bad version");
    uint8_t nbT  = f[5];
    uint8_t mode = f[6];
    pf.orig_nbytes        = get_u64(f+8);
    pf.pre_entropy_nbytes = get_u64(f+16);
    size_t off = 24;
    if(off + (size_t)nbT*8 > n) return fail("truncated transform table");
    for(uint8_t i=0;i<nbT;++i){
        StageMeta s;
        s.codec_id  = f[off+0];
        s.elt_width = f[off+1];
        s.param0    = f[off+2];
        s.n_elts    = get_u32(f+off+4);
        pf.stages_decode_order.push_back(s);
        off += 8;
    }
    pf.ent.mode = (EntropyMode)mode;
    pf.ent.pre_entropy_nbytes = pf.pre_entropy_nbytes;
    // All remaining-length checks use subtraction (rem = n - off) so a huge
    // attacker-controlled length can never wrap a `off + len` sum past `n`.
    if(mode == ENT_RANS){
        if(n - off < 8) return fail("truncated entropy header");
        uint8_t log2 = f[off]; off += 4;
        if(log2 >= 32) return fail("bad seg_size");
        pf.ent.seg_size = 1u << log2;
        pf.ent.nb_seg   = get_u32(f+off); off += 4;
        // nb_seg must exactly partition the pre-entropy stream into seg_size chunks.
        uint64_t expect_seg = pf.ent.seg_size ? ((pf.pre_entropy_nbytes + pf.ent.seg_size - 1) / pf.ent.seg_size) : ~0ull;
        if((uint64_t)pf.ent.nb_seg != expect_seg) return fail("nb_seg inconsistent with pre_entropy size");
        if(n - off < RANS_TABLE_BYTES) return fail("truncated table");
        rans_table_deserialize(f+off, pf.ent.table); off += RANS_TABLE_BYTES;
        if((uint64_t)(n - off) < (uint64_t)pf.ent.nb_seg*4) return fail("truncated seg_len");
        pf.ent.seg_len.resize(pf.ent.nb_seg);
        size_t total=0;
        for(uint32_t i=0;i<pf.ent.nb_seg;++i){ pf.ent.seg_len[i]=get_u32(f+off); off+=4; total+=pf.ent.seg_len[i]; }
        if(n - off < 4 || total > n - off - 4) return fail("truncated packed payload");
        pf.ent.packed.assign(f+off, f+off+total); off += total;
    } else {
        size_t payload = pf.pre_entropy_nbytes;
        if(n - off < 4 || payload > n - off - 4) return fail("truncated stored payload");
        pf.ent.stored.assign(f+off, f+off+payload); off += payload;
    }
    if(n - off < 4) return fail("missing checksum");
    pf.content_checksum = get_u32(f+off); off += 4;
    pf.ok = true;
    return pf;
}

// Validate that a parsed frame can be decoded into a buffer of `cap` bytes
// (the decoders allocate cap = orig_nbytes + 64). This walks the inverse-
// transform chain and rejects any frame whose sizes/widths/codecs are
// inconsistent or would write past `cap`, BEFORE any kernel or copy runs.
// Without this, a crafted (but structurally parseable) frame could drive an
// out-of-bounds device/host write. Returns false + err on rejection.
inline bool validate_decodable(const ParsedFrame& pf, size_t cap, std::string& err){
    auto bad = [&](const char* m){ err = m; return false; };
    size_t size = pf.pre_entropy_nbytes;          // entropy decode writes `size` bytes into the buffer
    if(size > cap) return bad("pre-entropy size exceeds buffer capacity");
    for(const auto& st : pf.stages_decode_order){
        const int W = st.elt_width;
        if(st.codec_id > CODEC_BITPACK) return bad("unknown codec id");
        // transpose is a pure byte permutation (any record width 1..255);
        // the numeric codecs use load_le and require a power-of-two width <= 8.
        if(st.codec_id == CODEC_TRANSPOSE){
            if(W < 1 || W > 255) return bad("invalid transpose width");
        } else {
            if(W!=1 && W!=2 && W!=4 && W!=8) return bad("invalid element width");
        }
        const size_t nelts = st.n_elts;
        size_t expected_in;
        if(st.codec_id == CODEC_BITPACK){
            const int nb = st.param0;
            if(nb < 1 || nb > 8*W) return bad("invalid bitpack nbits");
            expected_in = (nelts*(size_t)nb + 7) / 8;
        } else {
            expected_in = nelts*(size_t)W;
        }
        if(expected_in != size) return bad("stage input size mismatch");
        const size_t out = nelts*(size_t)W;
        if(out > cap) return bad("stage output exceeds buffer capacity");
        size = out;
    }
    if(size != pf.orig_nbytes) return bad("decoded size != orig_nbytes");
    return true;
}

} // namespace gpuzl
