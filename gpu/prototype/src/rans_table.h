// gpu/prototype/src/rans_table.h
//
// Host-side construction of the normalized rANS frequency table from a 256-bin
// byte histogram. Deterministic, so the CPU reference and the GPU path (which
// histograms on-device but normalizes here) build identical tables -> identical
// frames. Also (de)serializes the table into the frame (256 LE u16 = 512 bytes).
#pragma once

#include <cstdint>
#include <cstring>
#include "codec_core.cuh"

struct RansTable {
    uint16_t freq[256];                 // normalized; sum == RANS_PROB_SCALE
    uint16_t start[256];                // exclusive prefix sum of freq (cumfreq)
    uint8_t  slot2sym[RANS_PROB_SCALE]; // slot in [0,SCALE) -> symbol
    bool     valid = false;             // false if input was empty
};

// Build start[] (cumfreq) and slot2sym[] from freq[]. The slot writes are
// bounded to RANS_PROB_SCALE so a malformed/empty table can never overflow
// slot2sym (defensive against corrupt frames and the empty-input case).
inline void rans_table_finish(RansTable& t){
    uint32_t c = 0;
    for(int s=0;s<256;++s){
        t.start[s] = (uint16_t)c;
        uint32_t f = t.freq[s];
        for(uint32_t k=0; k<f && (c+k)<RANS_PROB_SCALE; ++k) t.slot2sym[c+k] = (uint8_t)s;
        c += f;
    }
    // c == RANS_PROB_SCALE for any well-formed (valid) table.
}

// Normalize a histogram to sum == RANS_PROB_SCALE. Returns false if total==0.
inline bool rans_build_table(const uint64_t counts[256], RansTable& t){
    uint64_t total = 0;
    for(int s=0;s<256;++s) total += counts[s];
    std::memset(t.freq, 0, sizeof(t.freq));
    if(total == 0){ t.valid = false; return false; }

    // Floor scaling, with a floor of 1 for any present symbol.
    uint32_t sum = 0;
    for(int s=0;s<256;++s){
        if(counts[s]==0){ t.freq[s]=0; continue; }
        uint64_t f = (counts[s] * (uint64_t)RANS_PROB_SCALE) / total;
        if(f==0) f=1;
        t.freq[s] = (uint16_t)f;
        sum += (uint32_t)f;
    }
    // Reconcile to exactly RANS_PROB_SCALE by adjusting the largest bucket(s).
    while(sum < RANS_PROB_SCALE){
        int best=-1; uint32_t bf=0;
        for(int s=0;s<256;++s) if(t.freq[s]>bf){ bf=t.freq[s]; best=s; }
        t.freq[best]++; sum++;
    }
    while(sum > RANS_PROB_SCALE){
        int best=-1; uint32_t bf=1; // only pick freq>1 so we never drop to 0
        for(int s=0;s<256;++s) if(t.freq[s]>bf){ bf=t.freq[s]; best=s; }
        t.freq[best]--; sum--;
    }
    t.valid = true;
    rans_table_finish(t);
    return true;
}

// Serialize freq[] as 256 little-endian u16 (512 bytes). Returns bytes written.
inline size_t rans_table_serialize(const RansTable& t, uint8_t* out){
    for(int s=0;s<256;++s){
        out[2*s+0] = (uint8_t)(t.freq[s] & 0xff);
        out[2*s+1] = (uint8_t)(t.freq[s] >> 8);
    }
    return 512;
}

// Rebuild a full table (freq/start/slot2sym) from serialized freq[].
inline void rans_table_deserialize(const uint8_t* in, RansTable& t){
    for(int s=0;s<256;++s)
        t.freq[s] = (uint16_t)(in[2*s+0] | (in[2*s+1] << 8));
    t.valid = true;
    rans_table_finish(t);
}

static const size_t RANS_TABLE_BYTES = 512;
