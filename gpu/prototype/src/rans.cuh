// gpu/prototype/src/rans.cuh
//
// Interleaved byte-wise rANS entropy backend (the GPU analog of OpenZL's serial
// FSE/Huffman). The byte stream is cut into fixed-size SEGMENTS; each segment
// is an independent rANS stream, so thousands of segments encode/decode fully
// in parallel. Tables are built on the host (rans_table.h) from a GPU histogram
// so the CPU reference and GPU produce byte-identical output.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "rans_table.h"

namespace gpuzl {

// Segment size in bytes (power of two). Smaller -> more independent rANS
// streams -> better GPU occupancy, at the cost of ~ (4B state + 4B len)/SEG
// side overhead. 512 keeps ~1.5% overhead while saturating the 5090's SMs.
// The chosen value is stored (log2) in the frame, so decode is self-describing.
static const uint32_t RANS_SEG = 512;

struct RansEncoded {
    uint8_t*              d_packed = nullptr; // device, compacted, total bytes (GPU path)
    std::vector<uint8_t>  packed;             // host packed bytes (CPU reference path)
    size_t                total    = 0;       // compressed byte count
    uint32_t              nb_seg   = 0;
    uint32_t              seg_size = RANS_SEG;
    std::vector<uint32_t> seg_len;             // per-segment compressed length
    RansTable             table;               // normalized frequency table
};

// Compute the byte histogram of d_in[0..L) on the GPU into host counts[256].
void rans_histogram(const uint8_t* d_in, size_t L, uint64_t counts[256], cudaStream_t s = 0);

// Encode d_in[0..L). Allocates enc.d_packed (caller frees with cudaFree).
void rans_encode(const uint8_t* d_in, size_t L, RansEncoded& enc, cudaStream_t s = 0);

// Decode `total` packed bytes back into d_out[0..L). Reconstructs offsets from
// seg_len and the slot table from `table`.
void rans_decode(const uint8_t* d_packed, size_t total, const RansTable& table,
                 const std::vector<uint32_t>& seg_len, uint32_t seg_size,
                 uint32_t nb_seg, size_t L, uint8_t* d_out, cudaStream_t s = 0);

} // namespace gpuzl
