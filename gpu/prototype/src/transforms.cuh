// gpu/prototype/src/transforms.cuh
//
// Host launchers for the GPU transform codecs. All pointers are DEVICE
// pointers. Element counts are in elements; W is the element width in bytes
// (1/2/4/8). Sizes follow OpenZL semantics:
//   zigzag/delta/transpose : output bytes == input bytes (== n*W)
//   bitpack                : output bytes == ceil(n*nbits/8)
//
// IMPORTANT (bitpack decode): the input device buffer must have >= 16 bytes of
// readable padding past its logical end (the pipeline guarantees this).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace gpuzl {

void zigzag_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);
void zigzag_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);

void delta_encode (const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);
void delta_decode (const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);

void transpose_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);
void transpose_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, cudaStream_t s = 0);

// Returns nbits in [1,64] derived from the max element value (OpenZL rule).
int  bitpack_compute_nbits(const uint8_t* d_in, size_t n, int W, cudaStream_t s = 0);
// d_out must be zeroed for the (ceil(n*nbits/8) rounded up to 4) bytes.
void bitpack_encode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, int nbits, cudaStream_t s = 0);
void bitpack_decode(const uint8_t* d_in, uint8_t* d_out, size_t n, int W, int nbits, cudaStream_t s = 0);

} // namespace gpuzl
