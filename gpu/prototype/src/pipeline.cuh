// gpu/prototype/src/pipeline.cuh
//
// Pipeline configuration + the host-facing GPU compress/decompress entry points.
// A pipeline is an ordered list of transform stages terminated by an entropy
// mode. The same PipelineConfig drives both the GPU path (pipeline.cu) and the
// CPU reference (cpu_pipeline.h), which must produce byte-identical frames.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "frame.h"

namespace gpuzl {

// A stage as specified by the user (n_elts / nbits are derived at encode time).
struct StageSpec { uint8_t codec_id; uint8_t elt_width; };

struct PipelineConfig {
    std::string            name;
    std::vector<StageSpec> stages;   // encode order
    EntropyMode            entropy;
};

// Demonstrative pipelines.
inline PipelineConfig cfg_store()                 { return {"store",            {}, ENT_STORED}; }
inline PipelineConfig cfg_rans_only()             { return {"rans",             {}, ENT_RANS}; }
inline PipelineConfig cfg_delta_zigzag_rans(int W){ return {"delta+zigzag+rans",{{CODEC_DELTA,(uint8_t)W},{CODEC_ZIGZAG,(uint8_t)W}}, ENT_RANS}; }
inline PipelineConfig cfg_delta_zigzag_bitpack(int W){ return {"delta+zigzag+bitpack",{{CODEC_DELTA,(uint8_t)W},{CODEC_ZIGZAG,(uint8_t)W},{CODEC_BITPACK,(uint8_t)W}}, ENT_STORED}; }
inline PipelineConfig cfg_transpose_rans(int W)   { return {"transpose+rans",   {{CODEC_TRANSPOSE,(uint8_t)W}}, ENT_RANS}; }

// GPU path. gpu_compress returns the frame bytes; gpu_decompress returns the
// reconstructed original (empty on parse error, with *ok=false).
std::vector<uint8_t> gpu_compress  (const uint8_t* input, size_t n, const PipelineConfig& cfg);
std::vector<uint8_t> gpu_decompress(const uint8_t* frame, size_t n, bool* ok = nullptr, std::string* err = nullptr);

} // namespace gpuzl
