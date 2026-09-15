// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <string>
#include "openzl/zl_opaque_types.h"

#include "openzl/zl_compressor.h"
#include "openzl/zl_localParams.h"

namespace openzl::custom_parsers {

// zstd advanced parameter ids (ZSTD_cParameter in zstd.h). The zstd codec
// forwards its LocalIntParams to ZSTD_CCtx_setParameter() by id.
constexpr int kZstdCompressionLevel = 100; // ZSTD_c_compressionLevel
constexpr int kZstdWindowLog        = 101; // ZSTD_c_windowLog
constexpr int kZstdEnableLDM        = 160; // ZSTD_c_enableLongDistanceMatching

// 128MB window. zstd caps the window at the input size when it knows it, so
// small inputs do not pay for this.
constexpr int kTextWindowLog = 27;

/**
 * @brief zstd with a 128MB window and long-distance matching, for text streams
 * whose redundancy is spread across the whole input (repeated prompts, shared
 * boilerplate) and so invisible to a default-window zstd.
 *
 * @param extraParams Local int params applied on top of window/LDM, e.g.
 * { kZstdCompressionLevel, level }. With none, the level follows the
 * compressor's global ZL_CParam_compressionLevel.
 */
ZL_GraphID ZL_Compressor_registerLongRangeZstdGraph(
        ZL_Compressor* compressor,
        const ZL_IntParam* extraParams = nullptr,
        size_t nbExtraParams           = 0);

ZL_GraphID ZL_Compressor_registerStringTokenize(ZL_Compressor* compressor);
ZL_GraphID registerNullAwareDispatch(
        ZL_Compressor* compressor,
        const std::string& name,
        const std::array<ZL_GraphID, 3>& successors);

} // namespace openzl::custom_parsers
