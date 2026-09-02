// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZL_CUSTOM_PARSERS_JSON_JSON_PROFILE_H
#define ZL_CUSTOM_PARSERS_JSON_JSON_PROFILE_H

#include "openzl/zl_compressor.h"

namespace openzl::custom_parsers {

/// Default zstd level for the text stream when none is given.
const int kJsonDefaultLevel = 6;

/**
 * @brief Registers the JSON / JSON-Lines compression graph.
 *
 * The input is lexed by json_lex into structure / values / keys / numbers /
 * raw-string streams. The dominant stream for real-world JSON is the string
 * values, so it gets the care: it is unbundled into content + lengths, and the
 * content goes to zstd at @p level with a 128MB window and long-distance
 * matching enabled. JSON-Lines corpora are records that share boilerplate
 * across the whole file, and a default-window zstd cannot see that
 * redundancy; enabling it here is most of what makes this profile beat plain
 * zstd on such inputs.
 *
 * @param level zstd compression level for the string content stream.
 * @returns The graph ID, or ZL_GRAPH_ILLEGAL on failure.
 */
ZL_GraphID ZL_createGraph_jsonCompressor(
        ZL_Compressor* compressor,
        int level) noexcept;

} // namespace openzl::custom_parsers

#endif // ZL_CUSTOM_PARSERS_JSON_JSON_PROFILE_H
