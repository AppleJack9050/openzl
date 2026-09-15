// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CUSTOM_PARSERS_SHARED_COMPONENTS_CLUSTERING_H
#define OPENZL_CUSTOM_PARSERS_SHARED_COMPONENTS_CLUSTERING_H

#include "openzl/zl_compressor.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Registers a generic clustering graph where clustering is still
 * unconfigured.
 *
 * @param compressor The compressor to register the graph with
 * @returns The graph ID registered for the clustering graph
 */
ZL_GraphID ZS2_createGraph_genericClustering(ZL_Compressor* compressor);

/**
 * @brief Same as ZS2_createGraph_genericClustering(), except that, untrained,
 * serial streams go to zstd with a 128MB window and long-distance matching
 * (appended as successor index 9) instead of the generic graph.
 * Numeric/struct/string streams are unchanged. Mirrors the text-stream setup
 * in custom_parsers/json/json_profile.cpp.
 *
 * @note This describes the UNTRAINED graph only. `zli train` rebuilds the
 * clustering config without any typeDefaults, so after training a serial tag
 * that was not placed in a trained cluster falls back to
 * ZL_GRAPH_COMPRESS_GENERIC. On the default training path ACE also replaces
 * every successor, including the long-range one, so trained clusters can only
 * select it with --no-ace-successors
 * (see tools/training/clustering/clustering_graph_trainer.cpp).
 *
 * @param compressor The compressor to register the graph with
 * @returns The graph ID registered for the clustering graph
 */
ZL_GraphID ZS2_createGraph_genericClustering_withLongRangeSerial(
        ZL_Compressor* compressor);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_CUSTOM_PARSERS_SHARED_COMPONENTS_CLUSTERING_H
