// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/shared_components/clustering.h"
#include "custom_parsers/shared_components/numeric_graphs.h"
#include "custom_parsers/shared_components/string_graphs.h"
#include "openzl/compress/graphs/generic_clustering_graph.h"
#include "openzl/zl_compressor.h"

#include <iterator>
#include <vector>

namespace {
constexpr size_t kNbBaseSuccessors = 9;

ZL_GraphID registerGenericClustering(
        ZL_Compressor* compressor,
        const ZL_ClusteringConfig* config,
        const ZL_GraphID* extraSuccessors,
        size_t nbExtraSuccessors)
{
    ZL_GraphID const baseSuccessors[] = {
        ZL_GRAPH_STORE,
        ZL_GRAPH_FIELD_LZ,
        ZL_GRAPH_ZSTD,
        ZL_GRAPH_COMPRESS_GENERIC,
        openzl::custom_parsers::ZL_Compressor_registerRangePack(compressor),
        openzl::custom_parsers::ZL_Compressor_registerRangePackZstd(compressor),
        openzl::custom_parsers::ZL_Compressor_registerTokenizeSorted(
                compressor),
        openzl::custom_parsers::ZL_Compressor_registerDeltaFieldLZ(compressor),
        openzl::custom_parsers::ZL_Compressor_registerStringTokenize(compressor)
    };
    static_assert(
            sizeof(baseSuccessors) / sizeof(baseSuccessors[0])
                    == kNbBaseSuccessors,
            "kNbBaseSuccessors must match the list above");
    std::vector<ZL_GraphID> successors(
            std::begin(baseSuccessors), std::end(baseSuccessors));
    successors.insert(
            successors.end(),
            extraSuccessors,
            extraSuccessors + nbExtraSuccessors);

    std::array<ZL_NodeID, 4> clusteringCodecs = { ZL_NODE_CONCAT_SERIAL,
                                                  ZL_NODE_CONCAT_STRUCT,
                                                  ZL_NODE_CONCAT_NUMERIC,
                                                  ZL_NODE_CONCAT_STRING };

    return ZL_Clustering_registerGraphWithCustomClusteringCodecs(
            compressor,
            config,
            successors.data(),
            successors.size(),
            clusteringCodecs.data(),
            clusteringCodecs.size());
}
} // namespace

ZL_GraphID ZS2_createGraph_genericClustering(ZL_Compressor* compressor)
{
    ZL_ClusteringConfig config = {};
    return registerGenericClustering(compressor, &config, nullptr, 0);
}

ZL_GraphID ZS2_createGraph_genericClustering_withLongRangeSerial(
        ZL_Compressor* compressor)
{
    // Level follows the global ZL_CParam_compressionLevel, so this differs
    // from the plain ZL_GRAPH_ZSTD successor only in window / LDM.
    ZL_GraphID const longRangeZstd =
            openzl::custom_parsers::ZL_Compressor_registerLongRangeZstdGraph(
                    compressor);

    // Untrained, the clustering graph sends any stream without a typeDefault
    // to ZL_GRAPH_COMPRESS_GENERIC. Route serial streams (parquet BYTE_ARRAY
    // columns, i.e. text) to long-range zstd instead; numeric/struct streams
    // keep their existing path. Still concatenated per tag (CONCAT_SERIAL).
    // Training does not preserve typeDefaults; see the @note in clustering.h.
    ZL_ClusteringConfig_TypeSuccessor serialDefault = {};
    serialDefault.type               = ZL_Type_serial;
    serialDefault.eltWidth           = 1;
    serialDefault.successorIdx       = kNbBaseSuccessors; // longRangeZstd
    serialDefault.clusteringCodecIdx = 0;                 // ZL_NODE_CONCAT_SERIAL

    ZL_ClusteringConfig config = {};
    config.typeDefaults        = &serialDefault;
    config.nbTypeDefaults      = 1;
    return registerGenericClustering(compressor, &config, &longRangeZstd, 1);
}
