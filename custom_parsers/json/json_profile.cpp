// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/json/json_profile.h"

#include "custom_parsers/json/json_lex.h"
#include "custom_parsers/shared_components/string_graphs.h"
#include "openzl/codecs/zl_ace.h"
#include "openzl/codecs/zl_conversion.h"
#include "openzl/codecs/zl_field_lz.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_parse_int.h"
#include "openzl/codecs/zl_tokenize.h"
#include "openzl/codecs/zl_zstd.h"
#include "openzl/zl_localParams.h"

namespace openzl::custom_parsers {

namespace {
/// Marks @p graph as a trainable placeholder: untrained it behaves exactly
/// like @p graph, and `zli train` may replace it with something better.
ZL_GraphID trainable(ZL_Compressor* compressor, ZL_GraphID graph)
{
    return ZL_Compressor_buildACEGraphWithDefault(compressor, graph);
}
} // namespace

ZL_GraphID ZL_createGraph_jsonCompressor(
        ZL_Compressor* compressor,
        int level) noexcept
{
    // The level must reach EVERY stream, not just the text one. Streams whose
    // successor is a generic graph read the compressor's global level, so set
    // it here -- otherwise --profile-arg is silently ignored on inputs whose
    // bytes happen to land in one of those streams (e.g. escape-heavy JSON,
    // where all strings go to the raw stream).
    (void)ZL_Compressor_setParameter(
            compressor, ZL_CParam_compressionLevel, level);

    ZL_NodeID const lexer = ZL_JsonLex_registerEncoder(compressor);

    // String values: string -> {content (serial), lengths (numeric)}.
    // Content is where the bytes are; give it zstd with long-range matching.
    ZL_IntParam const textLevel[1] = { { kZstdCompressionLevel, level } };
    ZL_GraphID const textGraph =
            ZL_Compressor_registerLongRangeZstdGraph(compressor, textLevel, 1);
    ZL_GraphID const valueSuccessors[2] = {
        textGraph, trainable(compressor, ZL_GRAPH_COMPRESS_GENERIC) };
    ZL_GraphID const valuesGraph = ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            ZL_NODE_SEPARATE_STRING_COMPONENTS,
            valueSuccessors,
            2);

    // 16% of value bytes in JSON-Lines corpora are exact repeats of another
    // value (enum-like fields such as roles, plus shared prompts), and they sit
    // interleaved between the large free-text values. Tokenizing first sends
    // each distinct value once (in first-seen order, so text locality is kept)
    // and replaces the repeats with an index stream that costs almost nothing.
    ZL_GraphID const dedupValuesGraph = ZL_Compressor_registerTokenizeGraph(
            compressor,
            ZL_Type_string,
            /* sort */ false,
            valuesGraph,
            trainable(compressor, ZL_GRAPH_COMPRESS_GENERIC));

    // Structure: tiny once the values are gone; plain zstd at the same level.
    ZL_GraphID const structGraph = trainable(
            compressor,
            ZL_Compressor_registerZstdGraph_withLevel(compressor, level));

    // Keys: a small alphabet repeated per record -> tokenize.
    ZL_GraphID const keysGraph = trainable(
            compressor, ZL_Compressor_registerStringTokenize(compressor));

    // Numbers arrive as ASCII. Parse to int64 where that is lossless and
    // compress them as numeric; anything else falls through to generic.
    ZL_GraphID const fieldLz = ZL_Compressor_registerFieldLZGraph(compressor);
    ZL_GraphID const numbersGraph = trainable(
            compressor,
            ZL_RES_value(ZL_Compressor_parameterizeTryParseIntGraph(
                    compressor, fieldLz, ZL_GRAPH_COMPRESS_GENERIC)));

    // Non-canonically-escaped strings are still text with the same
    // cross-record redundancy, so they get the same long-range zstd as values
    // rather than a default-window generic graph.
    ZL_GraphID const rawSuccessors[2] = { textGraph,
                                          trainable(
                                                  compressor,
                                                  ZL_GRAPH_COMPRESS_GENERIC) };
    ZL_GraphID const rawGraph = ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            ZL_NODE_SEPARATE_STRING_COMPONENTS,
            rawSuccessors,
            2);

    ZL_GraphID const successors[5] = {
        structGraph,               // 0: structure
        dedupValuesGraph,          // 1: string values (deduplicated)
        keysGraph,                 // 2: object keys
        numbersGraph,              // 3: numbers
        rawGraph,                  // 4: non-canonical strings, as written
    };
    return ZL_Compressor_registerStaticGraph_fromNode(
            compressor, lexer, successors, 5);
}

} // namespace openzl::custom_parsers
