// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/poly/StringView.hpp"
#include "openzl/zl_dtransform.h"

namespace openzl::custom_parsers {

/**
 * Helper function which can be called as part of deserializing a compressor
 * containing non-standard graphs or codecs.
 * This function can be called via calling createCompressorFromSerialized,
 * or if the caller has more dependencies to register, than it can be
 * called directly before registering those dependencies.
 * Graphs added to zstrong/custom_parsers/ should be added
 * to this function. It's okay for  dependencies to be registereed on the
 * compressor even if they are not part of the final graph of a compressor.
 */
void processDependencies(Compressor& compressor, poly::string_view serialized);

/**
 * This function should be used to deserialize compressors containing
 * non-standard graphs or codecs.
 */
std::unique_ptr<Compressor> createCompressorFromSerialized(
        poly::string_view serialized);

/**
 * Registers the decoders for every custom codec that a profile in
 * custom_parsers/ can emit (currently json_lex). Standard codecs need no
 * registration, but a custom one must be known to the DCtx before it meets
 * a frame that uses it, so call this on any DCtx that may decompress output
 * of those profiles.
 */
void registerCustomDecoders(ZL_DCtx* dctx);

} // namespace openzl::custom_parsers
