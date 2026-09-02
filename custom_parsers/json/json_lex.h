// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZL_CUSTOM_PARSERS_JSON_JSON_LEX_H
#define ZL_CUSTOM_PARSERS_JSON_JSON_LEX_H

#include "openzl/shared/portability.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

/// Transform ID of the json_lex codec. Must match between encoder and decoder.
#define ZL_JSON_LEX_TRANSFORM_ID 320
/// Anchor name of the json_lex node (looked up without the leading '!').
#define ZL_JSON_LEX_NODE_NAME "!json_lex"

/**
 * Registers the json_lex encoder: a lossless JSON / JSON-Lines lexer.
 *
 * Input: serial: JSON text. The transform accepts ANY input and always
 * round-trips it byte-for-byte; it is only *efficient* on JSON.
 *
 * Output 0: serial: the "structure" -- the input with every JSON value
 *           replaced by a one-byte token (see json_lex.cpp for the layout).
 * Output 1: string: string VALUES, stored UNESCAPED (\n -> newline, ...).
 * Output 2: string: object KEYS, stored unescaped.
 * Output 3: string: numbers, ASCII exactly as written.
 * Output 4: string: strings whose escaping is not canonical, stored exactly
 *           as written (between the quotes). Canonical means: re-escaping the
 *           unescaped bytes with the rules of Python's
 *           json.dumps(ensure_ascii=False) reproduces the original text.
 *           Any string that fails that test lands here, which is what keeps
 *           the transform lossless on arbitrary escaping styles.
 */
ZL_NodeID ZL_JsonLex_registerEncoder(ZL_Compressor* compressor);

/// Registers the matching decoder. Required before decompressing any frame
/// produced with the json profile.
ZL_Report ZL_JsonLex_registerDecoder(ZL_DCtx* dctx);

ZL_END_C_DECLS

#endif // ZL_CUSTOM_PARSERS_JSON_JSON_LEX_H
