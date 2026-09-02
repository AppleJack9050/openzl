// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "custom_parsers/json/json_lex.h"
#include "custom_parsers/json/json_profile.h"
#include "openzl/openzl.hpp"

namespace openzl::custom_parsers {
namespace {

class JsonLexTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        auto gid = ZL_createGraph_jsonCompressor(
                compressor_.get(), kJsonDefaultLevel);
        compressor_.selectStartingGraph(gid);
        cctx_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx_.refCompressor(compressor_);
        ASSERT_FALSE(ZL_isError(ZL_JsonLex_registerDecoder(dctx_.get())));
    }

    /// Compresses, decompresses, checks equality, returns the compressed form.
    std::string roundtrip(std::string_view input)
    {
        auto compressed = cctx_.compressSerial(input);
        auto regen      = dctx_.decompressSerial(compressed);
        EXPECT_EQ(regen, input);
        return compressed;
    }

    CCtx cctx_{};
    DCtx dctx_{};
    Compressor compressor_{};
};

TEST_F(JsonLexTest, Empty)
{
    roundtrip("");
}

TEST_F(JsonLexTest, TopLevelScalars)
{
    roundtrip("42");
    roundtrip("null");
    roundtrip("\"top-level string\"");
    roundtrip("-1.5e+10");
}

TEST_F(JsonLexTest, CanonicalEscapes)
{
    // Exactly what json.dumps(ensure_ascii=False) emits: stored unescaped.
    roundtrip(R"({"k": "a\"b\\c\nd\re\tf\bg\fh \u0001 \u001f"})");
}

TEST_F(JsonLexTest, NonCanonicalEscapes)
{
    // Solidus, printable \u, uppercase hex, short-form control written as \u,
    // a surrogate pair and lone surrogates: all kept as written.
    roundtrip(
            R"({"a": "\/", "b": "\u0041", "c": "\u001B", "d": "\u000a", "e": "\ud83d\ude00", "f": "\ud800", "g": "\udc00"})");
}

TEST_F(JsonLexTest, RawUtf8AndDel)
{
    roundtrip("{\"s\": \"\xe2\x9c\x85 \xf0\x9f\x9a\x80 \x7f\"}");
}

TEST_F(JsonLexTest, InvalidStrings)
{
    roundtrip("{\"tab\": \"a\tb\"}"); // raw control character inside a string
    roundtrip("{\"unterminated\": \"oops");
    roundtrip("{\"bad-escape\": \"\\x41\"}");
    roundtrip("\"\\u12\""); // truncated \u
}

TEST_F(JsonLexTest, NumbersAndLiterals)
{
    roundtrip(
            R"([-0, 0, 1, -1, 1.5, 1.5e+10, 1E-5, 1., -, 01, 007, 12abc, 1e, 1e+, .5, 5., true, false, null, truex, nullable, falsey])");
}

TEST_F(JsonLexTest, KeysVersusValues)
{
    roundtrip(R"({"k":"v","spaced"   :   "value", "x" :1, "quote-then-colon": "y":2})");
}

TEST_F(JsonLexTest, ReservedBytesOutsideStrings)
{
    // Token bytes and the escape byte as plain text must survive.
    roundtrip("S K R N T F Z \\ \\\\ SKRNTFZ");
    std::string all;
    for (int c = 0; c < 256; ++c) {
        all.push_back((char)c);
    }
    roundtrip(all);
}

TEST_F(JsonLexTest, Whitespace)
{
    roundtrip("  {  \"a\"  :  [ 1 , 2 ]  }  \r\n\t");
    roundtrip("{\"crlf\": \"x\"}\r\n{\"n\": 2}\n");
}

TEST_F(JsonLexTest, LargeString)
{
    std::string s = "{\"big\": \"";
    s.append(200000, 'x');
    s += "\"}";
    roundtrip(s);
}

TEST_F(JsonLexTest, JsonLinesCompress)
{
    std::string jsonl;
    for (int i = 0; i < 2000; ++i) {
        jsonl += "{\"id\": " + std::to_string(i)
                + ", \"role\": \"user\", \"text\": \"line "
                + std::to_string(i % 7) + " of the sample\"}\n";
    }
    auto compressed = roundtrip(jsonl);
    EXPECT_LT(compressed.size(), jsonl.size() / 10);
}

} // namespace
} // namespace openzl::custom_parsers
