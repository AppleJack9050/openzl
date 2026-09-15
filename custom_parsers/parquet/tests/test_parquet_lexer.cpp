// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <arrow/api.h>    // @manual
#include <arrow/io/api.h> // @manual
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <stdint.h>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

#include "custom_parsers/parquet/parquet_lexer.h"
#include "custom_parsers/parquet/tests/test_utils.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/xxhash.h"

namespace zstrong {
namespace parquet {
namespace testing {

namespace {
std::shared_ptr<arrow::Table> generate_table()
{
    auto i64array = to_arrow_array<int64_t>({ 100, 200, 300, 400, 500 });
    auto strarray = to_arrow_array<std::string>(
            { "hello", "world", "my", "name", "is" });

    std::shared_ptr<arrow::Schema> schema = arrow::schema(
            { arrow::field("int", arrow::int64()),
              arrow::field("str", arrow::utf8()) });

    return arrow::Table::Make(schema, { i64array, strarray });
}

std::shared_ptr<arrow::Table> generate_nested_table()
{
    // Top-Level Int Column
    auto i64array = to_arrow_array<int64_t>({ 100, 200, 300, 400, 500 });

    // Nested Struct Column
    auto i128 = arrow::fixed_size_binary(16);
    auto level2_t =
            arrow::struct_({ { "int", arrow::int32() }, { "struct", i128 } });
    auto level1_t =
            arrow::struct_({ { "str", arrow::utf8() }, { "2", level2_t } });
    std::shared_ptr<arrow::Schema> schema = arrow::schema(
            { arrow::field("int", arrow::int64()),
              arrow::field("1", level1_t) });

    // Fill in level 2
    auto i32array = to_arrow_array<int32_t>({ 1, 2, 3, 4, 5 });
    std::string i128data(16, 'a');
    auto i128array = to_arrow_array(
            { i128data, i128data, i128data, i128data, i128data }, 16);

    auto level2 = std::make_shared<arrow::StructArray>(arrow::StructArray(
            level2_t, 5, { i32array, i128array }, nullptr, 0, 0));

    // Fill in level 1
    auto strarray = to_arrow_array<std::string>(
            { "hello", "world", "my", "name", "is" });
    auto level1 = std::make_shared<arrow::StructArray>(arrow::StructArray(
            level1_t, 5, { strarray, level2 }, nullptr, 0, 0));

    // Create table from top-level columns
    return arrow::Table::Make(schema, { i64array, level1 });
}

// Columns with repetition levels, so data pages store both repetition and
// definition levels before the values.
std::shared_ptr<arrow::Table> generate_repeated_table()
{
    auto pool = arrow::default_memory_pool();

    // list<int32>: [[1, 2, 3], [], null, [4], [5, 6]]
    auto listType = arrow::list(arrow::field("element", arrow::int32(), true));
    auto listValues = std::make_shared<arrow::Int32Builder>(pool);
    arrow::ListBuilder listBuilder(pool, listValues, listType);
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(
            listValues->AppendValues(std::vector<int32_t>{ 1, 2, 3 }));
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listBuilder.AppendNull());
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listValues->Append(4));
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(
            listValues->AppendValues(std::vector<int32_t>{ 5, 6 }));
    std::shared_ptr<arrow::Array> listArray;
    PARQUET_THROW_NOT_OK(listBuilder.Finish(&listArray));

    // map<utf8, int64>: [{a: 1}, {}, null, {b: 2, c: 3}, {d: 4}]
    auto keys  = std::make_shared<arrow::StringBuilder>(pool);
    auto items = std::make_shared<arrow::Int64Builder>(pool);
    arrow::MapBuilder mapBuilder(pool, keys, items);
    auto appendEntry = [&](const std::string& key, int64_t item) {
        PARQUET_THROW_NOT_OK(keys->Append(key));
        PARQUET_THROW_NOT_OK(items->Append(item));
    };
    PARQUET_THROW_NOT_OK(mapBuilder.Append());
    appendEntry("a", 1);
    PARQUET_THROW_NOT_OK(mapBuilder.Append());
    PARQUET_THROW_NOT_OK(mapBuilder.AppendNull());
    PARQUET_THROW_NOT_OK(mapBuilder.Append());
    appendEntry("b", 2);
    appendEntry("c", 3);
    PARQUET_THROW_NOT_OK(mapBuilder.Append());
    appendEntry("d", 4);
    std::shared_ptr<arrow::Array> mapArray;
    PARQUET_THROW_NOT_OK(mapBuilder.Finish(&mapArray));

    // list<list<int64>>: [[[1], [2, 3]], [[]], null, [], [[4, 5, 6]]]
    auto innerType = arrow::list(arrow::field("element", arrow::int64(), true));
    auto outerType = arrow::list(arrow::field("element", innerType, true));
    auto innerValues = std::make_shared<arrow::Int64Builder>(pool);
    auto innerBuilder =
            std::make_shared<arrow::ListBuilder>(pool, innerValues, innerType);
    arrow::ListBuilder outerBuilder(pool, innerBuilder, outerType);
    PARQUET_THROW_NOT_OK(outerBuilder.Append());
    PARQUET_THROW_NOT_OK(innerBuilder->Append());
    PARQUET_THROW_NOT_OK(innerValues->Append(1));
    PARQUET_THROW_NOT_OK(innerBuilder->Append());
    PARQUET_THROW_NOT_OK(
            innerValues->AppendValues(std::vector<int64_t>{ 2, 3 }));
    PARQUET_THROW_NOT_OK(outerBuilder.Append());
    PARQUET_THROW_NOT_OK(innerBuilder->Append());
    PARQUET_THROW_NOT_OK(outerBuilder.AppendNull());
    PARQUET_THROW_NOT_OK(outerBuilder.Append());
    PARQUET_THROW_NOT_OK(outerBuilder.Append());
    PARQUET_THROW_NOT_OK(innerBuilder->Append());
    PARQUET_THROW_NOT_OK(
            innerValues->AppendValues(std::vector<int64_t>{ 4, 5, 6 }));
    std::shared_ptr<arrow::Array> nestedArray;
    PARQUET_THROW_NOT_OK(outerBuilder.Finish(&nestedArray));

    std::shared_ptr<arrow::Schema> schema = arrow::schema(
            { arrow::field("list", listType),
              arrow::field("map", mapArray->type()),
              arrow::field("nested", outerType) });
    return arrow::Table::Make(schema, { listArray, mapArray, nestedArray });
}

// Required columns have no definition levels, so data pages may store no
// levels at all before the values.
std::shared_ptr<arrow::Table> generate_required_table()
{
    auto pool = arrow::default_memory_pool();

    // The first four value bytes must not be mistaken for a level length
    auto reqArray = to_arrow_array<int64_t>(
            { std::numeric_limits<int64_t>::max(), 1, 2, 3, 4 });

    // Required list of required int32: [[1], [2, 3], [], [4], [5]]
    auto listType = arrow::list(arrow::field("element", arrow::int32(), false));
    auto listValues = std::make_shared<arrow::Int32Builder>(pool);
    arrow::ListBuilder listBuilder(pool, listValues, listType);
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listValues->Append(1));
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(
            listValues->AppendValues(std::vector<int32_t>{ 2, 3 }));
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listValues->Append(4));
    PARQUET_THROW_NOT_OK(listBuilder.Append());
    PARQUET_THROW_NOT_OK(listValues->Append(5));
    std::shared_ptr<arrow::Array> listArray;
    PARQUET_THROW_NOT_OK(listBuilder.Finish(&listArray));

    auto optArray =
            to_arrow_array<int32_t>({ 1, std::nullopt, 3, std::nullopt, 5 });

    std::shared_ptr<arrow::Schema> schema = arrow::schema(
            { arrow::field("req", arrow::int64(), false),
              arrow::field("req_list", listType, false),
              arrow::field("opt", arrow::int32()) });
    return arrow::Table::Make(schema, { reqArray, listArray, optArray });
}

struct Column {
    std::vector<std::string> path;
    ZL_Type dataType;
    size_t dataWidth;
    /// If set, the exact size of the values in each data page
    std::optional<size_t> valueBytes = std::nullopt;
};

uint32_t getTag(const std::vector<std::string>& path)
{
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    for (const auto& str : path) {
        size_t len = str.size();
        XXH3_64bits_update(&state, str.data(), len);
        XXH3_64bits_update(&state, &len, sizeof(size_t));
    }
    XXH64_hash_t result = XXH3_64bits_digest(&state);
    return (uint32_t)result;
}

void validateTokens(
        const std::vector<ZL_ParquetToken>& tokens,
        const std::string& input,
        const std::vector<Column>& columns,
        size_t num_row_groups)
{
    int i = 0;
    // Magic
    EXPECT_EQ(tokens.at(i).type, ZL_ParquetTokenType_Magic);

    // Data Headers + Pages
    for (size_t rg = 0; rg < num_row_groups; rg++) {
        for (size_t j = 0; j < columns.size(); j++) {
            i++;
            EXPECT_EQ(tokens.at(i).type, ZL_ParquetTokenType_PageHeader);
            i++;
            auto& token    = tokens.at(i);
            auto& expected = columns.at(j);
            EXPECT_EQ(token.type, ZL_ParquetTokenType_DataPage);
            EXPECT_EQ(token.tag, getTag(expected.path));
            EXPECT_EQ(token.dataType, expected.dataType);
            EXPECT_EQ(token.dataWidth, expected.dataWidth);
            EXPECT_EQ(token.size % token.dataWidth, 0);
            if (expected.valueBytes) {
                EXPECT_EQ(token.size, *expected.valueBytes);
            }
        }
    }
    // Footer
    i++;
    EXPECT_EQ(tokens.at(i).type, ZL_ParquetTokenType_Footer);

    // Token sizes should add up to the input size
    auto sum = 0;
    for (auto const& token : tokens) {
        sum += token.size;
        EXPECT_NE(token.ptr, nullptr);
    }
    EXPECT_EQ(sum, input.size());
}
} // namespace

TEST(ParquetLexerTest, TestInitValidParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_table(), 3);

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestInitNonParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);
    std::string input = "hello world";

    EXPECT_TRUE(ZL_isError(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr)));

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestInitInvalidMetadataSize)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_table(), 3);

    ZL_writeLE32(
            input.data() + (input.size() - 8),
            std::numeric_limits<uint32_t>::max());

    EXPECT_TRUE(ZL_isError(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr)));

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexValidParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_table(), 3);

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(15);

    auto const res =
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr);
    EXPECT_FALSE(ZL_isError(res));
    EXPECT_TRUE(ZL_ParquetLexer_finished(lexer));
    auto const numTokens = ZL_validResult(res);
    EXPECT_LT(numTokens, tokens.size());
    tokens.resize(numTokens);

    std::vector<Column> columns = { { { "int" }, ZL_Type_numeric, 8 },
                                    { { "str" }, ZL_Type_serial, 1 } };

    validateTokens(tokens, input, columns, 2);

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexNestedParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_nested_table(), 3);

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(20);

    auto const res =
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr);
    EXPECT_FALSE(ZL_isError(res));
    EXPECT_TRUE(ZL_ParquetLexer_finished(lexer));
    auto const numTokens = ZL_validResult(res);
    EXPECT_LT(numTokens, tokens.size());
    tokens.resize(numTokens);

    std::vector<Column> columns = {
        { { "int" }, ZL_Type_numeric, 8 },
        { { "1", "str" }, ZL_Type_serial, 1 },
        { { "1", "2", "int" }, ZL_Type_numeric, 4 },
        { { "1", "2", "struct" }, ZL_Type_struct, 16 }
    };

    validateTokens(tokens, input, columns, 2);

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexRepeatedParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_repeated_table());

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(20);

    auto const res =
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr);
    ASSERT_FALSE(ZL_isError(res));
    EXPECT_TRUE(ZL_ParquetLexer_finished(lexer));
    auto const numTokens = ZL_validResult(res);
    EXPECT_LT(numTokens, tokens.size());
    tokens.resize(numTokens);

    std::vector<Column> columns = {
        { { "list", "list", "element" }, ZL_Type_numeric, 4, 6 * 4 },
        { { "map", "key_value", "key" }, ZL_Type_serial, 1, 4 * (4 + 1) },
        { { "map", "key_value", "value" }, ZL_Type_numeric, 8, 4 * 8 },
        { { "nested", "list", "element", "list", "element" },
          ZL_Type_numeric,
          8,
          6 * 8 },
    };

    validateTokens(tokens, input, columns, 1);

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexRequiredParquet)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_required_table());

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(20);

    auto const res =
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr);
    ASSERT_FALSE(ZL_isError(res));
    EXPECT_TRUE(ZL_ParquetLexer_finished(lexer));
    auto const numTokens = ZL_validResult(res);
    EXPECT_LT(numTokens, tokens.size());
    tokens.resize(numTokens);

    std::vector<Column> columns = {
        { { "req" }, ZL_Type_numeric, 8, 5 * 8 },
        { { "req_list", "list", "element" }, ZL_Type_numeric, 4, 5 * 4 },
        { { "opt" }, ZL_Type_numeric, 4, 3 * 4 },
    };

    validateTokens(tokens, input, columns, 1);

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexEmptyRowGroup)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    // Empty row groups have zero-byte column chunks with no pages
    auto table = generate_table();
    auto empty = table->Slice(0, 0);
    auto input = to_canonical_parquet_row_groups(
            { empty, table, empty, empty, table, empty });

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(15);

    auto const res =
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr);
    ASSERT_FALSE(ZL_isError(res));
    EXPECT_TRUE(ZL_ParquetLexer_finished(lexer));
    auto const numTokens = ZL_validResult(res);
    EXPECT_LT(numTokens, tokens.size());
    tokens.resize(numTokens);

    std::vector<Column> columns = { { { "int" }, ZL_Type_numeric, 8 },
                                    { { "str" }, ZL_Type_serial, 1 } };

    validateTokens(tokens, input, columns, 2);

    ZL_ParquetLexer_free(lexer);
}

TEST(ParquetLexerTest, TestLexBytesBeforeFooter)
{
    auto lexer = ZL_ParquetLexer_create();
    EXPECT_NE(lexer, nullptr);

    auto input = to_canonical_parquet(generate_table());

    // Insert bytes that belong to no column chunk just before the metadata
    size_t const metadataSize = ZL_readLE32(input.data() + input.size() - 8);
    input.insert(input.size() - 8 - metadataSize, 8, '\0');

    ZL_REQUIRE_SUCCESS(
            ZL_ParquetLexer_init(lexer, input.data(), input.size(), nullptr));

    auto tokens = std::vector<ZL_ParquetToken>(20);
    EXPECT_TRUE(ZL_isError(
            ZL_ParquetLexer_lex(lexer, tokens.data(), tokens.size(), nullptr)));

    ZL_ParquetLexer_free(lexer);
}
} // namespace testing
} // namespace parquet
} // namespace zstrong
