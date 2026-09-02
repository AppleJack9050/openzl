// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/json/json_lex.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "openzl/common/assertion.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_data.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

namespace openzl::custom_parsers {
namespace {

/* Layout of the structure stream.
 *
 * The structure stream is the input with every JSON value replaced by one
 * token byte. Object keys, string values, numbers and the three literals each
 * get their own token, so the decoder knows which side stream to pull the
 * value from. None of the token bytes can occur outside a string literal in
 * valid JSON (true/false/null are themselves tokenized), so valid input never
 * needs the escape byte -- but arbitrary input still round-trips, because any
 * literal byte from this set is written as kEsc followed by the byte.
 */
enum Tok : char {
    kTokStr   = 'S', // string value  -> kStreamValues, stored unescaped
    kTokKey   = 'K', // object key    -> kStreamKeys, stored unescaped
    kTokRaw   = 'R', // string with non-canonical escapes -> kStreamRaw, as written
    kTokNum   = 'N', // number        -> kStreamNumbers, ASCII as written
    kTokTrue  = 'T',
    kTokFalse = 'F',
    kTokNull  = 'Z',
    kEsc      = '\\',
};

enum Stream : int {
    kStreamStruct  = 0,
    kStreamValues  = 1,
    kStreamKeys    = 2,
    kStreamNumbers = 3,
    kStreamRaw     = 4,
    kNbStreams     = 5,
};

inline bool isReserved(char c)
{
    switch (c) {
        case kTokStr:
        case kTokKey:
        case kTokRaw:
        case kTokNum:
        case kTokTrue:
        case kTokFalse:
        case kTokNull:
        case kEsc:
            return true;
        default:
            return false;
    }
}

inline bool isDigit(char c)
{
    return c >= '0' && c <= '9';
}

inline int hexVal(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

inline bool parseHex4(const char* p, uint32_t& v)
{
    v = 0;
    for (int i = 0; i < 4; ++i) {
        int const h = hexVal(p[i]);
        if (h < 0)
            return false;
        v = (v << 4) | (uint32_t)h;
    }
    return true;
}

inline bool isLowerHex4(const char* p)
{
    for (int i = 0; i < 4; ++i) {
        if (!(isDigit(p[i]) || (p[i] >= 'a' && p[i] <= 'f')))
            return false;
    }
    return true;
}

inline size_t utf8Len(uint32_t cp)
{
    return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
}

inline char* writeUtf8(char* out, uint32_t cp)
{
    if (cp < 0x80) {
        *out++ = (char)cp;
    } else if (cp < 0x800) {
        *out++ = (char)(0xC0 | (cp >> 6));
        *out++ = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *out++ = (char)(0xE0 | (cp >> 12));
        *out++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *out++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *out++ = (char)(0xF0 | (cp >> 18));
        *out++ = (char)(0x80 | ((cp >> 12) & 0x3F));
        *out++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *out++ = (char)(0x80 | (cp & 0x3F));
    }
    return out;
}

/* Canonical escaping: exactly the rules of Python's
 * json.dumps(ensure_ascii=False). Only '"', '\\' and control characters are
 * escaped; the five with a short form get it, the rest become \u00xx with
 * lowercase hex; everything else -- '/', DEL, UTF-8 -- is emitted verbatim.
 * The decoder re-escapes with these rules, so a string is only stored
 * unescaped if the encoder verified that this reproduces it exactly. */
constexpr size_t escapedLen(unsigned char c)
{
    if (c == '"' || c == '\\')
        return 2;
    if (c >= 0x20)
        return 1;
    switch (c) {
        case '\n':
        case '\r':
        case '\t':
        case '\b':
        case '\f':
            return 2;
        default:
            return 6;
    }
}

inline char* writeEscaped(char* out, unsigned char c)
{
    static const char kHex[] = "0123456789abcdef";
    if (c >= 0x20 && c != '"' && c != '\\') {
        *out++ = (char)c;
        return out;
    }
    *out++ = '\\';
    switch (c) {
        case '"':
            *out++ = '"';
            return out;
        case '\\':
            *out++ = '\\';
            return out;
        case '\n':
            *out++ = 'n';
            return out;
        case '\r':
            *out++ = 'r';
            return out;
        case '\t':
            *out++ = 't';
            return out;
        case '\b':
            *out++ = 'b';
            return out;
        case '\f':
            *out++ = 'f';
            return out;
        default:
            *out++ = 'u';
            *out++ = '0';
            *out++ = '0';
            *out++ = kHex[c >> 4];
            *out++ = kHex[c & 15];
            return out;
    }
}

struct ByteTable {
    uint8_t v[256];
};

/// Escaped length of every byte, for sizing and for finding the next byte
/// that needs escaping without a branch per byte.
constexpr ByteTable makeEscLenTable()
{
    ByteTable t{};
    for (int i = 0; i < 256; ++i) {
        t.v[i] = (uint8_t)escapedLen((unsigned char)i);
    }
    return t;
}
constexpr ByteTable kEscLen = makeEscLenTable();

/// 1 for bytes that end an ordinary run inside a string literal: the quote,
/// the backslash, and control characters.
constexpr ByteTable makeStrSpecialTable()
{
    ByteTable t{};
    for (int i = 0; i < 256; ++i) {
        t.v[i] = (i < 0x20 || i == '"' || i == '\\') ? 1 : 0;
    }
    return t;
}
constexpr ByteTable kStrSpecial = makeStrSpecialTable();

/// Escapes @p n bytes of @p p into @p o: copies runs of ordinary bytes in
/// bulk and only takes the per-byte path on the ~5% that need escaping.
inline char* writeEscapedField(char* o, const char* p, size_t n)
{
    const char* const end = p + n;
    while (p < end) {
        const char* q = p;
        while (q < end && kEscLen.v[(unsigned char)*q] == 1) {
            ++q;
        }
        std::memcpy(o, p, (size_t)(q - p));
        o += q - p;
        if (q < end) {
            o = writeEscaped(o, (unsigned char)*q);
            ++q;
        }
        p = q;
    }
    return o;
}

inline bool hasShortEscape(uint32_t cp)
{
    return cp == '\n' || cp == '\r' || cp == '\t' || cp == '\b' || cp == '\f';
}

/// Result of scanning the content of one string literal.
struct StrScan {
    size_t rawLen;   // bytes between the quotes, as written
    size_t unescLen; // bytes once unescaped (meaningful only when canonical)
    bool valid;      // well-formed and terminated by a quote
    bool canonical;  // re-escaping the unescaped bytes reproduces the raw text
};

/// Scans from just after an opening quote up to the closing quote, decoding
/// escapes only as far as needed to size the unescaped form and to decide
/// whether canonical re-escaping would reproduce the text.
StrScan scanString(const char* p, const char* const end)
{
    StrScan s{ 0, 0, false, true };
    const char* const start = p;
    while (p < end) {
        unsigned char const c = (unsigned char)*p;
        if (c == '"') {
            s.rawLen = (size_t)(p - start);
            s.valid  = true;
            return s;
        }
        if (c < 0x20) {
            return s; // control characters must be escaped inside strings
        }
        if (c != '\\') {
            // ordinary run: everything but '"', '\\' and control characters
            const char* q = p + 1;
            while (q < end && !kStrSpecial.v[(unsigned char)*q]) {
                ++q;
            }
            s.unescLen += (size_t)(q - p);
            p = q;
            continue;
        }
        if (p + 1 >= end) {
            return s;
        }
        switch (p[1]) {
            case '"':
            case '\\':
            case 'n':
            case 'r':
            case 't':
            case 'b':
            case 'f':
                // exactly what the canonical escaper emits for these bytes
                p += 2;
                ++s.unescLen;
                break;
            case '/':
                // canonical form is a bare '/'
                p += 2;
                ++s.unescLen;
                s.canonical = false;
                break;
            case 'u': {
                if (p + 6 > end) {
                    return s;
                }
                uint32_t cp;
                if (!parseHex4(p + 2, cp)) {
                    return s;
                }
                size_t adv = 6;
                if (cp >= 0xD800 && cp <= 0xDBFF && p + 12 <= end
                    && p[6] == '\\' && p[7] == 'u') {
                    uint32_t lo;
                    if (parseHex4(p + 8, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp  = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        adv = 12;
                    }
                }
                // The canonical escaper only emits \u00xx, lowercase, for
                // control characters without a short form. Anything else --
                // printable, non-ASCII, surrogates (paired or lone), uppercase
                // hex, or a control char that has a short form -- is written
                // some other way, so the string must be kept as written.
                if (cp >= 0x20 || hasShortEscape(cp) || !isLowerHex4(p + 2)) {
                    s.canonical = false;
                }
                s.unescLen += utf8Len(cp);
                p += adv;
                break;
            }
            default:
                return s; // invalid escape
        }
    }
    return s; // unterminated
}

/// Unescapes @p rawLen bytes of validated string content into @p out.
char* unescapeTo(char* out, const char* p, size_t rawLen)
{
    const char* const end = p + rawLen;
    while (p < end) {
        if (*p != '\\') {
            const char* q =
                    (const char*)std::memchr(p, '\\', (size_t)(end - p));
            if (q == nullptr) {
                q = end;
            }
            std::memcpy(out, p, (size_t)(q - p));
            out += q - p;
            p = q;
            continue;
        }
        switch (p[1]) {
            case '"':
                *out++ = '"';
                p += 2;
                break;
            case '\\':
                *out++ = '\\';
                p += 2;
                break;
            case '/':
                *out++ = '/';
                p += 2;
                break;
            case 'n':
                *out++ = '\n';
                p += 2;
                break;
            case 'r':
                *out++ = '\r';
                p += 2;
                break;
            case 't':
                *out++ = '\t';
                p += 2;
                break;
            case 'b':
                *out++ = '\b';
                p += 2;
                break;
            case 'f':
                *out++ = '\f';
                p += 2;
                break;
            case 'u': {
                uint32_t cp = 0;
                parseHex4(p + 2, cp);
                size_t adv = 6;
                if (cp >= 0xD800 && cp <= 0xDBFF && p + 12 <= end
                    && p[6] == '\\' && p[7] == 'u') {
                    uint32_t lo;
                    if (parseHex4(p + 8, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp  = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        adv = 12;
                    }
                }
                out = writeUtf8(out, cp);
                p += adv;
                break;
            }
            default:
                ZL_ASSERT_FAIL("unescapeTo called on an unvalidated string");
                *out++ = *p++;
                break;
        }
    }
    return out;
}

/// Length of the JSON number starting at @p p, or 0 if there is none.
/// Accepts the longest valid prefix, so "1.x" yields "1" and leaves ".x".
size_t scanNumber(const char* p, const char* const end)
{
    const char* q = p;
    if (q < end && *q == '-') {
        ++q;
    }
    if (q < end && *q == '0') {
        ++q;
    } else if (q < end && isDigit(*q)) {
        while (q < end && isDigit(*q)) {
            ++q;
        }
    } else {
        return 0;
    }
    if (q + 1 < end && *q == '.' && isDigit(q[1])) {
        q += 2;
        while (q < end && isDigit(*q)) {
            ++q;
        }
    }
    if (q < end && (*q == 'e' || *q == 'E')) {
        const char* r = q + 1;
        if (r < end && (*r == '+' || *r == '-')) {
            ++r;
        }
        if (r < end && isDigit(*r)) {
            while (r < end && isDigit(*r)) {
                ++r;
            }
            q = r;
        }
    }
    return (size_t)(q - p);
}

inline bool startsWith(
        const char* p,
        const char* const end,
        const char* lit,
        size_t n)
{
    return (size_t)(end - p) >= n && std::memcmp(p, lit, n) == 0;
}

/// A string literal is a key iff the next non-whitespace byte is ':'.
inline bool isKeyAhead(const char* p, const char* const end)
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
    return p < end && *p == ':';
}

/// One pass over the input, reporting each element to @p sink. Run once with
/// a counting sink to size the outputs exactly, then with a writing sink.
template <class Sink>
void lex(const char* p, const char* const end, Sink& sink)
{
    while (p < end) {
        char const c = *p;
        if (c == '"') {
            StrScan const s = scanString(p + 1, end);
            if (s.valid) {
                const char* const raw   = p + 1;
                const char* const after = raw + s.rawLen + 1; // past the quote
                if (!s.canonical) {
                    sink.token(kTokRaw);
                    sink.raw(raw, s.rawLen);
                } else if (isKeyAhead(after, end)) {
                    sink.token(kTokKey);
                    sink.key(raw, s.rawLen, s.unescLen);
                } else {
                    sink.token(kTokStr);
                    sink.value(raw, s.rawLen, s.unescLen);
                }
                p = after;
                continue;
            }
            sink.literal(c);
            ++p;
            continue;
        }
        if (c == '-' || isDigit(c)) {
            size_t const n = scanNumber(p, end);
            if (n > 0) {
                sink.token(kTokNum);
                sink.number(p, n);
                p += n;
                continue;
            }
        }
        if (c == 't' && startsWith(p, end, "true", 4)) {
            sink.token(kTokTrue);
            p += 4;
            continue;
        }
        if (c == 'f' && startsWith(p, end, "false", 5)) {
            sink.token(kTokFalse);
            p += 5;
            continue;
        }
        if (c == 'n' && startsWith(p, end, "null", 4)) {
            sink.token(kTokNull);
            p += 4;
            continue;
        }
        sink.literal(c);
        ++p;
    }
}

struct Counts {
    size_t structBytes = 0;
    size_t valueBytes  = 0;
    size_t keyBytes    = 0;
    size_t numberBytes = 0;
    size_t rawBytes    = 0;
    size_t nbValues    = 0;
    size_t nbKeys      = 0;
    size_t nbNumbers   = 0;
    size_t nbRaw       = 0;

    void token(char)
    {
        ++structBytes;
    }
    void literal(char c)
    {
        structBytes += isReserved(c) ? 2 : 1;
    }
    void value(const char*, size_t, size_t unescLen)
    {
        valueBytes += unescLen;
        ++nbValues;
    }
    void key(const char*, size_t, size_t unescLen)
    {
        keyBytes += unescLen;
        ++nbKeys;
    }
    void raw(const char*, size_t rawLen)
    {
        rawBytes += rawLen;
        ++nbRaw;
    }
    void number(const char*, size_t n)
    {
        numberBytes += n;
        ++nbNumbers;
    }
};

struct StrOut {
    char* data;
    uint32_t* lens;
};

struct Writer {
    char* st;
    StrOut values;
    StrOut keys;
    StrOut numbers;
    StrOut raws;

    void token(char t)
    {
        *st++ = t;
    }
    void literal(char c)
    {
        if (isReserved(c)) {
            *st++ = kEsc;
        }
        *st++ = c;
    }
    void value(const char* raw, size_t rawLen, size_t unescLen)
    {
        put(values, raw, rawLen, unescLen);
    }
    void key(const char* raw, size_t rawLen, size_t unescLen)
    {
        put(keys, raw, rawLen, unescLen);
    }
    void raw(const char* raw, size_t rawLen)
    {
        std::memcpy(raws.data, raw, rawLen);
        raws.data += rawLen;
        *raws.lens++ = (uint32_t)rawLen;
    }
    void number(const char* p, size_t n)
    {
        std::memcpy(numbers.data, p, n);
        numbers.data += n;
        *numbers.lens++ = (uint32_t)n;
    }

   private:
    static void put(StrOut& o, const char* raw, size_t rawLen, size_t unescLen)
    {
        char* const e = unescapeTo(o.data, raw, rawLen);
        ZL_ASSERT_EQ((size_t)(e - o.data), unescLen);
        (void)unescLen;
        o.data    = e;
        *o.lens++ = (uint32_t)unescLen;
    }
};

/// Creates string output @p idx sized for @p nb strings totalling @p bytes.
/// Capacities are floored at 1 so empty streams never hit a zero-size edge.
ZL_Output* createStringOut(
        ZL_Encoder* eictx,
        int idx,
        size_t nb,
        size_t bytes,
        StrOut& out)
{
    ZL_Output* const o = ZL_Encoder_createStringStream(
            eictx, idx, std::max<size_t>(nb, 1), std::max<size_t>(bytes, 1));
    if (o == nullptr) {
        return nullptr;
    }
    out.data = (char*)ZL_Output_ptr(o);
    out.lens = ZL_Output_stringLens(o);
    if (out.data == nullptr || out.lens == nullptr) {
        return nullptr;
    }
    return o;
}

ZL_Report jsonLexEncode(ZL_Encoder* eictx, const ZL_Input* in) noexcept
{
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_serial);
    const char* const src = (const char*)ZL_Input_ptr(in);
    size_t const size     = ZL_Input_numElts(in);
    ZL_RET_R_IF(
            node_invalid_input,
            size > UINT32_MAX,
            "json_lex: input larger than 4GB is not supported");

    Counts counts;
    lex(src, src + size, counts);

    ZL_Output* const st = ZL_Encoder_createTypedStream(
            eictx, kStreamStruct, std::max<size_t>(counts.structBytes, 1), 1);
    ZL_RET_R_IF_NULL(allocation, st);
    Writer w;
    w.st = (char*)ZL_Output_ptr(st);
    ZL_Output* const values = createStringOut(
            eictx, kStreamValues, counts.nbValues, counts.valueBytes, w.values);
    ZL_RET_R_IF_NULL(allocation, values);
    ZL_Output* const keys = createStringOut(
            eictx, kStreamKeys, counts.nbKeys, counts.keyBytes, w.keys);
    ZL_RET_R_IF_NULL(allocation, keys);
    ZL_Output* const numbers = createStringOut(
            eictx,
            kStreamNumbers,
            counts.nbNumbers,
            counts.numberBytes,
            w.numbers);
    ZL_RET_R_IF_NULL(allocation, numbers);
    ZL_Output* const raws = createStringOut(
            eictx, kStreamRaw, counts.nbRaw, counts.rawBytes, w.raws);
    ZL_RET_R_IF_NULL(allocation, raws);

    char* const stStart = w.st;
    lex(src, src + size, w);
    ZL_ASSERT_EQ((size_t)(w.st - stStart), counts.structBytes);
    (void)stStart;

    ZL_RET_R_IF_ERR(ZL_Output_commit(st, counts.structBytes));
    ZL_RET_R_IF_ERR(ZL_Output_commit(values, counts.nbValues));
    ZL_RET_R_IF_ERR(ZL_Output_commit(keys, counts.nbKeys));
    ZL_RET_R_IF_ERR(ZL_Output_commit(numbers, counts.nbNumbers));
    ZL_RET_R_IF_ERR(ZL_Output_commit(raws, counts.nbRaw));
    return ZL_returnSuccess();
}

/// Sequential reader over a string-typed input.
struct StrIn {
    const char* data;
    const uint32_t* lens;
    size_t nb;
    size_t total;
    size_t used = 0;

    explicit StrIn(const ZL_Input* in)
            : data((const char*)ZL_Input_ptr(in)),
              lens(ZL_Input_stringLens(in)),
              nb(ZL_Input_numElts(in)),
              total(ZL_Input_contentSize(in))
    {
    }

    /// Escaped size of the whole stream, for sizing the output exactly.
    size_t escapedTotal() const
    {
        size_t n = 0;
        for (size_t i = 0; i < total; ++i) {
            n += kEscLen.v[(unsigned char)data[i]];
        }
        return n;
    }

    /// Consumes the next field. Only valid while used < nb.
    void next(const char*& p, size_t& n)
    {
        n = lens[used++];
        p = data;
        data += n;
    }
};

ZL_Report jsonLexDecode(ZL_Decoder* dictx, const ZL_Input* ins[]) noexcept
{
    ZL_RET_R_IF_NE(
            corruption, ZL_Input_type(ins[kStreamStruct]), ZL_Type_serial);
    for (int i = kStreamValues; i < kNbStreams; ++i) {
        ZL_RET_R_IF_NE(corruption, ZL_Input_type(ins[i]), ZL_Type_string);
    }
    const char* const st = (const char*)ZL_Input_ptr(ins[kStreamStruct]);
    size_t const stSize  = ZL_Input_numElts(ins[kStreamStruct]);
    StrIn values(ins[kStreamValues]);
    StrIn keys(ins[kStreamKeys]);
    StrIn numbers(ins[kStreamNumbers]);
    StrIn raws(ins[kStreamRaw]);

    // Pass 1: validate the structure against the side streams and size the
    // output exactly.
    size_t outSize = 0, nbStr = 0, nbKey = 0, nbNum = 0, nbRaw = 0;
    for (size_t i = 0; i < stSize; ++i) {
        switch (st[i]) {
            case kEsc:
                ZL_RET_R_IF(
                        corruption,
                        i + 1 >= stSize,
                        "json_lex: dangling escape in structure stream");
                ++i;
                ++outSize;
                break;
            case kTokStr:
                ++nbStr;
                break;
            case kTokKey:
                ++nbKey;
                break;
            case kTokRaw:
                ++nbRaw;
                break;
            case kTokNum:
                ++nbNum;
                break;
            case kTokTrue:
            case kTokNull:
                outSize += 4;
                break;
            case kTokFalse:
                outSize += 5;
                break;
            default:
                ++outSize;
                break;
        }
    }
    ZL_RET_R_IF_NE(corruption, nbStr, values.nb);
    ZL_RET_R_IF_NE(corruption, nbKey, keys.nb);
    ZL_RET_R_IF_NE(corruption, nbNum, numbers.nb);
    ZL_RET_R_IF_NE(corruption, nbRaw, raws.nb);
    outSize += 2 * (nbStr + nbKey + nbRaw); // the quotes
    outSize += values.escapedTotal() + keys.escapedTotal();
    outSize += numbers.total + raws.total;

    ZL_Output* const out =
            ZL_Decoder_create1OutStream(dictx, std::max<size_t>(outSize, 1), 1);
    ZL_RET_R_IF_NULL(allocation, out);
    char* o             = (char*)ZL_Output_ptr(out);
    char* const oStart  = o;
    const char* field   = nullptr;
    size_t fieldLen     = 0;

    // Pass 2: rebuild.
    for (size_t i = 0; i < stSize; ++i) {
        switch (st[i]) {
            case kEsc:
                *o++ = st[++i];
                break;
            case kTokStr:
            case kTokKey: {
                StrIn& s = st[i] == kTokStr ? values : keys;
                s.next(field, fieldLen);
                *o++ = '"';
                o    = writeEscapedField(o, field, fieldLen);
                *o++ = '"';
                break;
            }
            case kTokRaw:
                raws.next(field, fieldLen);
                *o++ = '"';
                std::memcpy(o, field, fieldLen);
                o += fieldLen;
                *o++ = '"';
                break;
            case kTokNum:
                numbers.next(field, fieldLen);
                std::memcpy(o, field, fieldLen);
                o += fieldLen;
                break;
            case kTokTrue:
                std::memcpy(o, "true", 4);
                o += 4;
                break;
            case kTokFalse:
                std::memcpy(o, "false", 5);
                o += 5;
                break;
            case kTokNull:
                std::memcpy(o, "null", 4);
                o += 4;
                break;
            default:
                *o++ = st[i];
                break;
        }
    }
    ZL_ASSERT_EQ((size_t)(o - oStart), outSize);
    ZL_RET_R_IF_NE(corruption, (size_t)(o - oStart), outSize);
    ZL_RET_R_IF_ERR(ZL_Output_commit(out, outSize));
    return ZL_returnSuccess();
}

const ZL_Type kOutTypes[kNbStreams] = {
    ZL_Type_serial, // structure
    ZL_Type_string, // values
    ZL_Type_string, // keys
    ZL_Type_string, // numbers
    ZL_Type_string, // raw strings
};

ZL_TypedGraphDesc graphDesc()
{
    ZL_TypedGraphDesc gd = {};
    gd.CTid              = ZL_JSON_LEX_TRANSFORM_ID;
    gd.inStreamType      = ZL_Type_serial;
    gd.outStreamTypes    = kOutTypes;
    gd.nbOutStreams      = kNbStreams;
    return gd;
}

} // namespace
} // namespace openzl::custom_parsers

ZL_NodeID ZL_JsonLex_registerEncoder(ZL_Compressor* compressor)
{
    ZL_TypedEncoderDesc desc = {};
    desc.gd                  = openzl::custom_parsers::graphDesc();
    desc.transform_f         = openzl::custom_parsers::jsonLexEncode;
    desc.name                = ZL_JSON_LEX_NODE_NAME;
    return ZL_Compressor_registerTypedEncoder(compressor, &desc);
}

ZL_Report ZL_JsonLex_registerDecoder(ZL_DCtx* dctx)
{
    ZL_TypedDecoderDesc desc = {};
    desc.gd                  = openzl::custom_parsers::graphDesc();
    desc.transform_f         = openzl::custom_parsers::jsonLexDecode;
    desc.name                = ZL_JSON_LEX_NODE_NAME;
    return ZL_DCtx_registerTypedDecoder(dctx, &desc);
}
