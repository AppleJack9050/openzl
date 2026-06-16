// gpu/prototype/src/datagen.h
//
// Deterministic synthetic datasets used by the correctness tests and the
// benchmark. Each generator targets a structure that a particular codec in the
// pipeline is supposed to exploit, so we can see real compression happening.
//
// Pure host code, no CUDA. Deterministic (seeded) so tests are reproducible.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace datagen {

// xorshift64 — tiny deterministic PRNG (no <random> locale/impl variance).
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    std::uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    std::uint32_t u32() { return (std::uint32_t)(next() >> 11); }
    double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

enum class Kind {
    SequentialU32,    // 0,1,2,...  -> delta crushes this
    RandomWalkU32,    // cumulative small steps -> delta + zigzag + entropy
    LowCardU8,        // few distinct byte values -> tokenize + entropy
    StructAoS,        // array-of-structs records -> transpose helps
    RandomU8,         // incompressible baseline
    BiasedU8,         // skewed byte distribution -> entropy only
};

inline const char* kind_name(Kind k) {
    switch (k) {
        case Kind::SequentialU32: return "sequential_u32";
        case Kind::RandomWalkU32: return "randomwalk_u32";
        case Kind::LowCardU8:     return "lowcard_u8";
        case Kind::StructAoS:     return "struct_aos";
        case Kind::RandomU8:      return "random_u8";
        case Kind::BiasedU8:      return "biased_u8";
    }
    return "?";
}

// Generate `n_bytes` (approx) of data of the given kind. The returned buffer's
// size is rounded to the element width so codecs see whole elements.
inline std::vector<std::uint8_t> make(Kind k, size_t n_bytes, std::uint64_t seed = 12345) {
    Rng rng(seed);
    std::vector<std::uint8_t> out;

    switch (k) {
        case Kind::SequentialU32: {
            size_t n = n_bytes / 4;
            out.resize(n * 4);
            auto* p = reinterpret_cast<std::uint32_t*>(out.data());
            for (size_t i = 0; i < n; ++i) p[i] = (std::uint32_t)i;
            break;
        }
        case Kind::RandomWalkU32: {
            size_t n = n_bytes / 4;
            out.resize(n * 4);
            auto* p = reinterpret_cast<std::uint32_t*>(out.data());
            std::int64_t cur = 1'000'000;
            for (size_t i = 0; i < n; ++i) {
                int step = (int)(rng.u32() % 9) - 4; // [-4, 4]
                cur += step;
                p[i] = (std::uint32_t)cur;
            }
            break;
        }
        case Kind::LowCardU8: {
            out.resize(n_bytes);
            std::uint8_t alphabet[6] = {0x00, 0x10, 0x41, 0x42, 0x7F, 0xFE};
            for (size_t i = 0; i < n_bytes; ++i)
                out[i] = alphabet[rng.u32() % 6];
            break;
        }
        case Kind::StructAoS: {
            // 16-byte records: [u32 id][u32 ts][u32 flagsLow][u32 flagsHigh]
            // Fields are individually low-entropy but interleaved in memory,
            // so a transpose-by-field exposes the structure to delta/entropy.
            struct Rec { std::uint32_t id, ts, a, b; };
            size_t n = n_bytes / sizeof(Rec);
            out.resize(n * sizeof(Rec));
            auto* r = reinterpret_cast<Rec*>(out.data());
            std::uint32_t ts = 0;
            for (size_t i = 0; i < n; ++i) {
                r[i].id = (std::uint32_t)i;
                ts += 1 + (rng.u32() % 3);
                r[i].ts = ts;
                r[i].a = 0xABCD0000u | (rng.u32() & 0xFF);
                r[i].b = 0x0000DEADu;
            }
            break;
        }
        case Kind::RandomU8: {
            out.resize(n_bytes);
            for (size_t i = 0; i < n_bytes; ++i) out[i] = (std::uint8_t)rng.u32();
            break;
        }
        case Kind::BiasedU8: {
            // Geometric-ish skew: small values dominate.
            out.resize(n_bytes);
            for (size_t i = 0; i < n_bytes; ++i) {
                std::uint32_t v = 0;
                while ((rng.u32() & 0x3) == 0 && v < 255) ++v; // ~3/4 stop each step
                out[i] = (std::uint8_t)v;
            }
            break;
        }
    }
    return out;
}

} // namespace datagen
