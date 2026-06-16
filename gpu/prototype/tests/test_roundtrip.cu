// gpu/prototype/tests/test_roundtrip.cu
//
// Correctness harness. For every (dataset, pipeline) case it checks:
//   (1) the GPU frame is BYTE-IDENTICAL to the CPU reference frame,
//   (2) gpu_decompress(frame) reproduces the original exactly,
//   (3) cpu_decompress(frame) reproduces the original exactly (cross-check),
// plus edge cases (empty, single element, all-same, random/incompressible).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/pipeline.cuh"
#include "../src/cpu_pipeline.h"
#include "../src/datagen.h"

using namespace gpuzl;

static int g_pass = 0, g_fail = 0;

static bool bytes_eq(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b){
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size())==0);
}
// First differing offset for diagnostics.
static long first_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b){
    size_t m = std::min(a.size(), b.size());
    for(size_t i=0;i<m;++i) if(a[i]!=b[i]) return (long)i;
    if(a.size()!=b.size()) return (long)m;
    return -1;
}

static void run(const char* label, const std::vector<uint8_t>& input, const PipelineConfig& cfg){
    std::vector<uint8_t> gf = gpu_compress(input.data(), input.size(), cfg);
    std::vector<uint8_t> cf = cpu_compress(input.data(), input.size(), cfg);

    bool frame_match = bytes_eq(gf, cf);

    bool gok=false, cok=false; std::string ge, ce;
    std::vector<uint8_t> gout = gpu_decompress(gf.data(), gf.size(), &gok, &ge);
    std::vector<uint8_t> cout = cpu_decompress(gf.data(), gf.size(), &cok, &ce);

    bool grt = gok && bytes_eq(gout, input);
    bool crt = cok && bytes_eq(cout, input);

    bool pass = frame_match && grt && crt;
    double ratio = input.empty() ? 0.0 : (double)input.size() / (double)gf.size();

    printf("[%s] %-22s %-22s  in=%8zu frame=%8zu  ratio=%5.2fx  %s\n",
           pass?"PASS":"FAIL", label, cfg.name.c_str(), input.size(), gf.size(), ratio,
           pass?"":"<<<");
    if(!pass){
        if(!frame_match) printf("    frame mismatch: gpu=%zu cpu=%zu firstDiff=%ld\n", gf.size(), cf.size(), first_diff(gf,cf));
        if(!grt) printf("    gpu round-trip FAILED (ok=%d err=%s diff=%ld)\n", gok, ge.c_str(), first_diff(gout,input));
        if(!crt) printf("    cpu round-trip FAILED (ok=%d err=%s diff=%ld)\n", cok, ce.c_str(), first_diff(cout,input));
    }
    if(pass) ++g_pass; else ++g_fail;
}

int main(){
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s (sm_%d%d, %d SMs)\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

    // --- main dataset x pipeline matrix (a few MB each) ---
    const size_t MB = 1u<<20;
    {
        auto d = datagen::make(datagen::Kind::SequentialU32, 4*MB);
        run("sequential_u32", d, cfg_delta_zigzag_rans(4));
        run("sequential_u32", d, cfg_delta_zigzag_bitpack(4));
        run("sequential_u32", d, cfg_rans_only());
    }
    {
        auto d = datagen::make(datagen::Kind::RandomWalkU32, 4*MB);
        run("randomwalk_u32", d, cfg_delta_zigzag_rans(4));
        run("randomwalk_u32", d, cfg_delta_zigzag_bitpack(4));
        run("randomwalk_u32", d, cfg_rans_only());
    }
    {
        auto d = datagen::make(datagen::Kind::LowCardU8, 4*MB);
        run("lowcard_u8", d, cfg_rans_only());
        run("lowcard_u8", d, cfg_store());
    }
    {
        auto d = datagen::make(datagen::Kind::BiasedU8, 4*MB);
        run("biased_u8", d, cfg_rans_only());
    }
    {
        auto d = datagen::make(datagen::Kind::StructAoS, 4*MB);
        run("struct_aos", d, cfg_transpose_rans(16));
        run("struct_aos", d, cfg_rans_only());
    }
    {
        auto d = datagen::make(datagen::Kind::RandomU8, 4*MB);
        run("random_u8", d, cfg_rans_only());
        run("random_u8", d, cfg_store());
    }

    // --- width coverage for zigzag/delta (1/2/4/8) ---
    for(int W : {1,2,4,8}){
        auto d = datagen::make(datagen::Kind::RandomWalkU32, 1*MB); // 4-byte structured
        // reinterpret as width-W elements (size already multiple of 8)
        char lbl[32]; std::snprintf(lbl,sizeof(lbl),"rw_W%d", W);
        run(lbl, d, cfg_delta_zigzag_rans(W));
        run(lbl, d, cfg_delta_zigzag_bitpack(W));
    }

    // --- boundary coverage: bitpack nbits up to 64, full-width wraparound, 256-symbol alphabet ---
    {
        // random bytes reinterpreted as u64 elements -> delta/zigzag produce
        // near-full-width values -> bitpack picks nbits up to 64.
        auto d = datagen::make(datagen::Kind::RandomU8, 2*MB);
        run("rand_u64", d, cfg_delta_zigzag_bitpack(8));
        run("rand_u64", d, cfg_delta_zigzag_rans(8));
    }
    {
        // every one of the 256 byte symbols present -> full-alphabet rANS table.
        std::vector<uint8_t> alln(256*4096);
        for(size_t i=0;i<alln.size();++i) alln[i]=(uint8_t)(i & 0xff);
        run("all_256_syms", alln, cfg_rans_only());
    }
    {
        // size exactly on a rANS segment boundary (multiple of 512).
        auto d = datagen::make(datagen::Kind::BiasedU8, 512*200);
        run("seg_aligned", d, cfg_rans_only());
    }

    // --- edge cases ---
    {
        std::vector<uint8_t> empty;
        run("empty", empty, cfg_rans_only());
        run("empty", empty, cfg_store());
        run("empty", empty, cfg_delta_zigzag_rans(4));
    }
    {
        std::vector<uint8_t> one(4, 0); one[0]=0x2a; // single u32 element
        run("single_u32", one, cfg_delta_zigzag_rans(4));
        run("single_u32", one, cfg_delta_zigzag_bitpack(4));
    }
    {
        std::vector<uint8_t> same(64*1024, 0x7e); // all identical bytes
        run("all_same", same, cfg_rans_only());
    }
    {
        // odd size not divisible by 4: use byte pipelines only
        std::vector<uint8_t> odd(4096*3 + 7);
        datagen::Rng r(99); for(auto& b: odd) b=(uint8_t)r.u32();
        run("odd_random", odd, cfg_rans_only());
        run("odd_random", odd, cfg_store());
    }

    // --- negative tests: malformed/adversarial frames must be REJECTED, not crash ---
    {
        auto expect_reject = [&](const char* label, const std::vector<uint8_t>& f){
            bool gok=true, cok=true; std::string ge, ce;
            gpu_decompress(f.data(), f.size(), &gok, &ge);
            cpu_decompress(f.data(), f.size(), &cok, &ce);
            bool pass = (!gok && !cok); // both reject, neither crashes
            printf("[%s] %-22s reject  gpu_ok=%d cpu_ok=%d (%s)\n",
                   pass?"PASS":"FAIL", label, gok, cok, ge.empty()?ce.c_str():ge.c_str());
            if(pass) ++g_pass; else ++g_fail;
        };
        // a valid baseline frame to mutate
        std::vector<uint8_t> base = gpu_compress(
            datagen::make(datagen::Kind::BiasedU8, 64*1024).data(), 64*1024, cfg_rans_only());

        std::vector<uint8_t> truncated(base.begin(), base.end()-16);
        expect_reject("truncated", truncated);

        std::vector<uint8_t> badmagic = base; badmagic[0] ^= 0xff;
        expect_reject("bad_magic", badmagic);

        std::vector<uint8_t> badver = base; badver[4] = 99;
        expect_reject("bad_version", badver);

        // hand-crafted: claims orig=0 (cap=64) but a 10000-byte stored payload.
        // parse_frame accepts it; validate_decodable must reject before any write.
        {
            std::vector<uint8_t> f;
            f.insert(f.end(), FRAME_MAGIC, FRAME_MAGIC+4);
            put_u8(f,FRAME_VERSION); put_u8(f,0); put_u8(f,ENT_STORED); put_u8(f,0);
            put_u64(f, 0);      // orig_nbytes
            put_u64(f, 10000);  // pre_entropy_nbytes (L) >> cap
            f.resize(f.size()+10000, 0xAA);
            put_u32(f, 0);
            expect_reject("cap_overflow_stored", f);
        }
        // hand-crafted: near-2^64 pre_entropy must not wrap the bound check.
        {
            std::vector<uint8_t> f;
            f.insert(f.end(), FRAME_MAGIC, FRAME_MAGIC+4);
            put_u8(f,FRAME_VERSION); put_u8(f,0); put_u8(f,ENT_STORED); put_u8(f,0);
            put_u64(f, 0);
            put_u64(f, 0xFFFFFFFFFFFFFFECull); // huge L
            put_u8(f,1); put_u8(f,2); put_u8(f,3); put_u8(f,4);
            put_u32(f, 0);
            expect_reject("pre_entropy_overflow", f);
        }
    }

    printf("\n==== %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
