// gpu/prototype/bench/bench.cu
//
// Throughput benchmark on the RTX 5090. Two parts:
//   A) End-to-end host API (gpu_compress / gpu_decompress, includes H2D/D2H and
//      allocation) vs the single-threaded CPU reference.
//   B) Device-only kernel throughput (CUDA-event timed, data resident on the
//      GPU, buffers preallocated) for each stage in isolation.
//
// Throughput is reported as GB/s over the ORIGINAL (uncompressed) byte count.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#include "../src/pipeline.cuh"
#include "../src/cpu_pipeline.h"
#include "../src/transforms.cuh"
#include "../src/rans.cuh"
#include "../src/common.cuh"
#include "../src/datagen.h"

using namespace gpuzl;

static double now_ms(){ // host wall clock via CpuTimer-like
    static CpuTimer t; (void)t; return 0; }

// ---- Part A: end-to-end API throughput --------------------------------------
static void bench_endtoend(const char* label, const std::vector<uint8_t>& in, const PipelineConfig& cfg){
    // warmup + correctness sanity
    std::vector<uint8_t> frame = gpu_compress(in.data(), in.size(), cfg);
    bool ok=false; auto dec = gpu_decompress(frame.data(), frame.size(), &ok, nullptr);
    bool rt = ok && dec.size()==in.size() && (in.empty()||std::memcmp(dec.data(),in.data(),in.size())==0);

    const int K = 5;
    CpuTimer t;
    // compress
    t.start();
    for(int i=0;i<K;++i){ auto f = gpu_compress(in.data(), in.size(), cfg); asm volatile(""); (void)f; }
    double comp_ms = t.stop_ms()/K;
    // decompress
    t.start();
    for(int i=0;i<K;++i){ bool o; auto d = gpu_decompress(frame.data(), frame.size(), &o, nullptr); (void)d; }
    double decomp_ms = t.stop_ms()/K;
    // cpu compress (one shot — it's slow)
    t.start();
    auto cf = cpu_compress(in.data(), in.size(), cfg);
    double cpu_ms = t.stop_ms();

    double ratio = in.empty()?0:(double)in.size()/frame.size();
    printf("  %-22s %-20s ratio=%6.2fx  GPU comp=%7.2f GB/s  GPU decomp=%7.2f GB/s  CPU comp=%6.2f GB/s  speedup=%5.1fx  %s\n",
           label, cfg.name.c_str(), ratio,
           gbps(in.size(), comp_ms), gbps(in.size(), decomp_ms), gbps(in.size(), cpu_ms),
           cpu_ms>0? (cpu_ms/comp_ms):0.0, rt?"":"[RT FAIL]");
}

// ---- Part B: device-only kernel throughput ----------------------------------
static double time_kernel(int K, cudaStream_t s, void(*fn)(void*), void* ctx){
    GpuTimer g; fn(ctx); ZL_GPU_CHECK(cudaStreamSynchronize(s)); // warmup
    g.start();
    for(int i=0;i<K;++i) fn(ctx);
    return g.stop()/K;
}

struct StageCtx { const uint8_t* in; uint8_t* out; size_t n; int W; int nbits; cudaStream_t s; };

static void bench_kernels(size_t bytes){
    printf("\nPart B - device-only kernel throughput (data resident, events; %zu MB):\n", bytes>>20);
    cudaStream_t s; ZL_GPU_CHECK(cudaStreamCreate(&s));

    // structured numeric data so bitpack/rANS do real work
    auto host = datagen::make(datagen::Kind::RandomWalkU32, bytes);
    size_t n4 = host.size()/4;
    uint8_t *dA=nullptr,*dB=nullptr;
    ZL_GPU_CHECK(cudaMalloc(&dA, host.size()+64));
    ZL_GPU_CHECK(cudaMalloc(&dB, host.size()+64));
    ZL_GPU_CHECK(cudaMemcpy(dA, host.data(), host.size(), cudaMemcpyHostToDevice));
    const int K=20;

    double t_ze=0,t_zd=0,t_de=0,t_dd=0,t_re=0,t_rd=0;
    auto report=[&](const char* name, double ms){ printf("  %-26s %8.2f GB/s  (%.3f ms)\n", name, gbps(host.size(), ms), ms); };

    { GpuTimer g; zigzag_encode(dA,dB,n4,4,s); cudaStreamSynchronize(s); g.start(); for(int i=0;i<K;++i) zigzag_encode(dA,dB,n4,4,s); t_ze=g.stop()/K; report("zigzag encode  (u32)", t_ze); }
    { GpuTimer g; zigzag_decode(dA,dB,n4,4,s); cudaStreamSynchronize(s); g.start(); for(int i=0;i<K;++i) zigzag_decode(dA,dB,n4,4,s); t_zd=g.stop()/K; report("zigzag decode  (u32)", t_zd); }
    { GpuTimer g; delta_encode(dA,dB,n4,4,s);  cudaStreamSynchronize(s); g.start(); for(int i=0;i<K;++i) delta_encode(dA,dB,n4,4,s);  t_de=g.stop()/K; report("delta encode   (u32)", t_de); }
    { GpuTimer g; delta_decode(dA,dB,n4,4,s);  cudaStreamSynchronize(s); g.start(); for(int i=0;i<K;++i) delta_decode(dA,dB,n4,4,s);  t_dd=g.stop()/K; report("delta decode (scan,u32)", t_dd); }
    { GpuTimer g; transpose_encode(dA,dB,host.size()/16,16,s); cudaStreamSynchronize(s); g.start(); for(int i=0;i<K;++i) transpose_encode(dA,dB,host.size()/16,16,s); report("transpose encode (W16)", g.stop()/K); }

    // rANS encode/decode on the byte stream
    { RansEncoded enc; rans_encode(dA, host.size(), enc, s); cudaStreamSynchronize(s);
      GpuTimer g; g.start(); for(int i=0;i<K;++i){ RansEncoded e2; rans_encode(dA, host.size(), e2, s); if(e2.d_packed) cudaFreeAsync(e2.d_packed,s);}
      t_re=g.stop()/K; report("rANS encode", t_re);
      // decode
      uint8_t* dpk=nullptr; ZL_GPU_CHECK(cudaMalloc(&dpk, enc.total?enc.total:1));
      ZL_GPU_CHECK(cudaMemcpy(dpk, enc.d_packed, enc.total, cudaMemcpyDeviceToDevice));
      GpuTimer g2; rans_decode(dpk, enc.total, enc.table, enc.seg_len, enc.seg_size, enc.nb_seg, host.size(), dB, s); cudaStreamSynchronize(s);
      g2.start(); for(int i=0;i<K;++i) rans_decode(dpk, enc.total, enc.table, enc.seg_len, enc.seg_size, enc.nb_seg, host.size(), dB, s);
      t_rd=g2.stop()/K; report("rANS decode", t_rd);
      cudaFree(dpk); if(enc.d_packed) cudaFree(enc.d_packed);
    }

    printf("  --\n");
    report("FULL compress  (d+z+rANS)", t_de + t_ze + t_re);
    report("FULL decompress(rANS+z+d)", t_rd + t_zd + t_dd);

    cudaFree(dA); cudaFree(dB); cudaStreamDestroy(s);
    (void)now_ms; (void)time_kernel;
}

int main(int argc, char** argv){
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  sm_%d%d  %d SMs  %.0f GB  memBus=%d-bit\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem/1e9, p.memoryBusWidth);

    size_t MB = 1u<<20;
    size_t sz = (argc>1) ? (size_t)atoll(argv[1])*MB : 128*MB;

    printf("Part A - end-to-end host API throughput (incl. H2D/D2H + alloc; %zu MB):\n", sz>>20);
    bench_endtoend("randomwalk_u32", datagen::make(datagen::Kind::RandomWalkU32, sz), cfg_delta_zigzag_rans(4));
    bench_endtoend("randomwalk_u32", datagen::make(datagen::Kind::RandomWalkU32, sz), cfg_delta_zigzag_bitpack(4));
    bench_endtoend("biased_u8",      datagen::make(datagen::Kind::BiasedU8,      sz), cfg_rans_only());
    bench_endtoend("lowcard_u8",     datagen::make(datagen::Kind::LowCardU8,     sz), cfg_rans_only());
    bench_endtoend("struct_aos",     datagen::make(datagen::Kind::StructAoS,     sz), cfg_transpose_rans(16));
    bench_endtoend("sequential_u32", datagen::make(datagen::Kind::SequentialU32, sz), cfg_delta_zigzag_bitpack(4));

    bench_kernels(sz);
    return 0;
}
