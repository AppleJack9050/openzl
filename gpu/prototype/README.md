# GPU-accelerated OpenZL-style compression pipeline (prototype)

A self-contained CUDA prototype of a **whole format-aware compression pipeline**
in the spirit of [OpenZL](https://github.com/facebook/openzl) — a chain of
data-shaping *transform* codecs followed by an *entropy* stage — running
end-to-end on the GPU (encode **and** decode), tuned for the **NVIDIA RTX 5090**
(Blackwell, compute capability **`sm_120`**).

It is **inspired by, and faithful to, OpenZL's codec semantics**, but it is a
**standalone prototype with its own frame format** — it does *not* produce
OpenZL-compatible frames (byte-exact OpenZL frames would require reproducing the
serial FSE bitstream, which doesn't parallelize). The entropy stage instead uses
**interleaved rANS**, the GPU-appropriate analog of OpenZL's serial FSE/Huffman
(the same family of technique used by GPU entropy coders like Meta's DietGPU /
NVIDIA nvCOMP).

> Built and verified on the target hardware: nvcc 13.2, RTX 5090, driver 610.47,
> WSL2. All 29 correctness cases pass and the GPU path is `compute-sanitizer`
> clean.

---

## What's implemented

**Transform codecs** (GPU encode + decode, semantics mirrored from
`src/openzl/codecs/*/spec.md`):

| codec | forward | inverse | parallelism |
|-------|---------|---------|-------------|
| `delta` | `out[k]=src[k]-src[k-1]` (unsigned modular, widths 1/2/4/8) | inclusive modular **prefix scan** | encode: stencil; decode: CUB `DeviceScan` |
| `zigzag` | `p≥0→2p, n<0→2\|n\|-1` | `(z>>1) ^ -(z&1)` | embarrassingly parallel |
| `transpose` | byte-plane SoA: `out[p*N+e]=in[e*W+p]` | exact inverse | one thread per element |
| `bitpack` | LSB-first pack to `nbits=1+highbit(max)` | shift+mask unpack | atomicOr pack / closed-form unpack |

**Entropy backend:** interleaved byte-wise **rANS** (Ryg `rans_byte`, 32-bit
state, 12-bit probabilities). The byte stream is cut into fixed **512-byte
segments**, each an independent rANS stream, so thousands of segments
encode/decode fully in parallel. The normalized frequency table is built on the
host from a GPU histogram.

**Pipeline + frame:** a self-describing container (`frame.h`) with a header, an
ordered transform table (stored in decode order), the entropy section, and a
content checksum. The host (`pipeline.cu`) drives the stage chain; the data
plane — transforms, entropy, *and* the integrity checksum (a GPU-parallel
chunked FNV, `checksum.cu`) — is all on the GPU.

Pre-wired pipelines (`PipelineConfig`):
`delta+zigzag+rans`, `delta+zigzag+bitpack`, `transpose+rans`, `rans`, `store`.

---

## Build & run

Requirements: CUDA toolkit (nvcc) with `sm_120` support (CUDA ≥ 12.8; tested on
13.2), a C++17 host compiler.

```sh
cd gpu/prototype
make            # builds build/test_roundtrip and build/bench
make test       # correctness harness (round-trip + bit-exact GPU==CPU frame)
make bench      # throughput benchmark (default 128 MB)
./build/bench 256   # benchmark at 256 MB

# Different GPU? override the arch:
make ARCH=sm_90    # H100 / GH200
make ARCH=sm_89    # RTX 4090
```

---

## Correctness methodology

Three independent checks, all green:

1. **Round-trip** — `gpu_decompress(gpu_compress(x)) == x` for every case.
2. **Bit-exact CPU oracle** — a pure-host reference (`cpu_pipeline.h`) reuses the
   *same* `__host__ __device__` codec core, so the GPU frame must be
   **byte-for-byte identical** to the CPU frame. Any kernel bug surfaces as a
   frame diff, independent of round-trip. (Everything compiles with nvcc so host
   and device share one source of truth.)
3. **`compute-sanitizer --tool memcheck`** — 0 errors.

Coverage: 6 dataset shapes × the relevant pipelines, element widths 1/2/4/8,
boundary cases (bitpack `nbits`=64, full 256-symbol alphabet, segment-aligned
sizes), edge cases (empty, single element, all-identical bytes, non-aligned,
incompressible), and **adversarial frames** (truncated, bad magic/version,
size-overflow, near-2⁶⁴ length fields) which must be *rejected gracefully*
(`ok=false`, never an OOB/crash). `make test` → **38 passed, 0 failed**.

### Untrusted-frame hardening

`gpu_decompress` / `cpu_decompress` take raw frame bytes, so the decoder is an
untrusted-input boundary. `parse_frame` uses overflow-safe (`rem = n - off`)
bound checks, and a `validate_decodable()` gate walks the inverse-transform
chain *before any kernel runs* — rejecting any frame whose sizes, element
widths, codec ids, bitpack widths, or rANS segment count are inconsistent or
would write past the decode buffer. (These checks were added after an
adversarial review found that crafted-but-parseable frames could otherwise
drive out-of-bounds writes.)

---

## Benchmark results (RTX 5090, 256 MB, CUDA 13.2, WSL2)

**Device-only kernel throughput** (data resident on the GPU, CUDA-event timed —
this is the GPU's actual processing rate; transform numbers vary ±2× with
clock-boost state):

```
zigzag encode  (u32)         ~660-710 GB/s
zigzag decode  (u32)         ~660-710 GB/s
delta  encode  (u32)         ~680     GB/s
delta  decode  (scan, u32)   ~70-110  GB/s
transpose encode (W16)       ~680     GB/s
rANS   encode                ~10      GB/s
rANS   decode                ~37      GB/s
--
FULL compress   (delta+zigzag+rANS)   ~10 GB/s
FULL decompress (rANS+zigzag+delta)   ~26 GB/s
```

The single-threaded **CPU reference** runs the same pipeline at **~0.15 GB/s**,
so the resident GPU pipeline is **~65× faster at compress and ~170× at
decompress**. (The CPU reference is correctness-first, not an optimized SIMD
baseline like OpenZL's production CPU coder.)

**Compression ratios** (lossless) on the synthetic datasets:

| data | pipeline | ratio |
|------|----------|-------|
| sequential u32 | delta+zigzag+bitpack | **16.0×** |
| all-identical bytes | rans | **81.5×** |
| biased bytes | rans | 6.7× |
| random walk u32 | delta+zigzag+rans | 5.2× |
| low-cardinality bytes | rans | 3.0× |
| incompressible random | store | 1.00× (no expansion) |

**End-to-end host API** (`gpu_compress`/`gpu_decompress`, *including* pageable
H2D/D2H over PCIe and per-call allocation), 256 MB:

| data | pipeline | compress | decompress | vs CPU |
|------|----------|----------|------------|--------|
| sequential u32 | delta+zigzag+bitpack | 6.4 GB/s | 2.3 GB/s | 21× |
| biased bytes | rans | 3.1 GB/s | 2.4 GB/s | 47× |
| random walk u32 | delta+zigzag+rans | 2.5 GB/s | 1.9 GB/s | 18× |
| low-cardinality | rans | 2.0 GB/s | 2.1 GB/s | 11× |
| struct AoS | transpose+rans | 1.1 GB/s | 1.6 GB/s | 9× |

End-to-end is now bounded by **pageable-PCIe transfer and the rANS stage**, not
by host bookkeeping. (An earlier version was dominated by a serial host checksum
at ~1.5 GB/s; that hash now runs on the GPU, which is why the API throughput
jumped 2.5–6×.) Keeping data resident / batching and using pinned host memory
would close most of the remaining gap to the device-only numbers above.

---

## Performance notes & next steps

The **rANS stage is the compute bottleneck** (~12 GB/s encode). The current
design uses one thread per 512-byte segment, which means consecutive threads
read memory 512 bytes apart (uncoalesced) and each does a serial divide per
symbol. The standard fix — and the clearest next optimization — is the
**interleaved-warp layout** (à la DietGPU/nvCOMP): 32 lanes of a warp share one
stream and read *contiguous* bytes per step, making loads coalesce and hiding
divide latency. That alone should bring rANS into the ~100 GB/s range.

Other improvements, in rough priority:
- **Pinned host staging + transfer/compute overlap** to push the end-to-end API
  closer to the device-only numbers.
- **Persistent context** (reuse device buffers across calls) to remove per-call
  allocation.
- **`delta` decode** can use a single fused scan kernel instead of
  expand→CUB→contract (saves two memory passes).
- **`transpose`** shared-memory tiling for coalesced plane writes at large `W`.
- Per-field pipelines after `transpose` (the real OpenZL pattern) instead of a
  single linear chain.

### Known limits

- **Scale (32-bit fields).** rANS segment offsets and total are 64-bit (the
  packed stream may exceed 4 GiB), but `StageMeta.n_elts` and `nb_seg` are
  32-bit in the frame, capping a single stage at 2³² elements and the
  pre-entropy stream at ~2 TiB. Larger inputs would need 64-bit frame fields.
- **Corrupt-content trust.** The validator guarantees a malformed frame can't
  cause an OOB *write*; the GPU decode buffer is also +16-padded against renorm
  over-*reads*. A frame with valid lengths but deliberately corrupted rANS
  *bytes* decodes to wrong data that the content checksum then rejects — i.e.
  detected, not silently accepted. The CPU reference oracle is not hardened
  against adversarial content (it's a verification tool, fed trusted frames).

---

## Layout

```
src/
  codec_core.cuh    shared host+device element math + rANS core (one source of truth)
  rans_table.h      histogram -> normalized frequency table + (de)serialize
  transforms.cu/.cuh  GPU delta / zigzag / transpose / bitpack kernels + launchers
  rans.cu/.cuh      GPU histogram + per-segment rANS encode/decode + compaction
  frame.h           self-describing container (build_frame / parse_frame)
  pipeline.cu/.cuh  host orchestration of the GPU compress/decompress pipeline
  cpu_pipeline.h    pure-host bit-exact reference (verification oracle)
  datagen.h         deterministic synthetic datasets
tests/test_roundtrip.cu   correctness harness
bench/bench.cu            throughput benchmark
Makefile
```
