# gpu/prototype/Makefile — OpenZL-style GPU compression prototype
#
# Targets:
#   make            build the test + benchmark binaries
#   make test       build & run the correctness harness
#   make bench      build & run the GPU-vs-CPU benchmark
#   make clean
#
# Override the GPU arch if needed:  make ARCH=sm_90
NVCC  ?= nvcc
ARCH  ?= sm_120                      # RTX 5090 (Blackwell). sm_90=H100, sm_89=4090
NVCCFLAGS ?= -arch=$(ARCH) -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo
BUILD := build

SRC := src/transforms.cu src/rans.cu src/pipeline.cu src/checksum.cu
OBJ := $(SRC:src/%.cu=$(BUILD)/%.o)
# Header-only modules are included widely; rebuild every object if any changes.
HDR := $(wildcard src/*.cuh src/*.h)

.PHONY: all test bench clean
all: $(BUILD)/test_roundtrip $(BUILD)/bench

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cu $(HDR) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/test_roundtrip: tests/test_roundtrip.cu $(OBJ) $(HDR) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< $(OBJ) -o $@

$(BUILD)/bench: bench/bench.cu $(OBJ) $(HDR) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< $(OBJ) -o $@

test: $(BUILD)/test_roundtrip
	./$(BUILD)/test_roundtrip

bench: $(BUILD)/bench
	./$(BUILD)/bench

clean:
	rm -rf $(BUILD)
