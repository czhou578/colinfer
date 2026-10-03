# Microbenchmarks for DGX Spark (GB10).
#   make -C bench            builds bw_bench and every CUTLASS GEMM config (use -j)
#   make -C bench gemm -k    just the GEMM configs, keep going if one fails to compile
NVCC    ?= nvcc
ARCH    ?= sm_121a
CUTLASS ?= ../csrc/third_party/cutlass
FLAGS   ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
GEMM_FLAGS := $(FLAGS) --expt-relaxed-constexpr -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include
OUT     := build

# name : KIND : TILE_M : TILE_N : TILE_K : SCHED
# Valid SM120 shapes per media/docs/cpp/blackwell_functionality.md tables 16/17; the ones marked
# (x) are outside the documented set and are built -k to see whether the builder accepts them.
GEMM_CFGS := \
  nvfp4_128x128x128_pp:KIND_NVFP4:128:128:128:SCHED_PINGPONG \
  nvfp4_128x128x128_coop:KIND_NVFP4:128:128:128:SCHED_COOP \
  nvfp4_256x128x128_coop:KIND_NVFP4:256:128:128:SCHED_COOP \
  nvfp4_128x128x256_pp:KIND_NVFP4:128:128:256:SCHED_PINGPONG \
  nvfp4_128x128x256_coop:KIND_NVFP4:128:128:256:SCHED_COOP \
  nvfp4_64x128x128_pp:KIND_NVFP4:64:128:128:SCHED_PINGPONG \
  nvfp4_128x64x128_pp:KIND_NVFP4:128:64:128:SCHED_PINGPONG \
  fp8_128x128x128_coop:KIND_FP8:128:128:128:SCHED_COOP \
  fp8_128x128x128_pp:KIND_FP8:128:128:128:SCHED_PINGPONG \
  fp8_128x128x64_coop:KIND_FP8:128:128:64:SCHED_COOP \
  fp8_128x64x64_coop:KIND_FP8:128:64:64:SCHED_COOP \
  fp8_64x128x128_pp:KIND_FP8:64:128:128:SCHED_PINGPONG

GEMM_BINS := $(foreach c,$(GEMM_CFGS),$(OUT)/gemm_sm120_$(word 1,$(subst :, ,$c)))

all: $(OUT)/bw_bench gemm
gemm: $(GEMM_BINS)

$(OUT)/bw_bench: bw_bench.cu | $(OUT)
	$(NVCC) $(FLAGS) -o $@ $<

define GEMM_RULE
$(OUT)/gemm_sm120_$(word 1,$(subst :, ,$1)): gemm_sm120.cu | $(OUT)
	$(NVCC) $(GEMM_FLAGS) -D$(word 2,$(subst :, ,$1)) -DTILE_M=$(word 3,$(subst :, ,$1)) -DTILE_N=$(word 4,$(subst :, ,$1)) -DTILE_K=$(word 5,$(subst :, ,$1)) -D$(word 6,$(subst :, ,$1)) -o $$@ $$<
endef
$(foreach c,$(GEMM_CFGS),$(eval $(call GEMM_RULE,$c)))

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)

.PHONY: all gemm clean
