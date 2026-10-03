# Microbenchmarks for DGX Spark (GB10). Build: make -C bench
NVCC  ?= nvcc
ARCH  ?= sm_121a
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
OUT   := build

all: $(OUT)/bw_bench

$(OUT)/bw_bench: bw_bench.cu | $(OUT)
	$(NVCC) $(FLAGS) -o $@ $<

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)

.PHONY: all clean
