# Build all five versions into build/.
#   make        -> everything (CUDA only if nvcc is found)
#   make cpu    -> serial, openmp, mpi, hybrid
#   make cuda   -> CUDA only
#   make clean

CC      ?= gcc
MPICC   ?= mpicc
# Prefer the Linux toolkit: WSL's PATH also contains the Windows CUDA install.
NVCC    ?= $(shell ls /usr/local/cuda/bin/nvcc 2>/dev/null || command -v nvcc 2>/dev/null)
# Same optimisation level for every version so the comparison is fair.
CFLAGS  := -O3 -march=native -std=gnu11 -Wall -Wextra -Wno-unknown-pragmas
NVFLAGS := -O3 -arch=native -Xcompiler -Wall

BUILD   := build
HEADERS := src/fire_core.h src/fire_common.h
CPU_BIN := $(BUILD)/fire_serial $(BUILD)/fire_omp $(BUILD)/fire_mpi $(BUILD)/fire_hybrid

ifneq ($(NVCC),)
all: cpu cuda
else
all: cpu
	@echo "nvcc not found - skipped the CUDA build"
endif

cpu: $(CPU_BIN)
cuda: $(BUILD)/fire_cuda

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/fire_serial: src/fire_serial.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ -lm

$(BUILD)/fire_omp: src/fire_omp.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) -fopenmp $< -o $@ -lm

# Same source, two builds: without -fopenmp the omp pragma is ignored.
$(BUILD)/fire_mpi: src/fire_mpi.c $(HEADERS) | $(BUILD)
	$(MPICC) $(CFLAGS) $< -o $@ -lm

$(BUILD)/fire_hybrid: src/fire_mpi.c $(HEADERS) | $(BUILD)
	$(MPICC) $(CFLAGS) -fopenmp $< -o $@ -lm

$(BUILD)/fire_cuda: src/fire_cuda.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all cpu cuda clean
