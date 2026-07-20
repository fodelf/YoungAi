CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_TARGETS := ds4.o ds4_metal.o
QUALITY_OBJS := ../ds4.o ../ds4_metal.o
QUALITY_LINK := $(CC) $(CFLAGS) -I.. -o
else
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_TARGETS := ds4.o ds4_cuda.o
QUALITY_OBJS := ../ds4.o ../ds4_cuda.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I.. -o
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score go-onebit-test

all: deepseek4-quantize

GO1B_DIR := go-onebit
# module layout (DESIGN.md): cross-module includes resolve via -I, sources keep
# plain #include "header.h" — quant/latent/calib stay decoupled at file level.
CFLAGS += -I$(GO1B_DIR)/quant -I$(GO1B_DIR)/latent -I$(GO1B_DIR)/latent/legacy -I$(GO1B_DIR)/calib

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h $(GO1B_DIR)/quant/onebit_quant.c $(GO1B_DIR)/quant/onebit_quant.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c $(GO1B_DIR)/quant/onebit_quant.c -lm -pthread

# Go-domain 1-bit hypothesis-test driver (reads HF + cap_m1; streams experts).
calib_run: $(GO1B_DIR)/calib/calib_run.c $(GO1B_DIR)/calib/npy.c $(GO1B_DIR)/calib/hf_read.c $(GO1B_DIR)/quant/onebit_quant.c $(GO1B_DIR)/calib/layer_probe.c $(GO1B_DIR)/latent/hiddenvar_solve.c $(GO1B_DIR)/latent/legacy/solve_perexpert.c $(GO1B_DIR)/latent/legacy/solve_denoise.c $(GO1B_DIR)/latent/legacy/solve_genrf.c $(GO1B_DIR)/latent/solve_rrr.c $(GO1B_DIR)/latent/linalg_small.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ -lm -pthread

# Diagnostics variant: o_ref/ô rank, oracle rank-d Δo bound, x→Δo R², --target oref direct-gen.
calib_diag: $(GO1B_DIR)/calib/calib_diag.c $(GO1B_DIR)/calib/npy.c $(GO1B_DIR)/calib/hf_read.c $(GO1B_DIR)/quant/onebit_quant.c $(GO1B_DIR)/calib/layer_probe.c $(GO1B_DIR)/latent/hiddenvar_solve.c $(GO1B_DIR)/latent/linalg_small.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ -lm -pthread

# Append per-layer hidden-variable z^L correction tensors as F32 GGUF tensors.
# Standalone (no model/quants deps); writes the corr sidecar or a merged GGUF.
emit_z: $(GO1B_DIR)/latent/emit_z.c
	$(CC) -O3 -Wall -std=c99 -o $@ $<

# Emit per-layer 1-bit RESIDUAL Q1(W-Q1(W)) as go1b tensors (sidecar). Reads HF.
emit_residual: $(GO1B_DIR)/latent/legacy/emit_residual.c $(GO1B_DIR)/calib/hf_read.c $(GO1B_DIR)/quant/onebit_quant.c $(GO1B_DIR)/calib/npy.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ -lm -pthread

# Standalone self-tests for the go-onebit modules (pure C, no model load).
go-onebit-test:
	@mkdir -p $(GO1B_DIR)/.build
	$(CC) $(CFLAGS) -DGO1B_TEST    $(GO1B_DIR)/quant/onebit_quant.c -o $(GO1B_DIR)/.build/onebit_test -lm && $(GO1B_DIR)/.build/onebit_test
	$(CC) $(CFLAGS) -DLINALG_TEST  $(GO1B_DIR)/latent/linalg_small.c -o $(GO1B_DIR)/.build/linalg_test -lm && $(GO1B_DIR)/.build/linalg_test
	$(CC) $(CFLAGS) -DPROBE_TEST   $(GO1B_DIR)/calib/layer_probe.c  -o $(GO1B_DIR)/.build/probe_test  -lpthread -lm && $(GO1B_DIR)/.build/probe_test
	$(CC) $(CFLAGS) -DHVSOLVE_TEST $(GO1B_DIR)/latent/hiddenvar_solve.c $(GO1B_DIR)/latent/linalg_small.c -o $(GO1B_DIR)/.build/hv_test -lm && $(GO1B_DIR)/.build/hv_test
	$(CC) $(CFLAGS) -DNPY_TEST     $(GO1B_DIR)/calib/npy.c -o $(GO1B_DIR)/.build/npy_test -lm && $(GO1B_DIR)/.build/npy_test
	$(CC) $(CFLAGS) -DHFREAD_TEST  $(GO1B_DIR)/calib/hf_read.c -o $(GO1B_DIR)/.build/hfread_test -lm && $(GO1B_DIR)/.build/hfread_test
	$(CC) $(CFLAGS) -DPEREXPERT_TEST  $(GO1B_DIR)/latent/legacy/solve_perexpert.c $(GO1B_DIR)/latent/linalg_small.c -o $(GO1B_DIR)/.build/perexpert_test -lm && $(GO1B_DIR)/.build/perexpert_test
	$(CC) $(CFLAGS) -DCALIBDIAG_TEST  $(GO1B_DIR)/calib/calib_diag.c $(GO1B_DIR)/latent/hiddenvar_solve.c $(GO1B_DIR)/latent/linalg_small.c -o $(GO1B_DIR)/.build/calibdiag_test -lm -pthread && $(GO1B_DIR)/.build/calibdiag_test
	$(CC) $(CFLAGS) -DRRR_TEST     $(GO1B_DIR)/latent/solve_rrr.c $(GO1B_DIR)/latent/hiddenvar_solve.c $(GO1B_DIR)/latent/linalg_small.c -o $(GO1B_DIR)/.build/rrr_test -lm -pthread && $(GO1B_DIR)/.build/rrr_test

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize calib_run calib_diag emit_z quality-testing/score_official
	rm -rf $(GO1B_DIR)/.build
