CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 -I.
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc -I.

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

# ds4_distributed.c 已机械拆分为 src/dist/*.c(行为零变化, 拆分见 src/dist/dist_internal.h)。
# 对象放源旁; 两个平台段的 CORE_OBJS/CPU_CORE_OBJS 共用这一份列表。
DIST_OBJS = src/dist/dist_util.o src/dist/dist_transport.o src/dist/dist_framing.o \
    src/dist/dist_tp.o src/dist/dist_wire.o src/dist/dist_reg.o \
    src/dist/dist_coord_route.o src/dist/dist_coord_dispatch.o src/dist/dist_coord_eval.o \
    src/dist/dist_coord_gen.o src/dist/dist_prefill_pipe.o src/dist/dist_prefill.o \
    src/dist/dist_coord_recover.o src/dist/dist_coord_ctl.o src/dist/dist_kv_snapshot.o \
    src/dist/dist_payload.o src/dist/dist_coord_kv.o src/dist/dist_coord_session.o \
    src/dist/dist_coord_main.o src/dist/dist_worker_loop.o src/dist/dist_worker_route.o \
    src/dist/dist_worker_fwd.o src/dist/dist_worker_kv.o src/dist/dist_worker_exec.o \
    src/dist/dist_worker_prefetch.o src/dist/dist_worker_main.o src/dist/dist_cli.o \
    src/dist/dist_rfetch.o
# EVAL 模块(重构阶段3: ds4_eval.c 拆分为 src/eval/*.c)
EVAL_SRCS := $(wildcard src/eval/*.c)
EVAL_OBJS := $(EVAL_SRCS:.c=.o)
EVAL_CPU_OBJS := $(EVAL_SRCS:.c=_cpu.o)
# KV 模块(重构阶段3: ds4_kvstore.c 拆分为 src/kv/*.c; 纯主机 C, 各构建共用)
KV_SRCS := $(wildcard src/kv/*.c)
KV_OBJS := $(KV_SRCS:.c=.o)
# BENCH 模块(重构阶段3: ds4_bench.c 拆分为 src/bench/*.c)
BENCH_SRCS := $(wildcard src/bench/*.c)
BENCH_OBJS := $(BENCH_SRCS:.c=.o)
BENCH_CPU_OBJS := $(BENCH_SRCS:.c=_cpu.o)
# CLI 模块(重构阶段3: ds4_cli.c 拆分为 src/cli/*.c)
CLI_SRCS := $(wildcard src/cli/*.c)
CLI_OBJS := $(CLI_SRCS:.c=.o)
CLI_CPU_OBJS := $(CLI_SRCS:.c=_cpu.o)
# ds4-server 拆分件 (src/server/): 生产 .o、测试 .o(-DDS4_SERVER_TEST, 保留
# #ifdef DS4_SERVER_TEST 行为分支且不含生产 main)、CPU .o(-DDS4_NO_GPU) 三套
# 由同一批源文件编出。tests/server_tests_*.c 是原内嵌测试块, 只进 ds4_test。
SERVER_SRCS := $(wildcard src/server/*.c)
SERVER_OBJS := $(SERVER_SRCS:.c=.o)
SERVER_TEST_OBJS := $(SERVER_SRCS:.c=_test.o)
SERVER_CPU_OBJS := $(SERVER_SRCS:.c=_cpu.o)
SERVER_HDRS := src/server/server_internal.h src/server/server_types.h src/server/server_types2.h
SERVER_TESTS_SRCS := $(wildcard tests/server_tests_*.c)
SERVER_TESTS_OBJS := $(SERVER_TESTS_SRCS:.c=.o)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal -framework Accelerate
MM_OBJS = ds4_multimodal.o ds4_spatial.o ds4_css.o
CORE_OBJS = ds4.o ds4_corr.o ds4_zchain.o $(MM_OBJS) $(DIST_OBJS) ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_corr_cpu.o ds4_zchain.o $(MM_OBJS) $(DIST_OBJS)
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_SPARK_FLAGS := -DDS4_CUDA_SPARK_HBM_CACHE=1
MM_OBJS = ds4_multimodal.o ds4_spatial.o ds4_css.o
CORE_OBJS = ds4.o ds4_corr.o ds4_zchain.o $(MM_OBJS) $(DIST_OBJS) ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_corr_cpu.o ds4_zchain.o $(MM_OBJS) $(DIST_OBJS)
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
METAL_LDLIBS := $(LDLIBS)
endif

# 终端 agent: 原单文件 ds4_agent.c 机械拆分为 src/agent/*.c(跨文件接口在
# agent_internal.h)。对象放源文件旁; 与根目录惯例一致, 普通 .o 走 GPU 构建,
# _cpu.o 加 -DDS4_NO_GPU。新增 agent 源文件放进 src/agent/ 即自动入列。
AGENT_SRCS = $(sort $(wildcard src/agent/*.c))
AGENT_OBJS = $(AGENT_SRCS:.c=.o)
AGENT_CPU_OBJS = $(AGENT_SRCS:.c=_cpu.o)
AGENT_HDRS = src/agent/agent_internal.h src/agent/agent_types.h ds4.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression e0

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent modules

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: $(CLI_OBJS) linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJS) linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: $(BENCH_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(BENCH_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: $(EVAL_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(EVAL_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: $(AGENT_OBJS) ds4_web.o $(KV_OBJS) linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(AGENT_OBJS) ds4_web.o $(KV_OBJS) linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

# P4 多模态 image-family 编码器 (macOS Vision; opt-in, 不进默认 all 以免硬依赖
# swiftc)。mm-ui = 前端开发域 UI 结构草图 (生产默认, 引擎自动绑定 ./mm-ui);
# mm-ocr = 纯逐字 OCR (诊断用)。
mm-ui: tools/mm_ui.swift
	swiftc -O tools/mm_ui.swift -o $@

mm-ocr: tools/mm_ocr.swift
	swiftc -O tools/mm_ocr.swift -o $@

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) ds4_web.o $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench $(BENCH_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval $(EVAL_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent $(AGENT_CPU_OBJS) ds4_web.o $(KV_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10 with Spark HBM weight cache"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH= CFLAGS="$(CFLAGS) $(CUDA_SPARK_FLAGS)" NVCCFLAGS="$(NVCCFLAGS) $(CUDA_SPARK_FLAGS) -default-stream per-thread"

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

ds4: $(CLI_OBJS) linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(BENCH_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: $(EVAL_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: $(AGENT_OBJS) ds4_web.o $(KV_OBJS) linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) ds4_web.o $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench $(BENCH_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval $(EVAL_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent $(AGENT_CPU_OBJS) ds4_web.o $(KV_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_internal.h ds4_distributed.h ds4_gpu.h ds4_multimodal.h ds4_spatial.h ds4_css.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c


# Standalone capability modules (self-contained, no engine coupling): z 隐变量
# closed-form latent, 四损失 calibration losses, 后训练 closed-form quant
# optimization. Built with `all` so they stay green; linked only by the tools
# that consume them. (多模态 ds4_multimodal.o graduated into CORE_OBJS: the
# engine binds the image-family encoder and the server consumes image blocks.)
MODULE_OBJS = ds4_z.o ds4_loss.o ds4_posttrain.o

modules: $(MODULE_OBJS)

ds4_z.o: ds4_z.c ds4_z.h
	$(CC) $(CFLAGS) -c -o $@ ds4_z.c

ds4_loss.o: ds4_loss.c ds4_loss.h
	$(CC) $(CFLAGS) -c -o $@ ds4_loss.c

ds4_posttrain.o: ds4_posttrain.c ds4_posttrain.h
	$(CC) $(CFLAGS) -c -o $@ ds4_posttrain.c

ds4_multimodal.o: ds4_multimodal.c ds4_multimodal.h
	$(CC) $(CFLAGS) -c -o $@ ds4_multimodal.c

# 前端域多模态插件: 物理方位 (UI 草图 -> 空间关系) 与 CSS 理解 (草图 -> CSS
# 布局事实), 都是编码器文本的确定性闭式推导, 经 enricher 链挂在 image 族。
ds4_spatial.o: ds4_spatial.c ds4_spatial.h
	$(CC) $(CFLAGS) -c -o $@ ds4_spatial.c

ds4_css.o: ds4_css.c ds4_css.h ds4_spatial.h
	$(CC) $(CFLAGS) -c -o $@ ds4_css.c

# P2 管线步骤4: 激活对 -> 闭式求解 -> corr 侧车 GGUF (tools/zsolve.c)
zsolve: tools/zsolve.c ds4_z.o
	$(CC) $(CFLAGS) -o $@ tools/zsolve.c ds4_z.o -lm

ds4_corr.o: ds4_corr.c ds4_internal.h ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_corr.c

ds4_corr_cpu.o: ds4_corr.c ds4_internal.h ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_corr.c

# go-onebit DQZ2 运行时侧车 (纯主机 C, 无 GPU 引用 — CPU/Metal/CUDA 构建共用同一 .o)
ds4_zchain.o: ds4_zchain.c ds4_zchain.h
	$(CC) $(CFLAGS) -c -o $@ ds4_zchain.c

src/cli/%.o: src/cli/%.c src/cli/cli_internal.h ds4.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/cli/%_cpu.o: src/cli/%.c src/cli/cli_internal.h ds4.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

# src/dist 全组 .c 只包含 dist_internal.h(其再包含 dist_proto.h/dist_state.h 与根公共头),
# 头依赖对整组一致, 用静态模式规则替代逐文件规则。
$(DIST_OBJS): %.o: %.c src/dist/dist_internal.h src/dist/dist_proto.h src/dist/dist_state.h ds4_distributed.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ $<

SERVER_DEP_HDRS := $(SERVER_HDRS) ds4.h ds4_distributed.h ds4_kvstore.h ds4_multimodal.h rax.h

src/server/%_test.o: src/server/%.c $(SERVER_DEP_HDRS)
	$(CC) $(CFLAGS) -DDS4_SERVER_TEST -c -o $@ $<

src/server/%_cpu.o: src/server/%.c $(SERVER_DEP_HDRS)
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

src/server/%.o: src/server/%.c $(SERVER_DEP_HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

# generate_job 函数体分片 (见 server_generate.c 头注释)
SERVER_GEN_INCS := $(wildcard src/server/server_generate_body*.inc)
src/server/server_generate.o src/server/server_generate_test.o src/server/server_generate_cpu.o: $(SERVER_GEN_INCS)

tests/server_tests_%.o: tests/server_tests_%.c tests/server_tests_internal.h $(SERVER_DEP_HDRS)
	$(CC) $(CFLAGS) -DDS4_SERVER_TEST -DDS4_SERVER_TEST_NO_MAIN -c -o $@ $<

src/bench/%.o: src/bench/%.c src/bench/bench_internal.h ds4.h ds4_distributed.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/bench/%_cpu.o: src/bench/%.c src/bench/bench_internal.h ds4.h ds4_distributed.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

src/eval/%.o: src/eval/%.c src/eval/eval_internal.h ds4.h ds4_distributed.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/eval/%_cpu.o: src/eval/%.c src/eval/eval_internal.h ds4.h ds4_distributed.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

src/agent/%_cpu.o: src/agent/%.c $(AGENT_HDRS)
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

src/agent/%.o: src/agent/%.c $(AGENT_HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

src/kv/%.o: src/kv/%.c src/kv/kv_internal.h ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ $<

ds4_test.o: tests/ds4_test.c tests/server_tests_internal.h $(SERVER_DEP_HDRS) ds4_spatial.h ds4_css.h
	$(CC) $(CFLAGS) -DDS4_SERVER_TEST -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_internal.h ds4_distributed.h ds4_gpu.h ds4_multimodal.h ds4_spatial.h ds4_css.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c





ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

# src/common 共享格式库单测: 无模型/无 GPU, 纯主机 C。夹具路径相对仓库根。
ds4_unit: tests/unit/test_common.c src/common/ds4_quantfmt.c src/common/ds4_gguf.c \
          src/common/ds4_quantfmt.h src/common/ds4_gguf.h src/common/ds4_float.h src/common/ds4_fp8.h
	$(CC) $(CFLAGS) -Isrc/common -o $@ tests/unit/test_common.c \
	    src/common/ds4_quantfmt.c src/common/ds4_gguf.c $(LDLIBS)

test: ds4_test ds4-eval ds4_unit
	./ds4_unit
	./ds4-eval --self-test-extractors
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_unit e0-pingpong mm-ui mm-ocr *.o src/cli/*.o src/bench/*.o src/kv/*.o src/eval/*.o src/agent/*.o src/dist/*.o src/server/*.o tests/server_tests_*.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o

# Task 04 / E0: standalone thunderbolt ping-pong latency gate (no core deps, no
# model). Defined after the default targets so it never becomes the default goal.
e0: e0-pingpong
e0-pingpong: tools/e0_pingpong.c
	$(CC) $(CFLAGS) -o $@ tools/e0_pingpong.c -lm
