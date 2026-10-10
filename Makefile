CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 -I.
# -fno-common: 拆分后原 static 全局改为 metal_state.m 强定义+头文件 extern; Apple clang 的
# ObjC 前端默认仍给未初始化全局 common 链接, 大数组会触发 ld 的 __common 对齐告警。
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc -fno-common -I.

LDLIBS ?= -lm -pthread
# 监控页的硬件读数(src/server/server_monitor_hw.c): macOS 走 IOKit(GPU 负载 / 磁盘字节), 纯 C 接口, GPU 与 CPU 构建都链;
# Linux 走 dlopen(libnvidia-ml), 新 glibc 的 dlopen 在 libc 里, -ldl 只是给老 glibc 留的。
ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework IOKit -framework CoreFoundation
else
LDLIBS += -ldl
endif
METAL_SRCS := $(wildcard metal/*.metal)
# ds4_metal.m 已机械拆分为 src/metal/*.m(行为零变化, 跨文件接口在 src/metal/metal_internal.h)。
# 新增 metal 后端源文件放进 src/metal/ 即自动入列。
METAL_OBJC_SRCS := $(wildcard src/metal/*.m)
METAL_OBJS := $(METAL_OBJC_SRCS:.m=.o)

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
    src/dist/dist_cli_check.o \
    src/dist/dist_rfetch.o
# EVAL 模块(重构阶段3: ds4_eval.c 拆分为 src/eval/*.c)
EVAL_SRCS := $(wildcard src/eval/*.c)
EVAL_OBJS := $(EVAL_SRCS:.c=.o)
EVAL_CPU_OBJS := $(EVAL_SRCS:.c=_cpu.o)
# WEB 模块(重构阶段3: ds4_web.c 拆分为 src/web/*.c; 纯主机 C)
WEB_SRCS := $(wildcard src/web/*.c)
WEB_OBJS := $(WEB_SRCS:.c=.o)
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
SERVER_HDRS := src/server/server_internal.h src/server/server_types.h src/server/server_types2.h src/server/server_monitor.h
SERVER_TESTS_SRCS := $(wildcard tests/server_tests_*.c)
SERVER_TESTS_OBJS := $(SERVER_TESTS_SRCS:.c=.o)
# ds4_test 拆分件(重构阶段8: 原单文件 tests/ds4_test.c 按 suite 拆为 tests/t_*.c,
# 跨文件接口在 tests/test_internal.h)。通配模式 t_*.c 与 server_tests_*.c 不重叠,
# 新增 suite 源文件放进 tests/ 叫 t_xxx.c 即自动入列。
TESTS_SRCS := $(wildcard tests/t_*.c)
TESTS_OBJS := $(patsubst %.c,%.o,$(TESTS_SRCS))

# 引擎核心(重构阶段4: ds4.c 机械拆分为 src/core/*.c, 共享内部头 src/core/core_internal.h)
CORE_ENGINE_SRCS := $(wildcard src/core/*.c)
CORE_ENGINE_OBJS := $(CORE_ENGINE_SRCS:.c=.o)
CORE_ENGINE_CPU_OBJS := $(CORE_ENGINE_SRCS:.c=_cpu.o)

# src/common 共享格式库(纯主机 C, 无 GPU 引用 — 全部构建共用一个 .o)
COMMON_FMT_OBJS := src/common/ds4_quantfmt.o src/common/ds4_json.o
# 训练页四件(src/train/, 规则在下面; train_models.o = 模型页): 要在 ds4-server 的依赖行之前定义 —— 依赖列表在解析时就展开, 定义在后面 Linux 那条 $^ 就是空的(10-10 实撞)
TRAIN_API_OBJS := src/train/train_api.o src/train/train_runs.o src/train/train_ctl.o src/train/train_models.o \
                  src/train/train_child.o src/train/train_model.o src/train/train_job.o src/train/train_gate.o src/train/train_gen.o

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal -framework Accelerate
MM_OBJS = ds4_multimodal.o ds4_spatial.o ds4_css.o
CORE_OBJS = $(CORE_ENGINE_OBJS) $(COMMON_FMT_OBJS) ds4_corr.o ds4_zchain.o ds4_zfinetune.o ds4_z.o ds4_loss.o $(MM_OBJS) $(DIST_OBJS) $(METAL_OBJS)
CPU_CORE_OBJS = $(CORE_ENGINE_CPU_OBJS) $(COMMON_FMT_OBJS) ds4_corr_cpu.o ds4_zchain.o ds4_zfinetune.o ds4_z.o ds4_loss.o $(MM_OBJS) $(DIST_OBJS)
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
# -I. 与 CFLAGS/OBJCFLAGS 同款: src/cuda/ 分片里的根目录头(vq_fmt.h/ds4_gpu.h/
# ds4_iq2_tables_cuda.inc)按引用文件目录解析不到, 需要仓库根兜底。
# -default-stream per-thread 是引擎的契约(图捕获用 cudaStreamPerThread, 服务端多线程各走自己的默认流), 2026-10-07 前只有
# cuda-spark 目标带它、cuda-generic/cuda 不带 —— 同一源码编出两种流语义。现在进基础旗, 所有 CUDA 目标一致。
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread -I. -default-stream per-thread
MM_OBJS = ds4_multimodal.o ds4_spatial.o ds4_css.o
CORE_OBJS = $(CORE_ENGINE_OBJS) $(COMMON_FMT_OBJS) ds4_corr.o ds4_zchain.o ds4_zfinetune.o ds4_z.o ds4_loss.o $(MM_OBJS) $(DIST_OBJS) ds4_cuda.o
CPU_CORE_OBJS = $(CORE_ENGINE_CPU_OBJS) $(COMMON_FMT_OBJS) ds4_corr_cpu.o ds4_zchain.o ds4_zfinetune.o ds4_z.o ds4_loss.o $(MM_OBJS) $(DIST_OBJS)
CUDA_LDLIBS ?= -lm -ldl -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt
METAL_LDLIBS := $(LDLIBS)
endif

# 终端 agent: 原单文件 ds4_agent.c 机械拆分为 src/agent/*.c(跨文件接口在
# agent_internal.h)。对象放源文件旁; 与根目录惯例一致, 普通 .o 走 GPU 构建,
# _cpu.o 加 -DDS4_NO_GPU。新增 agent 源文件放进 src/agent/ 即自动入列。
AGENT_SRCS = $(sort $(wildcard src/agent/*.c))
AGENT_OBJS = $(AGENT_SRCS:.c=.o)
AGENT_CPU_OBJS = $(AGENT_SRCS:.c=_cpu.o)
AGENT_HDRS = src/agent/agent_internal.h src/agent/agent_types.h ds4.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h

.PHONY: all help clean test linecount cpu cuda cuda-spark cuda-generic cuda-regression e0

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-train modules

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: $(CLI_OBJS) linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJS) linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: $(BENCH_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(BENCH_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: $(EVAL_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(EVAL_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: $(AGENT_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(AGENT_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

# P4 多模态 image-family 编码器 (macOS Vision; opt-in, 不进默认 all 以免硬依赖
# swiftc)。mm-ui = 前端开发域 UI 结构草图 (生产默认, 引擎自动绑定 ./mm-ui);
# mm-ocr = 纯逐字 OCR (诊断用)。
mm-ui: tools/mm_ui.swift
	swiftc -O tools/mm_ui.swift -o $@

mm-ocr: tools/mm_ocr.swift
	swiftc -O tools/mm_ocr.swift -o $@

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench $(BENCH_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval $(EVAL_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-generic        Build CUDA for the local GPU (-arch=native; unified-memory vs discrete decided at runtime)"
	@echo "  make cuda-spark          Alias of cuda-generic (kept for scripts; no Spark-specific flags since 2026-10-07)"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

# 2026-09-06: 原来 CUDA_ARCH= 空(nvcc 默认 compute_75 PTX, 运行时 JIT 到 GB10 的 sm_121)。后果: __CUDA_ARCH__ 在设备
# 编译期是 750, sm_80+ 才有的指令(mma.m16n8k16 / ldmatrix / cp.async)一律编不进去 —— indexer 打分核的 #if >= 800 分支
# 被整个吃掉, 发出去的是空核(剖面 9.8 µs/发), 静默失效。改 native = 直出 sm_121 SASS。
cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-train CUDA_ARCH=native

# 2026-10-07: cuda-spark 不再是独立配方。原来它比 cuda-generic 多带 -DDS4_CUDA_SPARK_HBM_CACHE(启动权重缓存 / 专家收编 /
# 预算公式)和 -default-stream per-thread, cuda-generic 编出来的二进制放到 GB10 上会退化成"只缓骨架、每次专家读跨 C2C"。
# 现在统一内存与否由引擎运行时按设备属性判(ds4_gpu_unified_memory_host), per-thread 默认流进了 NVCCFLAGS, 两个目标同义;
# 保留这个名字只是给 sync_spark.sh 等脚本用。
cuda-spark: cuda-generic

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

ds4: $(CLI_OBJS) linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(BENCH_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: $(EVAL_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: $(AGENT_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench $(BENCH_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval $(EVAL_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke tests/cuda_sample_selftest
	./tests/cuda_long_context_smoke
	./tests/cuda_sample_selftest
endif

src/core/%.o: src/core/%.c $(wildcard src/core/*.h) ds4.h ds4_internal.h ds4_distributed.h ds4_gpu.h ds4_multimodal.h ds4_spatial.h ds4_css.h ds4_zchain.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/core/%_cpu.o: src/core/%.c $(wildcard src/core/*.h) ds4.h ds4_internal.h ds4_distributed.h ds4_multimodal.h ds4_spatial.h ds4_css.h ds4_zchain.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<

src/common/ds4_quantfmt.o: src/common/ds4_quantfmt.c src/common/ds4_quantfmt.h src/common/ds4_float.h
	$(CC) $(CFLAGS) -c -o $@ src/common/ds4_quantfmt.c

src/common/ds4_json.o: src/common/ds4_json.c src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ src/common/ds4_json.c

# 训练页(src/train/): 路由/产物解析/进程控制三件(TRAIN_API_OBJS)ds4-server 与 ds4-train 共用; ds4-train 只多一层薄 HTTP。
# 纯主机 C, 不链引擎; Mac/Linux 同一条规则(nvcc 不参与)
src/train/%.o: src/train/%.c src/train/train_internal.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds4-train: src/train/train_main.o src/train/train_http.o src/train/train_proxy.o $(TRAIN_API_OBJS) src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ src/train/train_main.o src/train/train_http.o src/train/train_proxy.o $(TRAIN_API_OBJS) src/common/ds4_json.o $(LDLIBS)


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

src/web/%.o: src/web/%.c src/web/web_internal.h ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/kv/%.o: src/kv/%.c src/kv/kv_internal.h ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ $<

# t_units.c 额外吃 src/core/core_internal.h(采样/惩罚单测直插引擎内部), 头依赖
# 对整组统一挂上 — 多算依赖只多触发重编, 不会漏。-Wno-unused-function: 共享助手
# 去 static 后个别 TU 只用到子集, 不为此加 #if 网。
tests/t_%.o: tests/t_%.c tests/test_internal.h tests/server_tests_internal.h $(SERVER_DEP_HDRS) ds4_spatial.h ds4_css.h src/core/core_internal.h ds4_internal.h
	$(CC) $(CFLAGS) -DDS4_SERVER_TEST -Wno-unused-function -c -o $@ $<

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c
tests/cuda_sample_selftest.o: tests/cuda_sample_selftest.c ds4_gpu.h ds4_gpu_v41.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_sample_selftest.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c






# src/metal 全组 .m 只包含 metal_internal.h(其再包含 metal_args.h/metal_expert.h 与根公共头),
# 头依赖对整组一致。metal_source.o 额外依赖 .metal kernel 文件(运行时拼接清单),
# metal_moe_vq.o/metal_routed_moe_batch.o 额外包含 vq_fmt.h。
METAL_INTERNAL_HDRS = src/metal/metal_internal.h src/metal/metal_args.h src/metal/metal_expert.h ds4_gpu.h ds4.h \
                      src/metal/metal_v41.h src/metal/metal_v41_args.h ds4_gpu_v41.h ds4_gpu_bwd.h ds4_gpu_core.h
src/metal/%.o: src/metal/%.m $(METAL_INTERNAL_HDRS)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

src/metal/metal_source.o: src/metal/metal_source.m $(METAL_INTERNAL_HDRS) $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

src/metal/metal_moe_vq.o: src/metal/metal_moe_vq.m $(METAL_INTERNAL_HDRS) vq_fmt.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

src/metal/metal_routed_moe_batch.o: src/metal/metal_routed_moe_batch.m $(METAL_INTERNAL_HDRS) vq_fmt.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

# ds4_cuda.cu 是聚合根: 实现在 src/cuda/*.inc.cu 分片(单 TU 纹理包含),
# 分片或共享前奏一变就得重编这个 .o。
CUDA_INC_SRCS := $(wildcard src/cuda/*.inc.cu) src/cuda/cuda_internal.cuh
ds4_cuda.o: ds4_cuda.cu $(CUDA_INC_SRCS) ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# 两个 CUDA 测试都链整套引擎对象: ds4_cuda.o 引用 core 的全局(g_ds4_v41_prof / g_ds4_v41_vq_group 等), 只链 ds4_cuda.o 会 undefined reference
tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
# 设备采样核的分布门(2026-09-28): 不要模型, 只要 CUDA 设备; 采样路没有逐字节金标, 门 = 频率等于目标分布 + 投机边缘分布不变 + 确定性
tests/cuda_sample_selftest: tests/cuda_sample_selftest.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(TRAIN_API_OBJS) $(CORE_OBJS) $(CUDA_LDLIBS)
endif

# src/common 共享格式库单测: 无模型/无 GPU, 纯主机 C。夹具路径相对仓库根。
ds4_unit: tests/unit/test_common.c tests/unit/test_zmod.c tests/unit/t_zfinetune.c src/common/ds4_quantfmt.c src/common/ds4_gguf.c \
          src/common/ds4_quantfmt.h src/common/ds4_gguf.h src/common/ds4_float.h src/common/ds4_fp8.h \
          ds4_z.c ds4_z.h ds4_zchain.c ds4_zchain.h ds4_zfinetune.c ds4_zfinetune.h ds4_loss.c ds4_loss.h
	$(CC) $(CFLAGS) -Isrc/common -I. -o $@ tests/unit/test_common.c tests/unit/test_zmod.c tests/unit/t_zfinetune.c \
	    src/common/ds4_quantfmt.c src/common/ds4_gguf.c ds4_z.c ds4_zchain.c ds4_zfinetune.c ds4_loss.c $(LDLIBS)

test: ds4_test ds4-eval ds4_unit linecount
	./ds4_unit
	./ds4-eval --self-test-extractors
	./ds4_test


# 500 行守卫(重构阶段9): 源文件单文件 ≤500 行, 豁免清单见 .linecount-exempt
# (vendored/单函数 EXCEPTION/冻结转录)。范围含 gguf-tools(批6 已落地)。
linecount:
	@ex=$$(grep -v '^#' .linecount-exempt | grep -v '^$$'); \
	viol=$$(find src tests metal gguf-tools ds4*.c ds4*.h vq_fmt.h rax.c rax.h rax_malloc.h linenoise.c linenoise.h \
	        \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.cu' -o -name '*.cuh' -o -name '*.metal' -o -name '*.inc' \) \
	        2>/dev/null | sort -u | grep -v -x -F "$$ex" \
	        | xargs wc -l 2>/dev/null | awk '$$2 != "total" && $$1 > 500 {print $$1, $$2}'); \
	if [ -n "$$viol" ]; then echo "linecount: 以下文件超 500 行(豁免走 .linecount-exempt, 要带理由):"; echo "$$viol"; exit 1; \
	else echo "linecount: ok (≤500 行, 豁免清单外零超标)"; fi
clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_unit e0-pingpong mm-ui mm-ocr *.o src/cli/*.o src/bench/*.o src/kv/*.o src/web/*.o src/eval/*.o src/agent/*.o src/dist/*.o src/server/*.o src/core/*.o src/common/*.o src/metal/*.o tests/server_tests_*.o tests/t_*.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o

# Task 04 / E0: standalone thunderbolt ping-pong latency gate (no core deps, no
# model). Defined after the default targets so it never becomes the default goal.
e0: e0-pingpong
e0-pingpong: tools/e0_pingpong.c
	$(CC) $(CFLAGS) -o $@ tools/e0_pingpong.c -lm
