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
# 个人助理模块(src/assistant/, 见下方 ds4-train 段): 变量要在用到它的规则之前定义, make 解析规则时就展开前置
ASST_SRCS := $(sort $(wildcard src/assistant/*.c) $(wildcard src/assistant/tools/*.c))
ASST_OBJS := $(ASST_SRCS:.c=.o)
# 沙盒(src/sandbox/, 总纲 docs/modules.md): 守门 + 内核锁 + 沙盒工具箱 + 云手机生命周期/adb; 云手机本体 ds4-phone 在 src/sandbox/phone/
SANDBOX_SRCS := $(sort $(wildcard src/sandbox/*.c))
SANDBOX_OBJS := $(SANDBOX_SRCS:.c=.o)
# 技能(src/skill/): 装/删/校验 SKILL.md + use_skill 渐进披露; 随包自带的技能在顶层 skills/
SKILL_OBJS := src/skill/skill.o src/skill/skill_seed.o
# 推送密码学(src/push/: P-256 / AES-GCM / HKDF / RFC 8291 加密 / VAPID JWT); 纯主机 C, 零依赖; 金标 tests/push/t_push.c
PUSH_SRCS := $(sort $(wildcard src/push/*.c))
PUSH_OBJS := $(PUSH_SRCS:.c=.o)
# 数据库连接器(src/db/: BSON / MongoDB 线协议 / SCRAM-SHA-256 / 只读查询, 方案稿 docs/assistant_connectors.md); 纯主机 C, 零依赖; 金标 tests/asst/t_asst_db.c
DB_SRCS := $(sort $(wildcard src/db/*.c))
DB_OBJS := $(DB_SRCS:.c=.o)
# 官方产出工具的格式基元(src/office/: zip/deflate/Markdown/docx/xlsx, 方案稿 docs/assistant_office.md); 纯主机 C, 零依赖
OFFICE_SRCS := $(sort $(wildcard src/office/*.c))
OFFICE_OBJS := $(OFFICE_SRCS:.c=.o)
# 视频格式基元(src/video/: H.264 Baseline 编码 + MP4 封装, 方案稿 docs/assistant_video.md); 纯主机 C, 零依赖(不用 ffmpeg)
VIDEO_SRCS := $(sort $(wildcard src/video/*.c))
VIDEO_OBJS := $(VIDEO_SRCS:.c=.o)
# 旁白小模型(src/tts/: Piper VITS 的 C 推理, 权重是自家 GGUF; 方案稿 docs/assistant_video.md §3.3); 纯主机 C, 只在沙盒子进程里按需加载
TTS_SRCS := $(sort $(wildcard src/tts/*.c))
TTS_OBJS := $(TTS_SRCS:.c=.o)
TTS_LINK := $(TTS_OBJS) src/common/ds4_gguf.o src/common/ds4_quantfmt.o
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
COMMON_FMT_OBJS := src/common/ds4_quantfmt.o src/common/ds4_json.o src/common/ds4_pretok.o src/common/ds4_sha256.o
# 模型模块(src/model/, 总纲 docs/modules.md): 模型子进程起停 + 模型清单/侧车 + 后训练作业与选轮 + /api/train|models 路由; ds4-server 与 ds4-train 共用。
# 要在 ds4-server 的依赖行之前定义 —— 依赖列表在解析时就展开, 定义在后面 Linux 那条 $^ 就是空的(10-10 实撞)
MODEL_OBJS := src/model/model_api.o src/model/model_train_runs.o src/model/model_proc.o src/model/model_catalog.o \
              src/model/model_child.o src/model/model_serve.o src/model/model_train_job.o src/model/model_train_pick.o
# 跟模型说话的 HTTP 客户端: 还靠助理的小 JSON 取值(asst_json.c), 只有 ds4-train 链; 小 JSON 并进 src/common 后并入 MODEL_OBJS
MODEL_CLIENT_OBJS := src/model/model_llm.o
# 语音识别(src/model/asr/, 方案稿 docs/assistant_asr.md): 住 ds4-server(要 GPU 原语 ds4_gpu_asr.h); ds4_test 的 --asr 门也链。CPU 变体只会报"要 GPU 构建"。
ASR_SRCS := $(wildcard src/model/asr/*.c)
ASR_OBJS := $(ASR_SRCS:.c=.o)
ASR_CPU_OBJS := $(ASR_SRCS:.c=_cpu.o)

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
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-train ds4-seg modules

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: $(CLI_OBJS) linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJS) linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)

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

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(ASR_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
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
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-train ds4-seg CUDA_ARCH=native

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

ds4-server: $(SERVER_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(BENCH_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: $(EVAL_OBJS) $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: $(AGENT_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: $(CLI_CPU_OBJS) $(SERVER_CPU_OBJS) $(ASR_CPU_OBJS) $(BENCH_CPU_OBJS) $(EVAL_CPU_OBJS) $(AGENT_CPU_OBJS) $(WEB_OBJS) $(KV_OBJS) linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 $(CLI_CPU_OBJS) linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server $(SERVER_CPU_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_CPU_OBJS) $(CPU_CORE_OBJS) $(LDLIBS)
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
src/common/ds4_pretok.o: src/common/ds4_pretok.c src/common/ds4_pretok.h src/common/ds4_unicode_tab.h
	$(CC) $(CFLAGS) -c -o $@ src/common/ds4_pretok.c
src/common/ds4_sha256.o: src/common/ds4_sha256.c src/common/ds4_sha256.h
	$(CC) $(CFLAGS) -c -o $@ src/common/ds4_sha256.c

# 工作台外壳(src/studio/): 只有 ds4-train 链; 路由/产物解析/进程控制在模型模块(MODEL_OBJS), ds4-server 也链。
# 纯主机 C, 不链引擎; Mac/Linux 同一条规则(nvcc 不参与)
src/studio/%.o: src/studio/%.c src/studio/studio_internal.h src/model/model_internal.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
src/model/%.o: src/model/%.c src/model/model_internal.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
src/model/asr/%.o: src/model/asr/%.c src/model/asr/ds4_asr.h src/model/asr/asr_internal.h ds4_gpu_asr.h src/common/ds4_st41.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
src/model/asr/%_cpu.o: src/model/asr/%.c src/model/asr/ds4_asr.h src/model/asr/asr_internal.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ $<
src/model/model_llm.o: src/assistant/asst_internal.h
src/model/model_catalog.o: src/common/ds4_gr_fnv.h
# 登录(studio_auth.o, 只有主进程有): ds4-server 不链它 —— 模型子进程只绑 127.0.0.1, 外面进来的请求都先过主进程的闸
STUDIO_AUTH_OBJS := src/studio/studio_auth.o src/studio/studio_keys.o src/common/ds4_sha256.o
src/studio/studio_auth.o src/studio/studio_keys.o: src/common/ds4_sha256.h
# 设置弹窗(云服务器 / frp 穿透 / HTTPS 证书, 只有主进程): 外部命令 ssh curl tar unzip openssl 都是 fork+exec, 不链任何库
STUDIO_NET_OBJS := src/studio/studio_net.o src/studio/studio_phone.o src/studio/studio_exec.o src/studio/studio_frp.o src/studio/studio_frps.o src/studio/studio_sshtun.o src/studio/studio_tunnel.o src/studio/studio_cert.o
src/studio/studio_frp.o: src/common/ds4_sha256.h
# 云手机运行时(src/sandbox/phone/, 方案稿 docs/assistant_phone.md): 独立小二进制, root 跑(sudoers 一行放行), 只在 Linux 有 —— 用的是 Linux 命名空间/cgroup/binder
ifneq ($(UNAME_S),Darwin)
PHONE_OBJS := src/sandbox/phone/phone_main.o src/sandbox/phone/phone_cg.o src/sandbox/phone/phone_net.o src/sandbox/phone/phone_mnt.o src/sandbox/phone/phone_rc.o
src/sandbox/phone/%.o: src/sandbox/phone/%.c src/sandbox/phone/phone_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds4-phone: $(PHONE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(PHONE_OBJS)
endif
# 云手机门 0 探针(tests/phone/adbprobe.c): 只链 adb 客户端与 ds4_buf, 不链引擎
tests/phone/adbprobe: tests/phone/adbprobe.c src/sandbox/sbx_adb.o src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ tests/phone/adbprobe.c src/sandbox/sbx_adb.o src/common/ds4_json.o
# 图像模块(改图工具): 要在 ds4-train 规则之前定义 —— 规则的依赖读到时就展开, 定义在后面依赖就是空的, 新加的 .c 不会被编(10-15 实撞 img_grid.o)
IMAGE_SRCS := $(sort $(wildcard src/image/*.c))
IMAGE_OBJS := $(IMAGE_SRCS:.c=.o)
ds4-train: src/studio/studio_main.o src/studio/studio_http.o src/studio/studio_proxy.o $(MODEL_OBJS) $(STUDIO_AUTH_OBJS) $(STUDIO_NET_OBJS) $(ASST_OBJS) $(SANDBOX_OBJS) $(SKILL_OBJS) $(MODEL_CLIENT_OBJS) $(PUSH_OBJS) $(DB_OBJS) $(OFFICE_OBJS) $(IMAGE_OBJS) $(VIDEO_OBJS) $(TTS_LINK) $(WEB_OBJS) src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ src/studio/studio_main.o src/studio/studio_http.o src/studio/studio_proxy.o $(MODEL_OBJS) $(STUDIO_AUTH_OBJS) $(STUDIO_NET_OBJS) $(ASST_OBJS) $(SANDBOX_OBJS) $(SKILL_OBJS) $(MODEL_CLIENT_OBJS) $(PUSH_OBJS) $(DB_OBJS) $(OFFICE_OBJS) $(IMAGE_OBJS) $(VIDEO_OBJS) $(TTS_LINK) $(WEB_OBJS) src/common/ds4_json.o $(LDLIBS)
src/push/%.o: src/push/%.c src/push/ds4_push.h src/common/ds4_sha256.h
	$(CC) $(CFLAGS) -c $< -o $@
src/db/%.o: src/db/%.c src/db/db.h src/db/bson.h src/db/mongo.h src/common/ds4_json.h src/common/ds4_sha256.h
	$(CC) $(CFLAGS) -c $< -o $@
ds4_push_test: tests/push/t_push.c $(PUSH_OBJS) src/common/ds4_sha256.o
	$(CC) $(CFLAGS) -o $@ tests/push/t_push.c $(PUSH_OBJS) src/common/ds4_sha256.o $(LDLIBS)
src/office/%.o: src/office/%.c src/office/ds4_office.h src/office/office_internal.h src/office/office_pdf.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
# 改图基元(src/image/, 方案稿 docs/assistant_image.md): 解码(stb_image 唯一实现点)/PNG 编码/几何调色/颜色掩膜/导向滤波; 纯主机 C
src/image/%.o: src/image/%.c src/image/ds4_image.h src/office/ds4_office.h stb_image.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds4_image_test: tests/image/t_image.c $(IMAGE_OBJS) src/office/office_flate.o src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# 点提示分割(src/seg/, 方案稿 docs/assistant_image.md §3.6): 独立小程序 ds4-seg, 直接读官方 .pt(zip+pickle), 算子走 GPU
# (Mac = seg_metal.m + metal/seg.metal + MPS GEMM; Linux = seg_cuda.cu + cuBLAS)。改图工具在沙盒里 exec 它, 进程退出即释放权重与 GPU。
SEG_C_SRCS := $(sort $(wildcard src/seg/*.c))
SEG_C_OBJS := $(SEG_C_SRCS:.c=.o)
SEG_HOST_OBJS := $(filter-out src/seg/seg_main.o,$(SEG_C_OBJS)) $(IMAGE_OBJS) src/office/office_flate.o src/common/ds4_json.o
src/seg/%.o: src/seg/%.c src/seg/ds4_seg.h src/seg/seg_ops.h src/seg/seg_internal.h src/office/ds4_office.h src/image/ds4_image.h
	$(CC) $(CFLAGS) -c -o $@ $<
ifeq ($(UNAME_S),Darwin)
SEG_GPU_OBJS := src/seg/seg_metal.o
SEG_LDLIBS := $(METAL_LDLIBS) -framework MetalPerformanceShaders
src/seg/seg_metal.o: src/seg/seg_metal.m src/seg/seg_ops.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<
ds4-seg: src/seg/seg_main.o $(SEG_HOST_OBJS) $(SEG_GPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(SEG_LDLIBS)
ds4_seg_test: tests/seg/t_seg.c tests/seg/t_seg_ops.c $(SEG_HOST_OBJS) $(SEG_GPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(SEG_LDLIBS)
else
SEG_GPU_OBJS := src/seg/seg_cuda.o
src/seg/seg_cuda.o: src/seg/seg_cuda.cu src/seg/seg_ops.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
ds4-seg: src/seg/seg_main.o $(SEG_HOST_OBJS) $(SEG_GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
ds4_seg_test: tests/seg/t_seg.c tests/seg/t_seg_ops.c $(SEG_HOST_OBJS) $(SEG_GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif
# 格式基元单测: 金标逐字节 + flate 往返 + 公式 + 解析器; 有 unzip/xmllint/textutil 的机器再用它们复验(没有就 SKIP)
ds4_office_test: tests/office/t_office.c $(OFFICE_OBJS) src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ tests/office/t_office.c $(OFFICE_OBJS) src/common/ds4_json.o $(LDLIBS)
src/video/%.o: src/video/%.c src/video/ds4_video.h src/video/video_internal.h src/video/video_aac.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
# 视频单测: 规范向量逐位 + 码表无前缀冲突 + 合成 1080p 编两遍逐字节同; macOS 上再让系统解码器(vidcheck)解回来与重建逐字节比
ds4_video_test: tests/video/t_video.c $(VIDEO_OBJS) src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ tests/video/t_video.c $(VIDEO_OBJS) src/common/ds4_json.o $(LDLIBS)
src/tts/%.o: src/tts/%.c src/tts/ds4_tts.h src/tts/tts_internal.h src/common/ds4_gguf.h
	$(CC) $(CFLAGS) -c -o $@ $<
# 旁白小模型判决: 同一串音素 id 与官方 piper 噪声全 0 的波形逐样本对拍(夹具 tests/tts/fixtures; 模型 gguf/tts/zh-huayan.gguf 缺就 SKIP)
ds4_tts_test: tests/tts/t_tts.c $(TTS_LINK)
	$(CC) $(CFLAGS) -o $@ tests/tts/t_tts.c $(TTS_LINK) $(LDLIBS)
tests/video/vidcheck: tests/video/vidcheck.m
	$(CC) -O2 -fobjc-arc -Wall -o $@ $< -framework Foundation -framework AVFoundation -framework CoreMedia -framework CoreVideo
ifeq ($(UNAME_S),Darwin)
ds4_video_test: tests/video/vidcheck
endif

# 个人助理(src/assistant/, 方案 docs/assistant.md): 住 ds4-train 主进程; 每次工具调用 fork 子进程先上内核沙盒
# (Linux = Landlock+seccomp, macOS = Seatbelt; 两个平台文件在对方平台上编成空对象)。纯主机 C, 不链引擎。
src/studio/studio_main.o src/studio/studio_phone.o: src/assistant/asst_internal.h
src/assistant/%.o: src/assistant/%.c src/assistant/asst_internal.h src/assistant/asst_verify.h src/common/ds4_json.h src/office/ds4_office.h src/video/ds4_video.h
	$(CC) $(CFLAGS) -c -o $@ $<
# 工具执行点(注册表 asst_tools.c 每行一个): 子目录要自己的规则, 落到默认规则上能编过但不跟头文件, 改了头不重编
src/assistant/tools/%.o: src/assistant/tools/%.c src/assistant/tools/tool_image.h src/image/ds4_image.h src/assistant/asst_internal.h src/assistant/asst_verify.h src/common/ds4_json.h src/office/ds4_office.h src/video/ds4_video.h src/tts/ds4_tts.h src/web/web_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<
src/sandbox/%.o: src/sandbox/%.c src/assistant/asst_internal.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
src/skill/%.o: src/skill/%.c src/assistant/asst_internal.h src/common/ds4_json.h
	$(CC) $(CFLAGS) -c -o $@ $<
# 助理页面端到端门(方案稿 docs/assistant.md §13.5): 假模型(叫 ds4-server 才能被主进程认领)+ 系统 Chrome 的 Playwright 脚本; 不进 make test(要浏览器)
tests/asst/fake/ds4-server: tests/asst/fake_model.c
	mkdir -p tests/asst/fake && $(CC) $(CFLAGS) -o $@ $<
asst-e2e: ds4-train tests/asst/fake/ds4-server
	node tests/asst/e2e_device.mjs
# 沙盒逃逸单测 + 纯逻辑单测 + 工作台登录单测(不要模型, 秒级; make test 里跑)
ds4_asst_test: tests/asst/t_asst_sbx.c tests/asst/t_asst_logic.c tests/asst/t_asst_video.c tests/asst/t_studio_auth.c tests/asst/t_studio_net.c tests/asst/t_studio_models.c tests/asst/t_asst_phone.c tests/asst/t_asst_pay.c tests/asst/t_asst_task.c tests/asst/t_asst_push.c tests/asst/t_asst_plan.c tests/asst/t_asst_memory.c tests/asst/t_asst_image.c tests/asst/t_asst_preview.c tests/asst/t_asst_market.c tests/asst/t_asst_lottery.c tests/asst/t_asst_db.c tests/asst/t_asst_verify.c $(ASST_OBJS) $(SANDBOX_OBJS) $(SKILL_OBJS) $(MODEL_CLIENT_OBJS) $(PUSH_OBJS) $(DB_OBJS) $(OFFICE_OBJS) $(IMAGE_OBJS) $(VIDEO_OBJS) $(TTS_LINK) $(WEB_OBJS) $(MODEL_OBJS) $(STUDIO_AUTH_OBJS) $(STUDIO_NET_OBJS) src/common/ds4_json.o
	$(CC) $(CFLAGS) -o $@ tests/asst/t_asst_sbx.c tests/asst/t_asst_logic.c tests/asst/t_asst_video.c tests/asst/t_studio_auth.c tests/asst/t_studio_net.c tests/asst/t_studio_models.c tests/asst/t_asst_phone.c tests/asst/t_asst_pay.c tests/asst/t_asst_task.c tests/asst/t_asst_push.c tests/asst/t_asst_plan.c tests/asst/t_asst_memory.c tests/asst/t_asst_image.c tests/asst/t_asst_preview.c tests/asst/t_asst_market.c tests/asst/t_asst_lottery.c tests/asst/t_asst_db.c tests/asst/t_asst_verify.c $(ASST_OBJS) $(SANDBOX_OBJS) $(SKILL_OBJS) $(MODEL_CLIENT_OBJS) $(PUSH_OBJS) $(DB_OBJS) $(OFFICE_OBJS) $(IMAGE_OBJS) $(VIDEO_OBJS) $(TTS_LINK) $(WEB_OBJS) $(MODEL_OBJS) $(STUDIO_AUTH_OBJS) $(STUDIO_NET_OBJS) src/common/ds4_json.o $(LDLIBS)


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

ds4_test: $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ $(TESTS_OBJS) $(SERVER_TEST_OBJS) $(SERVER_TESTS_OBJS) $(KV_OBJS) rax.o $(MODEL_OBJS) $(ASR_OBJS) $(CORE_OBJS) $(CUDA_LDLIBS)
endif

# src/common 共享格式库单测: 无模型/无 GPU, 纯主机 C。夹具路径相对仓库根。
ds4_unit: tests/unit/test_common.c tests/unit/test_zmod.c tests/unit/t_zfinetune.c tests/unit/t_sha256.c tests/unit/t_pretok.c tests/unit/pretok_cases.h \
          src/common/ds4_sha256.c src/common/ds4_sha256.h src/common/ds4_pretok.c src/common/ds4_pretok.h src/common/ds4_unicode_tab.h \
          src/common/ds4_quantfmt.c src/common/ds4_gguf.c src/common/ds4_quantfmt.h src/common/ds4_gguf.h src/common/ds4_float.h src/common/ds4_fp8.h \
          ds4_z.c ds4_z.h ds4_zchain.c ds4_zchain.h ds4_zfinetune.c ds4_zfinetune.h ds4_loss.c ds4_loss.h
	$(CC) $(CFLAGS) -Isrc/common -I. -o $@ tests/unit/test_common.c tests/unit/test_zmod.c tests/unit/t_zfinetune.c tests/unit/t_sha256.c tests/unit/t_pretok.c \
	    src/common/ds4_sha256.c src/common/ds4_pretok.c src/common/ds4_quantfmt.c src/common/ds4_gguf.c ds4_z.c ds4_zchain.c ds4_zfinetune.c ds4_loss.c $(LDLIBS)

test: ds4_test ds4-eval ds4_unit ds4_asst_test ds4_office_test ds4_image_test ds4_seg_test ds4_video_test ds4_tts_test ds4_push_test linecount
	./ds4_unit
	./ds4_office_test
	./ds4_image_test
	./ds4_seg_test
	./ds4_video_test
	./ds4_tts_test gguf/tts/zh-huayan.gguf tests/tts/fixtures zh
	./ds4_tts_test gguf/tts/en-lessac.gguf tests/tts/fixtures en
	./ds4_asst_test
	./ds4_push_test
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
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-phone ds4-seg ds4_cpu ds4_native ds4_server_test ds4_test ds4_unit ds4_asst_test ds4_office_test ds4_image_test ds4_seg_test ds4_video_test ds4_push_test tests/video/vidcheck e0-pingpong mm-ui mm-ocr *.o src/assistant/*.o src/assistant/tools/*.o src/sandbox/*.o src/sandbox/phone/*.o src/skill/*.o src/db/*.o src/model/*.o src/studio/*.o src/push/*.o src/office/*.o src/image/*.o src/seg/*.o src/video/*.o src/cli/*.o src/bench/*.o src/kv/*.o src/web/*.o src/eval/*.o src/agent/*.o src/dist/*.o src/server/*.o src/core/*.o src/common/*.o src/metal/*.o tests/server_tests_*.o tests/t_*.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o

# Task 04 / E0: standalone thunderbolt ping-pong latency gate (no core deps, no
# model). Defined after the default targets so it never becomes the default goal.
e0: e0-pingpong
e0-pingpong: tools/e0_pingpong.c
	$(CC) $(CFLAGS) -o $@ tools/e0_pingpong.c -lm

# 不加载模型的损失梯度门；与训练器链接同一套 CUDA 对象。
ds4_cuda.o src/core/core_ptrain_rankprob.o: ds4_gpu_rankprob.h
tests/cuda_rankprob_selftest.o: tests/cuda_rankprob_selftest.c ds4_gpu_rankprob.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<
tests/cuda_rankprob_selftest: tests/cuda_rankprob_selftest.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
