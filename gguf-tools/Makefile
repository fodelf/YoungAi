# gguf-tools/Makefile — 离线量化/反修/校准/评测工具全家的唯一构建入口(2026-08-25 重组批1)。
#
# 用法速查:
#   make all          # deepseek4-quantize + anchor_metrics + compare_scores
#   make amp          # 反修链(ds4quant_run/zlayer/dql_to_zchain/teacher_routed/zloss_solve)
#   make legacy       # (2026-08-26 清仓后为空; solvers 源仅作 calib_run 链接件)
#   make calib        # 校准/前向验证/npy 工具
#   make bench        # 评测/取证工具(pubbench/kl_forensic/...)
#   make tools-test   # 9 项纯 C 自测(原 go-onebit-test, 旧名保留一版别名)
#   make quality-score# 续写评分器(要先能构建父仓引擎)
#   make <工具名>     # 单个工具(如 make zlayer), 产物落在源文件旁(见各规则 -o)
#
# 目录(2026-08-25 批4 重组, go-onebit 目录消亡): quantize/=量化器族 amp/=反修链
# calib/=校准/验证 bench/=评测 legacy/=上一代口径 scripts/=战役脚本 data/=数据 docs/=文档
#
# 平台: Darwin 走 Accelerate; Linux 走 scipy_openblas32 用户态前缀(spark 无 root,
# 照 migrate/build_ctools.sh 历史特判), 装不到 BLAS 时降级为无 BLAS 标量构建(慢但能跑)。
# Linux 且 /usr/local/cuda/bin/nvcc 存在时自动叠 CUDA(zlayer/ds4quant_run/deepseek4-quantize)。
# make CUDA=0 强制关 CUDA(对拍参照)。

CC ?= cc
UNAME_S := $(shell uname -s)

QZ  := quantize
AMP := amp
CAL := calib
BEN := bench
LEG := legacy
SLV := legacy/solvers
MIG := migrate
QT  := bench/quality-testing
BUILD := .build

# 三档编译旗标 —— 别合并, 数值契约不同:
#   CFLAGS   老 Makefile 目标(量化器/自测): -std=c11 ISO 模式 ⇒ 隐含 -ffp-contract=off,
#            deepseek4-quantize 的 CPU↔GPU 逐字节对拍就建立在这上(见 build_dsq_gpu_spark 注)。
#   BARE     build_ctools.sh 移植目标: 无 -std(gnu 默认, fp-contract=fast), 金标(migrate/golden.txt)
#            是这个口径跑出来的, 换旗标=换数字。
CFLAGS ?= -O3 -Wall -Wextra -std=c11
CPPFLAGS ?= -D_GNU_SOURCE
BARE := -O3
INC := -I$(QZ) -I$(AMP) -I$(CAL) -I$(SLV)

# 聚合根分片依赖(2026-08-25 批6 拆分): 超 500 行源文件拆成 <名>_pK.inc.c, 由原路径
# 聚合根按序 #include(单 TU, 符号零变化)。分片只进【依赖列表】触发重建, 绝不进编译
# 命令行(它们不是独立 TU)。
QUANTS_INC := $(wildcard $(QZ)/quants_p*.inc.c)
DSQMAIN_INC := $(wildcard $(QZ)/deepseek4-quantize_p*.inc.c)
PUBBENCH_INC := $(wildcard $(BEN)/pubbench_p*.inc.c)
CALIBDIAG_INC := $(wildcard $(CAL)/calib_diag_p*.inc.c)
HVSOLVE_INC := $(wildcard $(AMP)/hiddenvar_solve_p*.inc.c)
RRR_INC := $(wildcard $(AMP)/solve_rrr_p*.inc.c)
CALIBRUN_INC := $(wildcard $(CAL)/calib_run_p*.inc.c)
VFWD_INC := $(wildcard $(CAL)/validate_fwd_p*.inc.c)
DSQFWD_INC := $(wildcard $(AMP)/ds4quant_fwd_p*.inc.c)
TRAJ_INC := $(wildcard $(CAL)/traj_analyze_p*.inc.c)
HFREAD_INC := $(wildcard $(CAL)/hf_read_p*.inc.c)
VQMERGE_INC := $(wildcard $(QZ)/vq_merge_v4_p*.inc.c)
ZLAYER_INC := $(wildcard $(AMP)/zlayer_p*.inc.c)
DSQRUN_INC := $(wildcard $(AMP)/ds4quant_run_p*.inc.c)

ifeq ($(UNAME_S),Darwin)
NATIVE ?= -mcpu=native
HAVE_BLAS := 1
BLAS_CFLAGS :=
BLAS_LIBS := -framework Accelerate
CUDA := 0
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal -framework Accelerate
# 父仓 Darwin 引擎对象集合(重构后 ds4.o/ds4_metal.o 已裂成对象群): 解析期不 wildcard
# (对象可能还没生成), quality-score 配方里先 make -C .. 再 shell find。
QUALITY_OBJS_CMD = find ../src/core ../src/dist ../src/metal -name '*.o'; \
                   echo ../src/common/ds4_quantfmt.o ../ds4_corr.o ../ds4_zchain.o \
                        ../ds4_multimodal.o ../ds4_spatial.o ../ds4_css.o
QUALITY_LINK = $(CC) -O3 $(NATIVE) -I..
else
NATIVE ?= -march=native
# scipy_openblas32 用户态前缀(spark 无 root): 符号带 scipy_ 前缀, -D 改名接上。
# 缺失时降级无 BLAS(BLAS_CFLAGS/BLAS_LIBS 全空, DQ_BLAS 不开, 走标量参考路)。
SO_DIR := $(HOME)/.local/lib/python3.12/site-packages/scipy_openblas32/lib
ifneq ($(wildcard $(SO_DIR)/libscipy_openblas.so),)
HAVE_BLAS := 1
BLAS_RENAMES := -Dcblas_sgemm=scipy_cblas_sgemm -Dcblas_dgemm=scipy_cblas_dgemm \
                -Dcblas_sgemv=scipy_cblas_sgemv -Dcblas_dsdot=scipy_cblas_dsdot \
                -Dcblas_dtrsm=scipy_cblas_dtrsm
BLAS_CFLAGS := $(BLAS_RENAMES) -I$(SO_DIR)/../include
BLAS_LIBS := $(SO_DIR)/libscipy_openblas.so -Wl,-rpath,$(SO_DIR)
else
HAVE_BLAS := 0
BLAS_CFLAGS :=
BLAS_LIBS :=
endif
# CUDA 能力探测(nvcc 在才启; make CUDA=0 强制关)
CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA ?= $(if $(wildcard $(NVCC)),1,0)
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/sbsa-linux/lib -lcudart -lstdc++
QUALITY_LDLIBS := -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_OBJS_CMD = find ../src/core ../src/dist -name '*.o'; \
                   echo ../src/common/ds4_quantfmt.o ../ds4_corr.o ../ds4_zchain.o \
                        ../ds4_multimodal.o ../ds4_spatial.o ../ds4_css.o ../ds4_cuda.o
QUALITY_LINK = $(NVCC) -O3 -I..
endif

# curl 特判(pubbench; spark 无 root): 系统缺 curl 头时用 apt-get download+dpkg -x 的
# 用户态头(~/.local/curl-dev), 链接直指系统运行时 libcurl.so.4(ABI 稳定)。
CURL_FLAGS := -lcurl
ifneq ($(UNAME_S),Darwin)
ifeq ($(wildcard /usr/include/curl/curl.h),)
ifeq ($(wildcard /usr/include/$(shell uname -m)-linux-gnu/curl/curl.h),)
CDEV := $(HOME)/.local/curl-dev/extract
CURL_FLAGS := -I$(CDEV)/usr/include/$(shell uname -m)-linux-gnu -I$(CDEV)/usr/include \
              /usr/lib/$(shell uname -m)-linux-gnu/libcurl.so.4
endif
endif
endif

# ---------------------------------------------------------------- 目标分组
.PHONY: all amp legacy calib bench tools-test go-onebit-test quality-score clean everything
.PHONY: deepseek4-quantize-alias anchor_metrics compare_scores ds4quant_run zlayer \
        dql_to_zchain teacher_routed zloss_solve \
        calib_run calib_diag validate_fwd traj_analyze cap_raw2npy vq_blob_truesize dump_gguf_meta \
        kl_forensic trace_ladder rec_fidelity pubbench pubbench_extract_test vq_merge_v4 \
        go2b_parity vq2q2k_probe vq_em deepseek4-quantize-cpu

all: deepseek4-quantize anchor_metrics compare_scores
amp: ds4quant_run zlayer dql_to_zchain teacher_routed zloss_solve
# legacy 二进制已随 2026-08-26 清仓删除(用户令); solvers 源仅作 calib_run 链接件存续。
legacy:
	@echo "legacy 目标已清空(2026-08-26 清仓)"
calib: calib_run calib_diag validate_fwd traj_analyze cap_raw2npy vq_blob_truesize dump_gguf_meta
bench: kl_forensic trace_ladder rec_fidelity pubbench pubbench_extract_test vq_merge_v4 \
       go2b_parity vq2q2k_probe vq_em
everything: all amp legacy calib bench

# ---------------------------------------------------------------- all
# deepseek4-quantize: Linux+CUDA 时 IQ2_XXS 编码器上 GPU。数值契约: CPU 侧 -std=c11 是
# ISO 模式(-ffp-contract=off), 所以 nvcc 必须 -fmad=false 且【禁 --use_fast_math】——
# 否则 GPU 与 CPU 差最后一个 ulp, 逐字节对拍必挂(见原 build_dsq_gpu_spark.sh)。
DSQ_SRC := $(QZ)/deepseek4-quantize.c $(QZ)/quants.c $(QZ)/onebit_quant.c
ifeq ($(UNAME_S)-$(CUDA),Linux-1)
deepseek4-quantize: $(DSQ_SRC) $(QUANTS_INC) $(DSQMAIN_INC) $(QZ)/quants.h $(QZ)/onebit_quant.h $(QZ)/quantize_gpu.cu
	$(NVCC) -O3 -arch=native -fmad=false -c $(QZ)/quantize_gpu.cu -o $(QZ)/quantize_gpu.o
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -DDS4Q_CUDA -o $@ $(DSQ_SRC) $(QZ)/quantize_gpu.o \
	    $(CUDA_LIBS) -lm -pthread
else
deepseek4-quantize: $(DSQ_SRC) $(QUANTS_INC) $(DSQMAIN_INC) $(QZ)/quants.h $(QZ)/onebit_quant.h
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(DSQ_SRC) -lm -pthread
endif

# 纯 CPU 参照(对拍用, Linux; Mac 上本来就是 CPU 版)
deepseek4-quantize-cpu: $(DSQ_SRC) $(QUANTS_INC) $(DSQMAIN_INC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(DSQ_SRC) -lm -pthread

anchor_metrics: $(BEN)/anchor_metrics
$(BEN)/anchor_metrics: $(BEN)/anchor_metrics.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread

compare_scores: $(QT)/compare_scores
$(QT)/compare_scores: $(QT)/compare_scores.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm

# ---------------------------------------------------------------- amp(反修链)
# ds4quant_run: -I.. 供 #include "ds4_z.c"/"ds4_loss.c"(仓库根, 与引擎同一份实现)。
# Darwin 由 ds4quant_fwd.c 的 __APPLE__ 分支自动接 Accelerate; Linux 用 scipy_openblas
# 改名, CUDA 在则 vq_gpu.o+cuBLAS 直传(原 build_quant_spark.sh)。
DSQR_FLAGS := $(BARE) $(NATIVE) -Wall -Wno-unused-parameter -Wno-unused-function -I..
ifeq ($(UNAME_S),Darwin)
ds4quant_run: $(AMP)/ds4quant_run
$(AMP)/ds4quant_run: $(AMP)/ds4quant_run.c $(DSQRUN_INC) $(AMP)/ds4quant_fwd.c $(DSQFWD_INC) $(QZ)/onebit_quant.c \
                     $(AMP)/ds4quant_qhelp.h $(QZ)/go2b_qc.h $(QZ)/vq_qc.h
	$(CC) $(DSQR_FLAGS) -o $@ $(AMP)/ds4quant_run.c $(BLAS_LIBS) -lm -lpthread
else
ifeq ($(CUDA),1)
$(QZ)/vq_gpu.o: $(QZ)/vq_gpu.cu
	$(NVCC) -O3 -arch=native -c $< -o $@
ds4quant_run: $(AMP)/ds4quant_run
$(AMP)/ds4quant_run: $(AMP)/ds4quant_run.c $(DSQRUN_INC) $(QZ)/vq_gpu.o
	$(CC) $(DSQR_FLAGS) $(if $(filter 1,$(HAVE_BLAS)),-DDQ_BLAS -DDS4QUANT_OPENBLAS $(BLAS_CFLAGS)) \
	    -DDS4QUANT_CUDA -I$(CUDA_HOME)/include -o $@ $(AMP)/ds4quant_run.c $(QZ)/vq_gpu.o \
	    $(CUDA_LIBS) -lcublas $(BLAS_LIBS) -lm -lpthread
else
ds4quant_run: $(AMP)/ds4quant_run
$(AMP)/ds4quant_run: $(AMP)/ds4quant_run.c $(DSQRUN_INC)
	$(CC) $(DSQR_FLAGS) $(if $(filter 1,$(HAVE_BLAS)),-DDQ_BLAS -DDS4QUANT_OPENBLAS $(BLAS_CFLAGS)) \
	    -o $@ $(AMP)/ds4quant_run.c $(BLAS_LIBS) -lm -lpthread
endif
endif

# zlayer: 反修解算器。Darwin=Accelerate; Linux=scipy_openblas(缺则无法链—— zlayer 的
# LAPACK potrf/potrs 在非 APPLE 分支写死 scipy_ 前缀, 无 BLAS 不可降级); CUDA 在则叠
# zlayer_gpu.o(cuBLAS 薄封装, GPU 缺时运行时自动回落 CPU 路)。
zlayer: $(AMP)/zlayer
ifeq ($(UNAME_S),Darwin)
$(AMP)/zlayer: $(AMP)/zlayer.c $(ZLAYER_INC) ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS -o $@ $(AMP)/zlayer.c $(BLAS_LIBS) -lm -lpthread
else
ifeq ($(HAVE_BLAS),0)
$(AMP)/zlayer:
	@echo "zlayer: Linux 上需要 scipy_openblas32($(SO_DIR)), 缺失 — 跳过(装上再 make zlayer)"; exit 1
else
ifeq ($(CUDA),1)
$(AMP)/zlayer_gpu.o: $(AMP)/zlayer_gpu.cu
	$(NVCC) -O3 -arch=native -c $< -o $@
$(AMP)/zlayer: $(AMP)/zlayer.c $(ZLAYER_INC) $(AMP)/zlayer_gpu.o ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS $(BLAS_CFLAGS) -DZL_CUDA -o $@ $(AMP)/zlayer.c \
	    $(AMP)/zlayer_gpu.o $(BLAS_LIBS) -lcublas -lcusolver $(CUDA_LIBS) -lm -lpthread
else
$(AMP)/zlayer: $(AMP)/zlayer.c $(ZLAYER_INC) ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS $(BLAS_CFLAGS) -o $@ $(AMP)/zlayer.c $(BLAS_LIBS) -lm -lpthread
endif
endif
endif

dql_to_zchain: $(AMP)/dql_to_zchain
$(AMP)/dql_to_zchain: $(AMP)/dql_to_zchain.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
# zloss_solve: 单层贪心四损失解算器(2026-08-26 用户设计)。-I.. 吃仓库根 ds4_z.c/
# ds4_loss.c(与引擎同一份实现); BLAS 在则开 DQ_BLAS 快路(ds4_z.c 内守卫)。
zloss_solve: $(AMP)/zloss_solve
ZLOSS_DEPS := $(AMP)/zloss_solve.c $(AMP)/zloss_arms.inc.c $(AMP)/zloss_gate.inc.c $(AMP)/linalg_small.c $(CAL)/npy.c ../ds4_z.c ../ds4_z.h ../ds4_loss.c ../ds4_loss.h
ifeq ($(HAVE_BLAS)-$(CUDA),1-1)
# GPU 路(2026-08-26 提速令"必须用GPU"): gemm/potrf/potrs/QR 走 zlayer_gpu.o(cuBLAS+cuSOLVER)
$(AMP)/zloss_solve: $(ZLOSS_DEPS) $(AMP)/zlayer_gpu.o
	$(CC) $(BARE) $(NATIVE) -I.. -I$(CAL) -DDQ_BLAS -DDQ_CUDA $(BLAS_CFLAGS) -o $@ $(AMP)/zloss_solve.c $(AMP)/linalg_small.c $(CAL)/npy.c $(AMP)/zlayer_gpu.o $(BLAS_LIBS) -lcublas -lcusolver $(CUDA_LIBS) -lm -lpthread
else ifeq ($(HAVE_BLAS),1)
$(AMP)/zloss_solve: $(ZLOSS_DEPS)
	$(CC) $(BARE) $(NATIVE) -I.. -I$(CAL) -DDQ_BLAS $(BLAS_CFLAGS) -o $@ $(AMP)/zloss_solve.c $(AMP)/linalg_small.c $(CAL)/npy.c $(BLAS_LIBS) -lm -lpthread
else
$(AMP)/zloss_solve: $(ZLOSS_DEPS)
	$(CC) $(BARE) $(NATIVE) -I.. -I$(CAL) -o $@ $(AMP)/zloss_solve.c $(AMP)/linalg_small.c $(CAL)/npy.c -lm -lpthread
endif

teacher_routed: teacher_routed.bin
.PHONY: teacher_routed.bin
teacher_routed.bin: $(AMP)/teacher_routed.c $(CAL)/npy.c $(CAL)/layer_probe.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -I$(QZ) -o teacher_routed $(AMP)/teacher_routed.c \
	    $(CAL)/npy.c $(CAL)/layer_probe.c -lm -lpthread

# ---------------------------------------------------------------- calib(校准/验证)
calib_run: calib_run.bin
.PHONY: calib_run.bin
CALIB_RUN_SRC := $(CAL)/calib_run.c $(CAL)/npy.c $(CAL)/hf_read.c $(QZ)/onebit_quant.c \
    $(CAL)/layer_probe.c $(AMP)/hiddenvar_solve.c $(SLV)/solve_perexpert.c $(SLV)/solve_denoise.c \
    $(SLV)/solve_genrf.c $(AMP)/solve_rrr.c $(AMP)/linalg_small.c
calib_run.bin: $(CALIB_RUN_SRC) $(CALIBRUN_INC) $(HVSOLVE_INC) $(RRR_INC) $(HFREAD_INC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o calib_run $(CALIB_RUN_SRC) -lm -pthread

calib_diag: calib_diag.bin
.PHONY: calib_diag.bin
CALIB_DIAG_SRC := $(CAL)/calib_diag.c $(CAL)/npy.c $(CAL)/hf_read.c $(QZ)/onebit_quant.c \
    $(CAL)/layer_probe.c $(AMP)/hiddenvar_solve.c $(AMP)/linalg_small.c
calib_diag.bin: $(CALIB_DIAG_SRC) $(CALIBDIAG_INC) $(HVSOLVE_INC) $(HFREAD_INC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o calib_diag $(CALIB_DIAG_SRC) -lm -pthread

validate_fwd: $(CAL)/validate_fwd
VFWD_SRC := $(CAL)/validate_fwd.c $(CAL)/npy.c $(CAL)/hf_read.c $(CAL)/layer_probe.c \
    $(QZ)/onebit_quant.c $(SLV)/solve_denoise.c $(SLV)/solve_genrf.c $(AMP)/linalg_small.c
$(CAL)/validate_fwd: $(VFWD_SRC) $(VFWD_INC) $(HFREAD_INC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(VFWD_SRC) -lm -pthread

traj_analyze: $(CAL)/traj_analyze
# 源列表显式化: 原配方用 $^ 当源, 加 *.inc.c 依赖后 $^ 会把分片当独立 TU 编译, 必须分开。
TRAJ_SRC := $(CAL)/traj_analyze.c $(CAL)/npy.c $(AMP)/linalg_small.c
$(CAL)/traj_analyze: $(TRAJ_SRC) $(TRAJ_INC)
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -I$(AMP) -o $@ $(TRAJ_SRC) -lm -lpthread

cap_raw2npy: cap_raw2npy.bin
.PHONY: cap_raw2npy.bin
cap_raw2npy.bin: $(CAL)/cap_raw2npy.c
	$(CC) $(BARE) $(NATIVE) -o cap_raw2npy $< -lm -lpthread

vq_blob_truesize: $(CAL)/vq_blob_truesize
$(CAL)/vq_blob_truesize: $(CAL)/vq_blob_truesize.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread

dump_gguf_meta: dump_gguf_meta.bin
.PHONY: dump_gguf_meta.bin
dump_gguf_meta.bin: $(QZ)/dump_gguf_meta.c
	$(CC) $(BARE) $(NATIVE) -o dump_gguf_meta $< -lm

# ---------------------------------------------------------------- bench(评测/取证)
kl_forensic: $(BEN)/kl_forensic
$(BEN)/kl_forensic: $(BEN)/kl_forensic.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
trace_ladder: $(BEN)/trace_ladder
$(BEN)/trace_ladder: $(BEN)/trace_ladder.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
rec_fidelity: $(BEN)/rec_fidelity
$(BEN)/rec_fidelity: $(BEN)/rec_fidelity.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -o $@ $(BEN)/rec_fidelity.c $(CAL)/npy.c -lm -lpthread

pubbench: $(BEN)/pubbench
$(BEN)/pubbench: $(BEN)/pubbench.c $(PUBBENCH_INC)
	$(CC) $(BARE) $(NATIVE) -o $@ $< $(CURL_FLAGS) -lz -lm -lpthread

pubbench_extract_test: $(BEN)/gates/pubbench_extract_test
$(BEN)/gates/pubbench_extract_test: $(BEN)/gates/pubbench_extract_test.c $(BEN)/pubbench.c $(PUBBENCH_INC)
	$(CC) $(BARE) -o $@ $(BEN)/gates/pubbench_extract_test.c $(CURL_FLAGS) -lz -lm -lpthread

vq_merge_v4: $(QZ)/vq_merge_v4
$(QZ)/vq_merge_v4: $(QZ)/vq_merge_v4.c $(VQMERGE_INC)
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm

go2b_parity: $(QZ)/go2b_parity
$(QZ)/go2b_parity: $(QZ)/go2b_parity.c $(QZ)/onebit_quant.h $(QZ)/go2b_qc.h
	$(CC) $(BARE) $(NATIVE) -ffast-math -I$(QZ) $(BLAS_CFLAGS) -o $@ $(QZ)/go2b_parity.c $(BLAS_LIBS) -lm -lpthread

vq2q2k_probe: $(QZ)/vq2q2k_probe
$(QZ)/vq2q2k_probe: $(QZ)/vq2q2k_probe.c $(QZ)/quants.c $(QUANTS_INC) $(QZ)/onebit_quant.c
	$(CC) $(BARE) $(NATIVE) -o $@ $(QZ)/vq2q2k_probe.c $(QZ)/quants.c $(QZ)/onebit_quant.c -lm -lpthread

# vq_em: GPU-EM 码本精修, Linux+CUDA 专属(st_bridge.c 桥接 st_read)。
ifeq ($(UNAME_S)-$(CUDA),Linux-1)
vq_em: vq_em.bin
.PHONY: vq_em.bin
vq_em.bin: $(QZ)/vq_em.cu $(QZ)/st_bridge.c
	$(NVCC) -O3 -arch=native -c $(QZ)/vq_em.cu -o $(QZ)/vq_em.o
	$(CC) $(BARE) $(NATIVE) -c $(QZ)/st_bridge.c -o $(QZ)/st_bridge.o
	$(CC) $(BARE) -o vq_em $(QZ)/vq_em.o $(QZ)/st_bridge.o $(CUDA_LIBS) -lcublas -lm -lpthread
else
vq_em:
	@echo "vq_em: Linux+CUDA 专属(GPU-EM), 本平台跳过"
endif

# ---------------------------------------------------------------- 自测(9 项, 纯 C 无模型)
tools-test:
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DGO1B_TEST    $(QZ)/onebit_quant.c -o $(BUILD)/onebit_test -lm && $(BUILD)/onebit_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DLINALG_TEST  $(AMP)/linalg_small.c -o $(BUILD)/linalg_test -lm && $(BUILD)/linalg_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DPROBE_TEST   $(CAL)/layer_probe.c  -o $(BUILD)/probe_test  -lpthread -lm && $(BUILD)/probe_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DHVSOLVE_TEST $(AMP)/hiddenvar_solve.c $(AMP)/linalg_small.c -o $(BUILD)/hv_test -lm && $(BUILD)/hv_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DNPY_TEST     $(CAL)/npy.c -o $(BUILD)/npy_test -lm && $(BUILD)/npy_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DHFREAD_TEST  $(CAL)/hf_read.c -o $(BUILD)/hfread_test -lm && $(BUILD)/hfread_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DPEREXPERT_TEST $(SLV)/solve_perexpert.c $(AMP)/linalg_small.c -o $(BUILD)/perexpert_test -lm && $(BUILD)/perexpert_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DCALIBDIAG_TEST $(CAL)/calib_diag.c $(AMP)/hiddenvar_solve.c $(AMP)/linalg_small.c -o $(BUILD)/calibdiag_test -lm -pthread && $(BUILD)/calibdiag_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DRRR_TEST     $(AMP)/solve_rrr.c $(AMP)/hiddenvar_solve.c $(AMP)/linalg_small.c -o $(BUILD)/rrr_test -lm -pthread && $(BUILD)/rrr_test

# 旧名别名(保留一版, 下一轮清)
go-onebit-test: tools-test

# ---------------------------------------------------------------- quality-score
# 父仓引擎对象已从 ds4.o/ds4_metal.o 裂成对象群(src/core|dist|metal + 根杂件): 先让父仓
# 把对象建出来(make -C .. ds4), 再在配方期用 shell find 收集 —— 解析期 $(wildcard) 会在
# 对象未生成时展空, 别改回去。
quality-score:
	$(MAKE) -C .. ds4
	$(QUALITY_LINK) -o $(QT)/score_official $(QT)/score_official.c \
	    $$( $(QUALITY_OBJS_CMD) ) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize deepseek4-quantize-cpu calib_run calib_diag \
	    teacher_routed cap_raw2npy dump_gguf_meta vq_em \
	    $(QT)/score_official $(QT)/compare_scores \
	    $(QZ)/quantize_gpu.o $(QZ)/vq_gpu.o $(QZ)/vq_merge_v4 $(QZ)/go2b_parity $(QZ)/vq2q2k_probe \
	    $(QZ)/vq_em.o $(QZ)/st_bridge.o \
	    $(AMP)/ds4quant_run $(AMP)/zlayer $(AMP)/zlayer_gpu.o \
	    $(AMP)/dql_to_zchain $(AMP)/zloss_solve \
	    $(CAL)/validate_fwd $(CAL)/traj_analyze $(CAL)/vq_blob_truesize \
	    $(BEN)/anchor_metrics $(BEN)/kl_forensic $(BEN)/trace_ladder $(BEN)/rec_fidelity \
	    $(BEN)/pubbench $(BEN)/gates/pubbench_extract_test \
	    ds4quant_fwd_test
	rm -rf $(BUILD)
