# gguf-tools/Makefile — 离线量化/反修/校准/评测工具全家的唯一构建入口(2026-08-25 重组批1)。
#
# 用法速查:
#   make all          # deepseek4-quantize + anchor_metrics + compare_scores
#   make amp          # 反修链全家(ds4quant_run/zlayer/amp_solve/合链/emit_z/探针...)
#   make legacy       # 上一代 zcache 口径工具(amp_solve_zc/emit_residual/...)
#   make calib        # 校准/前向验证/npy 工具
#   make bench        # 评测/取证工具(pubbench/kl_forensic/...)
#   make tools-test   # 9 项纯 C 自测(原 go-onebit-test, 旧名保留一版别名)
#   make quality-score# 续写评分器(要先能构建父仓引擎)
#   make <工具名>     # 单个工具(如 make zlayer), 产物落在源文件旁(见各规则 -o)
#
# 平台: Darwin 走 Accelerate; Linux 走 scipy_openblas32 用户态前缀(spark 无 root,
# 照 migrate/build_ctools.sh 历史特判), 装不到 BLAS 时降级为无 BLAS 标量构建(慢但能跑)。
# Linux 且 /usr/local/cuda/bin/nvcc 存在时自动叠 CUDA(zlayer/ds4quant_run/deepseek4-quantize)。
# make CUDA=0 强制关 CUDA(对拍参照)。

CC ?= cc
UNAME_S := $(shell uname -s)

G1B := go-onebit
CAL := $(G1B)/calib
QNT := $(G1B)/quant
LAT := $(G1B)/latent
LEG := $(G1B)/latent/legacy
MIG := $(G1B)/migrate
QT  := quality-testing
BUILD := $(G1B)/.build

# 三档编译旗标 —— 别合并, 数值契约不同:
#   CFLAGS   老 Makefile 目标(量化器/自测): -std=c11 ISO 模式 ⇒ 隐含 -ffp-contract=off,
#            deepseek4-quantize 的 CPU↔GPU 逐字节对拍就建立在这上(见 build_dsq_gpu_spark 注)。
#   BARE     build_ctools.sh 移植目标: 无 -std(gnu 默认, fp-contract=fast), 金标(migrate/golden.txt)
#            是这个口径跑出来的, 换旗标=换数字。
#   amp_solve_zc 单独 -ffp-contract=off(文件头明令: FMA 合并少一次舍入会真改数)。
CFLAGS ?= -O3 -Wall -Wextra -std=c11
CPPFLAGS ?= -D_GNU_SOURCE
BARE := -O3
INC := -I$(QNT) -I$(LAT) -I$(LEG) -I$(CAL)

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
.PHONY: deepseek4-quantize-alias anchor_metrics compare_scores ds4quant_run zlayer amp_solve \
        zrec_to_zchain dql_to_zchain zchain_merge route_alpha_set route_bias_rebake emit_z \
        teacher_routed ge_solve hsolve hbase dilution z_tune z_explore z_explore2 \
        amp_solve_zc emit_residual corr_stats lut_cov merge_residual min_go_check \
        calib_run calib_diag validate_fwd traj_analyze cap_raw2npy vq_blob_truesize dump_gguf_meta \
        kl_forensic trace_ladder rec_fidelity pubbench pubbench_extract_test vq_merge_v4 \
        go2b_parity vq2q2k_probe vq_em deepseek4-quantize-cpu

all: deepseek4-quantize anchor_metrics compare_scores
amp: ds4quant_run zlayer amp_solve zrec_to_zchain dql_to_zchain zchain_merge route_alpha_set \
     route_bias_rebake emit_z teacher_routed ge_solve hsolve hbase dilution z_tune z_explore z_explore2
legacy: amp_solve_zc emit_residual corr_stats lut_cov merge_residual min_go_check
calib: calib_run calib_diag validate_fwd traj_analyze cap_raw2npy vq_blob_truesize dump_gguf_meta
bench: kl_forensic trace_ladder rec_fidelity pubbench pubbench_extract_test vq_merge_v4 \
       go2b_parity vq2q2k_probe vq_em
everything: all amp legacy calib bench

# ---------------------------------------------------------------- all
# deepseek4-quantize: Linux+CUDA 时 IQ2_XXS 编码器上 GPU。数值契约: CPU 侧 -std=c11 是
# ISO 模式(-ffp-contract=off), 所以 nvcc 必须 -fmad=false 且【禁 --use_fast_math】——
# 否则 GPU 与 CPU 差最后一个 ulp, 逐字节对拍必挂(见原 build_dsq_gpu_spark.sh)。
DSQ_SRC := deepseek4-quantize.c quants.c $(QNT)/onebit_quant.c
ifeq ($(UNAME_S)-$(CUDA),Linux-1)
deepseek4-quantize: $(DSQ_SRC) quants.h $(QNT)/onebit_quant.h quantize_gpu.cu
	$(NVCC) -O3 -arch=native -fmad=false -c quantize_gpu.cu -o quantize_gpu.o
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -DDS4Q_CUDA -o $@ $(DSQ_SRC) quantize_gpu.o \
	    $(CUDA_LIBS) -lm -pthread
else
deepseek4-quantize: $(DSQ_SRC) quants.h $(QNT)/onebit_quant.h
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(DSQ_SRC) -lm -pthread
endif

# 纯 CPU 参照(对拍用, Linux; Mac 上本来就是 CPU 版)
deepseek4-quantize-cpu: $(DSQ_SRC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(DSQ_SRC) -lm -pthread

anchor_metrics: $(CAL)/anchor_metrics
$(CAL)/anchor_metrics: $(CAL)/anchor_metrics.c
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
ds4quant_run: $(QNT)/ds4quant_run
$(QNT)/ds4quant_run: $(QNT)/ds4quant_run.c $(QNT)/ds4quant_fwd.c $(QNT)/onebit_quant.c \
                     $(QNT)/ds4quant_qhelp.h $(QNT)/go2b_qc.h $(QNT)/vq_qc.h
	$(CC) $(DSQR_FLAGS) -o $@ $(QNT)/ds4quant_run.c $(BLAS_LIBS) -lm -lpthread
else
ifeq ($(CUDA),1)
$(QNT)/vq_gpu.o: $(QNT)/vq_gpu.cu
	$(NVCC) -O3 -arch=native -c $< -o $@
ds4quant_run: $(QNT)/ds4quant_run
$(QNT)/ds4quant_run: $(QNT)/ds4quant_run.c $(QNT)/vq_gpu.o
	$(CC) $(DSQR_FLAGS) $(if $(filter 1,$(HAVE_BLAS)),-DDQ_BLAS -DDS4QUANT_OPENBLAS $(BLAS_CFLAGS)) \
	    -DDS4QUANT_CUDA -I$(CUDA_HOME)/include -o $@ $(QNT)/ds4quant_run.c $(QNT)/vq_gpu.o \
	    $(CUDA_LIBS) -lcublas $(BLAS_LIBS) -lm -lpthread
else
ds4quant_run: $(QNT)/ds4quant_run
$(QNT)/ds4quant_run: $(QNT)/ds4quant_run.c
	$(CC) $(DSQR_FLAGS) $(if $(filter 1,$(HAVE_BLAS)),-DDQ_BLAS -DDS4QUANT_OPENBLAS $(BLAS_CFLAGS)) \
	    -o $@ $(QNT)/ds4quant_run.c $(BLAS_LIBS) -lm -lpthread
endif
endif

# zlayer: 反修解算器。Darwin=Accelerate; Linux=scipy_openblas(缺则无法链—— zlayer 的
# LAPACK potrf/potrs 在非 APPLE 分支写死 scipy_ 前缀, 无 BLAS 不可降级); CUDA 在则叠
# zlayer_gpu.o(cuBLAS 薄封装, GPU 缺时运行时自动回落 CPU 路)。
zlayer: $(CAL)/zlayer
ifeq ($(UNAME_S),Darwin)
$(CAL)/zlayer: $(CAL)/zlayer.c ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS -o $@ $(CAL)/zlayer.c $(BLAS_LIBS) -lm -lpthread
else
ifeq ($(HAVE_BLAS),0)
$(CAL)/zlayer:
	@echo "zlayer: Linux 上需要 scipy_openblas32($(SO_DIR)), 缺失 — 跳过(装上再 make zlayer)"; exit 1
else
ifeq ($(CUDA),1)
$(CAL)/zlayer_gpu.o: $(CAL)/zlayer_gpu.cu
	$(NVCC) -O3 -arch=native -c $< -o $@
$(CAL)/zlayer: $(CAL)/zlayer.c $(CAL)/zlayer_gpu.o ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS $(BLAS_CFLAGS) -DZL_CUDA -o $@ $(CAL)/zlayer.c \
	    $(CAL)/zlayer_gpu.o $(BLAS_LIBS) -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ -lm -lpthread
else
$(CAL)/zlayer: $(CAL)/zlayer.c ../src/common/ds4_quantfmt.c ../src/common/ds4_gguf.c
	$(CC) $(BARE) $(NATIVE) -DDQ_BLAS $(BLAS_CFLAGS) -o $@ $(CAL)/zlayer.c $(BLAS_LIBS) -lm -lpthread
endif
endif
endif

# amp_solve(换代解算器, capnpy 口径)与 zchain_merge/ge_solve: 产物落 gguf-tools 根
# (vqpipe/vqge/amp_campaign 按 $GT/<name> 找)。
amp_solve: amp_solve.bin
.PHONY: amp_solve.bin
amp_solve.bin: $(CAL)/amp_solve.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -o amp_solve $(CAL)/amp_solve.c $(CAL)/npy.c -lm -lpthread

zchain_merge: zchain_merge.bin
.PHONY: zchain_merge.bin
zchain_merge.bin: $(CAL)/zchain_merge.c
	$(CC) $(BARE) $(NATIVE) -o zchain_merge $< -lm -lpthread

ge_solve: ge_solve.bin
.PHONY: ge_solve.bin
ge_solve.bin: $(CAL)/ge_solve.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -I.. -o ge_solve $(CAL)/ge_solve.c $(CAL)/npy.c -lm -lpthread

zrec_to_zchain: $(CAL)/zrec_to_zchain
$(CAL)/zrec_to_zchain: $(CAL)/zrec_to_zchain.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
dql_to_zchain: $(CAL)/dql_to_zchain
$(CAL)/dql_to_zchain: $(CAL)/dql_to_zchain.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
route_alpha_set: $(CAL)/route_alpha_set
$(CAL)/route_alpha_set: $(CAL)/route_alpha_set.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
route_bias_rebake: $(CAL)/route_bias_rebake
$(CAL)/route_bias_rebake: $(CAL)/route_bias_rebake.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread

emit_z: emit_z.bin
.PHONY: emit_z.bin
emit_z.bin: $(LAT)/emit_z.c
	$(CC) -O3 -Wall -std=c99 -o emit_z $<

teacher_routed: teacher_routed.bin
.PHONY: teacher_routed.bin
teacher_routed.bin: teacher_routed.c $(CAL)/npy.c $(CAL)/layer_probe.c
	$(CC) $(BARE) $(NATIVE) -I. -I$(CAL) -I$(QNT) -o teacher_routed teacher_routed.c \
	    $(CAL)/npy.c $(CAL)/layer_probe.c -lm -lpthread

hsolve: $(CAL)/hsolve
$(CAL)/hsolve: $(CAL)/hsolve.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -o $@ $(CAL)/hsolve.c $(CAL)/npy.c -lm -lpthread
hbase: $(CAL)/hbase
$(CAL)/hbase: $(CAL)/hbase.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -o $@ $(CAL)/hbase.c $(CAL)/npy.c -lm -lpthread
dilution: $(CAL)/dilution
$(CAL)/dilution: $(CAL)/dilution.c $(CAL)/npy.c
	$(CC) $(BARE) $(NATIVE) -I. -I$(CAL) -o $@ $(CAL)/dilution.c $(CAL)/npy.c -lm -lpthread

# z 探针(z_tune/z_explore*): #include 仓库根 ds4_z.c/ds4_loss.c + quant 目录 st_read/fwd。
z_tune: $(QNT)/z_tune
$(QNT)/z_tune: $(QNT)/z_tune.c
	$(CC) $(BARE) $(NATIVE) -I.. -o $@ $< $(BLAS_LIBS) -lm -lpthread
z_explore: $(QNT)/z_explore
$(QNT)/z_explore: $(QNT)/z_explore.c
	$(CC) $(BARE) $(NATIVE) -I.. -o $@ $< $(BLAS_LIBS) -lm -lpthread
z_explore2: $(QNT)/z_explore2
$(QNT)/z_explore2: $(QNT)/z_explore2.c
	$(CC) $(BARE) $(NATIVE) -I.. -o $@ $< $(BLAS_LIBS) -lm -lpthread

# ---------------------------------------------------------------- legacy(上一代口径)
# amp_solve_zc: -ffp-contract=off 是正确性要求, 不是可选项(FMA 合并少一次舍入会真改数,
# 见文件头)。产线务必带 BLAS(无 BLAS 慢 15 倍但数值相同)。
amp_solve_zc: $(CAL)/amp_solve_zc
ifeq ($(UNAME_S),Darwin)
$(CAL)/amp_solve_zc: $(CAL)/amp_solve_zc.c
	$(CC) $(BARE) $(NATIVE) -ffp-contract=off -DDQ_BLAS -DACCELERATE_NEW_LAPACK \
	    -o $@ $< $(BLAS_LIBS) -lm -lpthread
else
ifeq ($(HAVE_BLAS),1)
$(CAL)/amp_solve_zc: $(CAL)/amp_solve_zc.c
	$(CC) $(BARE) $(NATIVE) -ffp-contract=off -DDQ_BLAS $(BLAS_CFLAGS) -o $@ $< $(BLAS_LIBS) -lm -lpthread
else
$(CAL)/amp_solve_zc: $(CAL)/amp_solve_zc.c
	$(CC) $(BARE) $(NATIVE) -ffp-contract=off -o $@ $< -lm -lpthread
endif
endif

emit_residual: emit_residual.bin
.PHONY: emit_residual.bin
emit_residual.bin: $(LEG)/emit_residual.c $(CAL)/hf_read.c $(QNT)/onebit_quant.c $(CAL)/npy.c
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o emit_residual $^ -lm -pthread

corr_stats: $(LEG)/corr_stats
$(LEG)/corr_stats: $(LEG)/corr_stats.c
	$(CC) $(BARE) $(NATIVE) $(INC) -o $@ $< -lm -lpthread
lut_cov: $(LEG)/lut_cov
$(LEG)/lut_cov: $(LEG)/lut_cov.c
	$(CC) $(BARE) $(NATIVE) $(INC) -o $@ $< -lm -lpthread
merge_residual: $(LEG)/merge_residual
$(LEG)/merge_residual: $(LEG)/merge_residual.c
	$(CC) $(BARE) $(NATIVE) $(INC) -o $@ $< -lm -lpthread
min_go_check: $(CAL)/min_go_check
$(CAL)/min_go_check: $(CAL)/min_go_check.c
	$(CC) $(BARE) $(NATIVE) $(INC) -o $@ $< -lm -lpthread

# ---------------------------------------------------------------- calib(校准/验证)
calib_run: calib_run.bin
.PHONY: calib_run.bin
CALIB_RUN_SRC := $(CAL)/calib_run.c $(CAL)/npy.c $(CAL)/hf_read.c $(QNT)/onebit_quant.c \
    $(CAL)/layer_probe.c $(LAT)/hiddenvar_solve.c $(LEG)/solve_perexpert.c $(LEG)/solve_denoise.c \
    $(LEG)/solve_genrf.c $(LAT)/solve_rrr.c $(LAT)/linalg_small.c
calib_run.bin: $(CALIB_RUN_SRC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o calib_run $(CALIB_RUN_SRC) -lm -pthread

calib_diag: calib_diag.bin
.PHONY: calib_diag.bin
CALIB_DIAG_SRC := $(CAL)/calib_diag.c $(CAL)/npy.c $(CAL)/hf_read.c $(QNT)/onebit_quant.c \
    $(CAL)/layer_probe.c $(LAT)/hiddenvar_solve.c $(LAT)/linalg_small.c
calib_diag.bin: $(CALIB_DIAG_SRC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o calib_diag $(CALIB_DIAG_SRC) -lm -pthread

validate_fwd: $(CAL)/validate_fwd
VFWD_SRC := $(CAL)/validate_fwd.c $(CAL)/npy.c $(CAL)/hf_read.c $(CAL)/layer_probe.c \
    $(QNT)/onebit_quant.c $(LEG)/solve_denoise.c $(LEG)/solve_genrf.c $(LAT)/linalg_small.c
$(CAL)/validate_fwd: $(VFWD_SRC)
	$(CC) $(CFLAGS) $(NATIVE) $(CPPFLAGS) $(INC) -o $@ $(VFWD_SRC) -lm -pthread

traj_analyze: $(CAL)/traj_analyze
$(CAL)/traj_analyze: $(CAL)/traj_analyze.c $(CAL)/npy.c $(LAT)/linalg_small.c
	$(CC) $(BARE) $(NATIVE) -I$(CAL) -I$(LAT) -o $@ $^ -lm -lpthread

cap_raw2npy: cap_raw2npy.bin
.PHONY: cap_raw2npy.bin
cap_raw2npy.bin: $(CAL)/cap_raw2npy.c
	$(CC) $(BARE) $(NATIVE) -o cap_raw2npy $< -lm -lpthread

vq_blob_truesize: $(CAL)/vq_blob_truesize
$(CAL)/vq_blob_truesize: $(CAL)/vq_blob_truesize.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread

dump_gguf_meta: dump_gguf_meta.bin
.PHONY: dump_gguf_meta.bin
dump_gguf_meta.bin: dump_gguf_meta.c
	$(CC) $(BARE) $(NATIVE) -o dump_gguf_meta $< -lm

# ---------------------------------------------------------------- bench(评测/取证)
kl_forensic: $(CAL)/kl_forensic
$(CAL)/kl_forensic: $(CAL)/kl_forensic.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
trace_ladder: $(CAL)/trace_ladder
$(CAL)/trace_ladder: $(CAL)/trace_ladder.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread
rec_fidelity: $(CAL)/rec_fidelity
$(CAL)/rec_fidelity: $(CAL)/rec_fidelity.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm -lpthread

pubbench: $(CAL)/pubbench
$(CAL)/pubbench: $(CAL)/pubbench.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< $(CURL_FLAGS) -lz -lm -lpthread

pubbench_extract_test: $(MIG)/pubbench_extract_test
$(MIG)/pubbench_extract_test: $(MIG)/pubbench_extract_test.c $(CAL)/pubbench.c
	$(CC) $(BARE) -o $@ $(MIG)/pubbench_extract_test.c $(CURL_FLAGS) -lz -lm -lpthread

vq_merge_v4: $(QNT)/vq_merge_v4
$(QNT)/vq_merge_v4: $(QNT)/vq_merge_v4.c
	$(CC) $(BARE) $(NATIVE) -o $@ $< -lm

go2b_parity: $(QNT)/go2b_parity
$(QNT)/go2b_parity: $(QNT)/go2b_parity.c $(QNT)/onebit_quant.h $(QNT)/go2b_qc.h
	$(CC) $(BARE) $(NATIVE) -ffast-math -I$(QNT) -o $@ $(QNT)/go2b_parity.c $(BLAS_LIBS) -lm -lpthread

vq2q2k_probe: $(QNT)/vq2q2k_probe
$(QNT)/vq2q2k_probe: $(QNT)/vq2q2k_probe.c quants.c $(QNT)/onebit_quant.c
	$(CC) $(BARE) $(NATIVE) -I. -o $@ $(QNT)/vq2q2k_probe.c quants.c $(QNT)/onebit_quant.c -lm -lpthread

# vq_em: GPU-EM 码本精修, Linux+CUDA 专属(st_bridge.c 桥接 st_read)。
ifeq ($(UNAME_S)-$(CUDA),Linux-1)
vq_em: vq_em.bin
.PHONY: vq_em.bin
vq_em.bin: $(CAL)/vq_em.cu $(CAL)/st_bridge.c
	$(NVCC) -O3 -arch=native -c $(CAL)/vq_em.cu -o $(CAL)/vq_em.o
	$(CC) $(BARE) $(NATIVE) -c $(CAL)/st_bridge.c -o $(CAL)/st_bridge.o
	$(CC) $(BARE) -o vq_em $(CAL)/vq_em.o $(CAL)/st_bridge.o $(CUDA_LIBS) -lcublas -lm -lpthread
else
vq_em:
	@echo "vq_em: Linux+CUDA 专属(GPU-EM), 本平台跳过"
endif

# ---------------------------------------------------------------- 自测(9 项, 纯 C 无模型)
tools-test:
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DGO1B_TEST    $(QNT)/onebit_quant.c -o $(BUILD)/onebit_test -lm && $(BUILD)/onebit_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DLINALG_TEST  $(LAT)/linalg_small.c -o $(BUILD)/linalg_test -lm && $(BUILD)/linalg_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DPROBE_TEST   $(CAL)/layer_probe.c  -o $(BUILD)/probe_test  -lpthread -lm && $(BUILD)/probe_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DHVSOLVE_TEST $(LAT)/hiddenvar_solve.c $(LAT)/linalg_small.c -o $(BUILD)/hv_test -lm && $(BUILD)/hv_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DNPY_TEST     $(CAL)/npy.c -o $(BUILD)/npy_test -lm && $(BUILD)/npy_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DHFREAD_TEST  $(CAL)/hf_read.c -o $(BUILD)/hfread_test -lm && $(BUILD)/hfread_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DPEREXPERT_TEST $(LEG)/solve_perexpert.c $(LAT)/linalg_small.c -o $(BUILD)/perexpert_test -lm && $(BUILD)/perexpert_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DCALIBDIAG_TEST $(CAL)/calib_diag.c $(LAT)/hiddenvar_solve.c $(LAT)/linalg_small.c -o $(BUILD)/calibdiag_test -lm -pthread && $(BUILD)/calibdiag_test
	$(CC) $(CFLAGS) $(NATIVE) $(INC) -DRRR_TEST     $(LAT)/solve_rrr.c $(LAT)/hiddenvar_solve.c $(LAT)/linalg_small.c -o $(BUILD)/rrr_test -lm -pthread && $(BUILD)/rrr_test

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
	rm -f deepseek4-quantize deepseek4-quantize-cpu quantize_gpu.o calib_run calib_diag emit_z \
	    emit_residual amp_solve zchain_merge ge_solve teacher_routed cap_raw2npy dump_gguf_meta vq_em \
	    $(QT)/score_official $(QT)/compare_scores \
	    $(CAL)/anchor_metrics $(CAL)/zlayer $(CAL)/zlayer_gpu.o $(CAL)/zrec_to_zchain \
	    $(CAL)/dql_to_zchain $(CAL)/route_alpha_set $(CAL)/route_bias_rebake $(CAL)/hsolve \
	    $(CAL)/hbase $(CAL)/dilution $(CAL)/amp_solve_zc $(CAL)/kl_forensic $(CAL)/trace_ladder \
	    $(CAL)/rec_fidelity $(CAL)/pubbench $(CAL)/vq_blob_truesize $(CAL)/validate_fwd \
	    $(CAL)/traj_analyze $(CAL)/min_go_check $(CAL)/vq_em.o $(CAL)/st_bridge.o \
	    $(QNT)/ds4quant_run $(QNT)/vq_gpu.o $(QNT)/vq_merge_v4 $(QNT)/go2b_parity \
	    $(QNT)/vq2q2k_probe $(QNT)/z_tune $(QNT)/z_explore $(QNT)/z_explore2 \
	    $(LEG)/corr_stats $(LEG)/lut_cov $(LEG)/merge_residual $(MIG)/pubbench_extract_test
	rm -rf $(BUILD)
