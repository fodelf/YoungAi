#!/bin/bash
# build_dsq_gpu_spark.sh — spark(GB10/Linux aarch64) 上把标量量化器 deepseek4-quantize
# 编成"IQ2_XXS 编码器走 CUDA"的版本(2026-08-19, 用户铁律"spark 重计算必须 GPU 化")。
#
# 数值契约: CPU 侧 gcc -std=c11 是 ISO 模式 → -ffp-contract=off(实测 asm 里 0 条 fmadd),
# 所以 nvcc 必须 -fmad=false 且不开 --use_fast_math, 否则 FMA 融合/快速除法会让
# GPU 结果与 CPU 差最后一个 ulp, 逐字节对拍必挂。别"顺手"加 --use_fast_math。
#
# 铁律: spark 上 nvcc 串行编译(禁并行 + 禁与大 CPU 任务并发, 历史内核崩溃在案)。
#
# 用法:
#   bash build_dsq_gpu_spark.sh          # 编 CUDA 版 -> deepseek4-quantize
#   bash build_dsq_gpu_spark.sh cpu      # 编纯 CPU 版 -> deepseek4-quantize-cpu(对拍参照)
#
# 运行期开关: DS4Q_GPU=0 关 GPU 回落 CPU; DS4Q_GPU_VERIFY=1 每张量再跑一遍 CPU 逐字节对拍;
#            DS4Q_GPU_STREAMS(默认4) / DS4Q_GPU_BLOCK(默认128)。
set -euo pipefail
cd "$(dirname "$0")/../../"          # -> gguf-tools/
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
G1="go-onebit"
INC="-I$G1/quant -I$G1/latent -I$G1/latent/legacy -I$G1/calib"
CFLAGS="-O3 -Wall -Wextra -std=c11 -march=native -D_GNU_SOURCE $INC"
SRC="deepseek4-quantize.c quants.c $G1/quant/onebit_quant.c"

if [ "${1:-cuda}" = "cpu" ]; then
    gcc $CFLAGS -o deepseek4-quantize-cpu $SRC -lm -pthread
    echo "build ✓ deepseek4-quantize-cpu (纯 CPU 参照) $(stat -c %s deepseek4-quantize-cpu) B"
    exit 0
fi

[ -x "$NVCC" ] || { echo "nvcc 缺: $NVCC"; exit 1; }
"$NVCC" -O3 -arch=native -fmad=false -c quantize_gpu.cu -o quantize_gpu.o
gcc $CFLAGS -DDS4Q_CUDA -o deepseek4-quantize $SRC quantize_gpu.o \
    -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/targets/sbsa-linux/lib" \
    -lcudart -lstdc++ -lm -pthread
echo "build ✓ deepseek4-quantize (IQ2_XXS on CUDA) $(stat -c %s deepseek4-quantize) B"
