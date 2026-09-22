#!/bin/bash
# build_dsq_gpu_spark.sh — 转调壳(2026-08-25 批1): deepseek4-quantize 构建收进
# gguf-tools/Makefile。数值契约(nvcc -fmad=false, 禁 --use_fast_math)也在那边, 别绕过。
# 用法不变: bash build_dsq_gpu_spark.sh        # CUDA 版(nvcc 缺时 Makefile 自动退 CPU)
#          bash build_dsq_gpu_spark.sh cpu    # 纯 CPU 参照 -> deepseek4-quantize-cpu
set -euo pipefail
GT="$(cd "$(dirname "$0")/../.." && pwd)/gguf-tools"
if [ "${1:-cuda}" = "cpu" ]; then
    exec make -C "$GT" deepseek4-quantize-cpu
fi
exec make -C "$GT" deepseek4-quantize
