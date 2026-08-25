#!/bin/bash
# build_quant_spark.sh — 转调壳(2026-08-25 批1): ds4quant_run 构建收进 gguf-tools/Makefile
# (scipy_openblas32 改名/CUDA 探测/cuBLAS 都在那边)。用法不变: bash build_quant_spark.sh [nocuda]
set -euo pipefail
GT="$(cd "$(dirname "$0")/../.." && pwd)/gguf-tools"
if [ "${1:-cuda}" = "nocuda" ]; then
    exec make -C "$GT" ds4quant_run CUDA=0
fi
exec make -C "$GT" ds4quant_run
