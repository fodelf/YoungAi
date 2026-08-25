#!/bin/bash
# build_quant_spark.sh — spark(GB10/Linux aarch64) 量化器编译(固化, 之前只活在会话里)。
# BLAS=scipy-openblas32(符号 scipy_ 前缀, -D 改名); CUDA=1 加 cuBLAS 大 GEMM 直传(统一内存)。
# 用法: bash build_quant_spark.sh [nocuda]
set -euo pipefail
cd "$(dirname "$0")/../quant"
SO_DIR="$HOME/.local/lib/python3.12/site-packages/scipy_openblas32/lib"
[ -f "$SO_DIR/libscipy_openblas.so" ] || { echo "scipy_openblas32 缺"; exit 1; }
CUDA_FLAGS=""
VQGPU_OBJ=""
if [ "${1:-cuda}" != "nocuda" ]; then
    /usr/local/cuda/bin/nvcc -O3 -arch=native -c vq_gpu.cu -o vq_gpu.o
    VQGPU_OBJ="vq_gpu.o"
    CUDA_FLAGS="-DDS4QUANT_CUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcudart -lstdc++"
fi
gcc -O3 -march=native -Wall -Wno-unused-parameter -Wno-unused-function \
    -DDQ_BLAS -DDS4QUANT_OPENBLAS \
    -Dcblas_sgemm=scipy_cblas_sgemm -Dcblas_sgemv=scipy_cblas_sgemv -Dcblas_dsdot=scipy_cblas_dsdot \
    -I"$HOME/ds4-main" -I"$SO_DIR/../include" \
    -o ds4quant_run ds4quant_run.c $VQGPU_OBJ \
    $CUDA_FLAGS \
    "$SO_DIR/libscipy_openblas.so" -Wl,-rpath,"$SO_DIR" \
    -lm -lpthread
echo "build ✓ $(ls -la ds4quant_run | awk '{print $5}') B"
