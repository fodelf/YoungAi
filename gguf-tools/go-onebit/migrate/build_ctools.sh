#!/bin/bash
# build_ctools.sh — 迁移后 C 工具集中构建(2026-08-25)。用法: bash build_ctools.sh [工具名...]
# 默认构建全部。产物=calib/<name>(与源同目录, .gitignore 不入库)。
set -e
CAL="$(cd "$(dirname "$0")/../calib" && pwd)"
TOOLS="${*:-anchor_metrics kl_forensic trace_ladder rec_fidelity zrec_to_zchain dql_to_zchain zlayer vq_merge_v4 vq_blob_truesize route_bias_rebake route_alpha_set amp_solve_zc pubbench pubbench_extract_test}"
for t in $TOOLS; do
  case "$t" in
    zlayer)
      if [ "$(uname)" = Darwin ]; then
        gcc -O3 -march=native -DDQ_BLAS -o "$CAL/$t" "$CAL/$t.c" -framework Accelerate -lm -lpthread
      else
        SO_DIR="$HOME/.local/lib/python3.12/site-packages/scipy_openblas32/lib"
        gcc -O3 -march=native -DDQ_BLAS -Dcblas_sgemm=scipy_cblas_sgemm -Dcblas_dgemm=scipy_cblas_dgemm \
          -I"$SO_DIR/../include" -o "$CAL/$t" "$CAL/$t.c" "$SO_DIR/libscipy_openblas.so" \
          -Wl,-rpath,"$SO_DIR" -lm -lpthread
      fi ;;
    vq_merge_v4) gcc -O3 -march=native -o "$CAL/../quant/$t" "$CAL/../quant/$t.c" -lm ;;
    # pubbench 要 libcurl(HTTP)+zlib(数据集 .gz); 抽取器金标回归单独一个驱动
    pubbench) gcc -O3 -march=native -o "$CAL/$t" "$CAL/$t.c" -lcurl -lz -lm -lpthread ;;
    pubbench_extract_test)
      gcc -O3 -o "$CAL/../migrate/$t" "$CAL/../migrate/$t.c" -lcurl -lz -lm -lpthread
      echo "build ✓ $CAL/../migrate/$t"; continue ;;
    *) gcc -O3 -march=native -o "$CAL/$t" "$CAL/$t.c" -lm -lpthread ;;
  esac
  echo "build ✓ $CAL/$t"
done
