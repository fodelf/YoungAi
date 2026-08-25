#!/bin/bash
# build_ctools.sh — 转调壳(2026-08-25 批1): 构建逻辑已收进 gguf-tools/Makefile,
# 本脚本只做参数转发, 保住老调用方的入口。用法不变: bash build_ctools.sh [工具名...]
# 默认构建原默认清单。平台特判(Accelerate/scipy_openblas/CUDA/curl)都在 Makefile 里。
set -e
GT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${*:-anchor_metrics kl_forensic trace_ladder rec_fidelity zrec_to_zchain dql_to_zchain zlayer vq_merge_v4 vq_blob_truesize route_bias_rebake route_alpha_set amp_solve_zc pubbench pubbench_extract_test}"
exec make -C "$GT" $TOOLS
