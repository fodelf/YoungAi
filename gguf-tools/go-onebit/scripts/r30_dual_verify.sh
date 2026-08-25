#!/bin/bash
# r30_dual_verify.sh — ds4-r30(平行架构一遍反修版)双机层切片流水线验证(2026-08-04 用户令:
# 编程场景直接双机测代码, 否决通用语料对表)。
# 复用 tools/mtp_pipe_q2_speed.sh 全套安全闸(两机 12G 红线+RSS 看门狗+超时双杀)。
# 质量判据 = 默认 twoSum BOS 裸续写(正确输出=双重循环 twoSum + threeSum 自然续写);
# 速度判据 = coordinator 日志 prefill/gen t/s。
# 用法: ./r30_dual_verify.sh            # 冒烟(twoSum, NPRED=96)
#       PROMPT='...' NPRED=256 ./r30_dual_verify.sh   # 自定义编程 prompt
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_REL="gguf/go-onebit/ds4-r30.gguf"
[ -f "$ROOT/$MODEL_REL" ] || { echo "本机模型缺: $ROOT/$MODEL_REL(先从 M1 传)" >&2; exit 2; }
export MODEL="$MODEL_REL"
export CTX="${CTX:-4096}" NPRED="${NPRED:-96}" SEED="${SEED:-1}"
exec "$ROOT/tools/mtp_pipe_q2_speed.sh"
