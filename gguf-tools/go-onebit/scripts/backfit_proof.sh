#!/bin/bash
# backfit_proof.sh — 铁证: 全局回扫的"反修前层"真的【原地改写前层 z 载荷】, 不是往文件尾巴粘标量。
#
# 做法(单线程=确定性): 同一 2 层配方跑两遍 —
#   A) DS4_GSWEEP 未设 → 只前向, 写层文件, 不反修           → $DA
#   B) DS4_GSWEEP=1    → 前向 + backfit_layer_z 用最终KL重解z → $DB
# 前向确定 → A/B 反修前字节完全一致; 差异 = 纯 backfit 增量。
# 判据:
#   · cmp 若相同 → 本 tiny 配方无最终KL改善(反修永不劣化, 看 "z保持" 行, 机制仍在跑)。
#   · cmp 若不同 且 同大小 → ✓ 原地改写 z 系数(旧逻辑会让文件"变长"=追加标量, 是伪反修)。
# 用法: ./scripts/backfit_proof.sh [ids=/tmp/rr_code.ids] [ntok=24]
set -euo pipefail
cd "$(dirname "$0")/../quant"
../scripts/quant_verify.sh build
HF="${DS4_HF:-/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base}"
IDS="${1:-/tmp/rr_code.ids}"; NTOK="${2:-24}"
[ -f "$IDS" ] || { echo "缺 ids: $IDS (编程域校准语料)"; exit 1; }
ANCH="/tmp/ds4quant_anchor_proof_nl2_s${NTOK}.bin"
DA=/tmp/backfit_proof_A; DB=/tmp/backfit_proof_B
rm -rf "$DA" "$DB"; mkdir -p "$DA" "$DB"
# 单线程: 专家和的归约顺序固定 → 前向逐位可复现(否则 A/B 反修前就抖动, 污染判据)
COMMON="DS4_HF=$HF DS4_NL=2 DS4_LCFG=gg DS4_COADAPT=1 DS4_ANCHOR=$ANCH DS4_THREADS=1"
echo "[A] 无反修(GSWEEP 未设) → $DA"
( eval "export $COMMON DS4_LAYER_DIR=$DA"; exec ./ds4quant_run "$IDS" "$NTOK" ) \
    >/tmp/backfit_A.out 2>/tmp/backfit_A.log || { echo "A 失败:"; tail -15 /tmp/backfit_A.log; exit 1; }
echo "[B] 反修(GSWEEP=1) → $DB"
( eval "export $COMMON DS4_LAYER_DIR=$DB DS4_GSWEEP=1"; exec ./ds4quant_run "$IDS" "$NTOK" ) \
    >/tmp/backfit_B.out 2>/tmp/backfit_B.log || { echo "B 失败:"; tail -15 /tmp/backfit_B.log; exit 1; }
echo "===== 反修日志(B, 每前层以最终KL重解) ====="
grep -E "GSWEEP|✓改写层文件|z保持" /tmp/backfit_B.out || echo "(无 GSWEEP 行 — 检查 /tmp/backfit_B.log)"
for LF in dql_L00.bin dql_L01.bin; do
  [ -f "$DA/$LF" ] && [ -f "$DB/$LF" ] || continue
  echo "===== $LF : A(无反修) vs B(反修) ====="
  ls -la "$DA/$LF" "$DB/$LF"
  if cmp -s "$DA/$LF" "$DB/$LF"; then
    echo "  → 相同: 本层反修无最终KL改善即不改(永不劣化)。机制在跑, 见上 z保持 行。"
  else
    SA=$(stat -f%z "$DA/$LF"); SB=$(stat -f%z "$DB/$LF")
    if [ "$SA" = "$SB" ]; then echo "  → 不同 且 同大小 ($SA) = ✓★原地改写 z 系数, 非追加标量★"
    else echo "  → 不同 但 大小变了 (A=$SA B=$SB) — 异常, 不该发生"; fi
    echo "  首批差异字节(位置 A值 B值):"; cmp -l "$DA/$LF" "$DB/$LF" | head -6
  fi
done
echo "[提示] 这是 2 层 tiny 证明(证'文件真的被改'); 真实质量以 ./quant_layer.sh 全 43 层端到端为准。"
