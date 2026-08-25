#!/bin/bash
# vqge.sh — VQ86 GE 全层反修(2026-08-23): FP 锚教师 → GE 每专家门控(C) → 合链 → wt2 还原率判决。
# 判决主尺 = 分布还原率 Σmin(还原率铁律), 对标: 裸 vq86h 0.7306, 目标 ≥0.90。
# 单层验证已过(L20 GE held 18% 真泛化, cos 转正); z^L 在本底座 held 负, 弃。
set -uo pipefail
ROOT="$HOME/ds4-main"
D2="$ROOT/gguf/go-onebit/vqhalf"
GT="$ROOT/gguf-tools"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
LOG(){ echo "[vqge $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "${A:-99}" -lt 4 ] && { echo "[watchdog] ★杀★" >&2; pkill -9 -f 'ds4 --cuda'; pkill -9 -f ge_solve; pkill -9 -f teacher_routed; break; }
    sleep 5
  done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
mkdir -p "$D2/ge_c"

# 教师双并发(锚口径, 跳过已有) ∥ GE 流水(教师就绪即解)
( for L in $(seq 0 2 42); do
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] && continue
    "$GT/teacher_routed" --hf "$HF" --cap "$D2/capnpy_a" --anchor "$D2/anchor_vqhalf_a_s8192.bin" \
        --layers $L-$L --ntok 8192 --threads 9 --swlim 10 || exit 1
  done ) & TE=$!
( for L in $(seq 1 2 42); do
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] && continue
    "$GT/teacher_routed" --hf "$HF" --cap "$D2/capnpy_a" --anchor "$D2/anchor_vqhalf_a_s8192.bin" \
        --layers $L-$L --ntok 8192 --threads 9 --swlim 10 || exit 1
  done ) & TO=$!
for L in $(seq 0 42); do
    [ -s "$D2/ge_c/zrec_L$(printf %02d $L).bin" ] && continue
    for i in $(seq 1 720); do [ -s "$D2/capnpy_a/routed_L$L.npy" ] && break; sleep 2; done
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] || DIE "教师 L$L 超时"
    "$GT/ge_solve" --cap "$D2/capnpy_a" --dql "$D2/vq86h/layers" --layers $L \
        --out "$D2/ge_c" --threads 14 || DIE "GE L$L 失败"
done
wait "$TE" "$TO" 2>/dev/null || true
N=$(ls "$D2/ge_c"/zrec_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || DIE "GE 不齐 $N/43"
LOG "GE 43/43 收官"

"$GT/zchain_merge" "$D2/ge_c" "$D2/zchain_ge.bin" 43 || DIE "合链失败"
cd "$ROOT"
LOG "终判 wt2: vq86h+GE (主尺 Σmin, 对标裸 0.7306 / 目标 0.90)"
timeout --foreground 3000 ./ds4 --cuda -m "$ROOT/gguf/ds4-vq86h.gguf" --zchain "$D2/zchain_ge.bin" \
    --score-ids "$G7/wt2.ids" --score-out /tmp/vq86h_ge_wt2.bin </dev/null 2>&1 | tail -1
"$GT/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student /tmp/vq86h_ge_wt2.bin --tail 3 2>&1 | head -14
LOG "vqge 收官"
