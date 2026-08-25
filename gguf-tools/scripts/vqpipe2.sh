#!/bin/bash
# vqpipe2.sh — VQ86 放大器全链 v3(2026-08-23): 压到半小时的并行重排。
# 时间线: [取料#2(GPU)] ∥ [npy+教师双并发(CPU)] → 复现闸 → 解算(26s/层)流水 → 合链 → 终判。
# 教师=靶子生产非拟合动作, 提前到复现闸前(若不复现→产物整体作废重来, cmp 挡在解算前);
# 解算(拟合)严格等 cmp 过。全链零 Python。
set -uo pipefail
ROOT="$HOME/ds4-main"
D2="$ROOT/gguf/go-onebit/vqhalf"
GT="$ROOT/gguf-tools"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
M="$ROOT/gguf/ds4-vq86h.gguf"
LOG(){ echo "[vqpipe2 $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

( while true; do
    A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "${A:-99}" -lt 4 ] && { echo "[watchdog] MemAvailable=${A}GB <4GB ★杀★" >&2
        pkill -9 -f 'ds4 --cuda'; pkill -9 -f amp_solve; pkill -9 -f teacher_routed; break; }
    sleep 5
  done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

[ -f "$D2/cap_a/.done" ] || DIE "取料#1 未完成(先让 vqpipe v2 跑完①)"
cd "$ROOT"

# ── 并行段: 取料#2(GPU) ∥ npy + 教师双并发(CPU) ──
# cap_a/.verified = 复现闸已过的终态标记(cmp 后落), 在则跳过整个复现遍
if [ -f "$D2/cap_a/.verified" ]; then GPU_PID=0
elif [ ! -f "$D2/cap_a2/.done" ]; then
    rm -rf "$D2/cap_a2"; mkdir -p "$D2/cap_a2"
    LOG "取料#2 复现遍(GPU) 发车"
    ( timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$D2/vqhalf_a.ids" \
        --score-out /tmp/vq86h_capa2.bin --cap-dir "$D2/cap_a2" </dev/null 2>&1 | tail -1
      touch "$D2/cap_a2/.done" ) &
    GPU_PID=$!
else GPU_PID=0; fi

mkdir -p "$D2/capnpy_a" "$D2/amp_c"
[ -s "$D2/capnpy_a/ffn_in_L42.npy" ] || \
    "$GT/cap_raw2npy" --raw "$D2/cap_a" --out "$D2/capnpy_a" --ntok 8192 || DIE "npy 失败"

LOG "教师双并发发车(偶/奇层, 各 9 线程)"
( for L in $(seq 0 2 42); do
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] && continue
    "$GT/teacher_routed" --hf "$HF" --cap "$D2/capnpy_a" --layers $L-$L \
        --ntok 8192 --threads 9 --swlim 60 || exit 1
  done ) & TE_PID=$!
( for L in $(seq 1 2 42); do
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] && continue
    "$GT/teacher_routed" --hf "$HF" --cap "$D2/capnpy_a" --layers $L-$L \
        --ntok 8192 --threads 9 --swlim 60 || exit 1
  done ) & TO_PID=$!

# ── 复现闸(解算前的硬线; .verified 在则已过) ──
if [ ! -f "$D2/cap_a/.verified" ]; then
    [ "$GPU_PID" != 0 ] && { wait "$GPU_PID" || true; }
    [ -s "$D2/cap_a2/raw_ffn_in_L42" ] || DIE "取料#2 没落盘"
    for L in 0 16 32 42; do
        cmp "$D2/cap_a/raw_ffn_in_L$L" "$D2/cap_a2/raw_ffn_in_L$L" || DIE "L$L ffn_in 不复现, 产物作废"
        cmp "$D2/cap_a/raw_route_L$L"  "$D2/cap_a2/raw_route_L$L"  || DIE "L$L route 不复现, 产物作废"
    done
    LOG "复现 ✓ (L0/16/32/42 逐位一致)"
    touch "$D2/cap_a/.verified"
    rm -rf "$D2/cap_a2"
fi

# ── 解算流水: 教师产物就绪即解(26s/层) ──
for L in $(seq 0 42); do
    [ -s "$D2/amp_c/zrec_L$(printf %02d $L).bin" ] && continue
    for i in $(seq 1 360); do [ -s "$D2/capnpy_a/routed_L$L.npy" ] && break; sleep 2; done
    [ -s "$D2/capnpy_a/routed_L$L.npy" ] || DIE "教师 L$L 超时"
    "$GT/amp_solve" --cap "$D2/capnpy_a" --out "$D2/amp_c" --layers $L --threads 12 || DIE "解算 L$L 失败"
done
wait "$TE_PID" "$TO_PID" 2>/dev/null || true
N=$(ls "$D2/amp_c"/zrec_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || DIE "解算不齐 $N/43"
LOG "教师+解算 43/43 收官"

# ── 合链(C) + 终判(C) ──
"$GT/zchain_merge" "$D2/amp_c" "$D2/zchain_vq86h.bin" 43 || DIE "合链失败"
LOG "终判 wt2: vq86h+放大器"
timeout --foreground 3000 ./ds4 --cuda -m "$M" --zchain "$D2/zchain_vq86h.bin" \
    --score-ids "$G7/wt2.ids" --score-out /tmp/vq86h_amp_wt2.bin </dev/null 2>&1 | tail -1
echo "══ wt2 五指标 vq86h+放大器 (对表: 裸 4.1380 / base86p 4.1222) ══"
"$GT/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student /tmp/vq86h_amp_wt2.bin --tail 3 2>&1 | head -14
LOG "vqpipe2 收官"
