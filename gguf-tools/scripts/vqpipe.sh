#!/bin/bash
# vqpipe.sh — VQ86 放大器全链(v2, 2026-08-23): 取料→复现闸→教师∥解算流水线→合链→终判。
# ★链上零 Python(铁律): cap_raw2npy / teacher_routed / amp_solve / zchain_merge /
#   anchor_metrics 全 C。★复现闸先于一切拟合(铁律: 捕获即验, 不复现不许拿去拟合)。
# 并行边界: 取料#2(GPU) ∥ npy 转换(CPU, 纯格式转换非拟合); 教师/解算等 cmp 过后起跑,
#   教师出一层解算跟一层(限 1 并发)。
set -uo pipefail
ROOT="$HOME/ds4-main"
D2="$ROOT/gguf/go-onebit/vqhalf"
GT="$ROOT/gguf-tools"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
M="$ROOT/gguf/ds4-vq86h.gguf"
LOG(){ echo "[vqpipe $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

WD=""
watchdog_start(){
    ( while true; do
        A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "${A:-99}" -lt 4 ] && { echo "[watchdog] MemAvailable=${A}GB <4GB ★杀本段★" >&2
            pkill -9 -f 'ds4 --cuda'; pkill -9 -f amp_solve; pkill -9 -f teacher_routed; break; }
        sleep 5
      done ) & WD=$!
}
watchdog_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }

[ -s "$M" ] || DIE "vq86h.gguf 缺"
[ -s "$D2/vqhalf_a.ids" ] || DIE "放大器半 ids 缺"
watchdog_start

# ① 取料#1(解码路=部署同路)
if [ ! -s "$D2/cap_a/.done" ]; then
    LOG "①取料#1 解码路 on vq86h × 放大器半"
    rm -rf "$D2/cap_a" "$D2/cap_a2" "$D2/capnpy_a"; mkdir -p "$D2/cap_a" "$D2/cap_a2"
    timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$D2/vqhalf_a.ids" \
        --score-out /tmp/vq86h_capa.bin --cap-dir "$D2/cap_a" </dev/null 2>&1 | tail -1
    [ -s "$D2/cap_a/raw_ffn_in_L42" ] || DIE "取料#1 没落盘"
    touch "$D2/cap_a/.done"
fi
cd "$ROOT"

# ② 取料#2(GPU) ∥ npy 转换(CPU, 纯格式转换)
if [ ! -s "$D2/cap_a2/.done" ]; then
    LOG "②取料#2 复现遍(GPU) ∥ npy 转换(CPU)"
    ( timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$D2/vqhalf_a.ids" \
        --score-out /tmp/vq86h_capa2.bin --cap-dir "$D2/cap_a2" </dev/null 2>&1 | tail -1
      touch "$D2/cap_a2/.done" ) &
    GPU_PID=$!
    mkdir -p "$D2/capnpy_a"
    "$GT/cap_raw2npy" --raw "$D2/cap_a" --out "$D2/capnpy_a" --ntok 8192 || DIE "npy 失败"
    wait "$GPU_PID"
fi
[ -s "$D2/cap_a2/raw_ffn_in_L42" ] || DIE "取料#2 没落盘"

# ③ 复现闸(先于一切拟合): 浅/中/深三档 ffn_in+route 逐位比对
for L in 0 16 32 42; do
    cmp "$D2/cap_a/raw_ffn_in_L$L" "$D2/cap_a2/raw_ffn_in_L$L" || DIE "L$L ffn_in 不复现, 产物作废"
    cmp "$D2/cap_a/raw_route_L$L"  "$D2/cap_a2/raw_route_L$L"  || DIE "L$L route 不复现, 产物作废"
done
LOG "③复现 ✓ (L0/16/32/42 逐位一致), 清检查遍"
rm -rf "$D2/cap_a2"

# ④ 教师→解算流水线(全 C; 教师 14 线程, 解算 14 线程, 限 1 并发跟层)
mkdir -p "$D2/amp_c"
SOLVE_PID=0
for L in $(seq 0 42); do
    if [ ! -s "$D2/capnpy_a/routed_L$L.npy" ]; then
        "$GT/teacher_routed" --hf "$HF" --cap "$D2/capnpy_a" \
            --layers $L-$L --ntok 8192 --threads 14 --swlim 60 || DIE "教师 L$L 失败"
    fi
    [ "$SOLVE_PID" != 0 ] && { wait "$SOLVE_PID" || DIE "解算失败(前层)"; }
    "$GT/amp_solve" --cap "$D2/capnpy_a" --out "$D2/amp_c" \
        --layers $L --threads 14 & SOLVE_PID=$!
done
wait "$SOLVE_PID" || DIE "解算失败(末层)"
N=$(ls "$D2/amp_c"/zrec_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || DIE "解算不齐 $N/43"
LOG "④教师+解算 43/43 收官"

# ⑤ 合链(C) + 终判(C 判决器)
"$GT/zchain_merge" "$D2/amp_c" "$D2/zchain_vq86h.bin" 43 || DIE "合链失败"
LOG "⑤终判 wt2: vq86h+放大器"
timeout --foreground 3000 ./ds4 --cuda -m "$M" --zchain "$D2/zchain_vq86h.bin" \
    --score-ids "$G7/wt2.ids" --score-out /tmp/vq86h_amp_wt2.bin </dev/null 2>&1 | tail -1
echo "══ wt2 五指标 vq86h+放大器 (对表: 裸 4.1380 / base86p 4.1222) ══"
"$GT/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student /tmp/vq86h_amp_wt2.bin --tail 3 2>&1 | head -14
watchdog_stop
LOG "vqpipe 收官"
