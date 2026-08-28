#!/bin/bash
# merge_base86p.sh — base86p 冠军(KLD 0.4361) 合并 GGUF + zchain 侧车抽取。
# ★2026-08-20 参数化(c86 战役复用): 层目录/输出/骨架全 env 可覆盖, 默认=base86p 原语义。
#   M86_LAYERS=层目录  M86_MDL=输出gguf  M86_SKEL=骨架  M86_ZCH=dql内嵌z抽取目标(空=跳过,
#   INJ=2 外挂时代 dql 内嵌链不是产物)。幂等: M86_MDL 已在且非空则跳过合并。
set -uo pipefail
ROOT="$HOME/ds4-main"
R30="$ROOT/gguf/go-onebit/r30"
LAYERS="${M86_LAYERS:-$R30/base86p/layers}"
MDL="${M86_MDL:-$ROOT/gguf/ds4-base86p.gguf}"
SKEL="${M86_SKEL:-$R30/r30_skeleton.gguf}"
ZCH="${M86_ZCH-$R30/base86p/zchain_base86p.bin}"   # M86_ZCH= (空串)显式跳过抽取
LOG(){ echo "[merge86p $(date +%H:%M:%S)] $*"; }

cd "$ROOT"
if [ -s "$MDL" ]; then LOG "已在 $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}'), 跳过"; exit 0; fi
LOG "manifest 生成"
for L in $(seq 0 42); do
    F="$LAYERS/$(printf 'dql_vq_L%02d.bin' $L)"
    [ -f "$F" ] || { LOG "★层缺 $F★"; exit 2; }
    echo "$L $(stat -c %s "$F")"
done > "$LAYERS/manifest.txt"
if [ -n "$ZCH" ]; then
    LOG "zchain 抽取 → $ZCH"
    "$(dirname "$0")/../amp/dql_to_zchain" "$LAYERS" "$ZCH" 43 \
        || { LOG "★zchain 失败★"; exit 3; }
fi
FREE=$(df -BG --output=avail "$ROOT/gguf" | sed -n 2p | tr -dc 0-9)
[ "${FREE:-0}" -ge 95 ] || { LOG "★盘不足★"; exit 4; }
LOG "合并发车(全 VQ --no-down)"
"$(dirname "$0")/../quantize/vq_merge_v4" --merge \
    --skeleton "$SKEL" \
    --blob-sizes "$LAYERS/manifest.txt" --no-down \
    --dql-host 127.0.0.1 --dql-dir "$LAYERS" \
    --out "$MDL" || { LOG "★合并失败★"; exit 5; }
LOG "合并完: $(ls -la "$MDL" | awk '{printf "%.2f GB", $5/1e9}')"

# ★路由反修落地: α·Δb 烘进 blk.L.exp_probs_b.bias(冠军 r64 原样)★
# 2026-08-28 补: 反修段(DS4_ROUTE_BIAS_FIT)会产出 Δb, 但本脚本此前【零处理】—— Δb 算了
# 没人用, 等于路由反修白做。冠军是在合并时烘进去, 引擎侧零改动(只改选择分不动权重分)。
# α=2.5 是冠军定值(实测曲线 ≤1.0 阈下无效 / 1.5→80.3 / 2.0→81.6 / ★2.5→84.2★ / 3.0 过冲回落)。
# route_alpha_set 幂等: 首跑存裸态快照 <model>.bias0.bin, 之后每次从快照绝对重写, α 可来回扫。
RB="${M86_RB:-$(dirname "$LAYERS")/route_bias_r30.bin}"
if [ -s "$RB" ]; then
    # ★α 必须来自本底座实扫, 不能照搬冠军的 2.5★(2026-08-28 用户纠: 底座与语料都换了,
    # 峰位没理由不动; fable5 3300 前车之鉴=迁移版 Δb 直接搬净负)。rb_alpha.txt 由
    # amp_campaign.sh champrb 段扫出后写入; 缺文件时退回 2.5 并【显式告警】。
    RBA="${M86_RB_ALPHA:-}"
    if [ -z "$RBA" ]; then
        AF="$(dirname "$LAYERS")/rb_alpha.txt"
        if [ -s "$AF" ]; then RBA=$(tr -d " \n" < "$AF"); LOG "α=$RBA (本底座实扫, 来自 $AF)"
        else RBA=2.5; LOG "★警告: 未见 $AF, 退回冠军值 α=2.5 —— 该值是 r64 在别的底座/语料上扫的, 未必适配本底座★"; fi
    fi
    "$ROOT/gguf-tools/route_alpha_set" "$MDL" "$RB" "$RBA" \
        && LOG "路由偏置已烘: α=$RBA Δb=$(ls -l "$RB" | awk '{printf "%.1fKB", $5/1024}')" \
        || { LOG "★路由偏置烘焙失败★"; exit 6; }
else
    LOG "★Δb 不在($RB) — 路由反修未落地, 合并出的是无路由修正版★"
fi
