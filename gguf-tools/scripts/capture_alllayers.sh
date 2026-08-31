#!/usr/bin/env bash
# capture_alllayers.sh — 全 43 层激活捕获 (杠杆① scale 重算用)。
# 短 prefill 一趟, 双机各存本机层 (M4=0:19, M1=20:output), harvest 合并到 M1
# (HF fp8 权重在 M1, scale 重算就地跑)。只需 raw_ffn_in(X) + raw_route(选专家)。
# 小盘: NTOK≈2048 × 43层 × 4096 × 2B ≈ 720MB, M1 7G 盘够。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main; DPORT=5599
MODEL=${MODEL:-gguf/ds4-mono-mixed.gguf}   # 2026-07-21: env 可覆盖(v2 捕获用)
# (env 大扫除 2026-08-31: EXPERT_OFFLOAD=1 归 AUTO 按 --mem-budget-mb 判定; GATHER_THREADS
#  写死 8; NO_MODEL_WARMUP 已删; PREFETCH_AHEAD=0 不再需要 — 捕获仪器武装时引擎自动关预取;
#  层过滤 env(原 DS4_CAP_LAYERS)已死 — 双机各自只算本片层, 捕获天然只落本片)
FLAGSTR="--reverse-connect --prefill-chunk 2048 --mem-budget-mb 12000"
COORD_FLAGS="--dist-prefill-cap 2048"   # 只 coordinator 侧认
# 校准 prompt: 默认 DSML seg0; PROMPT_FILE 可覆盖 (代码域校准用 gocode_calib.txt)
PROMPT_FILE=${PROMPT_FILE:-$ROOT/cap_dsml/seg0.txt}
PROMPT=$(cat "$PROMPT_FILE")
CAP=${CAP:-capall}   # CAP=capcode 代码域

ssh "$M1" "pkill -f 'role worker'; mkdir -p /tmp/${CAP} && rm -f /tmp/${CAP}/raw_* /tmp/ds4_worker_cap.log" 2>/dev/null; sleep 2
ssh "$M1" "( cd $M1DIR && nohup ./ds4 -m $MODEL $FLAGSTR --cap-dir /tmp/${CAP} --role worker --listen $M1 $DPORT --layers 20:output -c 4096 --temp 0 --nothink ) > /tmp/ds4_worker_cap.log 2>&1 < /dev/null & echo ok"
until ssh "$M1" "grep -q 'waiting for coordinator' /tmp/ds4_worker_cap.log" 2>/dev/null; do sleep 3; done
echo "[capall] worker 就绪, 起 coordinator (M4 存 L0-19)"
mkdir -p /tmp/${CAP}_m4 && rm -f /tmp/${CAP}_m4/raw_*
# 双机 12G 看门狗
( sleep 15; while pgrep -f "role coordinator" >/dev/null; do
    cp=$(pgrep -f "role coordinator"|head -1); kb=$(ps -o rss= -p "$cp" 2>/dev/null|tr -d ' '); g=$((${kb:-0}/1048576))
    rkb=$(ssh "$M1" "pgrep -f 'role worker'|head -1|xargs -I{} ps -o rss= -p {}" 2>/dev/null|tr -d ' '); rg=$((${rkb:-0}/1048576))
    [ "$g" -gt 12 ] || [ "$rg" -gt 12 ] && { echo "[capall] MEM-BREACH L=${g}G R=${rg}G"; kill "$cp"; ssh "$M1" "pkill -f 'role worker'"; break; }
    sleep 20; done ) & WD=$!
( cd "$ROOT" && ./ds4 -m "$MODEL" $FLAGSTR $COORD_FLAGS --cap-dir /tmp/${CAP}_m4 --role coordinator --coordinator "$M1" "$DPORT" --layers 0:19 -c 4096 -n 1 --temp 0 --nothink -p "$PROMPT" > /tmp/${CAP}_coord.out 2> /tmp/${CAP}_coord.log )
kill "$WD" 2>/dev/null; ssh "$M1" "pkill -f 'role worker'" 2>/dev/null; sleep 3
# harvest: M4 的 L0-19 送 M1, 与 M1 的 L20-42 合并
echo "[capall] harvest L0-19 (M4) → M1"
scp -q /tmp/${CAP}_m4/raw_ffn_in_L* /tmp/${CAP}_m4/raw_route_L* "$M1:/tmp/${CAP}/" 2>/dev/null
n4=$(ls /tmp/${CAP}_m4/raw_ffn_in_L* 2>/dev/null|wc -l|tr -d ' ')
nall=$(ssh "$M1" "ls /tmp/${CAP}/raw_ffn_in_L* 2>/dev/null|wc -l"|tr -d ' ')
echo "[capall] 完成: M4 存 $n4 层, M1 合计 $nall 层 (期望 43)"
