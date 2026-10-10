#!/bin/bash
# release_e2e_spark.sh — 发布包全链路闭环复验(2026-10-10): 解压到一个不含源码的新目录 → 全路径执行 ds4-train → 模型页加载 → 聊天 → 上传料 → 训练
#   → 训完没自动挂 → 开关挂 ③ → 聊天 → 卸。全程只打页面接口(与浏览器同一套), 不碰任何脚本。
#   用法: release_e2e_spark.sh <包.tar.gz> <新目录> <已有的 gguf/hub 目录(硬链接进来, 代替 113 GB 下载)> [页面端口 8100] [料 jsonl]
#   ★会停生产的 ds4-train + ds4-server(模型只能装一份), 跑完按 gguf/serve_pick.txt 把生产装回来(8000)★。日志 /tmp/release_e2e_<时间>.log。
set -uo pipefail
PACK="${1:?包}"; NEW="${2:?新目录}"; HUB="${3:?已有 gguf/hub}"; PORT="${4:-8100}"; JS="${5:-$HOME/ds4-main/gguf-tools/data/posttrain/jsonl_smoke.jsonl}"
U="http://127.0.0.1:$PORT"; LOG="/tmp/release_e2e_$(date +%m%d%H%M).log"
C() { curl -s --noproxy '*' --max-time 900 "$@"; }
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "★$*★"; exit 1; }
ask() { local q="$1" body a; body=$(jq -cn --arg q "$q" '{model:"ds4",messages:[{role:"user",content:$q}],temperature:0,max_tokens:120,stream:false}')
        a=$(C -H 'Content-Type: application/json' -d "$body" "$U/v1/chat/completions" | jq -r '.choices[0].message.content // (.error|tostring) // .'); say "问: $q"; say "答: $a"; }
wait_mode() { local want="$1" tmo="$2" i m; for i in $(seq 1 "$tmo"); do m=$(C "$U/api/train/status" | jq -r .mode 2>/dev/null); [ "$m" = "$want" ] && return 0; sleep 5; done; return 1; }

say "日志 $LOG"
[ -s "$PACK" ] || die "没有包 $PACK"
[ -e "$NEW" ] && die "$NEW 已存在(要干净目录)"
mkdir -p "$NEW" && tar -xzf "$PACK" -C "$NEW" --strip-components=1 || die "解压失败"
say "解压: $(ls "$NEW" | tr '\n' ' ')"
say "包里的脚本: $(ls "$NEW/gguf-tools/scripts" | tr '\n' ' ')"
mkdir -p "$NEW/gguf" && cp -al "$HUB" "$NEW/gguf/hub" || die "硬链接 hub 失败"
say "hub(硬链接): $(ls "$NEW/gguf/hub" | tr '\n' ' ')"
# 停生产(模型只能装一份)
pkill -x ds4-train; pkill -x ds4-server; for i in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
pgrep -x ds4-server >/dev/null && die "生产 ds4-server 停不掉"
say "生产已停, MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
( cd /tmp && nohup "$NEW/ds4-train" --port "$PORT" > "$NEW/ds4-train.log" 2>&1 </dev/null & )
sleep 2
say "主进程: $(C "$U/api/train/status" | jq -c '{mode,root}')"
LST=$(C "$U/api/models/list"); GG=$(echo "$LST" | jq -r '.local[0].path // ""'); SC=$(echo "$LST" | jq -r '.local[0].sidecars[0] // ""')
[ -n "$GG" ] || die "模型页没列出本机模型: $(echo "$LST" | cut -c1-300)"
say "模型页: $GG + 侧车 $SC"
say "加载: $(C -H 'Content-Type: application/json' -d "{\"gguf\":\"$GG\",\"sidecar\":\"$SC\"}" "$U/api/models/load")"
wait_mode serving 150 || die "10 分钟没装好: $(tail -5 "$NEW/gguf/v41/posttrain/ui_logs/"serve_*.log 2>/dev/null | tail -3)"
say "装好: $(C "$U/api/models/list" | jq -c .loaded)"
ask "用一句话介绍一下你自己。"
NAME=$(basename "$JS")
say "上传: $(C -X POST --data-binary "@$JS" "$U/api/train/upload?kind=jsonl&name=$NAME")"
say "发车: $(C -H 'Content-Type: application/json' -d "{\"data\":\"gguf-tools/data/posttrain/$NAME\",\"epochs\":\"1\",\"layers\":\"0-39\",\"lr\":\"2e-4\",\"extra\":\"\"}" "$U/api/train/start")"
T0=$(date +%s)
until [ "$(C "$U/api/train/status" | jq -r .mode)" = serving ]; do
    sleep 30; s=$(C "$U/api/train/status"); say "$(( ($(date +%s)-T0)/60 ))min $(echo "$s" | jq -c '{mode,phase,run,mem_avail_mb}')"
    [ $(( $(date +%s)-T0 )) -gt 3000 ] && die "50 分钟没收工"
done
RUN=$(C "$U/api/train/runs" | jq -r '.[0].name'); say "趟 $RUN: $(C "$U/api/train/run?name=$RUN" | jq -c '{status,pick,step0,e:[.epochs[]|{n,eval,dpp:.gate.dpp,pass:.gate.pass}]}')"
say "作业日志尾: $(tail -4 "$NEW/gguf/v41/posttrain/ui_logs/"cycle_*.log | tr '\n' '|' | cut -c1-600)"
LST=$(C "$U/api/models/list"); LD=$(echo "$LST" | jq -c .loaded); say "训后装着: $LD"
[ "$(echo "$LD" | jq -r '.posttrain // ""')" = "" ] && say "训后 ③ 没自动挂 ✓" || say "★训后 ③ 自动挂了★"
ZC=$(echo "$LD" | jq -r '.zchain // ""'); PT=$(echo "$LST" | jq -r --arg z "$ZC" '(.fnv[$z]) as $w | [.posttrains[]|select(.fnv==$w)][0].path // ""')
if [ -n "$PT" ]; then
    say "开关挂 ③ $PT: $(C -H 'Content-Type: application/json' -d "{\"zchain\":\"$ZC\",\"posttrain\":\"$PT\"}" "$U/api/models/plugins")"; sleep 6
    say "挂后装着: $(C "$U/api/models/list" | jq -c .loaded)"
    ask "2026年9月1日，A股上涨、下跌、平盘家数及涨跌比分别是多少？"
    say "卸 ③: $(C -H 'Content-Type: application/json' -d "{\"zchain\":\"$ZC\",\"posttrain\":\"\"}" "$U/api/models/plugins")"; sleep 6
    say "卸后装着: $(C "$U/api/models/list" | jq -c .loaded)"
else say "没有配得上当前侧车的 ③(没过门)"; fi
# 收尾: 停测试的主进程与模型, 生产主进程回 8000 并按它的 serve_pick 起模型
pkill -x ds4-train; pkill -x ds4-server; for i in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
cd "$HOME/ds4-main" && bash gguf-tools/scripts/train_ui_spark.sh start 8000 >/dev/null; sleep 2
say "生产主进程起服: $(curl -s --noproxy '*' -X POST http://127.0.0.1:8000/api/train/serve)"
for i in $(seq 1 150); do [ "$(curl -s --noproxy '*' http://127.0.0.1:8000/api/train/status | jq -r .mode)" = serving ] && break; sleep 5; done
say "生产: $(curl -s --noproxy '*' http://127.0.0.1:8000/api/models/list | jq -c .loaded)"
say "完; 测试目录 $NEW 留着(硬链接不占空间), 日志 $LOG"
