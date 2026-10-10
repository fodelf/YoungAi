#!/bin/bash
# studio_posttrain_probe.sh — 从工作台接口走一遍"上传 jsonl → 开始训练 → 训完自动挂 ③ → 聊天对比"(2026-10-10)。
#   用法: studio_posttrain_probe.sh <jsonl> <问题文件(一行一题)> [页面端口 8000] [轮数 3]
#   全程只打主进程 ds4-train 的 HTTP 接口, 与浏览器页面同一套(上传 = POST /api/train/upload, 发车 = POST /api/train/start,
#   聊天 = POST /v1/chat/completions 经主进程转发, 开关 ③ = POST /api/models/plugins); 不碰脚本内部, 这样验的就是用户在页面上
#   真实走的那条路。三组回答原样进日志: 训前 / 训后挂 ③ / 训后卸 ③ —— 后两组一对比就知道 ③ 有没有写进东西。
#   收尾把 ③ 卸掉(serve_pick.txt 回到训前那套), 训出的目录留在 gguf/v41/posttrain/ 下, 页面"后训练"开关随时能再挂。
#   日志 /tmp/studio_probe_<MMDDHHMM>.log。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
JS="${1:?jsonl}"; QS="${2:?问题文件}"; PORT="${3:-8000}"; EP="${4:-3}"
U="http://127.0.0.1:$PORT"
LOG="/tmp/studio_probe_$(date +%m%d%H%M).log"
C() { curl -s --noproxy '*' --max-time 900 "$@"; }
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# 逐题问(温度 0, 上限 200 token 只为探针快; 页面本身不传这两项), 回答原样记
ask() {
    local tag="$1" q body a
    while IFS= read -r q; do
        [ -z "$q" ] && continue
        body=$(jq -cn --arg q "$q" '{model:"ds4",messages:[{role:"user",content:$q}],temperature:0,max_tokens:200,stream:false}')
        a=$(C -H 'Content-Type: application/json' -d "$body" "$U/v1/chat/completions" | jq -r '.choices[0].message.content // (.error|tostring) // .')
        printf '[%s] 问: %s\n答: %s\n\n' "$tag" "$q" "$a" | tee -a "$LOG"
    done < "$QS"
}
status() { C "$U/api/train/status"; }

say "日志 $LOG; 页面 $U"
mode=$(status | jq -r .mode)
[ "$mode" = serving ] || { say "主进程不在 serving 态(mode=$mode), 不发车"; exit 1; }

NAME="$(basename "$JS")"
say "上传 $JS → $(C -X POST --data-binary "@$JS" "$U/api/train/upload?kind=jsonl&name=$NAME")"
DATA="gguf-tools/data/posttrain/$NAME"
[ -f "$DATA" ] || { say "上传后盘上没有 $DATA"; exit 1; }
say "训前装着: $(C "$U/api/models/list" | jq -c .loaded)"
ask "训前"

say "发车: $(C -H 'Content-Type: application/json' -d "{\"data\":\"$DATA\",\"epochs\":\"$EP\",\"layers\":\"0-39\",\"lr\":\"2e-4\",\"extra\":\"\"}" "$U/api/train/start")"
T0=$(date +%s); RUN=""
while :; do
    sleep 30
    s=$(status); m=$(echo "$s" | jq -r .mode); pr=$(echo "$s" | jq -c .proc)
    [ -z "$RUN" ] && RUN=$(C "$U/api/train/runs" | jq -r --arg d "$DATA" '[.[]|select(.data==$d)]|sort_by(-.started)|.[0].name // ""')
    r=$(C "$U/api/train/run?name=$RUN" 2>/dev/null)
    st=$(echo "$r" | jq -r '.status // "?"'); last=$(echo "$r" | jq -c '{steps_done,last:.last.loss,ep:.last.ep,ntr,epochs:[.epochs[]|{n,eval,pass:.gate.pass,dpp:.gate.dpp}]}' 2>/dev/null)
    say "$(( ($(date +%s)-T0)/60 ))min mode=$m proc=$pr run=$RUN status=$st $last"
    if [ "$m" = serving ] && [ "$st" != running ] && [ "$(echo "$pr" | jq -r .server)" = true ]; then break; fi
    [ $(( $(date +%s)-T0 )) -gt 5400 ] && { say "90 分钟没收工, 停"; exit 1; }
done
say "趟 $RUN 收工: $(echo "$r" | jq -c '{status,pick,step0,epochs:[.epochs[]|{n,eval,train,hold,gate}]}')"
say "探针原文:"; echo "$r" | jq -r '.probes|to_entries[]|"--- \(.key) ---\n\(.value)"' | tee -a "$LOG"
sleep 5
LST=$(C "$U/api/models/list"); LD=$(echo "$LST" | jq -c .loaded); say "训后装着: $LD"
ZC=$(echo "$LD" | jq -r '.zchain // "none"')
# 训完不自动挂(10-10 用户定): 装回来的必须还是训前那套; ③ = 最新一份指纹配得上当前侧车的(页面开关挂的就是它)
[ "$(echo "$LD" | jq -r '.posttrain // ""')" = "" ] && say "训后 ③ 没自动挂上 ✓" || say "★训后 ③ 自动挂上了, 违反'训练完就是训练完'★"
PT=$(echo "$LST" | jq -r --arg z "${ZC}" '(.fnv[$z]) as $w | [.posttrains[]|select(.fnv==$w)][0].path // ""')
if [ -z "$PT" ]; then say "没有配得上当前侧车的 ③(没过门或没选轮), 只问一组"; ask "训后·无③"; exit 2; fi
say "开关挂 ③ $PT: $(C -H 'Content-Type: application/json' -d "{\"zchain\":\"$ZC\",\"posttrain\":\"$PT\"}" "$U/api/models/plugins")"
sleep 6
say "挂后装着: $(C "$U/api/models/list" | jq -c .loaded)"
ask "训后·挂③"
say "卸 ③: $(C -H 'Content-Type: application/json' -d "{\"zchain\":\"$ZC\",\"posttrain\":\"\"}" "$U/api/models/plugins")"
sleep 6
say "卸后装着: $(C "$U/api/models/list" | jq -c .loaded)"
ask "训后·卸③"
say "serve_pick.txt 现在: $(tr '\n' ' ' < gguf/serve_pick.txt)"
say "完; 日志 $LOG"
