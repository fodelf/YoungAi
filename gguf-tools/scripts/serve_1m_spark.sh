#!/bin/bash
# serve_1m_spark.sh — 在 spark 本机起 ds4-server(DeepSeek V4.1 现役对), 局域网可达(给 quant_trading_flow 用)。
# ★以后只有 V4.1★(用户 2026-09-19): V4 那套(q8ve GGUF + zchain.bin 单文件 + 外置 drafter + 1M 上下文)不再是任何默认或退路。
#
# 用法: serve_1m_spark.sh [start|stop|status|smoke] [GGUF] [zchain目录] [额外引擎参数...]
#   三文件部署的第三件用额外参数挂: ... --posttrain <后训练目录>
#   start   起跑清单(机器空/内存/模型/链) → nohup 起服务 → 等 /v1/models 可达 → 余量判 → 冒烟(不过就停服务) → 看门狗
#   stop    只杀 ds4-server 进程(清理=关进程, 不删文件)
#   status  进程 + 端口 + 内存
#   smoke   对已起的服务打一条中文金融问答(温 0, 48 token), 原样打印回答
#
# V4.1 服务路(src/server/server_generate_v41.c)的事实, 起服务前先知道:
#   - 上下文硬上限 1048576(2026-09-22 抬到 1M; 2026-09-21 是 524288, 再之前 32768, 8 万 token 的
#     新闻提示进不来)。--ctx 给大了服务端自己压回来。开大的代价是建状态时一次吃掉的固定显存, 见下面内存账。
#   - 只有贪心解码(请求里的 temperature 被忽略, 日志会提示); 没有 KV 复用, 每条请求整段预填(15k token 约 40 s @334 t/s)。
#   - 不建 V4 会话、不开磁盘 KV、不批处理 —— 所以这里不传 --kv-disk-*/--prefill-chunk/--batch, 传了也是空转。
# 内存账(121 GB spark, 09-19 实测): 启动缓存拷主干 106.5 GiB 到 MemAvailable 12 GB 停(三塔被挤在外面, 服务不用它们);
#   起来后 MemAvailable 11.1 GB, 首条请求的前向缓冲分完后稳定 6.4~6.7 GB, 请求之间不再涨。
#   起服地板 10 GB(在分请求缓冲之前量), 看门狗红线 2.5 GB(连续两次就杀服务: 08-24 那次是 available → 0 才假死)。
#   ★别手动调大 --mem-budget-mb 硬塞★(spark_95g_server_memory_wall)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
CMD="${1:-start}"
MDL="${2:-$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4-mtpnative.gguf}"   # ① 现役量化模型
ZCH="${3:-$ROOT/gguf/v41/gr-fin-40-fp4}"                                      # ② 现役反修插件目录(与 z_nightly_spark.sh 同一对)
# 第 4 个参数起原样透传给 ds4-server(如 --posttrain <③目录>)。老调用方传的 spec/plain(V4 的 drafter 模式)在 V4.1 上没有意义, 跳过。
EXTRA=("${@:4}")
if [ "${EXTRA[0]:-}" = spec ] || [ "${EXTRA[0]:-}" = plain ]; then EXTRA=("${EXTRA[@]:1}"); fi
# CTX = 1M(用户定, 2026-09-22): 引擎 V4.1 路的硬上限同日抬到 1048576(core_v41.h 里有内存账),
# 以前这里写 500000 是我按"够用就行"自己定的 —— 不该自己定。调用方(crewAI)也是按 1M 报窗口的。
PORT=8000; CTX=1048576; BUDGET_MB=110000; MAXOUT=16384
LOGF="$HOME/ds4-server-1m.log"
LOG(){ echo "[serve1m $(date +%H:%M:%S)] $*"; }
alive(){ pgrep -x ds4-server >/dev/null; }
wait_up(){ local i; for i in $(seq 1 "$1"); do curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0; alive || return 1; sleep 5; done; return 1; }

smoke(){   # 温 0 / 48 token / 关思考(deepseek-chat), 原样贴回答: 判读交给人。返回非 0 = 服务答不了
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-chat","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"用一句话解释什么是市盈率(PE), 并说明它偏高通常意味着什么。"}]}' \
        | python3 -c '
import sys, json
raw = sys.stdin.read()
try: r = json.loads(raw)
except Exception: print("★冒烟: 回复不是 JSON★", raw[:300]); sys.exit(1)
if "choices" not in r: print("★冒烟: 服务报错★", json.dumps(r, ensure_ascii=False)[:300]); sys.exit(1)
m = r["choices"][0]["message"]; print("── 回答 ──"); print(m.get("content", "")); print("── usage ──", r.get("usage"))' 2>&1
}

# 看门狗: 与服务同生共死, 每 5 s 看一次 MemAvailable; 连续两次低于红线就杀服务(请求中断可重试, 机器假死不可)
WD_KILL_MB=2500
WDLOG="$HOME/ds4-server-watchdog.log"
watchdog(){
    local a bad=0
    while pgrep -x ds4-server >/dev/null; do
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$a" -lt "$WD_KILL_MB" ]; then bad=$((bad+1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then
            echo "[watchdog $(date +%H:%M:%S)] ★MemAvailable ${a} MB < ${WD_KILL_MB}, 连续两次, 杀 ds4-server★" >>"$WDLOG"
            pkill -x ds4-server; sleep 3; pgrep -x ds4-server >/dev/null && pkill -9 -x ds4-server
            return 1
        fi
        sleep 5
    done
    echo "[watchdog $(date +%H:%M:%S)] 服务已退出, 看门狗收工" >>"$WDLOG"
}

case "$CMD" in
  status)
    alive && { LOG "运行中 pid $(pgrep -x ds4-server | head -1)"; ss -ltnp 2>/dev/null | grep ":$PORT " | cut -c1-120; } || LOG "未运行"
    free -g | sed -n '1,2p'; exit 0;;
  stop)
    alive || { LOG "本就未运行"; exit 0; }
    pkill -x ds4-server; sleep 3; alive && pkill -9 -x ds4-server; LOG "已停"; exit 0;;
  smoke) smoke; exit 0;;
  start) ;;
  *) echo "用法: $0 [start|stop|status|smoke] [GGUF] [zchain目录] [额外引擎参数...]"; exit 2;;
esac

# ---- 起跑清单 ----
[ -s "$MDL" ] || { LOG "★模型缺 $MDL★"; exit 2; }
[ -d "$ZCH" ] && ls "$ZCH"/gr_L*.bin >/dev/null 2>&1 || { LOG "★zchain 目录缺或没有 gr_L*.bin: $ZCH★"; exit 2; }
# pgrep -x 精确进程名: -f 会自匹配 ssh 远程命令行(实撞 08-18/09-05)
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer v41_amp_run; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空(实例锁: 大模型进程只许一个): $BUSY★"; ps -o pid,comm -p $(echo $BUSY | tr ' ' ',') ; exit 3; }
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$AVAIL" -ge 100000 ] || { LOG "★available ${AVAIL} MB < 100000, 不起★"; exit 3; }
LOG "模型 $MDL ($(stat -c %s "$MDL" | awk '{printf "%.1f GB", $1/1e9}')) zchain $ZCH ctx $CTX 预算 $BUDGET_MB MB 输出上限 $MAXOUT available $AVAIL MB${EXTRA[*]:+ 额外: ${EXTRA[*]}}"

LOG "起服务(V4.1 纯解码)"
# --max-output-tokens: 服务端硬上限。crewAI 不传 max_tokens, 服务默认 384K ⇒ 一次跑飞 = 单 worker 被占几小时。
nohup ./ds4-server --cuda -m "$MDL" --zchain "$ZCH" --ctx "$CTX" --mem-budget-mb "$BUDGET_MB" \
    --host 0.0.0.0 --port "$PORT" --max-output-tokens "$MAXOUT" ${EXTRA[@]+"${EXTRA[@]}"} > "$LOGF" 2>&1 </dev/null &
if ! wait_up 120; then
    LOG "★没起来, 日志尾:★"; tail -6 "$LOGF" | cut -c1-200
    alive && pkill -x ds4-server; exit 4
fi
MEM_FLOOR_MB=10000
a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
LOG "起来后 MemAvailable ${a} MB(地板 $MEM_FLOOR_MB) | $(free -m | awk 'NR==2{printf "used %d buff/cache %d", $3, $6} NR==3{printf " swap %d", $3}')"
if [ "$a" -lt "$MEM_FLOOR_MB" ]; then LOG "★余量 ${a} MB < ${MEM_FLOOR_MB}, 会 swap/读盘假死, 停★"; pkill -x ds4-server; sleep 4; exit 4; fi
LOG "服务可达 http://$(hostname -I | awk '{print $1}'):$PORT/v1  (日志 $LOGF)"
grep -E "V4.1 服务路|budget gate|内存地板|listening" "$LOGF" | head -6 | cut -c1-160
# ★冒烟不过 = 服务没用★(09-19 实撞: 服务可达、/v1/models 正常, 每条 chat 却回 "cuda prefill failed", 下游整批报错还以为是模型的事)
if ! smoke; then LOG "★冒烟失败, 停服务★"; pkill -x ds4-server; sleep 3; exit 5; fi
export -f watchdog; export WD_KILL_MB WDLOG
nohup bash -c watchdog >/dev/null 2>&1 </dev/null &
LOG "看门狗已起(红线 ${WD_KILL_MB} MB, 日志 $WDLOG)"
free -g | sed -n '2p'
LOG "SERVE1M_UP"
