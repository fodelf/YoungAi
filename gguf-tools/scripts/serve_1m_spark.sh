#!/bin/bash
# serve_1m_spark.sh — 在 spark 本机起 ds4-server: 1M 上下文 + 投机解码, 局域网可达(给 quant_trading_flow 用)。
#
# 用法: serve_1m_spark.sh [start|stop|status|smoke] [GGUF] [zchain] [spec|plain] [额外引擎参数...]
#   额外参数原样透传, 三文件部署用: ... plain --finetune <微调侧车>
#   start   前台清单(机器空/内存/模型/链) → nohup 起服务 → 等 /v1/models 可达 → 打一条 8 token 冒烟请求
#   stop    只杀 ds4-server 进程(清理=关进程, 不删文件)
#   status  进程 + 端口 + 内存
#   smoke   对已起的服务打一条中文金融问答(温 0, 48 token), 原样打印回答
#
# 内存账(121 GB spark, 09-07 实跑口径): 模型 mmap 87 GB(q8ve) + 1M 上下文缓冲 11.6 GiB + 压缩 KV 7.6 GiB
#   + drafter 11.3 GB ≈ 118 GB > 110000 预算 ⇒ L1 静态闸会拒绝启动(闸拒绝 = 正常, 不是 OOM); 拒绝时脚本自动
#   退到不带 drafter 重试(≈106 GB, 闸放行则跑)。★别手动调大 --mem-budget-mb 硬塞★: 08-24 95 GB 模型 + 长解码把
#   available 打到 0, 机器假死数十分钟(spark_95g_server_memory_wall)。
# 投机: 1M 处 spec 23.4 vs plain 18.6 t/s(fable5 09-07 链 63), 短提示 38 vs 34 —— 恒开, 输出与纯解码逐字节同。
# zchain: 合一 GGUF 已内嵌 dql 层 op, 但完整部署链(sweep 导出)在工作区 zchain.bin, 与 speed_champ_spark.sh 同式挂上。
# 磁盘 KV: ~/ds4-kv 8 GiB —— crewAI 多 agent 会反复带同一前缀(系统提示+行情), 命中即免重算前缀。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
CMD="${1:-start}"; MDL="${2:-$ROOT/gguf/ds4-champ86q8ve.gguf}"
WS="$(basename "$MDL" .gguf)"; WS="${WS#ds4-}"
# 工作区根: champ86*(通用冠军)在 vqhalf, fin86*(金融)在 vqfin
case "$WS" in fin86*) VQH="$ROOT/gguf/go-onebit/vqfin";; *) VQH="$ROOT/gguf/go-onebit/vqhalf";; esac
ZCH="${3:-$VQH/$WS/zchain.bin}"
DRAFT="$ROOT/gguf/ds4-dspark-ve-q4-aq4.gguf"   # 09-07 链 28 定型: 注意力 Q4_K 档 drafter, 接受率不变、draft 段最快
PORT=8000; CTX=1048576; BUDGET_MB=110000
KVD="$HOME/ds4-kv"; LOGF="$HOME/ds4-server-1m.log"
LOG(){ echo "[serve1m $(date +%H:%M:%S)] $*"; }
alive(){ pgrep -x ds4-server >/dev/null; }
wait_up(){ local i; for i in $(seq 1 "$1"); do curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0; alive || return 1; sleep 5; done; return 1; }

smoke(){   # 温 0 / 48 token / 关思考(deepseek-chat), 原样贴回答: 判读交给人
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-chat","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"用一句话解释什么是市盈率(PE), 并说明它偏高通常意味着什么。"}]}' \
        | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("── 回答 ──"); print(m.get("content","")); print("── usage ──", r.get("usage"))' 2>&1
}

case "$CMD" in
  status)
    alive && { LOG "运行中 pid $(pgrep -x ds4-server | head -1)"; ss -ltnp 2>/dev/null | grep ":$PORT " | cut -c1-120; } || LOG "未运行"
    free -g | sed -n '1,2p'; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null; exit 0;;
  stop)
    alive || { LOG "本就未运行"; exit 0; }
    pkill -x ds4-server; sleep 3; alive && pkill -9 -x ds4-server; LOG "已停"; exit 0;;
  smoke) smoke; exit 0;;
  start) ;;
  *) echo "用法: $0 [start|stop|status|smoke] [GGUF] [zchain]"; exit 2;;
esac

# ---- 起跑清单 ----
[ -s "$MDL" ] || { LOG "★模型缺 $MDL★"; exit 2; }
[ -s "$ZCH" ] || { LOG "★zchain 缺 $ZCH(合一 GGUF 必须配它的部署链)★"; exit 2; }
[ -s "$DRAFT" ] || { LOG "★drafter 缺 $DRAFT★"; exit 2; }
# pgrep -x 精确进程名: -f 会自匹配 ssh 远程命令行(实撞 08-18/09-05)
# pgrep -x 只认 ≤15 字符的进程名(comm 截断), deepseek4-quantize 18 字符要走 -f 括号写法(不自匹配本脚本命令行)
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done; pgrep -f '^\./deepseek4-quant[i]ze')
[ -z "$BUSY" ] || { LOG "★机器非空(实例锁: 大模型进程只许一个): $BUSY★"; ps -o pid,comm -p $(echo $BUSY | tr ' ' ',') ; exit 3; }
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$AVAIL" -ge 100000 ] || { LOG "★available ${AVAIL} MB < 100000, 不起★"; exit 3; }
mkdir -p "$KVD"
LOG "模型 $MDL ($(stat -c %s "$MDL" | awk '{printf "%.1f GB", $1/1e9}')) zchain $ZCH ctx $CTX 预算 $BUDGET_MB MB available $AVAIL MB"

launch(){   # $1=标签, 其余=额外参数; 返回 0=服务可达
    LOG "起服务($1)"
    # --max-output-tokens 16384: 服务端硬上限。crewAI 不传 max_tokens, 服务默认 384K ⇒ 一次跑飞 = 单 worker 被占 5 小时;
    # 金融分析单次回答 16K token(≈13 分钟@21 t/s)绰绰有余。
    nohup ./ds4-server --cuda -m "$MDL" --zchain "$ZCH" --ctx "$CTX" --mem-budget-mb "$BUDGET_MB" \
        --host 0.0.0.0 --port "$PORT" --prefill-chunk 2048 --max-output-tokens 16384 \
        --kv-disk-dir "$KVD" --kv-disk-space-mb 8192 ${EXTRA[@]+"${EXTRA[@]}"} "${@:2}" > "$LOGF" 2>&1 </dev/null &
    if wait_up 120; then return 0; fi
    LOG "★($1)没起来, 日志尾:★"; tail -6 "$LOGF" | cut -c1-200
    alive && pkill -x ds4-server; sleep 2; return 1
}
# ★起来后按真实余量判, 不信 L1 闸★(09-08 实撞): L1 闸只算常驻骨干 8.2 GiB, 对 VQ blob 专家(73 GB)是瞎的 —— q8ve + 1M + drafter
# 闸放行, 实际 used 116/121 GB, available 5 GB, swap 960 MB, buff/cache 274 MB: 首请求 prefill 85 s, 解码 12.5 t/s(曲线 21.6),
# "CUDA q8 fp16 cache disabled after allocation failure"。这就是 08-24 假死的前兆(spark_95g_server_memory_wall)。
# 地板 10 GB: 5 GB 时已在 swap, 留 2 倍余量给 crewAI 长提示的上下文增长与 kv-disk 写缓存。
MEM_FLOOR_MB=10000
MODE="${4:-spec}"   # spec=先试投机 drafter, 余量不够自动退纯解码; plain=直接纯解码
# 第 5 个参数起原样透传给 ds4-server(2026-09-08 三文件部署: --finetune <微调侧车>)。
# 只透传, 不解释 —— 引擎认什么这里就能传什么, 脚本不做第二套参数表。
EXTRA=("${@:5}")
up_ok(){   # $1=标签; 服务可达后读一次真实余量, 不够就停(清理=关进程)
    local a; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    LOG "$1 起来后 MemAvailable ${a} MB(地板 $MEM_FLOOR_MB) | $(free -m | awk 'NR==2{printf "used %d buff/cache %d", $3, $6} NR==3{printf " swap %d", $3}')"
    [ "$a" -ge "$MEM_FLOOR_MB" ] && return 0
    LOG "★$1: 余量 ${a} MB < ${MEM_FLOOR_MB}, 会 swap/读盘假死, 停★"; pkill -x ds4-server; sleep 4; return 1
}
UP=""
if [ "$MODE" = spec ]; then
    if launch "投机 drafter" --spec --draft-gguf "$DRAFT" && up_ok "投机 drafter"; then UP=spec; else LOG "退到纯解码"; fi
fi
if [ -z "$UP" ]; then
    if launch "纯解码" && up_ok "纯解码"; then UP=plain
    else LOG "★纯解码 1M 也不够内存: 要么改小 CTX 常量(512k 省 ~9.6 GiB), 要么换更小骨架 —— 由用户定★"; exit 4; fi
fi
LOG "服务可达 http://$(hostname -I | awk '{print $1}'):$PORT/v1  (日志 $LOGF)"
grep -E "KV policy|budget|resident|spec|drafter" "$LOGF" | head -8 | cut -c1-160
free -g | sed -n '2p'
smoke
LOG "SERVE1M_UP"
