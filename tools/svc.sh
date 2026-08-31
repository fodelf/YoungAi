#!/usr/bin/env bash
# svc.sh — 双机 mono 常驻推理服务 (探针协议的服务版: 起一次, 之后所有验证零启动成本)。
#
# 铁律依据: fast-probe protocol (worker 常驻复用, 禁逐跑杀); 内存安全 (12G 红线+看门狗)。
# 拓扑: M4 ds4-server coordinator(0:19, HTTP :8013) + M1 worker(20:output, 常驻,
#       coordinator 断开自动清 session 等重连 —— 天生支持复用)。
#
# 用法:
#   tools/svc.sh up            # 幂等: 已在跑则直接报状态; 否则起 worker+server+看门狗
#   tools/svc.sh status        # 两机进程/内存/端口一眼看全
#   tools/svc.sh probe '提示词' [max_tokens]   # 秒发一条 /v1/messages 快探 (temp0)
#   tools/svc.sh claude '任务'  # 用 Claude Code CLI 打常驻服务跑任务 (env 全套已配好)
#   tools/svc.sh down          # 两边只杀进程 (不删任何文件)
#
# ⚠ 实例锁: 服务常驻期间, 单机 ds4/ds4_test 等大模型进程起不来 (AGENT.md 防双载)。
#   需要跑 ds4_test 时先 `svc.sh down`。
set -uo pipefail
M1=${M1:-192.168.1.2}
M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
DIR=${DIR:-/Users/fodelf/git/ds4-main}
PORT=${PORT:-8013}
DPORT=${DPORT:-5599}
MODEL=${MODEL:-gguf/go-onebit/ds4-code1b.gguf}   # 2026-07-22 主线=v3(编程全域基线)
CTX=${CTX:-65536}
CAP=${CAP:-512}
# CORR=/abs/path/sidecar.gguf 挂 corr 侧车 (P2 后训练产物)。corr 校正的层跑在哪台,
# 哪台就要加载 → 两端都传 (对不持有对应层的一端是无操作)。M1 侧用同名相对路径:
# 侧车须先 scp 到 $M1DIR/$(basename)。
CORR=${CORR:-}
# SOUL=行为示例文件 (P3, 默认修复灵魂; 空串禁用)。只 server 侧 (prompt 渲染)。
SOUL=${SOUL-gguf-tools/data/corpus/soul/soul_server_v3.txt}
# 2026-08-31 env 大扫除后, 行为杠杆全部走 CLI flag:
#   IO 杠杆(pread/prefetch/event-drain/gather 线程=8)已写死或随 offload 自适应, 无需再传;
#   批注入+自由区 96 预算已写死(2026-07-16 CC 攻3 A/B 定版); offload 由 --mem-budget-mb
#   喂给 AUTO 判定(80G 模型 vs 12G 预算 ⇒ 必然流式, 与旧 EXPERT_OFFLOAD=1 同效)。
COMMON_FLAGS="--reverse-connect --prefill-chunk 2048 --mem-budget-mb 12000"
# RESID=相对路径的热专家残差侧车(2026-07-14 方向A): 两端都要(各自持有的层
# 才用得上自己那部分)。侧车须已 scp 到 $M1DIR/同名相对路径。空=不挂。
RESID=${RESID-gguf/sidecars/code-hot-res-prog.gguf}   # 显式空=裸; 默认挂 prog 残差(2026-07-23 换代, 旧 v3p 版已删)
# BASE_NATIVE=1(默认开): go-onebit 是 BASE 底模, chat 角色帧会出符号汤 → 母语骨架渲染
# (# User:/# Assistant: 注释体) + 默认 stop; 工具帧仍由 --tool-primer 强制。只 server 侧生效。
BASE_NATIVE=${BASE_NATIVE:-1}
SRV_FLAGS=""
[ "$BASE_NATIVE" != 0 ] && SRV_FLAGS="$SRV_FLAGS --base-native"
# KNOWLEDGE=知识环检索库(2026-07-23): server 端 knowledge-primer 参考注入, 治无据知识问答
# 复读/幻觉(g1/g4/singleflight/read/commit 5针 A/B 全过)。纯 prompt 零模型零体积。默认挂。
KNOWLEDGE=${KNOWLEDGE-gguf-tools/data/corpus/knowledge.txt}
[ -n "$KNOWLEDGE" ] && SRV_FLAGS="$SRV_FLAGS --knowledge $KNOWLEDGE"
# COMPACT=1(2026-07-14 速度): 结构 token 只把语义锚点送进 KV(text 输出仍是完整合法 DSML)。
# 实测 inject 占热轮 68%(32.8s/48.4s) → edit 68 tok → ~20 tok。质量 A/B 后定版。
[ -n "${COMPACT:-}" ] && SRV_FLAGS="$SRV_FLAGS --primer-compact"
# POOL_MB=专家常驻池(2026-07-14 速度): 热专家反复命中 → 钉进 GPU 池免每 token 重拉 SSD。
# 保守起步(教训: 过大的显式 RAM 缓存会饿死 page cache 反而变慢, 3.84→1.84 实测)。
# 自动热锁: 每 N token 统计命中, top-K 常驻。空=不开池(今日基线 1.18 t/s)。
# PIN=1(2026-07-14 用户洞察: 活跃专家进RAM): 静态白名单钉 code 域 top-k 活跃专家进
# GPU 常驻池(绕过自动热锁 warmup, 确定性)。表 /tmp/code_pin.{coord,worker}.pinned 由
# gen_pinned.py(已删, 见 git 历史)当年从 s305 锚生成, 按层切(coord 0-19 / worker 20-42);
# 现只消费 /tmp 既有表, 重新生成需 checkout 历史。池 5G(装下 ~4-4.5G
# top64 + 余量); 内存: backbone 4G + 池 5G = 9G < 12G 红线, 看门狗兜底。
PIN=${PIN:-}
POOL_MB=${POOL_MB:-}
[ -n "$PIN" ] && POOL_MB=${POOL_MB:-5000}
POOL_FLAGS=""
if [ -n "$POOL_MB" ]; then
  POOL_FLAGS="--expert-pool-mb $POOL_MB --expert-pool-prefetch-top ${POOL_PREFETCH_TOP:-16}"
  # 无静态钉时用自动热锁(auto-pin 间隔已写死默认 32)
  [ -z "$PIN" ] && POOL_FLAGS="$POOL_FLAGS --expert-pool-auto-pin-top ${POOL_PIN_TOP:-32}"
fi
PIN_COORD=${PIN_COORD:-/tmp/code_pin.coord.pinned}
PIN_WORKER=${PIN_WORKER:-/tmp/code_pin.worker.pinned}
# PINRAM=1 (2026-07-17 诊断: decode 两机合计 ~1.2GB/token 全冷读 SSD, page cache 命中≈0):
# mlock 频率钉 (quant 无关, 对 go1b 生效; GPU 池对 go1b 是 no-op) — base 专家 +
# 残差侧车热槽都钉进 RAM。bit-exact 纯驻留。文件即现有 /tmp/code_pin.*.pinned
# (L 前缀/分号格式已被解析器兼容)。TOPK=32 → 每机 ~2.0-2.4G base + 同量残差,
# RSS ~5G < 12G 红线。默认关 = 今日基线。
PINRAM=${PINRAM:-}
CPINRAM=""; WPINRAM=""
if [ -n "$PINRAM" ]; then
  # TOPK 已写死 256(实际钉数由 mlock 预算封顶, 3000MB ≈ 旧 TOPK=32 的量级)
  PINRAM_COMMON="--expert-pin-mlock-mb ${PINRAM_MB:-3000} --resid-pin-mlock-mb ${PINRAM_RESID_MB:-3000}"
  CPINRAM="--expert-pin-file $PIN_COORD $PINRAM_COMMON"
  WPINRAM="--expert-pin-file $PIN_WORKER $PINRAM_COMMON"
fi
# ZCHAIN(2026-08-06 用户令"修通op"): 反修动态系数(GL/dyn2/dyn8/GE)的运行时通路。
# 规则=文件驱动零开关: <MODEL>.zchain.bin 在则两端固定挂载(与模型同目录同名衍生, 随模型分发)。
# ZCH=path 可显式覆盖(r29_bench 等 lane 用; 这是脚本变量, 落地仍走 --zchain flag)。
ZCH=${ZCH-$MODEL.zchain.bin}
[ -f "$DIR/$ZCH" ] || ZCH=""
# (NOGE/CS_CAP/PIPE_CHUNK/PROFILE 钩子已随 env 大扫除清退: GE 恒开; copy-spec 族 08-05
#  已删, 那两个 env 早无读取者; 诊断开关连代码一起删了。)
log(){ echo "[svc] $*"; }

# pgrep 锚定真二进制: 裸 "ds4-server.*$PORT"/"role worker" 会匹配看门狗自身命令行与 bash 包壳
# (2026-07-18 实证: server 单侧重启后看门狗 head -1 选中自己, 12G 红线双侧静默失效)
worker_pid(){ ssh "$M1" "pgrep -f '^\./ds4 .*role worker' | head -1" 2>/dev/null; }
server_pid(){ pgrep -f "^\./ds4-server .*--port $PORT" | head -1; }

status(){
  sp=$(server_pid); wp=$(worker_pid)
  if [ -n "$sp" ]; then
    kb=$(ps -o rss= -p "$sp" | tr -d ' '); log "server: pid=$sp rss=$(awk -v k="${kb:-0}" 'BEGIN{printf "%.1f",k/1048576}')G port=$PORT"
  else log "server: 不在"; fi
  if [ -n "$wp" ]; then
    rkb=$(ssh "$M1" "ps -o rss= -p $wp" 2>/dev/null | tr -d ' '); log "worker(M1): pid=$wp rss=$(awk -v k="${rkb:-0}" 'BEGIN{printf "%.1f",k/1048576}')G"
  else log "worker(M1): 不在"; fi
}

# 看门狗存在性/清理必须锚定 ^svc_watchdog_marker (argv[0] 被 exec -a 改名=行首):
# 裸子串会匹配任何命令行里含该字面量的宿主 shell (2026-07-18 实证: 探针任务链自含
# pkill/pgrep 语句→up 误判已在跳过拉起, 且裸 pkill 有误杀无辜 shell 风险)
ensure_watchdog(){
  if ! pgrep -f "^svc_watchdog_marker" >/dev/null 2>&1; then
    # 逻辑在 tools/svc_watchdog.sh (独立文件躲开 bash 3.2 嵌套引号错乱, 见该文件头注)
    nohup perl -e 'setpgrp(0,0); exec @ARGV or die $!' bash -c 'exec -a svc_watchdog_marker bash "$@"' _ "$DIR/tools/svc_watchdog.sh" "$PORT" "$M1" > /tmp/ds4-svc-watchdog.log 2>&1 &
    log "看门狗常驻 (12G 红线)"
  fi
}

up(){
  if [ -n "$(server_pid)" ] && [ -n "$(worker_pid)" ]; then log "已常驻:"; ensure_watchdog; status; return 0; fi
  if [ -z "$(worker_pid)" ]; then
    log "起 M1 worker (常驻)…"
    # 整个后台列表包 ( ) 重定向全 fd, 否则中间子壳持 sshd 管道 → ssh 挂到 worker 退出
    # PIN: worker 侧远程 cat 自己的白名单(\$( ) 转义 → 在 M1 shell 展开; 值无空格安全)
    WPIN=""; [ -n "$PIN" ] && WPIN="--expert-pool-pinned \$(cat $PIN_WORKER)"
    ssh "$M1" "rm -f /tmp/ds4_worker_svc.log; ( cd $M1DIR && nohup ./ds4 -m $MODEL ${CORR:+--corr $(basename "$CORR")} ${ZCH:+--zchain $ZCH} ${RESID:+--residual $RESID} $COMMON_FLAGS $POOL_FLAGS $WPIN $WPINRAM --role worker --listen $M1 $DPORT --layers 20:output -c $CTX --temp 0 --nothink ) > /tmp/ds4_worker_svc.log 2>&1 < /dev/null & echo ok" 2>/dev/null
    until ssh "$M1" "grep -q 'waiting for coordinator' /tmp/ds4_worker_svc.log" 2>/dev/null; do sleep 3; done
  fi
  if [ -z "$(server_pid)" ]; then
    log "起本机 ds4-server coordinator (常驻)…"
    rm -f /tmp/ds4-svc.log
    CPIN=""; [ -n "$PIN" ] && CPIN="--expert-pool-pinned $(cat "$PIN_COORD")"
    # perl setpgid: server 自成进程组 —— 宿主 shell/任务被按组清理时不连带杀 server
    # (2026-07-17 实证两次: 后台任务清理连带 SERVER_GONE)。macOS 无 setsid(1), 用 perl。
    ( cd "$DIR" && nohup perl -e 'setpgrp(0,0); exec @ARGV or die $!' ./ds4-server -m "$MODEL" ${CORR:+--corr "$CORR"} ${ZCH:+--zchain "$ZCH"} ${RESID:+--residual "$RESID"} $COMMON_FLAGS --dist-prefill-cap 2048 $POOL_FLAGS $CPIN $CPINRAM $SRV_FLAGS --role coordinator --coordinator "$M1" "$DPORT" --layers 0:19 -c "$CTX" --port "$PORT" --kv-disk-dir /tmp/ds4-kv-svc --kv-disk-space-mb 8192 --max-output-tokens "$CAP" --nothink --tool-primer ${SOUL:+--soul "$SOUL"} --trace /tmp/ds4-svc-trace.txt > /tmp/ds4-svc.log 2>&1 & )
    until grep -qE "listening|refusing" /tmp/ds4-svc.log 2>/dev/null; do sleep 3; done
    grep -q refusing /tmp/ds4-svc.log && { log "实例锁: 有别的 ds4 进程 (ds4_test?) 先退出它"; tail -2 /tmp/ds4-svc.log; return 1; }
  fi
  # 看门狗 (常驻, 12G 红线两边同杀; perl setpgrp 自立进程组防宿主任务组清理连带 — 07-18 实证)
  ensure_watchdog
  status
}

probe(){
  local p=${1:?prompt}; local n=${2:-64}
  # python3 内联清零(2026-08-25 迁移 Wave C): jq 等价替换(两机均有 /usr/bin/jq);
  # stderr 诊断行的 usage 展示格式从 python dict 变 jq JSON, 非判决数字, 语义同。
  curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' \
    -d "$(jq -n --arg p "$p" --argjson n "$n" '{model:"deepseek-chat",max_tokens:$n,temperature:0,messages:[{role:"user",content:$p}]}')" \
    | jq -r '.content[0].text, ("-- \(.stop_reason) \(.usage|tojson)\n" | stderr | empty)'
}

cc(){
  local task=${1:?task}
  env -u http_proxy -u https_proxy NO_PROXY='*' no_proxy='*' \
    ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_API_KEY=svc \
    ANTHROPIC_MODEL=deepseek-chat ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat \
    API_TIMEOUT_MS=3600000 DISABLE_NON_ESSENTIAL_MODEL_CALLS=1 MAX_THINKING_TOKENS=0 \
    claude -p "$task" --model deepseek-chat --permission-mode bypassPermissions
}

down(){
  log "停两边 (只杀进程, 不删文件)"
  pkill -f "^\./ds4-server .*--port $PORT" 2>/dev/null; pkill -f "^svc_watchdog_marker" 2>/dev/null
  ssh "$M1" "pkill -f 'role worker'" 2>/dev/null
  sleep 1; status
}

case "${1:-status}" in
  up) up ;;
  status) status ;;
  probe) shift; probe "$@" ;;
  claude) shift; cc "$@" ;;
  down) down ;;
  *) echo "用法: svc.sh up|status|probe '提示词' [n]|claude '任务'|down"; exit 2 ;;
esac
