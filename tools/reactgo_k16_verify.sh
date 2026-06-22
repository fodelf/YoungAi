#!/usr/bin/env bash
# react/go Mode P (keep-map / shrunken) 单机验证驱动 —— M-1.3 + B2b。
# (react-go-execution-plan.md §6 M-1 / §7 表 M-1.2/M-1.3。仿 tools/mtp_pipe_q2_speed.sh 安全结构。)
#
# 验证的是 *shrunken / keep-map* 模型 (shrink_gguf.py 产, 带 ds4.expert_keep_map.* KV)。
# 这类模型运行时走 GPU route_translate (orig id -> compact slot), B2b 的 clamp 计数器就挂在这条路径上。
# 整模型 (full GGUF) 不走 route_translate (ds4_gpu_translate_expert_ids 对无 keep-map 早退), 故本脚本
# 必须喂 keep-map GGUF 才有意义 (full 模型上 G1 会因"无 route_translate 路径"判 SKIP)。
#
# 门 (各自可关):
#   G1  B2b clamp verify : DS4_VERIFY_ROUTE_CLAMP=1 跑一遍, route_translate clamp 计数必须 == 0。
#                          Mode P 下冷专家应被 (非 hash 层 router-side -inf 屏蔽 / hash 层 B2a 全 keep)
#                          挡在 top-k 外, 永不被钳到 slot 0。clamp>0 = 屏蔽泄漏 (退出时 fail-loud 打印
#                          逐 (layer,expert) 计数)。这是 B2b 的唯一硬判据。
#   G2  smoke 连贯       : 生成文本不是"数字汤" (keep-map 屏蔽未实现时的旧症状; memory「k16 GGUF
#                          stale/broken」)。启发式: 非空 + 唯一 token 占比不过低 (非单 token 复读)。
#   G3  ds4-eval q1..q4  : 确定性 token 数 sanity (--temp 0 --seed 1); 与 README 期望人工对齐 (opt-in)。
#
# 在哪跑: 持有 shrunken Mode P GGUF 的机器 (= M1 worker; q2/k16 模板在 M1, 本机磁盘放不下不恢复)。
#   单机, 不双机。B2b 是 ds4_metal.m 内置纯 env 开关, 双机 mtp_pipe 跑也可直接 export
#   DS4_VERIFY_ROUTE_CLAMP=1 拿到每台机各自的 clamp 统计 (见文末)。
#
# 安全闸 (铁律, 硬约束):
#   - 载模型脚本: 默认 *不* 载模型。须显式 RUN=1 才真正起 ds4 (逐次授权; RUN 未设只打印计划 + 内存安全预检)。
#   - DS4_MEM_BUDGET_MB + ds4 的 L1 resident gate 在启动期拒绝超预算 (over-budget refuse startup)。
#   - 后台 RSS 看门狗: 每秒查 ds4 进程 RSS, 超 MAX_GB 立即杀 + abort。RUN_TIMEOUT_SEC 兜底。
#   - cleanup (trap INT TERM EXIT): 只杀 ds4 进程, *绝不删任何文件* (不碰模型)。
set -uo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

# ---------------- 配置 (均可 env 覆盖) ----------------
# shrunken Mode P keep-map GGUF。默认指向 quantize_reactgo.sh step-4 的 hot-only 产物;
# 也可指向任何 shrink_gguf.py 产的 k16/k6 GGUF (memory: 单机 k16/k6-imatrix 是免双机复现入口)。
MODEL=${MODEL:-gguf/ds4-reactgo-hot-iq2xxs.gguf}
MAX_GB=${MAX_GB:-12}                       # RSS 看门狗红线 (两机各 12/12 GiB 铁律)
MEM_BUDGET_MB=${MEM_BUDGET_MB:-12000}      # DS4_MEM_BUDGET_MB: ds4 L1 resident gate 硬预算
PROMPT=${PROMPT:-"写一个 Python 函数判断字符串是否为回文，并解释它的原理。"}
NPRED=${NPRED:-64}                         # 生成 token 数 (clamp 验证只需够触发各层 route_translate)
SEED=${SEED:-1}
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-600}    # 单步跑超过该秒数即杀 + abort

GATE_CLAMP=${GATE_CLAMP:-1}                # G1 B2b clamp (核心)
GATE_SMOKE=${GATE_SMOKE:-1}               # G2 smoke 连贯
GATE_EVAL=${GATE_EVAL:-0}                 # G3 ds4-eval q1..q4 (较重, opt-in)
EVAL_QUESTIONS=${EVAL_QUESTIONS:-4}
EVAL_TOKENS=${EVAL_TOKENS:-2048}

CLAMP_LOG=/tmp/reactgo_k16_clamp.log       # G1 stderr (含退出时 clamp dump)
CLAMP_OUT=/tmp/reactgo_k16_clamp.out       # G1 stdout (生成文本, G2 复用)
EVAL_LOG=/tmp/reactgo_k16_eval.log
DS4_PID=""

log(){ echo "[k16-verify] $*"; }
hr(){ printf '%s\n' "------------------------------------------------------------"; }

cleanup(){
  trap - INT TERM EXIT
  # 铁律: 只杀进程, 不删任何文件。
  [ -n "$DS4_PID" ] && kill "$DS4_PID" 2>/dev/null || true
  # 兜底清掉本脚本可能遗留的本模型 ds4 (精确匹配 MODEL, 不误伤别的 ds4)。
  pkill -f "ds4 -m $MODEL" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# RSS(GiB) of a pid; 空串若进程没了。
rss_gb(){ local kb; kb=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$kb" ] && awk -v k="$kb" 'BEGIN{printf "%.2f", k/1048576}'; }
# a > b ?
over(){ awk -v a="${1:-0}" -v b="$2" 'BEGIN{exit !(a>b)}'; }

# 后台起一条 ds4 生成, 带 RSS 看门狗 + 超时。$1=额外 env 串, $2=stderr 文件, $3=stdout 文件, 其余=ds4 参数。
# 返回 ds4 退出码; 看门狗触发则返回 124(超时)/137(超内存)。
run_ds4_guarded(){
  local extra_env="$1" errf="$2" outf="$3"; shift 3
  : >"$errf"; : >"$outf"
  # shellcheck disable=SC2086
  env $extra_env DS4_MEM_BUDGET_MB="$MEM_BUDGET_MB" \
    ./ds4 -m "$MODEL" "$@" >"$outf" 2>"$errf" &
  DS4_PID=$!
  local rc=0 i
  for i in $(seq 1 "$RUN_TIMEOUT_SEC"); do
    if ! kill -0 "$DS4_PID" 2>/dev/null; then wait "$DS4_PID"; rc=$?; DS4_PID=""; return $rc; fi
    local g; g=$(rss_gb "$DS4_PID")
    printf "\r[k16-verify] RSS=%sG/%dG  t=%ss/%ss   " "${g:-?}" "$MAX_GB" "$i" "$RUN_TIMEOUT_SEC"
    if [ -n "$g" ] && over "$g" "$MAX_GB"; then
      echo; log "RSS ${g}G 超 ${MAX_GB}G 红线 → 杀进程 abort"
      kill "$DS4_PID" 2>/dev/null || true; wait "$DS4_PID" 2>/dev/null || true; DS4_PID=""
      return 137
    fi
    sleep 1
  done
  echo; log "运行超 ${RUN_TIMEOUT_SEC}s → 杀进程 abort"
  kill "$DS4_PID" 2>/dev/null || true; wait "$DS4_PID" 2>/dev/null || true; DS4_PID=""
  return 124
}

# ---------------- 预检 + 计划 ----------------
hr; log "react/go Mode P (keep-map) 单机验证 —— M-1.3 + B2b"; hr
log "MODEL          = $MODEL"
log "内存预算        = DS4_MEM_BUDGET_MB=$MEM_BUDGET_MB (L1 gate 硬闸) + RSS 看门狗红线 ${MAX_GB}G"
log "门             = clamp:$GATE_CLAMP smoke:$GATE_SMOKE eval:$GATE_EVAL"
log "生成            = -n $NPRED --temp 0 --seed $SEED"

ok_pre=1
if [ ! -f "$MODEL" ]; then
  log "✗ 模型不存在: $MODEL"
  log "  (本机 gguf/ 只有 mask-*.bin; shrunken/k16 GGUF 在 M1。在 M1 上跑本脚本, 或设 MODEL=<M1上的keep-map GGUF>。)"
  ok_pre=0
fi
for b in ds4 ds4-eval; do
  [ -x "./$b" ] || { log "✗ 缺二进制 ./$b (先 make)"; ok_pre=0; }
done
# keep-map 自检: route_translate / B2b 只对带 ds4.expert_keep_map.* 的 GGUF 生效。
if [ -f "$MODEL" ] && command -v strings >/dev/null 2>&1; then
  if strings -n 8 "$MODEL" 2>/dev/null | grep -q "expert_keep_map"; then
    log "✓ keep-map GGUF (含 ds4.expert_keep_map.*) → 会走 route_translate → B2b 计数有效"
  else
    log "⚠ 未在 $MODEL 探到 ds4.expert_keep_map.* —— 若是 full 模型, G1 clamp 会判 SKIP (full 不走 route_translate)"
  fi
fi

if [ "${RUN:-0}" != 1 ]; then
  hr
  log "DRY-RUN (默认)。这是*载模型*脚本, 按铁律须逐次授权:"
  log "  确认内存安全 (预算 ${MEM_BUDGET_MB}MB + 看门狗 ${MAX_GB}G + ds4 L1 gate) 后, 用 RUN=1 真正跑:"
  log "    RUN=1 MODEL=$MODEL bash tools/reactgo_k16_verify.sh"
  log "  逐门单跑示例: RUN=1 GATE_SMOKE=0 GATE_EVAL=0 bash tools/reactgo_k16_verify.sh   # 只 G1 clamp"
  [ "$ok_pre" = 1 ] && log "预检: ✓ 就绪 (等授权)" || log "预检: ✗ 见上 (修好再跑)"
  hr
  exit 0
fi
[ "$ok_pre" = 1 ] || { log "预检未过, 拒绝载模型。"; exit 1; }

# ---------------- G1: B2b clamp verify ----------------
G1=skip G2=skip G3=skip
if [ "$GATE_CLAMP" = 1 ]; then
  hr; log "G1 B2b clamp verify (DS4_VERIFY_ROUTE_CLAMP=1) …"
  run_ds4_guarded "DS4_VERIFY_ROUTE_CLAMP=1" "$CLAMP_LOG" "$CLAMP_OUT" \
    --temp 0 --seed "$SEED" -n "$NPRED" -p "$PROMPT"
  rc=$?; echo
  if [ "$rc" != 0 ]; then
    log "G1 ✗ ds4 退出码 $rc (124=超时/137=超内存/其它=载模型或推理失败)。日志尾:"
    tail -8 "$CLAMP_LOG" | sed 's/^/    /'
    G1=fail
  elif ! grep -q "route-translate clamp\|ROUTE-TRANSLATE CLAMP" "$CLAMP_LOG"; then
    log "G1 ⚠ 未见 B2b clamp dump 行 —— MODEL 多半是 full 模型 (无 keep-map → 不走 route_translate)。判 SKIP。"
    grep -i "verify enabled" "$CLAMP_LOG" | sed 's/^/    /' || true
    G1=skip
  elif grep -q "CLAMP LEAK" "$CLAMP_LOG"; then
    log "G1 ✗ *** clamp 泄漏 *** Mode P 屏蔽漏了冷专家:"
    grep -i "clamp" "$CLAMP_LOG" | sed 's/^/    /'
    G1=fail
  else
    log "G1 ✓ clamp 计数 = 0:"
    grep -i "clamp count = 0\|clamp verify" "$CLAMP_LOG" | sed 's/^/    /'
    G1=pass
  fi
fi

# ---------------- G2: smoke 连贯 (无数字汤) ----------------
if [ "$GATE_SMOKE" = 1 ]; then
  hr; log "G2 smoke 连贯 (无数字汤) …"
  if [ ! -s "$CLAMP_OUT" ]; then
    # G1 没跑或没产出 → 单独跑一遍生成
    run_ds4_guarded "" "$CLAMP_LOG.smoke" "$CLAMP_OUT" --temp 0 --seed "$SEED" -n "$NPRED" -p "$PROMPT"; echo
  fi
  if [ ! -s "$CLAMP_OUT" ]; then
    log "G2 ✗ 无生成输出"; G2=fail
  else
    # 启发式: 唯一"词"占比。数字汤/单 token 复读 → 唯一占比极低。
    read -r ratio words uniq < <(awk '{for(i=1;i<=NF;i++){n++;c[$i]++}} END{u=length(c); if(n==0){print "0 0 0"}else{printf "%.2f %d %d", u/n, n, u}}' "$CLAMP_OUT")
    log "G2 输出 words=$words uniq=$uniq uniq_ratio=$ratio (前 200 字符):"
    head -c 200 "$CLAMP_OUT" | sed 's/^/    /'; echo
    if awk -v r="${ratio:-0}" -v n="${words:-0}" 'BEGIN{exit !(n>=8 && r>=0.30)}'; then
      log "G2 ✓ 输出连贯 (非数字汤)"; G2=pass
    else
      log "G2 ✗ 疑似数字汤/退化 (words<8 或 uniq_ratio<0.30) —— 人工核对上面输出"; G2=fail
    fi
  fi
fi

# ---------------- G3: ds4-eval q1..q4 ----------------
if [ "$GATE_EVAL" = 1 ]; then
  hr; log "G3 ds4-eval q1..q$EVAL_QUESTIONS (--temp 0 --seed $SEED) …"
  # ds4-eval 自带看门狗较弱, 这里直接前台跑但仍受 DS4_MEM_BUDGET_MB / L1 gate 约束。
  env DS4_MEM_BUDGET_MB="$MEM_BUDGET_MB" \
    ./ds4-eval -m "$MODEL" --plain --questions "$EVAL_QUESTIONS" --tokens "$EVAL_TOKENS" \
    --temp 0 --seed "$SEED" >"$EVAL_LOG" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    log "G3 ✓ ds4-eval 完成 (token 数与 README 期望人工对齐):"; tail -12 "$EVAL_LOG" | sed 's/^/    /'; G3=pass
  else
    log "G3 ✗ ds4-eval 退出码 $rc:"; tail -8 "$EVAL_LOG" | sed 's/^/    /'; G3=fail
  fi
fi

# ---------------- 汇总 ----------------
hr; log "汇总: G1(clamp)=$G1  G2(smoke)=$G2  G3(eval)=$G3"
log "  日志: G1=$CLAMP_LOG  out=$CLAMP_OUT  G3=$EVAL_LOG"
hr
# 任一 fail → 非零退出 (CI/门用)。skip 不算 fail。
case "$G1$G2$G3" in
  *fail*) log "结果: ✗ 有门未过"; exit 1 ;;
  *)      log "结果: ✓ 跑过的门全过 (skip 不计)"; exit 0 ;;
esac

# ---- 附: 在现有双机 mtp_pipe 跑里拿同样的 B2b clamp 统计 ----
# B2b 是 ds4_metal.m 内置、纯 env 开关, 与 run 模式无关。要在双机 Mode P 跑里验 clamp:
#   两端 ds4 都加 DS4_VERIFY_ROUTE_CLAMP=1 (mtp_pipe 的 LOCAL_RUN_ENV / REMOTE_RUN_ENV),
#   退出时各自 stderr 打印各机的 clamp 计数 (coordinator → /tmp/mtp_pipe_coord.log,
#   worker → M1:/tmp/mtp_pipe_worker.log)。两边都 "clamp count = 0" 才算 Mode P 屏蔽干净。
