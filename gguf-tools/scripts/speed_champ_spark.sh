#!/bin/bash
# speed_champ_spark.sh — 冠军态在 spark 引擎上的速度尺(在 spark 本机跑)。
#
# 部署形态 = 合一 GGUF(merge_base86p.sh 从 <工作区>/layers 合并, 专家全 VQ --no-down)
# + --zchain <工作区>/zchain.bin(sweep 导出的完整部署链, lfile 全语义)。
# 骨架 + --vq-dir 直读挂不起来(骨架无 ffn_exps_vq.blob ⇒ 绑定判专家张量必需, 实撞 09-05),
# 所以必须先合并; 合并时 M86_RB=/dev/null 跳过路由偏置(判决态 caliper 日志零 --route-bias)。
# 不测"裸底座"对照: 合一 GGUF 自带 dql 层文件内嵌 op(ds4.zchain.present), 不挂 --zchain
# 也不是裸态, 标成裸=假对照。放大器链的开销账见 fable5 nsys(zchain ≈62µs/层 ≈9%)。
#
# 两步: ①贪心 48 token 探针(证明链真武装+输出连贯, 顺带 t/s) ②ds4-bench 上下文前沿曲线。
# 用法: speed_champ_spark.sh [工作区=champ86amp] [标签=工作区名] [ctx上限=8192] [GGUF=gguf/ds4-<工作区>.gguf]
#                            [步长=2048 | x2(倍增)] [gen=128] [zchain=<工作区>/zchain.bin | none] [语料=promessi_sposi.txt]
# 产物: gguf/go-onebit/vqhalf/<工作区>/speed/{probe_*.txt,*_zchain.csv,*.log}
# 长上下文(09-06 1M 尺): 步长 x2 走 2048→4096→…→1048576 十个前沿, 线性 2048 步到 1M 是 512 个前沿跑不完;
# zchain=none 给没有放大器链的原始 GGUF(ds4flash IQ2 原版)用, 此时 CSV 仍叫 *_zchain.csv 但链未挂。
# 语料必须 ≥ ctx 上限 token 数, 不够 ds4-bench 报 "prompt has N tokens, need at least" 直接退出(1M 用 x4 拼接版)。
#
# 内存账(121 GiB 机器): 合一 GGUF ≈83 GB 映射 + q8 repack ~6 GB; --mem-budget-mb 110000 ⇒
# 引擎 L1 闸 85%=91.3 GiB 拒载 / 看门狗 90%=96.7 GiB 跳闸; 外加本脚本 available<6 GiB
# 外部看门狗(95 GB 模型把 spark 吃到 available=0 假死在案)。
set -uo pipefail
ROOT="$HOME/ds4-main"; VQH="$ROOT/gguf/go-onebit/vqhalf"
WS="${1:-champ86amp}"; TAG="${2:-$WS}"; CTXMAX="${3:-8192}"
# 工作区给绝对路径时(2026-09-08 fin profile: gguf/go-onebit/vqfin/fin86q8ve)按路径用, 名字取 basename; 否则仍是 vqhalf 下的名字
case "$WS" in /*) WSD="$WS"; WS="$(basename "$WSD")";; *) WSD="$VQH/$WS";; esac
MDL="${4:-$ROOT/gguf/ds4-$WS.gguf}"; STEP="${5:-2048}"; GEN="${6:-128}"
# 默认语料=英文 README×80(09-07 用户令: 测试一律不用意语, promessi 退役); 不入库, 缺则现生成(与 speed_longctx_spark.sh 同式)
ZCH="${7:-$WSD/zchain.bin}"; CORPUS="${8:-speed-bench/readme_en_x80.txt}"
EXTRA="${9:-}"   # 透传给 ds4-bench 的额外参数(如 1M 尺: "--prefill-chunk 2048 --gen-final-only"), 空格分隔
OUT="$WSD/speed"; mkdir -p "$OUT"
if [ "$STEP" = x2 ]; then STEPARGS=(--step-mul 2); else STEPARGS=(--step-incr "$STEP"); fi
BUDGET_MB=110000
LOG(){ echo "[speed $(date +%H:%M:%S)] $*"; }
cd "$ROOT" || exit 1
[ "$CORPUS" != speed-bench/readme_en_x80.txt ] || [ -s "$CORPUS" ] || for i in $(seq 80); do cat README.md; echo; done > "$CORPUS"

# ---- 起跑清单: 模型在/链在/机器空/内存够 ----
[ -s "$MDL" ] || { LOG "★合一 GGUF 缺 $MDL (先跑 merge_base86p.sh)★"; exit 2; }
[ "$ZCH" = none ] || [ -s "$ZCH" ] || { LOG "★zchain 缺 $ZCH★"; exit 2; }
[ -s "$CORPUS" ] || { LOG "★语料缺 $CORPUS★"; exit 2; }
# pgrep -x 按进程名精确匹配: -f 会自匹配到含模式串的父壳(ssh 远程命令行, 实撞 09-05)。
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done)
if [ -n "$BUSY" ]; then
    LOG "★机器非空(实例锁: 大模型进程只许一个)★"; ps -o pid,comm,args -p $(echo $BUSY | tr ' ' ',') | cut -c1-160; exit 3
fi
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$AVAIL" -ge 100000 ] || { LOG "★available ${AVAIL} MB < 100000, 不起跑★"; exit 3; }
LOG "模型 $(stat -c %s "$MDL") B, zchain $([ "$ZCH" = none ] && echo 无 || stat -c %s "$ZCH") B, available ${AVAIL} MB, ctx≤$CTXMAX 步 $STEP gen $GEN"

# ---- 外部看门狗: 只杀进程不删文件 ----
watch(){ while kill -0 "$1" 2>/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [ "$a" -lt 6000 ] && { LOG "★外部看门狗: available ${a} MB < 6000, 杀 $1★"; kill -9 "$1"; }
    sleep 2; done; }
run(){ "$@" & local p=$!; watch $p & local w=$!; wait $p; local rc=$?; kill $w 2>/dev/null; wait $w 2>/dev/null; return $rc; }

COMMON=(--cuda -m "$MDL" --mem-budget-mb "$BUDGET_MB")
[ "$ZCH" = none ] || COMMON+=(--zchain "$ZCH")

# ---- ① 探针 ----
PROMPT="Explain in plain words how a Redis stream differs from a Redis list."
LOG "① 探针: 贪心 48 token 带链"
run ./ds4 "${COMMON[@]}" --temp 0 -n 48 -p "$PROMPT" > "$OUT/probe_$TAG.txt" 2> "$OUT/probe_$TAG.log"
RC=$?
grep -h "zchain\|ZCHAIN\|VQ\|L1 budget\|t/s\|abort\|refus" "$OUT/probe_$TAG.log" | cut -c1-200
LOG "探针 rc=$RC 输出:"; cat "$OUT/probe_$TAG.txt"; echo
[ "$RC" = 0 ] || { LOG "★探针失败, 不进曲线★"; exit 4; }

# ---- ② 带链曲线 ----
LOG "② 曲线: 2048..$CTXMAX 步 $STEP, gen $GEN, 语料 $CORPUS${EXTRA:+, 额外: $EXTRA}"
# shellcheck disable=SC2086  # EXTRA 按空格拆成多个参数是有意的
run ./ds4-bench "${COMMON[@]}" --prompt-file "$CORPUS" \
    --ctx-start 2048 --ctx-max "$CTXMAX" "${STEPARGS[@]}" --gen-tokens "$GEN" $EXTRA \
    --csv "$OUT/${TAG}_zchain.csv" 2> "$OUT/bench_${TAG}_zchain.log"
LOG "曲线 rc=$? csv:"; cat "$OUT/${TAG}_zchain.csv" 2>/dev/null
LOG "收官: $OUT"
