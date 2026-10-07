#!/bin/bash
# multi_probe_spark.sh — 合批解码探针的成组跑法(2026-09-30, batch.md; spark 本机)。
#
# 对同一 12k 提示按路数 N 各跑一趟 --multi-probe(纯墙钟, -n 步), 再对最后一个 N 跑一趟带 --v41-prof 的逐段账(6 步够, 逐段计时有同步开销)。
# 为什么要脚本: 每趟装一次 113 GB 模型(2~3 分钟), 一组四趟十几分钟, 手敲 ssh 会被会话超时杀掉(实撞 09-30 16:3x)。
# 输出: /tmp/mp_N<N>[_tag].err 里 "[multi]" 行 = 每步中位毫秒 / 每路与总 t/s / 逐字节门; "[multi-prof]" 行 = 逐段毫秒。
# 投机默认开(每路各出草稿, 验证行拼进一次前向); 量纯解码合批传 --no-dspark。
# 用法: multi_probe_spark.sh [--nsys] ["N 列表"="1 3 8"] [步数=64] [提示字符=40000] [标签=""] [引擎额外参数...(如 --no-dspark / --no-lanes / --dspark-verify K)]
#   --nsys = 每趟套 nsys, 事后出逐类 + 逐核名的独占时间表(/tmp/mp_N<N><标签>_excl.txt), 见下面的说明。
# 第 5 个参数 = 提示源文件(默认 promessi_sposi.txt; 投机的账要换 speed-bench/fin_chat_prompt.txt 这种草稿器猜得中的文本 —— promessi 单请求路接受只 0.83/5)。
# ★源文件以 .ids 结尾 = 真实请求的 token id(如 /tmp/cfo_prompt.ids, 45 t/s 那条 CFO 请求), 整份按 id 喂(--gen-ids), 第 3 参数给 0★
#   (2026-09-30 用户: 并发的基线是产品单路 = 真实请求 + 投机 45 t/s, 三路必须跟它同一条请求、同一口径比; 真实请求只有 id 准, 文本重分词拼不回去)。
#   第 3 参数 0 也表示不跑末尾的逐段账(那是纯解码诊断用的, 看速度不需要)。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# ★--nsys(第 1 个位置之前, 2026-10-01)★: 每趟套 nsys 采 GPU 时间线(图按节点记), 事后出 d0a 第 ⑥ 张表(逐类 + 逐核名独占, speed-bench/d0a_excl_timeline.awk)。
# 为什么: 合批 N=3 一步 48 ms 对字节地板 34, 多出的 14 ms 是哪一发核、是独占还是被盖着, 墙钟与 --v41-prof 逐段账都答不了(逐段账每段同步, 把道间重叠拆散了)。
# 铁律(memory feedback_speed_per_kernel_then_fix): 速度问题先出逐核表再照最大项动刀。nsys 下步长会虚高 ~10%, 只看比例与独占, 不拿它报 t/s。
NSYS=0; BIN=ds4
while :; do   # 前置选项: --nsys / --bin <二进制>(A/B: 改前留档 ds4.base_xxx 对现在的 ds4, 同机同趟; 2026-10-01)
  case "${1:-}" in
    --nsys) NSYS=1; shift ;;
    --bin) BIN="${2:?--bin 要二进制名}"; shift 2 ;;
    *) break ;;
  esac
done
[ -x "./$BIN" ] || { echo "★没有 ./$BIN★"; exit 1; }
NLIST="${1:-1 3 8}"; STEPS="${2:-64}"; CHARS="${3:-40000}"; TAG="${4:-}"; SRC="${5:-speed-bench/promessi_sposi.txt}"
shift 5 2>/dev/null || shift $#
EXTRA=("$@")
if [ "$NSYS" = 1 ]; then command -v nsys >/dev/null || { echo "★没有 nsys★"; exit 1; }; fi
M=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
Z=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine
case "$SRC" in
  *.ids) [ -s "$SRC" ] || { echo "★没有 $SRC★"; exit 1; }; P="$SRC"; PARG=(--gen-ids "$P") ;;
  *)     P=/tmp/mp_prompt_$(basename "$SRC" .txt)_${CHARS}.txt; head -c "$CHARS" "$SRC" > "$P"; PARG=(--prompt-file "$P") ;;
esac
LOGP(){ echo "[multi_probe $(date +%H:%M:%S)] $*"; }
# ★内存看门狗★(2026-09-30, 1M 档起跑前置): 每 5 秒读一次 MemAvailable, 低于红线就杀 ds4(别 OOM 是最高约束: 1M 一路的状态 3.1 GB + KV 0.94 + 草稿 0.3,
# 三路合批时三份 KV 都活着; 模型装完这台机器 MemAvailable ≈ 11.8 GB)。红线 1500 MB = 系统自己还要的页缓存下限, 跌破就是账算错了, 停车看账。
WD_RED_MB=1500
wd_run(){   # wd_run <标签> <命令...>: 带看门狗跑一趟(自己不打印 —— 调用处把 stdout 定向到吐字文件, 打印会污染 md5); 事后用 wd_report 看最低点
  local tag="$1"; shift
  local memlog="/tmp/mp_mem${tag}.txt"; : > "$memlog"
  ( while true; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); echo "$(date +%H:%M:%S) $a" >> "$memlog"
      if [ "$a" -lt "$WD_RED_MB" ]; then echo "★看门狗★ MemAvailable ${a} MB < ${WD_RED_MB}, 杀 ds4" >> "$memlog"; pkill -x "$BIN"; fi; sleep 5; done ) &
  local wd=$!
  "$@"
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
}
wd_report(){ local memlog="/tmp/mp_mem$1.txt"
  echo "  MemAvailable 最低 $(awk 'NF==2{print $2}' "$memlog" | sort -n | head -1) MB(采样 $(wc -l < "$memlog") 次)$(grep -q 看门狗 "$memlog" && echo ' ★看门狗动过手★')"; }
LAST=""
for N in $NLIST; do
  LOGP "N=$N $([ "$NSYS" = 1 ] && echo nsys || echo 纯墙钟) $STEPS 步 ${EXTRA[*]:-}"
  WRAP=()
  if [ "$NSYS" = 1 ]; then
    rm -f "/tmp/mp_N${N}${TAG}.nsys-rep" "/tmp/mp_N${N}${TAG}.sqlite" "/tmp/mp_N${N}${TAG}_trace_cuda_gpu_trace.csv"
    WRAP=(nsys profile -o "/tmp/mp_N${N}${TAG}" --force-overwrite true -t cuda --cuda-graph-trace=node)   # 按节点记, 图里的核才看得见(d0a 09-18 实撞)
  fi
  wd_run "N${N}${TAG}" ${WRAP[@]+"${WRAP[@]}"} ./"$BIN" -m "$M" --zchain "$Z" --temp 0 -n "$STEPS" --multi-probe "$N" "${PARG[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
      > "/tmp/mp_N${N}${TAG}.out" 2> "/tmp/mp_N${N}${TAG}.err"
  wd_report "N${N}${TAG}"
  grep -a "\[multi\]\|失败\|error" "/tmp/mp_N${N}${TAG}.err" | tail -3 | cut -c1-220
  echo "  文本 md5 $(md5sum < "/tmp/mp_N${N}${TAG}.out" | cut -c1-8) ($(wc -c < "/tmp/mp_N${N}${TAG}.out") 字节)"
  if [ "$NSYS" = 1 ]; then
    # 等映射卸载干净再导出(单实例锁 + 110 GB 卸载要几秒; d0a 同一处实撞); 导出前删旧 sqlite, 否则 nsys stats 复用上一版的账
    for _ in $(seq 1 60); do pgrep -x "$BIN" >/dev/null || break; sleep 2; done
    rm -f "/tmp/mp_N${N}${TAG}.sqlite"
    nsys stats --report cuda_gpu_trace --format csv -o "/tmp/mp_N${N}${TAG}_trace" "/tmp/mp_N${N}${TAG}.nsys-rep" >/dev/null 2>&1
    CSV="/tmp/mp_N${N}${TAG}_trace_cuda_gpu_trace.csv"
    if [ -s "$CSV" ]; then
      echo "== N=$N 独占时间分账(d0a ⑥; 只看 verify_n* 那一类 = 合批解码步; prefill 类是预填块)"
      gawk -f speed-bench/d0a_excl_timeline.awk "$CSV" | tee "/tmp/mp_N${N}${TAG}_excl.txt"
    else echo "★N=$N 没导出 GPU 时间线(nsys 没采到核? 看 /tmp/mp_N${N}${TAG}.err)★"; fi
  fi
  LAST="$N"
done
# 逐段账只在 ≤200k 字符的提示上跑: --v41-prof 逐段同步把预填拖慢 7×(106k 三路 17 min 没预填完第一路, 09-30 实撞), 长上下文的税由纯墙钟那趟给
if [ -n "$LAST" ] && [ "$CHARS" -gt 0 ] && [ "$CHARS" -le 200000 ]; then
  LOGP "N=$LAST 逐段账(--v41-prof, 6 步)"
  ./"$BIN" -m "$M" --zchain "$Z" --temp 0 -n 6 --multi-probe "$LAST" --v41-prof "${PARG[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
      > "/tmp/mp_N${LAST}p${TAG}.out" 2> "/tmp/mp_N${LAST}p${TAG}.err"
  grep -a "\[multi-prof\]" "/tmp/mp_N${LAST}p${TAG}.err" | tail -2 | cut -c1-300
fi
echo MULTI_PROBE_DONE
