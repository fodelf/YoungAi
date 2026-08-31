#!/bin/sh
# mono_single_run.sh — 本机(M4)单机跑 go2b 全 EF mono, offload, 写 Go 代码。
# 单机在一块 GPU 上 → 无跨 GPU fp 漂移问题, 不需要 math_safe(单机已验证能写出干净 twoSum 函数体)。
#
# 已验证配置 (2026-07-06, env 大扫除 2026-08-31 迁 flag):
#   专家流式 offload 由 AUTO 按内存预算判定(原 EXPERT_OFFLOAD=1 env 已删)
#   --prefill-chunk 512         prefill 分块(内存有界)
#   (原 REPEAT_FREQ=1 repeat penalty 路已随 env 大扫除从引擎删除, 恒裸解码)
#   --ctx 4096                  KV 有界(避免默认 32768 撑内存)
#   BOS 前缀 (<｜begin▁of▁sentence｜>)  裸续写绕 chat 模板 → 直接续写代码
#   空默认系统提示词 + 防-panic 压力守卫 = 二进制默认 (ds4_cli.c / ds4.c)
# 内存安全三重: 引擎压力守卫(默认) + 本脚本外部 pressure-level 看门狗 + perl 超时。
#
# 用法:
#   ./mono_single_run.sh                      # 默认: Go twoSum BOS 裸续写, n=100
#   ./mono_single_run.sh "PROMPT" [NPRED]     # 自定义 prompt / token 数
#   PROMPT=$'<｜begin▁of▁sentence｜>...' ./mono_single_run.sh   # 自己的裸续写题
#   聊天式(非代码续写): 传不带 BOS 前缀的普通问句即可(走 chat 模板, 空系统提示词)。
#   env 可覆盖: MODEL / CTX / NPRED / PROMPT / OUT。
set -u
ROOT=/Users/fodelf/git/ds4-main
MODEL=${MODEL:-$ROOT/gguf/ds4-mono-mixed.gguf}
CTX=${CTX:-4096}
NPRED=${NPRED:-${2:-100}}
OUT=${OUT:-/tmp/mono_single.out}
LOG=${LOG:-/tmp/mono_single.log}
DEFAULT_PROMPT='<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {'
PROMPT=${1:-${PROMPT:-$DEFAULT_PROMPT}}

# 实例锁: 清掉本机残留 ds4(单机独占; 只杀进程不删文件)。避免"another ds4 already running; refusing to start"。
pkill -f 'ds4 --role' 2>/dev/null; pkill -f "ds4 -m " 2>/dev/null; sleep 1

# 外部内存看门狗(引擎已有同款, 这是双保险): 系统压力持续 CRITICAL ~6s → 杀 ds4, 抢在内核 watchdog panic 前。
# 用 pressure-level(1正常/2警告/4临界), 不用 free%(offload 下 page cache 填满 free% 本就低, 会误杀)。
( crit=0; while :; do
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
    if [ "$lvl" -ge 4 ]; then crit=$((crit+1)); else crit=0; fi
    [ "$crit" -ge 3 ] && { echo "[watchdog] 系统压力 critical → 杀 ds4" >&2; pkill -f "ds4 -m $MODEL"; break; }
    sleep 2
  done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT INT TERM

echo "=== mono 单机 $(date +%H:%M:%S) model=$(basename "$MODEL") ctx=$CTX n=$NPRED ===" >&2
echo "---------- 生成(原始输出) ----------"
perl -e 'alarm 600; exec @ARGV' \
  "$ROOT/ds4" -m "$MODEL" --prefill-chunk 512 --ctx "$CTX" --temp 0 -n "$NPRED" -p "$PROMPT" --metal \
  2> "$LOG" | tee "$OUT"
rc=$?
kill $WD 2>/dev/null
echo ""
echo "---------- 结束 (退出 $rc) ----------"
echo "原始输出: $OUT   诊断日志(t/s/内存): $LOG" >&2
grep -aE "t/s|prefill|generation" "$LOG" 2>/dev/null | tail -2 >&2
