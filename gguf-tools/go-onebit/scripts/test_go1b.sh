#!/bin/sh
# test_go1b.sh — 跑 go1b(1-bit) + z(四损失隐变量) 模型看输出, 一条命令自测.
#
# 关键(本次调试结论): go1b 量化自 DeepSeek-V4-Flash-BASE 基座模型. 基座没在 chat 标记
# (<｜User｜>/<｜Assistant｜>) 上训练过 -> 用 ds4 默认的 -p 会被 ds4_encode_chat_prompt 套进
# chat 模板, 基座就乱讲/答非所问. 本脚本默认把 prompt 以 <｜begin▁of▁sentence｜> 开头, 让 ds4
# 走 RAW 裸续写(is_rendered_chat_prompt() 命中 -> 原样 tokenize), 这才是基座该用的方式.
#
# 用法:
#   ./test_go1b.sh                      # 默认: Go Add 函数补全, +z, temp 0, -n 64
#   ./test_go1b.sh --preset hello       # 内置题: add | hello | fib | fact
#   ./test_go1b.sh -p 'func main() {'   # 自定义裸 prompt(自动加 BOS 前缀)
#   ./test_go1b.sh -n 128 -t 0.7        # 更长 + 采样(逃贪心循环)
#   ./test_go1b.sh --no-z               # 关 z(纯 1-bit base)做对照
#   ./test_go1b.sh --chat -p 'Write...' # 故意走 chat 模板(看基座乱讲, 仅对照)
#   ./test_go1b.sh --no-build           # 跳过编译(确定没改代码时快跑)
#   DS4_REPEAT_FREQ=3 ./test_go1b.sh    # 叠 repeat penalty 压退化循环
#
# 默认先 `make ds4` 增量编译最新代码再跑(没改源码就秒过); --no-build 跳过.
# offload 跑(45.6G 模型 mmap, 16G 内存安全, RSS 峰 ~3-4G); 前台等输出, ~2-6min.
set -e
ROOT="${ROOT:-/Users/fodelf/git/ds4-main}"
MODEL="${MODEL:-$ROOT/gguf/ds4-go1b.gguf}"
CORR="${CORR:-$ROOT/gguf/ds4-go1b-corr.gguf}"
BOS='<｜begin▁of▁sentence｜>'
N=64; TEMP=0; USE_Z=1; CHAT=0; PRESET=add; PROMPT=""; BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--prompt) PROMPT="$2"; shift 2;;
    -n|--tokens) N="$2"; shift 2;;
    -t|--temp)   TEMP="$2"; shift 2;;
    --no-z)      USE_Z=0; shift;;
    --chat)      CHAT=1; shift;;
    --no-build)  BUILD=0; shift;;
    --preset)    PRESET="$2"; shift 2;;
    -m|--model)  MODEL="$2"; shift 2;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (try --help)"; exit 1;;
  esac
done
[ -f "$MODEL" ] || { echo "model not found: $MODEL  (set MODEL=... or -m)"; exit 1; }
# 先编译最新 ds4(增量, 没改源码就秒过); --no-build 跳过
if [ "$BUILD" = 1 ]; then
  echo "=== build: make ds4 (增量编译最新代码) ==="
  ( cd "$ROOT" && make ds4 ) > /tmp/go1b_build.log 2>&1 \
    || { echo "BUILD FAILED:"; tail -20 /tmp/go1b_build.log; exit 1; }
  tail -1 /tmp/go1b_build.log; echo "build OK: $(ls -la "$ROOT/ds4" | awk '{print $5" bytes "$(NF-1)" "$NF}')"
fi
# 内置裸代码补全前缀(基座该续写出函数体)
if [ -z "$PROMPT" ]; then
  case "$PRESET" in
    add)   PROMPT='package main

// Add returns the sum of two integers.
func Add(a int, b int) int {';;
    hello) PROMPT='package main

import "fmt"

func main() {
	';;
    fib)   PROMPT='package main

// Fib returns the n-th Fibonacci number.
func Fib(n int) int {';;
    fact)  PROMPT='The sum of 2 and 3 is ';;
    *) echo "unknown preset: $PRESET (add|hello|fib|fact)"; exit 1;;
  esac
fi
# 写 prompt 文件: raw 模式加 BOS 前缀(绕模板); chat 模式不加(让 ds4 套模板)
PF=/tmp/go1b_test_prompt.txt
if [ "$CHAT" = 1 ]; then MODE=chat; printf '%s' "$PROMPT" > "$PF"
else MODE=raw; printf '%s%s' "$BOS" "$PROMPT" > "$PF"; fi
# z 开关
ZFLAG=""; ZDESC="NO z (纯 1-bit base)"
if [ "$USE_Z" = 1 ]; then
  [ -f "$CORR" ] && { ZFLAG="--corr $CORR"; ZDESC="+z (1-bit + 四损失隐变量)"; } \
                 || echo "warn: corr sidecar 不存在($CORR), 退回纯 1-bit"
fi
echo "=== go1b test | $ZDESC | mode=$MODE | temp=$TEMP | n=$N ==="
echo "--- prompt ---"; printf '%s\n' "$PROMPT"
echo "--- output (基座续写; ds4 不回显 prompt) ---"
cd "$ROOT"
./ds4 -m "$MODEL" $ZFLAG --prompt-file "$PF" -n "$N" --temp "$TEMP" --metal \
      > /tmp/go1b_test_out.txt 2> /tmp/go1b_test_err.txt || {
  echo "ds4 failed:"; tail -8 /tmp/go1b_test_err.txt; exit 1; }
cat /tmp/go1b_test_out.txt
echo ""
echo "--- status ---"
grep -iE 'correction loaded|prefill:|generation:' /tmp/go1b_test_err.txt || true
