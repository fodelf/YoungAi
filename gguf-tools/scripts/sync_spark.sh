#!/bin/bash
# sync_spark.sh — 把 Mac 上的源码同步到 spark 并重编(2026-09-13)。
#
# 为什么要这个脚本: Mac 是唯一编辑源, spark 是唯一跑真模型的机器, 每轮改动都要"传过去再编"。
# 手敲 rsync 排除列表迟早漏一项 —— 漏了 gguf/ 就是几百 GB 白传, 漏了 *.o 就是拿 Mac 的
# Mach-O 目标文件去污染 Linux 编译(实撞过: bench 下一串工具变成 Mach-O, 在 spark 上跑不了)。
#
# 用法: sync_spark.sh [engine|tools|all]   默认 all
#   engine = make cuda-spark(引擎二进制 ds4 等)
#   tools  = make -C gguf-tools anchor_metrics v41_amp_run
# 不传目标就两样都编。★只同步源码与脚本, 不碰对方的 gguf/ 与产物目录★。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHAT="${1:-all}"
REMOTE="spark"
RDIR="ds4-main"

cd "$ROOT" || exit 1
# ★不覆盖正在跑的脚本★(2026-09-13 实撞): rsync 把 spark 上正在执行的 z_nightly_spark.sh 换掉了 ——
# bash 是按文件偏移边读边执行的, 换掉 = 从某一行起执行的是新文件的内容, 而且不会报错。
# 所以同步前先看对面有没有在跑本仓的长任务, 有就拒绝。
if ssh "$REMOTE" 'pgrep -f "z_nightly_spar[k]|v41_amp_ru[n]|ds[4] --cuda|speed-benc[h]/" >/dev/null'; then
    echo "★对面有本仓的任务在跑, 拒绝同步(先停它, 或等它跑完)★"
    ssh "$REMOTE" 'pgrep -a -f "z_nightly_spar[k]|v41_amp_ru[n]|ds[4] --cuda|speed-benc[h]/" | head -3'
    exit 1
fi
echo "[sync] $ROOT → $REMOTE:$RDIR"
rsync -az --delete \
    --exclude '.git' --exclude 'gguf' --exclude 'hf' --exclude '*.o' --exclude '*.dSYM' \
    --exclude 'ds4' --exclude 'ds4-server' --exclude 'ds4-bench' --exclude 'ds4-eval' \
    `# ★护住基线二进制★(2026-09-16 实撞): 判决前会 cp ds4 ds4.base 留一份改造前的, 而它只在 spark 上、` \
    `# Mac 这边没有 ⇒ --delete 一来就把它删了, 门跑起来只报"无基线", 白跑一趟六次装模型。` \
    --exclude '*.base' --exclude 'ds4.base*' \
    --exclude 'ds4-agent' --exclude 'ds4_test' --exclude 'ds4_unit' --exclude 'zsolve' \
    `# ★工具二进制一律不同步★(2026-09-17 实撞): Mac 上编过一次的 gguf-tools 产物会被推到 Linux,` \
    `# 覆盖那边的原生版本。症状是跑起来报 "Syntax error: ( unexpected"(内核认不出 Mach-O,` \
    `# 退给 /bin/sh 当脚本读) —— 一眼看不出是二进制串了台。ELF/Mach-O 只按扩展名认不出来,` \
    `# 所以按"无扩展名的可执行文件"整类排除, 各机器自己 make。` \
    --exclude 'gguf-tools/bench/anchor_metrics' --exclude 'gguf-tools/bench/pubbench' \
    --exclude 'gguf-tools/bench/dspark_agree' --exclude 'gguf-tools/bench/kl_forensic' \
    --exclude 'gguf-tools/quantize/v41_to_gguf' --exclude 'gguf-tools/quantize/dump_gguf_meta' \
    --exclude 'gguf-tools/quantize/deepseek4-quantize' --exclude 'gguf-tools/dspark_align' \
    --exclude 'gguf-tools/amp/v41_amp_run' --exclude 'gguf-tools/amp/finetune_solve' \
    --exclude 'speed-bench' --exclude 'reports' --exclude 'notes' \
    ./ "$REMOTE:$RDIR/" || { echo "★rsync 失败★"; exit 1; }
# ★speed-bench 的脚本要单独推一遍★(2026-09-16): 上面整目录排除了 speed-bench —— 因为那边放着几份
# 大语料, 而且 spark 上还有本机生成的长提示文件, 带 --delete 的整目录同步会把它们删掉。
# 但尺子脚本(*.sh)必须跟着代码走, 不然改了门这边看不见, 判决还是按旧尺出的。只推 .sh, 不删任何东西。
rsync -az speed-bench/*.sh "$REMOTE:$RDIR/speed-bench/" || { echo "★speed-bench 脚本同步失败★"; exit 1; }

if [ "$WHAT" = engine ] || [ "$WHAT" = all ]; then
    echo "[sync] make cuda-spark"
    ssh "$REMOTE" "cd $RDIR && make cuda-spark 2>&1 | tail -5" || { echo "★引擎编译失败★"; exit 1; }
fi
if [ "$WHAT" = tools ] || [ "$WHAT" = all ]; then
    # ★两个都要编★(2026-09-13 实撞): 只编了判决器, 解算器还是旧二进制 —— 新加的 "@行号文件"
    # 写法它不认识, 报的却是"行段不合法", 查了半天才发现是二进制没换。
    echo "[sync] make -C gguf-tools anchor_metrics v41_amp_run"
    ssh "$REMOTE" "cd $RDIR && make -C gguf-tools anchor_metrics 2>&1 | tail -3 && make -C gguf-tools v41_amp_run 2>&1 | tail -3" \
        || { echo "★工具编译失败★"; exit 1; }
fi
echo "[sync] 完成"
