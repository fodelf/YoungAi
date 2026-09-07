#!/bin/bash
# speed_longctx_spark.sh — 长上下文合成点(1M / 512k)速度(spark 本机跑, 当前 ./ds4-bench)。
#
# 为什么: 8192 曲线量的是短上下文; 1M 的账(indexer 全扫/top-k/ratio-128 注意力)只有把上下文真填到
# 1M 才露出来。--fill-ctx 合成填充(内容零, 只算速度), 在 1046528 处再真预填 2048 + 生成 64。
# 用法: speed_longctx_spark.sh <tag> [工作区=champ86q4k] [drafter GGUF]   产物: vqhalf/<ws>/speed/fill_{1m,512k}_<tag>.csv
#   第 3 个参数给了就走投机(--spec --draft-gguf, 09-07): 量 MTP 在 1M/512k 的真速度, 账还打 spec 账(轮数/接受率)。
# 内存账同 speed_champ_spark.sh(--mem-budget-mb 110000; 1M 上下文缓冲 11.6 GiB + kv 7.6 GiB, 09-07 实跑 39 GiB 余;
# drafter 再加 11.4 GB, 超预算由引擎 L1 静态闸拒绝启动而不是 OOM)。
set -uo pipefail
TAG="${1:?tag}"; WS="${2:-champ86q4k}"; DRAFT="${3:-}"
# 英文语料 = README×80(≥1M token, 4.9 MB), 现生成不入库
[ -s speed-bench/readme_en_x80.txt ] || for i in $(seq 80); do cat README.md; echo; done > speed-bench/readme_en_x80.txt
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
OUT="$ROOT/gguf/go-onebit/vqhalf/$WS/speed"; mkdir -p "$OUT"
LOG(){ echo "[longctx-$TAG $(date +%H:%M:%S)] $*"; }
BUSY=$(for p in ds4 ds4-bench ds4-server; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
M=(--cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000
   --prefill-chunk 2048 --gen-tokens 64 --prompt-file speed-bench/readme_en_x80.txt)   # 09-07 用户令: 测试禁用意语, 英文 README×80(≥1M token)
if [ -n "$DRAFT" ]; then [ -s "$DRAFT" ] || { LOG "★drafter 缺 $DRAFT★"; exit 2; }; M+=(--spec --draft-gguf "$DRAFT"); fi
run(){ # $1=名 $2=fill $3=ctx
    LOG "== $1"
    ./ds4-bench "${M[@]}" --fill-ctx "$2" --ctx-start "$3" --ctx-max "$3" --csv "$OUT/fill_$1_$TAG.csv" 2>&1 \
        | grep -E "KV policy|failed|error|spec 账|spec 逐位|拒绝|budget" | cut -c1-200
    cat "$OUT/fill_$1_$TAG.csv"
}
run 1m 1046528 1048576
run 512k 522240 524288
LOG "LONGCTX_${TAG}_END"
