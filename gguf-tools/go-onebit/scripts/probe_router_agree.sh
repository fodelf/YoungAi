#!/bin/bash
# 路由分歧针 (8-12 分钟): 引擎自路由 vs 教师锚路由的逐层 top-8 重合率。
# 判什么: 引擎 KL 1.14 vs 量化器 teacher-forced 回放 0.68 的 0.46 缺口
#         是否来自"量化后 router 换人"(口径差, 引擎无罪) 而非引擎数值 bug。
# 用法: probe_router_agree.sh [S_take=32] [chunk=8]
set -euo pipefail
cd /Users/fodelf/git/ds4-main
S=${1:-32}; CH=${2:-8}
M1=192.168.1.2
ANCHOR_M1=/Users/fodelf/ds4-main/gguf/go-onebit/r30/anchor_r30_s1716.bin
PY=gguf-tools/go-onebit/scripts/probe_router_agree.py
MODEL=gguf/go-onebit/ds4-r30.gguf
IDS=/tmp/ids32.txt
DUMP=/tmp/rid32; rm -rf $DUMP; mkdir -p $DUMP

echo "== [1/3] M1 抽教师 ridx 前 $S 位置 =="
scp -q $PY $M1:/tmp/probe_router_agree.py
ssh $M1 "python3 /tmp/probe_router_agree.py extract $ANCHOR_M1 $S /tmp/teacher_ridx$S.i32"
scp -q $M1:/tmp/teacher_ridx$S.i32 $DUMP/

echo "== [2/3] 引擎 $S 位置前向 + 逐层 dump ffn_moe_topk (裸=空链) =="
DS4_EVAL_IDS=$IDS DS4_EVAL_NO_BOS=1 DS4_EVAL_CHUNK=$CH \
DS4_EVAL_LOGITS=$DUMP/ev_bare.bin \
DS4_METAL_GRAPH_DUMP_PREFIX=$DUMP/eng \
DS4_METAL_GRAPH_DUMP_NAME=ffn_moe_topk \
DS4_METAL_GRAPH_DUMP_LAYER=all \
DS4_METAL_EXPERT_OFFLOAD=1 \
./ds4 -m $MODEL --zchain /tmp/zl_empty.bin 2>&1 | /usr/bin/grep -v "^ds4: dumped" | tail -6

echo "== [3/3] 逐层重合率表 =="
python3 $PY compare $DUMP/teacher_ridx$S.i32 $DUMP eng $S $CH
