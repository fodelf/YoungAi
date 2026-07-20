#!/bin/sh
# r5_residual_dual.sh — R5-B 热专家残差插件：全 M1 发射（M4 的 HF 副本已删，
# 只有 M1 有 HF 原始 shard），分三波控峰值磁盘 ≤ ~12.5G（M1 空闲 24G）。
#
#   wave0: L25-42 —— M1 的 v2 稀疏切片自带这些层的底座字节，直接发射
#   wave1: L0-12  —— 从 M4 v2 splice 底座字节到自建临时稀疏文件 → 发射 → 删临时
#   wave2: L13-24 —— 同 wave1
#   merge: 三个 partial scp 回 M4 → merge_residual → gguf/sidecars/go-hot-res.gguf
#
# 幂等：每波输出存在即跳过；临时底座文件是本脚本自建的，用完即删（非用户数据）。
# 用法: r5_residual_dual.sh [ACTIVE_FILE_ON_M1] [OUT_GGUF] [PART_TAG]
set -e
ROOT=/Users/fodelf/git/ds4-main
GT=$ROOT/gguf-tools
M1=192.168.1.2
M1ROOT=/Users/fodelf/ds4-main
ACTIVE_M1=${1:-$M1ROOT/active32.txt}
OUT=${2:-$ROOT/gguf/sidecars/go-hot-res.gguf}
TAG=${3:-}
V2=$ROOT/gguf/ds4-go1b-v2.gguf
PY=$M1ROOT/cap_work/venv/bin/python
RS=$M1ROOT/gguf-tools/go-onebit/cluster/gguf_range_stream.py

# ---- wave0: L25-42（全本地）----
if ! ssh -o BatchMode=yes $M1 "test -f /tmp/res_m1a$TAG.gguf"; then
  echo "[wave0] M1 L25-42 发射"
  ssh -o BatchMode=yes $M1 "cd $M1ROOT/gguf-tools && ./emit_residual \
    --hf $M1ROOT/hf/DeepSeek-V4-Flash-Base --base-gguf $M1ROOT/gguf/ds4-go1b-v2.gguf \
    --out /tmp/res_m1a$TAG.gguf --layers \$(seq -s, 25 42) --active-experts $ACTIVE_M1 \
    < /dev/null 2>&1"
else echo "[wave0] 已存在，跳过"; fi

# ---- wave1/wave2: 需要 M4 底座字节的层 ----
V2SIZE=$(stat -f%z "$V2")
for W in "1 0-12 res_m1b" "2 13-24 res_m1c"; do
  set -- $W; N=$1; RANGE=$2; PART=$3$TAG
  if ssh -o BatchMode=yes $M1 "test -f /tmp/$PART.gguf"; then echo "[wave$N] 已存在，跳过"; continue; fi
  echo "[wave$N] 播种临时底座头 + splice L$RANGE (~10.5G)"
  # 头部（data0 前）字节 + 稀疏 truncate 到全尺寸 → bg_open 可解析、洞读零
  dd if="$V2" bs=1m count=6 2>/dev/null | ssh -o BatchMode=yes $M1 \
    "cat > /tmp/v2base_tmp.gguf && $PY -c \"import os;os.truncate('/tmp/v2base_tmp.gguf',$V2SIZE)\""
  python3 $GT/go-onebit/cluster/gguf_range_stream.py send "$V2" "$RANGE" \
    | ssh -o BatchMode=yes $M1 "$PY $RS recv /tmp/v2base_tmp.gguf" 2>&1 | tail -2
  echo "[wave$N] 发射 L$RANGE"
  ssh -o BatchMode=yes $M1 "cd $M1ROOT/gguf-tools && ./emit_residual \
    --hf $M1ROOT/hf/DeepSeek-V4-Flash-Base --base-gguf /tmp/v2base_tmp.gguf \
    --out /tmp/$PART.gguf --layers \$(echo \$(seq $(echo $RANGE | tr - ' ')) | tr ' ' ,) \
    --active-experts $ACTIVE_M1 < /dev/null 2>&1; \
    rm -f /tmp/v2base_tmp.gguf"
done

# ---- merge on M4 ----
echo "[merge] 取回 partial 并合并"
scp -o BatchMode=yes -q $M1:/tmp/res_m1a$TAG.gguf $M1:/tmp/res_m1b$TAG.gguf $M1:/tmp/res_m1c$TAG.gguf /tmp/
mkdir -p $ROOT/gguf/sidecars
$GT/merge_residual --out "$OUT" /tmp/res_m1b$TAG.gguf /tmp/res_m1c$TAG.gguf /tmp/res_m1a$TAG.gguf
ls -lh "$OUT"
echo R5-RESIDUAL-DUAL-DONE
