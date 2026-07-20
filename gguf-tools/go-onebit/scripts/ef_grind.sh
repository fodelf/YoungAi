#!/bin/sh
# ef_grind.sh — 健壮单机(M4)EF 编码 go2b 层: 流式 pack + 每步验证 + 重试 + 输出验满。
# 治此前一长串静默瞬态失败(ffn_in scp 丢/pack 短传/热专家内存卡)。
# 用法: sh ef_grind.sh L1 L2 ...   (在 M4 跑; HF 从 M1 流)
set -u
M1=192.168.1.2; M1ROOT=/Users/fodelf/ds4-main; ROOT=/Users/fodelf/git/ds4-main
ART=$ROOT/gguf/v3-artifacts; W=/tmp/efm4; mkdir -p "$W"
MONO=$ROOT/gguf/ds4-mono-mixed.gguf   # 每层 EF 后原地 splice 进 mono(零增长)再删 bin, M4 盘不堆积
PROBE=$ROOT/gguf-tools/go-onebit/go_heldout_48.txt   # 每层 splice 后短 Go NLL 快检切片(~50 token, ds4 要求>32)
FULL=1600000000   # 完整 go2b 层字节下限(实际~1.72G)
LOG=/tmp/ef_grind.log; : > "$LOG"
DONE=/tmp/ef_grind.done; touch "$DONE"   # 已 splice 进 mono 的层, 幂等跳过

fetch() {  # $1=远端路径 $2=本地路径 $3=最小字节; 重试3次验大小
  for t in 1 2 3; do
    scp -o BatchMode=yes -q "$M1:$1" "$2" 2>/dev/null
    sz=$(stat -f%z "$2" 2>/dev/null || echo 0)
    [ "$sz" -ge "$3" ] && return 0
    echo "  fetch retry$t $2 (sz=$sz<$3)" >>"$LOG"; sleep 3
  done
  return 1
}
stream() {  # $1=layer $2=part $3=out $4=min
  for t in 1 2 3; do
    ssh -o BatchMode=yes $M1 "cd $M1ROOT && DS4_HF=$M1ROOT/hf/DeepSeek-V4-Flash-Base python3 gguf-tools/go-onebit/quant/pack_stream.py --layer $1 --part $2" > "$3" 2>/dev/null
    sz=$(stat -f%z "$3" 2>/dev/null || echo 0)
    [ "$sz" -ge "$4" ] && return 0
    echo "  stream retry$t L$1 $2 (sz=$sz)" >>"$LOG"; sleep 3
  done
  return 1
}

for L in "$@"; do
  grep -qx "L$L" "$DONE" && { echo "L$L 已 splice 跳过" | tee -a "$LOG"; continue; }
  O=$ART/go2b_ef_L$L.bin
  f=$(df -m /System/Volumes/Data|tail -1|awk '{print $4}'); [ "$f" -lt 9000 ] && { echo "L$L 盘${f}M<9G 停" | tee -a "$LOG"; break; }
  echo "=== L$L $(date +%H:%M:%S) 盘${f}M ===" | tee -a "$LOG"
  rm -f "$O" "$W"/pk.* "$W"/ffn_in_L$L.npy "$W"/route_L$L.npy   # 清残留(含 828M 截断 bin)
  fetch "$M1ROOT/cap_ef2/ffn_in_L$L.npy" "$W/ffn_in_L$L.npy" 20000000 || { echo "L$L ffn_in 取失败,跳" | tee -a "$LOG"; continue; }
  fetch "$M1ROOT/cap_ef2/route_L$L.npy" "$W/route_L$L.npy" 10000 || { echo "L$L route 取失败,跳" | tee -a "$LOG"; continue; }
  stream "$L" w8 "$W/pk.w8" 6000000000 || { echo "L$L pk.w8 流失败,跳" | tee -a "$LOG"; continue; }
  stream "$L" si "$W/pk.si" 1000000 || { echo "L$L pk.si 流失败,跳" | tee -a "$LOG"; continue; }
  python3 -c "import json;K={'gate':{'rows':2048,'cols':4096,'nblk':16,'si_shape':[16,32]},'up':{'rows':2048,'cols':4096,'nblk':16,'si_shape':[16,32]},'down':{'rows':4096,'cols':2048,'nblk':8,'si_shape':[32,16]}};json.dump({'layer':$L,'kinds':K},open('$W/pk.meta.json','w'))"
  python3 -u "$ROOT/gguf-tools/go-onebit/quant/ef_layer.py" --layer "$L" --pack "$W/pk" --ffn-in "$W/ffn_in_L$L.npy" --route "$W/route_L$L.npy" --out "$O" 2>&1 | grep -aE "EF-LAYER-OK|Error|Traceback" | tee -a "$LOG"
  sz=$(stat -f%z "$O" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$FULL" ]; then
    python3 -u "$ROOT/gguf-tools/go-onebit/quant/splice_go2b_inplace.py" --gguf "$MONO" --layer "$L" --bin "$O" 2>&1 | grep -aE "OK|Error|Traceback|assert" | tee -a "$LOG"
    echo "L$L" >> "$DONE"; echo "L$L OK+spliced ($sz)" | tee -a "$LOG"; rm -f "$O"   # splice 后删 bin, 盘保持
  else
    echo "L$L 输出不满($sz), 删" | tee -a "$LOG"; rm -f "$O"
  fi
  rm -f "$W"/pk.* "$W"/ffn_in_L$L.npy "$W"/route_L$L.npy
done
echo "EF-GRIND-DONE $(date +%H:%M:%S)" | tee -a "$LOG"
