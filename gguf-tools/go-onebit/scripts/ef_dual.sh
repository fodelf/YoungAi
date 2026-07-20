#!/bin/sh
# ef_dual.sh — 双机认领制 EF 重编 16 个 go2b 最差层 (M1+M4 各拉层并行)。
# 每层: 定位 ffn_in/route → ef_layer.py(HF pack + EF) → 写 gguf/v3-artifacts/go2b_ef_L{L}.bin。
# 认领凭据在 M1: EFDONE/claim_L{L}(mkdir原子) / 完成=bin 存在且非空。
# 用法: sh ef_dual.sh m1|m4   (两机各跑一个; m4 自动从 M1 流 HF pack)
set -u
M1=192.168.1.2; M1ROOT=/Users/fodelf/ds4-main; ROOT=/Users/fodelf/git/ds4-main
SSH="ssh -o BatchMode=yes $M1"
WHO=${1:?m1|m4}
ART=$ROOT/gguf/v3-artifacts                 # M4 侧产物目录 (M1 跑时写本地再回传)
LOCALART=$([ "$WHO" = m1 ] && echo $M1ROOT/gguf/v3-artifacts || echo $ART)
EFDONE=$M1ROOT/zdump/ef_done
LAYERS="2 23 24 25 26 27 28 29 30 31 32 33 34 36 38 40"
LOG=/tmp/ef_dual_$WHO.log
$SSH "mkdir -p $EFDONE $M1ROOT/gguf/v3-artifacts"
mkdir -p "$ART" /tmp/efwork

# ffn_in/route 源目录 (按层): L2=cap_l05, L23-36=cap_ef, L38-42=cap_v2r1_deep, L0-5=cap_l05
capdir() {
  L=$1
  for d in cap_ef cap_l05 cap_v2r1_deep cap_v2r2; do
    $SSH "test -f $M1ROOT/$d/ffn_in_L$L.npy" && { echo "$M1ROOT/$d"; return 0; }
  done
  return 1
}

while :; do
  L=""
  for c in $LAYERS; do
    # 完成判据: bin 存在
    $SSH "test -s $M1ROOT/gguf/v3-artifacts/go2b_ef_L$c.bin" && continue
    $SSH "mkdir $EFDONE/claim_L$c 2>/dev/null" || continue
    CAP=$(capdir "$c") || { echo "L$c: ffn_in 未就绪(等补采), 释放" | tee -a "$LOG"; $SSH "rmdir $EFDONE/claim_L$c"; continue; }
    L=$c; CAPD=$CAP; break
  done
  if [ -z "$L" ]; then
    N=$($SSH "ls $M1ROOT/gguf/v3-artifacts/go2b_ef_L*.bin 2>/dev/null | wc -l | tr -d ' '")
    [ "$N" -ge 16 ] && { echo "$WHO: 16 层全 EF 完成" | tee -a "$LOG"; break; }
    echo "$WHO: 无可认领(等补采) done=$N/16 $(date +%H:%M:%S)" | tee -a "$LOG"; sleep 120; continue
  fi
  echo "=== EF L$L $WHO cap=$CAPD $(date +%H:%M:%S) ===" | tee -a "$LOG"
  FIN=/tmp/efwork/fin_$L.npy; RT=/tmp/efwork/rt_$L.npy; PK=/tmp/efwork/pk_$L; OUT=/tmp/efwork/go2b_ef_L$L.bin
  if [ "$WHO" = m1 ]; then
    cp "$CAPD/ffn_in_L$L.npy" "$FIN"; cp "$CAPD/route_L$L.npy" "$RT"
  else
    scp -o BatchMode=yes -q "$M1:$CAPD/ffn_in_L$L.npy" "$FIN"; scp -o BatchMode=yes -q "$M1:$CAPD/route_L$L.npy" "$RT"
  fi
  python3 -u "$ROOT/gguf-tools/go-onebit/quant/ef_layer.py" --layer "$L" --pack "$PK" \
      --ffn-in "$FIN" --route "$RT" --out "$OUT" 2>&1 | grep -aE "EF-LAYER-OK|Error" | tee -a "$LOG"
  if [ -s "$OUT" ]; then
    if [ "$WHO" = m1 ]; then mv "$OUT" "$M1ROOT/gguf/v3-artifacts/"; else scp -o BatchMode=yes -q "$OUT" "$M1:$M1ROOT/gguf/v3-artifacts/"; fi
  else
    echo "$WHO L$L FAILED" | tee -a "$LOG"; $SSH "rmdir $EFDONE/claim_L$L 2>/dev/null"
  fi
  rm -f "$FIN" "$RT" "$PK".* "$OUT"
done
echo "$WHO EF-DUAL-EXIT $(date +%H:%M:%S)" | tee -a "$LOG"
