#!/bin/sh
# m4_lane.sh — M4 参战 lane (跑在 M4): 与 M1 池共享 done/claim 认领协议, 从 L42 往下抢
# (M1 三路从低层往上, 两端夹击). 每层: M1 打包 → TB 线传(实测~650MB/s) → M4 GPTQ → sign 回传 → M1 拼接.
# 错误处理: 逐步显式检错(不依赖 set -e — 在 `(...)||` 下会被 POSIX 语义整体抑制, 烟测踩过);
# 任一步失败: 保留日志/释放认领/清两侧临时件, M1 池会捡走该层. 包在本地落地后立刻删 M1 侧(防 /tmp 爆盘).
# 用法: sh m4_lane.sh [LANE_ID]   (多开即多 lane)
set -u
M1=192.168.1.2
M1ROOT=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
DONE=$M1ROOT/zdump/v3_gptq1_done
LANE=${1:-1}
LO=${LO:-6}; HI=${HI:-42}   # 认领层范围 (L0-5 补采后: LO=0 HI=5)
CAPDIRS=${CAPDIRS:-}          # 额外 cap 目录 (逗号前缀拼给 pack_layer --capdirs)
# 实例锁 (macOS 无 flock, 用 mkdir 原子性): 同 LANE 双开会共用包名互踩 (烟测踩过: 1/4 尺寸 si)
LOCK=/tmp/m4lane$LANE.lockdir
mkdir "$LOCK" 2>/dev/null || { echo "m4lane$LANE already running (rm -rf $LOCK 强启)"; exit 1; }
TMP=/tmp/m4lane$LANE.$$; mkdir -p "$TMP"   # 每实例独立工作区+包名, 双保险
SSH="ssh -o BatchMode=yes $M1"
EXT="w8 si s b ffn_in.npy route.npy check.npy meta.json"

wd() { # 本机看门狗: 本 lane 的 pack_compute 单进程 >3.5G 杀
  while :; do
    for P in $(pgrep -f "gptq1_pack_compute.*m4lane$LANE"); do
      R=$(ps -o rss= -p "$P" 2>/dev/null | tr -d " ")
      [ -n "$R" ] && [ "$R" -gt 4718592 ] && { echo "M4-WDKILL pid=$P rss=${R}KB"; kill -9 "$P"; }
    done
    sleep 20
  done
}
wd & WD=$!
trap 'kill $WD 2>/dev/null; rm -rf "$LOCK" "$TMP"' EXIT

release() { # $1=L $2=原因 — 释放认领+清两侧临时件, 保留 $TMP/L$1.log ($PK 在调用点已就位)
  echo "m4lane$LANE L$1 FAILED($2), release claim"
  $SSH "rm -rf $DONE/claim_L$1; rm -f /tmp/$PK.* /tmp/signs_L$1.bin" 2>/dev/null
  rm -f "$TMP/$PK".* "$TMP/signs_L$1.bin"
}

while :; do
  L=""
  for c in $(seq "$HI" -1 "$LO"); do
    $SSH "[ -f $DONE/L$c ] && exit 1; mkdir $DONE/claim_L$c 2>/dev/null" && { L=$c; break; }
  done
  [ -z "$L" ] && { echo "m4lane$LANE: no layers left"; break; }
  PK="pk${LANE}x$$_L$L"
  echo "=== L$L m4lane$LANE $(date +%H:%M:%S) ===" | $SSH "tee -a /tmp/v3_gptq1.log"
  T0=$(date +%s)

  PO=$($SSH "cd $M1ROOT && DS4_HF=$M1ROOT/hf/DeepSeek-V4-Flash-Base python3 gguf-tools/go-onebit/quant/pack_layer.py --layer $L --out /tmp/$PK ${CAPDIRS:+--capdirs $CAPDIRS} && echo PACK-VERIFIED" 2>&1 | tail -4)
  printf '%s\n' "$PO"
  printf '%s' "$PO" | grep -q PACK-VERIFIED || { release "$L" pack; sleep 10; continue; }

  ok=1
  for x in $EXT; do
    scp -o BatchMode=yes -q "$M1:/tmp/$PK.$x" "$TMP/" || { ok=0; break; }
  done
  [ "$ok" = 1 ] || { release "$L" scp-in; sleep 10; continue; }
  $SSH "rm -f /tmp/$PK.*"                       # 包落地即清 M1 侧, 防 /tmp 爆盘
  T1=$(date +%s); echo "m4lane$LANE L$L pack+ship $((T1-T0))s"

  python3 -u "$ROOT/gguf-tools/go-onebit/quant/gptq1_pack_compute.py" \
    --pack "$TMP/$PK" --signs-out "$TMP/signs_L$L.bin" \
    --json "$TMP/v3_gptq1_L$L.json" > "$TMP/L$L.log" 2>&1 \
    || { tail -3 "$TMP/L$L.log"; release "$L" compute; sleep 10; continue; }
  grep -aE "SUMMARY|decode-parity" "$TMP/L$L.log" | $SSH "tee -a /tmp/v3_gptq1.log" || true

  scp -o BatchMode=yes -q "$TMP/signs_L$L.bin" "$TMP/v3_gptq1_L$L.json" "$TMP/L$L.log" "$M1:/tmp/" \
    || { release "$L" scp-out; sleep 10; continue; }
  $SSH "cd $M1ROOT && python3 gguf-tools/go-onebit/quant/splice_signs.py --gguf gguf/ds4-go1b-v3.gguf --layer $L --signs /tmp/signs_L$L.bin && mv -f /tmp/L$L.log /tmp/v3_gptq1_L$L.log && touch $DONE/L$L && rm -rf $DONE/claim_L$L && rm -f /tmp/signs_L$L.bin" \
    || { release "$L" splice; sleep 10; continue; }

  T2=$(date +%s); echo "m4lane$LANE L$L total $((T2-T0))s"
  find "$TMP" -name "$PK*" -delete 2>/dev/null; rm -f "$TMP/signs_L$L.bin"
done
