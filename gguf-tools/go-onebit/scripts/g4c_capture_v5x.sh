#!/bin/bash
# g4c_capture_v5x.sh — v5 语料分块 FP 态 X 捕获(全 43 层, 2026-07-28)。
# 用户令改向: 每层短针扫最小 bit → 最小体积 → 需要全层 X, 一遍前向全 dump。
# 机制: 每块 DS4_FP_ONLY 建锚(NL=43), ANC_BUILD 内 DS4_BF_DUMPXR=0..42 落
#       /tmp/xr_x_L*.npy → 收进块目录; 锚文件即建即删; 拼接后逐层删块文件(盘 ~9G)。
# 前置: g4c_cells.py 已产 /tmp/g4c_chunks/chunk_*.ids; ds4quant_run 已带 XR-FP 补丁重编。
# 产物: /tmp/g4c_x_L00..L42.npy (全 4699 行拼接, 行号=g4c_cells.tsv global; 43×77MB≈3.3G)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
QDIR="$ROOT/gguf-tools/go-onebit/quant"
CH=/tmp/g4c_chunks
cd "$QDIR"; [ -x ./ds4quant_run ] || { echo "[g4c] ds4quant_run 缺失" >&2; exit 2; }
ls "$CH"/chunk_*.ids >/dev/null 2>&1 || { echo "[g4c] 块缺失, 先跑 g4c_cells.py" >&2; exit 2; }

# 内存看门狗(铁律): footprint >11G 杀捕获进程
WDOG(){ while true; do
    P=$(pgrep -nf "ds4quant_run $CH" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    if [ -n "${MB:-}" ] && [ "$MB" -gt 11264 ]; then
        echo "[g4c][watchdog] ${MB}MB >11G → 杀捕获进程" >&2; kill -9 "$P" 2>/dev/null || true; fi
    sleep 5; done }
WDOG & WPID=$!
trap 'kill $WPID 2>/dev/null || true' EXIT

NCH=$(ls "$CH"/chunk_*.ids | wc -l | tr -d ' ')
DUMP=$(seq -s, 0 42)
i=0
for f in "$CH"/chunk_*.ids; do
    c=$(basename "$f" .ids); n=$(grep -c . "$f")
    echo "[g4c] 块 $((i+1))/$NCH ($c, $n tok) FP 前向(全43层 dump)…" >&2
    rm -f /tmp/xr_x_L*.npy /tmp/g4c_anchor.bin
    DS4_ANCHOR=/tmp/g4c_anchor.bin DS4_NL=43 DS4_FP_ONLY=1 DS4_BF_DUMPXR="$DUMP" \
      DS4_THREADS="${DS4_THREADS:-6}" ./ds4quant_run "$f" "$n" > "/tmp/g4c_${c}.log" 2>&1 \
      || { echo "[g4c] 块 $c 前向失败:" >&2; tail -5 "/tmp/g4c_${c}.log" >&2; exit 1; }
    [ -f /tmp/xr_x_L00.npy ] && [ -f /tmp/xr_x_L42.npy ] \
      || { echo "[g4c] 块 $c 未落全层 X(XR-FP 补丁没生效?)" >&2; exit 1; }
    for x in /tmp/xr_x_L*.npy; do mv "$x" "$CH/$(basename "$x" .npy)_$c.npy"; done
    rm -f /tmp/g4c_anchor.bin
    i=$((i+1))
done
python3 - <<'PEOF' >&2
import glob, os, numpy as np
for L in range(43):
    fs = sorted(glob.glob(f"/tmp/g4c_chunks/xr_x_L{L:02d}_chunk_*.npy"))
    X = np.concatenate([np.load(f) for f in fs])
    np.save(f"/tmp/g4c_x_L{L:02d}.npy", X)
    for f in fs: os.remove(f)                    # 逐层删块文件控盘
    if L % 10 == 0 or L == 42:
        print(f"[g4c] L{L:02d}: {len(fs)} 块拼接 {X.shape}")
PEOF
echo "[g4c] 捕获完成" >&2
