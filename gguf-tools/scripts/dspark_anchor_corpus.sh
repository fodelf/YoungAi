#!/bin/bash
# dspark_anchor_corpus.sh — 用**与量化同一套的冻结全场景小语料**驱动 drafter 锚捕获
# (2026-08-21 用户纠正: 此前用我自编的两个提示当校准分布, 覆盖面太窄)。
# 语料 = gguf-tools/data/corpus 下的冻结集(zh 技术散文/英文通用/编程/金融/冷门),
# 每块取一段当 prompt, 部署态(q2 drafter + SPEC)生成一小段, 逐块落锚后拼接。
# 用法: [CHUNKS=12] [NGEN=120] bash dspark_anchor_corpus.sh <out_anchor.bin>
set -uo pipefail
ROOT="$HOME/ds4-main"
C="$ROOT/gguf-tools/data/corpus"
OUT="${1:-$ROOT/gguf/go-onebit/r30/dspark/anchor_corpus.bin}"
TMP="$(dirname "$OUT")/.anchor_parts"
STUDENT="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
MAIN="$ROOT/gguf/ds4-allq2.gguf"
MAIN_ZC="$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin"
NGEN="${NGEN:-120}"
CHUNKS="${CHUNKS:-12}"
mkdir -p "$TMP"; rm -f "$TMP"/*.bin
cd "$ROOT"

# 冻结语料池(全场景): 每个文件按行取样, 跳过分节标记行
FILES=("$C/calib_general_v1.txt" "$C/calib_prog_v5.txt" "$C/calib_cold_v1.txt" "$C/calib_fin_v2.txt" "$C/calib_v2.txt")
python3 - "$TMP" "$CHUNKS" "${FILES[@]}" <<'PY'
import sys, os, random
tmp, n = sys.argv[1], int(sys.argv[2])
files = [f for f in sys.argv[3:] if os.path.exists(f)]
segs = []
for f in files:
    lines = [l.strip() for l in open(f, encoding="utf-8", errors="replace")
             if len(l.strip()) > 40 and not l.strip().startswith("====")]
    # 每文件均分取样, 保证场景覆盖而不是被大文件淹没
    if not lines: continue
    step = max(1, len(lines) // max(1, n // len(files) + 1))
    segs += lines[::step][: n // len(files) + 1]
random.Random(7).shuffle(segs)
segs = segs[:n]
for i, s in enumerate(segs):
    open(os.path.join(tmp, f"p{i:02d}.txt"), "w", encoding="utf-8").write(s[:1200])
print(f"语料分块: {len(segs)} 段, 来自 {len(files)} 个冻结文件")
PY

i=0
for f in "$TMP"/p*.txt; do
    P="$(cat "$f")"
    # ⚠DSPARK_ANCHOR 捕获诊断已随 env 大扫除删除(2026-08-31): 本脚本的锚采集腿失效,
    # 跑出来的 a*.bin 不会生成。需要 drafter 锚时按新机制重建采集器再启用。
    timeout 900 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" --spec --temp 0 -n "$NGEN" \
        -p "$P" </dev/null >/dev/null 2>&1
    sz=$(stat -c%s "$TMP/a$(printf %02d $i).bin" 2>/dev/null || echo 0)
    echo "  块 $i: 锚 $((sz/1000000))MB"
    i=$((i+1))
done
cat "$TMP"/a*.bin > "$OUT"
ls -l "$OUT"
