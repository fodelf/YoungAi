#!/bin/bash
# build_hard_v2.sh — ③rr_hard 大样(2026-07-24, 按序执行第3项): 硬多样文本 ≥400 token。
# 源: hard_eval.txt(现役64tok判决集为前缀, 连续性) + books 不同章节节选(与 calib_cold_v1 用章严格不同,
# calib用了 ch1-06/ch2-01/ch2-02/appendix-a → 这里用 ch3/ch4/ch5/ch6 — 校准/判决隔离)。
# 产物: corpus/hard_eval_v2.txt(repo) + /tmp/rr_hard_v2.ids
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CD="$ROOT/gguf-tools/go-onebit/corpus"
OUT="$CD/hard_eval_v2.txt"
: > "$OUT"
cat "$CD/hard_eval.txt" >> "$OUT"; echo >> "$OUT"
i=0
for b in "$CD"/books/*ch3-0[123]* "$CD"/books/*ch4-0[12]* "$CD"/books/*ch5-0[12]* "$CD"/books/*ch6-0[12]*; do
    [ -f "$b" ] || continue
    head -c 900 "$b" >> "$OUT"; echo >> "$OUT"
    i=$((i+1)); [ "$i" -ge 6 ] && break
done
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$OUT" <<'PEOF'
import sys
p = sys.argv[1]
t = open(p, "rb").read().decode("utf-8", errors="ignore")   # 先读后写(open"w"即截断, 顺序错=清空文件)
open(p, "w").write(t)
PEOF
wc -c "$OUT"
HF_TOK="${DS4_HF_TOK:-$ROOT/hf/DeepSeek-V4-Flash-Backbone/tokenizer.json}"
[ -f "$HF_TOK" ] || HF_TOK="$ROOT/hf/DeepSeek-V4-Flash-Base/tokenizer.json"
"$PY" - "$HF_TOK" "$OUT" /tmp/rr_hard_v2.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok")
PEOF
