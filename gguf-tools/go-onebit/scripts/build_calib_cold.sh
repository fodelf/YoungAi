#!/bin/bash
# build_calib_cold.sh — ②冷专家覆盖: 大杂烩校准语料(2026-07-24, 按序执行第2项)。
# 背景: S=530 校准下平均 64/256 专家/层零校准行(最差层94)=胡言触发面根源。
# 设计: 10语言×(mixed+complex)节选 + Go书籍散文/代码混排节选 + 现役calib为前缀(连续性),
#       目标 ~2600 token; ★判决语料(coding_hard/hard_eval→rr_code/rr_hard)独立不混入(防泄漏)★。
# 产物: corpus/calib_cold_v1.txt(repo固化) + /tmp/rr_calib_cold_v1.ids(分词, TTL易失可重生)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CD="$ROOT/gguf-tools/go-onebit/corpus"
OUT="$CD/calib_cold_v1.txt"
: > "$OUT"
cat "$CD/calib_prog_v1.txt" >> "$OUT"; echo >> "$OUT"          # 现役校准为前缀
cat "$CD/calib_prog_v2.txt" >> "$OUT"; echo >> "$OUT"
for lang in go rust python javascript typescript java c shell gin; do
    for tier in mixed complex; do
        f="$CD/build/${lang}_${tier}.txt"
        [ -f "$f" ] && { head -c 700 "$f" >> "$OUT"; echo >> "$OUT"; }
    done
done
# Go 书籍节选(中文散文+代码混排 → 路由撒得开): 取 3 章各 1200B
i=0
for b in "$CD"/books/*ch1-06* "$CD"/books/*ch2-0[12]* "$CD"/books/*appendix-a*; do
    [ -f "$b" ] || continue
    head -c 1200 "$b" >> "$OUT"; echo >> "$OUT"
    i=$((i+1)); [ "$i" -ge 3 ] && break
done
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$OUT" <<'PEOF'
import sys
p = sys.argv[1]
t = open(p, "rb").read().decode("utf-8", errors="ignore")   # head -c 截断的半个多字节字符清掉
open(p, "w").write(t)
PEOF
wc -c "$OUT"
HF_TOK="${DS4_HF_TOK:-$ROOT/hf/DeepSeek-V4-Flash-Backbone/tokenizer.json}"
[ -f "$HF_TOK" ] || HF_TOK="$ROOT/hf/DeepSeek-V4-Flash-Base/tokenizer.json"
"$PY" - "$HF_TOK" "$OUT" /tmp/rr_calib_cold_v1.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok")
PEOF
