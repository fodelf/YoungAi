#!/bin/bash
# make_calib_s305.sh — "同S换语料域"判决锚: 四支柱分层等份 ~305 tok 校准流。
# 与 v2 的 rr_code.ids(S=305 纯代码)同量级, 只换构成 → 隔离"数据构成"效应。
# 切片全部真语料: soul规约/方法论穿透段/issuefix线程头/skills Phase1/真项目map惯用法。
# 出: calib_v2_s512.txt + /tmp/rr_calib_s512.ids;  硬断言: hard_eval 判决锚不得混入。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
OUT=calib_v2_s512.txt

{ sed -n '3,10p'  soul/honesty_v1.txt                                   # 规约首条
  sed -n '/## Cache penetration/,/Bloom filter in front/p' method/methodology_core.md
  head -c 300 build/ref_pillars/p1_issuefix.txt; echo
  sed -n '/### Phase 1/,/Reproduce Consistently/p' \
      skills/obra__superpowers__skills_systematic-debugging_SKILL.md
  grep -A14 "func TestDuplicatedLinks" raw/go/avelino__awesome-go.code.txt | head -15
} > "$OUT"

HARD_SIG="$(head -c 120 hard_eval.txt | tail -c 60)"
grep -qF "$HARD_SIG" "$OUT" && { echo "[FATAL] 判决锚污染"; exit 1; }

HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$HF/tokenizer.json" "$OUT" /tmp/rr_calib_s512.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok (v2 判决口径 S=512 档)")
PEOF
wc -c "$OUT"
