#!/bin/bash
# make_calib_s305.sh — "同S换语料域"判决锚: 四支柱分层等份 ~305 tok 校准流。
# 与 v2 的 rr_code.ids(S=305 纯代码)同量级, 只换构成 → 隔离"数据构成"效应。
# 切片全部真语料: soul规约/方法论穿透段/issuefix线程头/skills Phase1/真项目map惯用法。
# 出: calib_v2_s512b.txt + /tmp/rr_calib_s512b.ids;  硬断言: hard_eval 判决锚不得混入。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
OUT=calib_v2_s512b.txt

{ grep -A30 "func TestDuplicatedLinks" raw/go/avelino__awesome-go.code.txt | head -31
  head -c 700 build/ref_pillars/p1_issuefix.txt; echo
  sed -n '/## Cache penetration/,/hammer the database/p' method/methodology_core.md
  sed -n '/### Phase 1/,/exact solution/p' \
      skills/obra__superpowers__skills_systematic-debugging_SKILL.md
  sed -n '3,6p'   soul/honesty_v1.txt                                   # 规约首条(截断牺牲位)
} > "$OUT"

HARD_SIG="$(head -c 120 hard_eval.txt | tail -c 60)"
grep -qF "$HARD_SIG" "$OUT" && { echo "[FATAL] 判决锚污染"; exit 1; }

HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$HF/tokenizer.json" "$OUT" /tmp/rr_calib_s512b.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok (v2 判决口径 S=512b 代码主导档)")
PEOF
wc -c "$OUT"

# ★第0分钟构成审计(2026-07-16 铁律: 构成错误必须在起跑前拦住, 不是20h后)★
PYA="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PYA" ] || PYA=python3
"$PYA" - "$HF/tokenizer.json" "$OUT" <<'AEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2]).read()
# 粗分: 代码类=含 { } := func 行, 其余=散文
code_b = sum(len(l)+1 for l in text.split('\n') if any(k in l for k in ('{','}',':=','func ','\t')))
total = tok.encode(text, add_special_tokens=False)
code_t = tok.encode('\n'.join(l for l in text.split('\n') if any(k in l for k in ('{','}',':=','func ','\t'))), add_special_tokens=False)
pct = 100*len(code_t.ids)/max(1,len(total.ids))
print(f"[构成审计] 总={len(total.ids)}tok 代码类≈{len(code_t.ids)}tok ({pct:.0f}%) 散文≈{100-pct:.0f}%")
print(f"[构成审计] 判据: 代码主导档要求代码类≥50%; 现值{'✓过' if pct>=50 else '✗不过, 拒发'}")
AEOF
