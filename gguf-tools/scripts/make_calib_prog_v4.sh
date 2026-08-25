#!/bin/bash
# make_calib_prog_v4.sh — 编程全域校准锚 v4(2026-07-27): ★fit/held 显式分区★。
# 破案背景(fable5): ds4quant_run 的 n_fit=S×3/4 是位置性切分 — v1 的 Go 槽整个落
# 在 held 区(拟合剂量=0), v3 的 Go 尾 25% 也会被切。v4 = 显式两区:
#   FIT 区(前段): 全部治疗剂量 — Go 重剂量(LFU 文件形态+gin) + 六语言真代码(v2 同款切片)
#   HELD 区(尾段): 判决集 — 各语言同函数的【后续行】(与 fit 行零重叠, 温和相关性已知
#     并接受, 判决锚不混入纪律仍守) + 非代码尾巴
# 配套: 输出 /tmp/rr_calib_prog_v4.meta (NTOK/NFIT), 战役 env 用 DS4_NFIT 显式钉边界。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
OUT=../data/corpus/calib_prog_v4.txt   # 批4: 语料产物落 data/corpus/
FIT=/tmp/calib_v4_fit.txt
HELD=/tmp/calib_v4_held.txt

slice() {  # slice <lang> <signature> <lines>
  grep -h -A"$(($3 - 1))" -F "$2" raw/"$1"/*.code.txt 2>/dev/null | head -"$3"
}
slice_tail() {  # 同签名函数的后续行(fit 行之后, 零行重叠): 首个命中窗口的 rows N..N+M-1
  grep -m1 -h -A"$(($3 + $4 - 1))" -F "$2" raw/"$1"/*.code.txt 2>/dev/null | head -"$(($3 + $4))" | tail -"$4"
}

{ # ══ FIT 区 ══
  echo "// ==== go ===="
  grep -h -A53 "^package cache" raw/go/TheAlgorithms__Go.code.txt | head -54
  echo
  slice go "func (engine *Engine) Handler() http.Handler {" 9
  echo "// ==== python ===="     ; slice python "    def route(self, rule: str" 9
  echo "// ==== javascript ====" ; slice javascript "function dispatchRequest(config) {" 8
  echo "// ==== rust ===="       ; slice rust "pub fn is_match<P: AsRef<Path>>" 6
  echo "// ==== typescript ====" ; slice typescript "export function reactive(target: object) {" 7
  echo "// ==== c ===="          ; slice c "static void assoc_expand(" 6
  echo "// ==== java ===="       ; slice java "public <T> T fromJson(String json, Class<T> classOfT)" 6
} > "$FIT"

{ # ══ HELD 区 ══
  echo "// ==== held_go ===="
  grep -h -A67 "^package cache" raw/go/TheAlgorithms__Go.code.txt | tail -13
  echo "// ==== held_python ===="     ; slice_tail python "    def route(self, rule: str" 9 6
  echo "// ==== held_javascript ====" ; slice_tail javascript "function dispatchRequest(config) {" 8 6
  echo "// ==== held_rust ===="       ; slice_tail rust "pub fn is_match<P: AsRef<Path>>" 6 5
  echo "// ==== held_typescript ====" ; slice_tail typescript "export function reactive(target: object) {" 7 5
  echo "// ==== held_c ===="          ; slice_tail c "static void assoc_expand(" 6 5
  echo "// ==== held_java ===="       ; slice_tail java "public <T> T fromJson(String json, Class<T> classOfT)" 6 5
  echo "// ==== issuefix ===="
  f=$(ls raw/python/*.issues.txt 2>/dev/null | head -1)
  [ -n "$f" ] && { head -c 180 "$f"; echo; }
  echo "// ==== method ===="
  sed -n '/## Cache penetration/,/hammer the database/p' method/methodology_core.md
} > "$HELD"

cat "$FIT" "$HELD" > "$OUT"

# fit/held 行级零重叠断言
python3 - "$FIT" "$HELD" <<'CEOF'
import sys
fit = set(l for l in open(sys.argv[1]) if len(l.strip()) > 12)
dup = [l for l in open(sys.argv[2]) if l in fit]
if dup:
    print(f"[FATAL] fit/held 行重叠 {len(dup)} 行, 首条: {dup[0][:60]!r}"); sys.exit(1)
CEOF
# 判决锚污染双签名硬断言
python3 - "$OUT" <<'CEOF'
import sys
out = open(sys.argv[1], 'rb').read()
for judge in ('hard_eval.txt', 'coding_hard.txt'):
    sig = open(judge, 'rb').read()[60:120]
    if sig in out:
        print(f"[FATAL] {judge} 判决锚污染"); sys.exit(1)
CEOF

HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$HF/tokenizer.json" "$FIT" "$OUT" /tmp/rr_calib_prog_v4.ids /tmp/rr_calib_prog_v4.meta <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
nfit = len(tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids)
ids = tok.encode(open(sys.argv[3]).read(), add_special_tokens=False).ids
open(sys.argv[4], "w").write("\n".join(map(str, ids)) + "\n")
open(sys.argv[5], "w").write(f"NTOK={len(ids)}\nNFIT={nfit}\n")
assert nfit < len(ids) - 32, "held 区太小"
print(f"v4: NTOK={len(ids)} NFIT={nfit} held={len(ids)-nfit} "
      f"(fit占比 {100*nfit//len(ids)}%)")
PEOF
wc -c "$OUT"
