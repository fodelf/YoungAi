#!/bin/bash
# make_calib_prog_v3.sh — 编程全域校准锚 v3(2026-07-27): ★Go 剂量修正★。
# 破案背景(fable5 2026-07-27): "注释逃逸"是全域病, v1/v2 按 15 针面板把重剂量配给
# py/js/rust, Go 只留 7 行 awesome-go TestDuplicatedLinks(goquery 链接测试, if 中截断)
# ——今日 HumanEval 双语基准 Py 1躲/Go 6躲 = 剂量表的镜像。teacher 前向证实原始模型
# fork 边际健康(Go/15 +3.07), 病在量化侧欠校准。
# v3 修正: Go 升重剂量档, 两切片全换真材实料(harvest 库存里本来就有):
#   A) TheAlgorithms__Go `package cache`→LFU.Get — ★文件形态完整(package/import 行)★
#      + 注释→签名→真实现, 正对躲题病灶形态(H7: 无 package 的伪 Go 题面加重躲题);
#      LFU≠判决锚(go_013=LRU, 不同算法不同 API, 判决锚不混入纪律仍守)。
#   B) gin Engine.Handler — 真项目招牌逻辑(与其他语言切片同选择器纪律)。
# 其余语言切片与 v2 逐字节一致(它们的剂量是对的, 不动)。
# 出: calib_prog_v3.txt + /tmp/rr_calib_prog_v3.ids
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
OUT=calib_prog_v3.txt

slice() {  # slice <lang> <signature> <lines>
  grep -h -A"$(($3 - 1))" -F "$2" raw/"$1"/*.code.txt 2>/dev/null | head -"$3"
}

{ # ---- 支柱1: 多语言真项目代码(前置, 截断安全区) ----
  echo "// ==== python ===="     ; slice python "    def route(self, rule: str" 9
  echo "// ==== javascript ====" ; slice javascript "function dispatchRequest(config) {" 8
  echo "// ==== rust ===="       ; slice rust "pub fn is_match<P: AsRef<Path>>" 6
  echo "// ==== typescript ====" ; slice typescript "export function reactive(target: object) {" 7
  echo "// ==== c ===="          ; slice c "static void assoc_expand(" 6
  echo "// ==== java ===="       ; slice java "public <T> T fromJson(String json, Class<T> classOfT)" 6
  # ★go 重剂量档(v3 修正位)★
  echo "// ==== go ===="
  grep -h -A53 "^package cache" raw/go/TheAlgorithms__Go.code.txt | head -54
  echo
  slice go "func (engine *Engine) Handler() http.Handler {" 9
  # ---- 支柱2: issuefix 线程头(牺牲区首段) ----
  echo "// ==== issuefix ===="
  f=$(ls raw/python/*.issues.txt 2>/dev/null | head -1)
  [ -n "$f" ] && { head -c 180 "$f"; echo; }
  # ---- 支柱3: 方法论(穿透段) ----
  echo "// ==== method ===="
  sed -n '/## Cache penetration/,/hammer the database/p' method/methodology_core.md
  # ---- 支柱4: soul 规约首条(最尾牺牲位) ----
  echo "// ==== soul ===="
  sed -n '3,4p' soul/honesty_v1.txt
} > "$OUT"

# ---- 判决锚污染双签名硬断言 ----
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
"$PY" - "$HF/tokenizer.json" "$OUT" /tmp/rr_calib_prog_v3.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok (v3: Go 剂量修正档)")
PEOF
wc -c "$OUT"

# ---- ★第0分钟构成审计★ (窗放宽 610→900: Go 补剂量净增 ~250tok, 非零和挤占 —
#      v3 语义=修 bug 不重配他语言; 未来重拟合 DS4_NTOK 相应取 ≥900) ----
"$PY" - "$HF/tokenizer.json" "$OUT" <<'AEOF'
import sys, re
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2]).read()
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
total = n(text)
NONCODE = {"issuefix", "method", "soul"}
parts = re.split(r"// ==== (\w+)[^\n]*====\n", text)
segs = {}
for i in range(1, len(parts) - 1, 2):
    segs[parts[i]] = segs.get(parts[i], 0) + n(parts[i + 1])
code_t = sum(v for k, v in segs.items() if k not in NONCODE)
print(f"[构成审计] 总={total}tok 代码类≈{code_t}tok ({100*code_t//max(1,total)}%)")
for k, v in segs.items():
    cls = "非代码" if k in NONCODE else "代码"
    print(f"[切片] {k:<11} {v:>4}tok ({100*v//max(1,total)}%) {cls}")
head_t = total - sum(segs.get(k, 0) for k in NONCODE)
ok = code_t >= total * 0.70 and head_t <= 900
print(f"[构成审计] 判据: 代码≥70% 且代码区≤900tok; 现值 code={100*code_t//max(1,total)}% head={head_t}tok "
      f"{'✓过' if ok else '✗不过, 拒发'}")
sys.exit(0 if ok else 1)
AEOF
