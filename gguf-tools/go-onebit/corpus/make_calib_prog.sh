#!/bin/bash
# make_calib_prog.sh — 编程全域四支柱校准锚(2026-07-20 域放大)。
# 构成决策(单一方案): 代码主导 ~85%(构成扫描判决 rr_code 强单调于代码占比; soul/方法论
# 行为收益走 server soul 层更便宜, 锚内只留牺牲位小尾巴), 多语言按 15 针面板弱点加权:
# python/js/rust 权重高(注释逃逸重灾), c/java/ts 中, go 保留槽位防退化。
# 纪律(承 make_calib_s512b.sh): 代码前置(NTOK 截断只吃尾)/判决锚不得混入(hard_eval+
# coding_hard 双签名硬断言)/第0分钟构成审计+语言切片审计。
# 前置: scripts/harvest_prog.sh 已跑完(raw/<lang>/*.code.txt 在位)。
# 出: calib_prog_v1.txt + /tmp/rr_calib_prog_v1.ids
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
OUT=calib_prog_v1.txt

# 选择器=惯用逻辑代码签名定位(跨该语言全部 shard, 首个命中=确定性)。
# ⚠ 教训(2026-07-20 v1 首构): "首文件前 N 行"选择器抽到 license 头/纯注释块/#ifdef 噪声
# ——注释块恰是 15 针面板要治的病灶, 锚里放注释=反向校准。签名钉死真逻辑代码。
slice() {  # slice <lang> <signature> <lines>
  grep -h -A"$(($3 - 1))" -F "$2" raw/"$1"/*.code.txt 2>/dev/null | head -"$3"
}

# 行数按实测 token 密度定(py 6.4/js 8.5/rust 10.3/ts 9.2/c 6.2/java 9.1 tok/行),
# 目标: 代码+标记 ≈ 530 tok 全落 NTOK=512 有效窗内, issue/method/soul 尾巴=截断牺牲位。
{ # ---- 支柱1: 多语言真项目代码(前置, 截断安全区; 签名=各库招牌逻辑函数) ----
  echo "// ==== python ===="     ; slice python "    def route(self, rule: str" 9
  echo "// ==== javascript ====" ; slice javascript "function dispatchRequest(config) {" 8
  echo "// ==== rust ===="       ; slice rust "pub fn is_match<P: AsRef<Path>>" 6
  echo "// ==== typescript ====" ; slice typescript "export function reactive(target: object) {" 7
  echo "// ==== c ===="          ; slice c "static void assoc_expand(" 6
  echo "// ==== java ===="       ; slice java "public <T> T fromJson(String json, Class<T> classOfT)" 6
  # go 槽位: 沿用 v3p 在案真项目切片(防 Go 退化)
  echo "// ==== go ===="
  grep -A6 "func TestDuplicatedLinks" raw/go/avelino__awesome-go.code.txt | head -7
  # ---- 支柱2: issuefix 线程头(真 issue, 牺牲区首段) ----
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

# ---- 判决锚污染双签名硬断言(字节子串真语义; grep -F 对含换行签名=任意行命中假警报) ----
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
"$PY" - "$HF/tokenizer.json" "$OUT" /tmp/rr_calib_prog_v1.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok (编程全域 v1 代码主导档)")
PEOF
wc -c "$OUT"

# ---- ★第0分钟构成审计 + 语言切片审计★ ----
"$PY" - "$HF/tokenizer.json" "$OUT" <<'AEOF'
import sys, re
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2]).read()
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
total = n(text)
NONCODE = {"issuefix", "method", "soul"}
# 分段: 按 // ==== <tag> ==== 界标; 代码=语言段, 非代码=issuefix/method/soul 段
parts = re.split(r"// ==== (\w+)[^\n]*====\n", text)
segs = {}
for i in range(1, len(parts) - 1, 2):
    segs[parts[i]] = segs.get(parts[i], 0) + n(parts[i + 1])
code_t = sum(v for k, v in segs.items() if k not in NONCODE)
print(f"[构成审计] 总={total}tok 代码类≈{code_t}tok ({100*code_t//max(1,total)}%) 非代码≈{total-code_t}tok")
for k, v in segs.items():
    cls = "非代码" if k in NONCODE else "代码"
    print(f"[切片] {k:<11} {v:>4}tok ({100*v//max(1,total)}%) {cls}")
# 双判据: 代码主导≥70%; 代码区(含标记)整体落进 NTOK=512 有效窗(允差 5%)
head_t = total - sum(segs.get(k, 0) for k in NONCODE)
ok = code_t >= total * 0.70 and head_t <= 512 * 1.05
print(f"[构成审计] 判据: 代码≥70% 且代码区≤538tok(512 窗+5%); "
      f"现值 code={100*code_t//max(1,total)}% head={head_t}tok "
      f"{'✓过' if ok else '✗不过, 拒发'}")
sys.exit(0 if ok else 1)
AEOF
