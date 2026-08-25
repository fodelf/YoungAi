#!/bin/bash
# regen_ids.sh — 重建 /tmp/rr_code.ids 与 /tmp/rr_hard.ids(2026-07-14)
# 背景: macOS /tmp 清理器按 3 天 TTL 删了两份校准/判决语料(教训: /tmp 非持久)。
# 源文本(repo 固化): corpus/coding_hard.txt → rr_code.ids | corpus/hard_eval.txt → rr_hard.ids
# 分词: HF tokenizer.json(与 ds4 运行时词表一致, pyfwd 同款), 无特殊 token, 逐行十进制 id。
# 验证: 重建后跑 NL=2 全F自检, VERDICT ratio=1.0000 = 与既有锚逐字节同源(锚仍有效)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"
[ -x "$PY" ] || PY=python3
"$PY" - "$HF/tokenizer.json" "$ROOT/gguf-tools/go-onebit/corpus/coding_hard.txt" /tmp/rr_code.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok")
PEOF
"$PY" - "$HF/tokenizer.json" "$ROOT/gguf-tools/go-onebit/corpus/hard_eval.txt" /tmp/rr_hard.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok")
PEOF
touch /tmp/rr_code.ids /tmp/rr_hard.ids   # 刷新 atime/mtime 抗 TTL(仍建议定期重跑本脚本)
echo "[regen_ids] done — 验证: quant_verify.sh selftest 或 NL=2 全F ratio=1.0000"

# ==== 通用混合语料(2026-08-09 北极星全能力: z天花板测试起用) ====
# calib_general_v1.txt(fit) + held_general_v1.txt(held) → 拼接单流 rr_general_s.ids(fit在前)
# 产物入项目 g7/(非/tmp); NFIT=fit段token数, 由本脚本打印, 消费端(zlayer/锚生成)用它切界。
"$PY" - "$HF/tokenizer.json" "$ROOT/gguf-tools/go-onebit/corpus" "$ROOT/gguf/go-onebit/g7" <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1]); C=sys.argv[2]; G=sys.argv[3]
fit = tok.encode(open(f"{C}/calib_general_v1.txt").read(), add_special_tokens=False).ids
hld = tok.encode(open(f"{C}/held_general_v1.txt").read(), add_special_tokens=False).ids
open(f"{G}/rr_general_s.ids","w").write("\n".join(map(str, fit+hld))+"\n")
print(f"rr_general_s.ids: fit={len(fit)} held={len(hld)} S={len(fit)+len(hld)}")
PEOF
