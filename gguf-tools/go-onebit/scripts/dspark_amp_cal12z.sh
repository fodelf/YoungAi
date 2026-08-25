#!/bin/bash
# dspark_amp_cal12z.sh — drafter 放大器: 用**当初主放大器同一份冻结全场景小语料 cal12z**
# 做校准(2026-08-21 用户裁决: 不再自编提示, 不再靠 A/B 读噪声; 若结果不对就是算法问题)。
#
# cal12z.ids = 2048 token 的全场景冻结集(amp86_spark.sh 里主模型放大器的锚源)。
# 引擎没有 ids 入口 ⇒ 用 GGUF 词表解回文本(字节级 BPE, Ġ=空格)再分块当 prompt,
# 在部署态(q2 drafter + SPEC)捕锚, 分布与量化一致。
# 段: decode → capture → fit×3 → chain。
set -uo pipefail
ROOT="$HOME/ds4-main"
D="$ROOT/gguf/go-onebit/r30/dspark"
IDS="${IDS:-$ROOT/gguf/go-onebit/g7/cal12z.ids}"
MAIN="$ROOT/gguf/ds4-allq2.gguf"
MAIN_ZC="$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin"
STUDENT="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
HF="$ROOT/hf/DeepSeek-V4-Flash-DSpark"
TXT="$D/cal12z_chunks"
ANCHOR="$D/anchor_cal12z.bin"
NGEN="${NGEN:-100}"
CHUNK_TOK="${CHUNK_TOK:-128}"
cd "$ROOT"
mkdir -p "$TXT" "$D/cz"; rm -f "$TXT"/*.txt "$TXT"/*.bin

echo "[1/4] cal12z.ids → 文本分块 (每块 ${CHUNK_TOK} token)"
python3 - "$MAIN" "$IDS" "$TXT" "$CHUNK_TOK" <<'PY'
import sys
from gguf import GGUFReader
mdl, idsf, out, ct = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
R = GGUFReader(mdl)
f = R.fields["tokenizer.ggml.tokens"]
def tok(i):
    return bytes(f.parts[f.data[i]]).decode("utf-8", "replace")
ids = [int(x) for x in open(idsf) if x.strip()]
# 字节级 BPE: Ġ=空格, Ċ=换行
def detok(seq):
    s = "".join(tok(i) for i in seq)
    return s.replace("Ġ", " ").replace("Ċ", "\n")
n = 0
for i in range(0, len(ids), ct):
    seg = detok(ids[i:i+ct]).strip()
    if len(seg) < 40: continue
    open(f"{out}/c{n:02d}.txt", "w", encoding="utf-8").write(seg)
    n += 1
print(f"  {len(ids)} token → {n} 块")
PY

echo "[2/4] 部署态捕锚(单进程 REPL 连喂所有块 — 每块单独起进程要重做 80GB 装载+74.9GiB CUDA 注册, 纯浪费)"
# 每块压成一行(REPL 按行读), 制表符还原换行不影响分布
python3 - "$TXT" <<'PY'
import sys, glob, os
tmp = sys.argv[1]
with open(os.path.join(tmp, "all_prompts.txt"), "w", encoding="utf-8") as w:
    for f in sorted(glob.glob(os.path.join(tmp, "c*.txt"))):
        w.write(open(f, encoding="utf-8").read().replace("\n", " ").strip() + "\n")
print("  合并提示行数:", sum(1 for _ in open(os.path.join(tmp, "all_prompts.txt"), encoding="utf-8")))
PY
env DS4_DRAFT_GGUF="$STUDENT" DS4_DSPARK_SPEC=1 DS4_DSPARK_ANCHOR="$ANCHOR" \
    timeout 3600 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" --temp 0 -n "$NGEN" \
    < "$TXT/all_prompts.txt" 2>&1 | grep -acE "generation" | sed "s/^/  完成生成段数: /"
ls -l "$ANCHOR"

echo "[3/4] 三块解算"
for B in 0 1 2; do
    timeout 3600 python3 -u gguf-tools/go-onebit/zlever/dspark_amp_fit.py \
        "$HF" "$STUDENT" "$ANCHOR" "$D/cz" "$B" 2>&1 | grep -aE "锚行|数据|★"
done

echo "[4/4] 成链"
"$(dirname "$0")/../calib/zrec_to_zchain" "$D/cz" "$D/zchain_drafter_cal12z.bin" 3 2>&1 | tail -1
ls -l "$D/zchain_drafter_cal12z.bin"
