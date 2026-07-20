#!/bin/bash
# build_pillars.sh — 四支柱后训练语料 v2 一键线(2026-07-15)。
# 支柱1 Go真项目+issue→fix对 | 支柱2 书/算法/方法论 | 支柱3 工程化skills | 支柱4 soul行为
# 三出口:
#   build/ref_pillars/*.txt  — DS4_REF_CORPUS 分片(各≤8MB, ds4_mtp 上限)
#   calib_v2.txt + /tmp/rr_calib_v2.ids — 量化校准锚扩容(S=305 → ~2-4k, 数据scaling判决)
#   pillar_probes.txt — 分钟级快判集(已在 repo)
# 铁律护栏: hard_eval.txt(rr_hard 判决锚)永不进校准 — 脚本内硬断言。
#   env: SKIP_HARVEST=1 只蒸馏; CALIB_KB=24; DS4_HF=hf路径(tokenizer)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
OUTREF=build/ref_pillars; mkdir -p "$OUTREF" raw

if [ "${SKIP_HARVEST:-0}" != "1" ]; then
  # 支柱1 需要 gh auth(~800 API call); PILLAR1=0 显式跳过(无 gh 时不静默降级)。
  if [ "${PILLAR1:-1}" = "1" ]; then
    gh auth status >/dev/null 2>&1 || { echo "[err] 支柱1 需要 gh auth login; 或显式 PILLAR1=0 跳过"; exit 1; }
    echo "== [1/3 采集] 支柱1: issue→fix 对(~10-20min, ~800 API call)"
    OUT=raw/issuefix python3 harvest_issue_fixes.py
  else
    echo "== [1/3 采集] 支柱1 显式跳过(PILLAR1=0)"
  fi
  echo "== [2/3 采集] 支柱2: 算法/系统设计定点仓库"
  REPOS="TheAlgorithms/Go,donnemartin/system-design-primer" KEYWORD=cs \
    EXTS=".go,.md" MAX_REPO_MB=1500 ISSUES_PER_REPO=0 OUT=raw python3 harvest_repos.py
  echo "== [3/3 采集] 支柱3: skills 仓库(superpowers 级)"
  OUT=skills WORK=raw/skills python3 harvest_skills.py
else
  echo "== 采集跳过(SKIP_HARVEST=1), 只蒸馏"
fi

echo "== [蒸馏] 出口1: REF_CORPUS 分片(≤8MB 尾截)"
CAPB=$((8*1024*1024 - 4096))
cap() { head -c "$CAPB"; }
cat raw/go/*.code.txt        2>/dev/null | cap > "$OUTREF/p1_go_code.txt"      || true
cat raw/issuefix/*.issuefix.txt 2>/dev/null | cap > "$OUTREF/p1_issuefix.txt"  || true
# books/method: 排除翻译版 README-xx*(纯重复噪声)
( find books method -type f \( -name '*.md' -o -name '*.txt' \) ! -name 'README-??*' -print0 2>/dev/null \
  | sort -z | xargs -0 cat 2>/dev/null ) | cap > "$OUTREF/p2_books_method.txt" || true
cat raw/cs/*.code.txt        2>/dev/null | cap > "$OUTREF/p2_algorithms.txt"   || true
cat method/methodology_core.md            | cap > "$OUTREF/p2_method_core.txt"
# skills: 精选仓(superpowers/anthropics)必须先进(字典序会让21MB聚合仓灌满8MB上限)
{ cat skills/obra__superpowers__* skills/anthropics__skills__* 2>/dev/null
  find skills -type f ! -name 'obra__superpowers__*' ! -name 'anthropics__skills__*' -print0 \
    | sort -z | xargs -0 cat 2>/dev/null
} | cap > "$OUTREF/p3_skills.txt" || true
cat soul/*.txt                            | cap > "$OUTREF/p4_soul.txt"
for f in "$OUTREF"/*.txt; do [ -s "$f" ] || { echo "[warn] 空分片: $f(对应采集缺失?)"; rm -f "$f"; }; done
ls -la "$OUTREF"

echo "== [蒸馏] 出口2: calib_v2.txt(四支柱均衡头切, 上限 ${CALIB_KB:-24}KB)"
{ cat soul/*.txt
  head -c 6000 method/methodology_core.md
  head -c 5000 "$OUTREF/p1_issuefix.txt"   2>/dev/null || true
  head -c 5000 "$OUTREF/p1_go_code.txt"    2>/dev/null || true
  head -c 4000 "$OUTREF/p3_skills.txt"     2>/dev/null || true
  head -c 3000 "$OUTREF/p2_algorithms.txt" 2>/dev/null || true
} | head -c $(( ${CALIB_KB:-24} * 1024 )) > calib_v2.txt

# 铁律断言: rr_hard 判决语料不得进校准(取 hard_eval 首行实体串查污染)
HARD_SIG="$(head -c 120 hard_eval.txt | tail -c 60)"
if grep -qF "$HARD_SIG" calib_v2.txt; then
  echo "[FATAL] calib_v2.txt 污染: 含 hard_eval.txt 内容(判决锚必须 held-out)"; exit 1
fi
echo "calib_v2.txt: $(wc -c < calib_v2.txt) bytes"

echo "== [蒸馏] 出口2b: tokenize → /tmp/rr_calib_v2.ids(regen_ids.sh 同款)"
HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"
PY="${DS4_PYVENV:-/tmp/go_venv/bin/python3}"; [ -x "$PY" ] || PY=python3
"$PY" - "$HF/tokenizer.json" calib_v2.txt /tmp/rr_calib_v2.ids <<'PEOF'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
open(sys.argv[3], "w").write("\n".join(map(str, ids)) + "\n")
print(f"{sys.argv[3]}: {len(ids)} tok (S 扩容判决口径)")
PEOF

echo "== 完成。消费入口:"
echo "   DS4_REF_CORPUS=\$(ls $PWD/$OUTREF/*.txt | tr '\n' ':')   # knowledge-MTP/copy-spec"
echo "   /tmp/rr_calib_v2.ids                                      # 量化校准锚(满档 S 扩容)"
echo "   pillar_probes.txt                                         # 分钟级快判集"
