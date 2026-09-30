#!/bin/bash
# docend_corpus_build.sh — "文档结尾"判决料(2026-09-30, 用户"先 1"): 整篇文档 + 真正的 EOS, 量量化把"停"这个决策压掉几倍。
#
# 为什么要它: 现役所有拟合料/判决料都是从文档中间切出来的 128~512 token 窗, 一个文档结尾都没有 —— EOS 这种最基本的
#   "停"决策从来没进过任何一份拟合或判决。09-30 在真实 CFO 序列上量到: FP 在"该收口"的位置给 </think> 22%~49%,
#   量化学生只给 1%~6%(引擎与官方代码同权重同数)。这份料回答"这是思考专属还是决策 token 的通病": 拿整篇开源文档
#   按预训练样子 <BOS> 正文 <EOS> 拼起来, 在每个 EOS 位置比 FP 与学生给 EOS 的概率(bugmd_ids_tools.py tokprobe)。
# 料: gguf-tools/data/corpus/{law,med,sci,code}/*.jsonl(一行一篇 {"text"}), 每个子域取 PER_SUB 篇、篇长在 [MIN_TOK, MAX_TOK] token
#   之间的整篇(不截断: 截断的文档没有真结尾)。同一子域按行号等距取, 不挑内容。
# 产物: gguf/go-onebit/docend/docend.ids(一行一个 id; 每篇 = BOS + 正文 + EOS 拼接) + docend.manifest.txt(篇号 来源 行号 起点 EOS位置 篇长)。
# 用法: bash gguf-tools/scripts/docend_corpus_build.sh [PER_SUB=1] [MIN_TOK=300] [MAX_TOK=1200]   (spark 本机跑, 分词器走 ~/v41env)
# 出错会怎样: 某子域凑不够 PER_SUB 篇 → 打 ★ 少收, 不硬停(判决料不要求每域等量); 分词器装不上 → 直接退。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
PER_SUB="${1:-1}"; MIN_TOK="${2:-300}"; MAX_TOK="${3:-1200}"
OUT="$ROOT/gguf/go-onebit/docend"; mkdir -p "$OUT"
~/v41env/bin/python - "$ROOT" "$OUT" "$PER_SUB" "$MIN_TOK" "$MAX_TOK" <<'PYEOF'
import glob, json, os, sys
root, out, per_sub, min_tok, max_tok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(root, "hf/DeepSeek-V4.1-Flash"), trust_remote_code=True)
BOS, EOS = tok.bos_token_id, tok.eos_token_id
assert BOS is not None and EOS is not None, "分词器没有 BOS/EOS"
# 子域清单: 中英各半, 散文为主, 代码只收四种"文件有自然结尾"的
SUBS = ["law/law_case_zh", "law/law_statute_en", "law/law_contract_en",
        "med/med_encyc_zh", "med/med_paper_en", "med/med_case_en",
        "sci/sci_paper_zh", "sci/sci_arxiv_en", "sci/sci_paper_hss_zh",
        "code/code_python", "code/code_markdown", "code/code_shell", "code/code_sql"]
ids_all, manifest, pos = [], [], 0
for sub in SUBS:
    path = os.path.join(root, "gguf-tools/data/corpus", sub + ".jsonl")
    if not os.path.exists(path):
        print("★缺 %s, 跳过★" % path); continue
    lines = open(path, encoding="utf-8").read().split("\n")
    cands = []
    for ln, s in enumerate(lines):
        if not s.strip(): continue
        try: text = json.loads(s)["text"]
        except Exception: continue
        if not text.strip(): continue
        enc = tok.encode(text, add_special_tokens=False)
        if min_tok <= len(enc) <= max_tok: cands.append((ln, enc))
        if len(cands) >= per_sub * 8: break          # 够挑了(等距取 per_sub 篇), 不用把整个子域都分词
    if not cands:
        print("★%s 没有 [%d,%d] token 的整篇★" % (sub, min_tok, max_tok)); continue
    step = max(1, len(cands) // per_sub)
    picked = cands[::step][:per_sub]
    if len(picked) < per_sub: print("★%s 只凑到 %d/%d 篇★" % (sub, len(picked), per_sub))
    for ln, enc in picked:
        doc = [BOS] + enc + [EOS]
        manifest.append((len(manifest), sub, ln, pos, pos + len(doc) - 1, len(enc)))   # EOS 的绝对位置 = pos+len(doc)-1
        ids_all.extend(doc); pos += len(doc)
with open(os.path.join(out, "docend.ids"), "w") as f: f.write("\n".join(map(str, ids_all)) + "\n")
with open(os.path.join(out, "docend.manifest.txt"), "w", encoding="utf-8") as f:
    f.write("# 篇号 子域 源行号 起点(BOS位置) EOS位置 正文token数; BOS=%d EOS=%d; 总 token %d\n" % (BOS, EOS, len(ids_all)))
    for m in manifest: f.write("%d %s %d %d %d %d\n" % m)
print("docend.ids: %d 篇, %d token(BOS %d / EOS %d) → %s" % (len(manifest), len(ids_all), BOS, EOS, out))
for m in manifest: print("  篇 %2d  %-22s 行 %5d  起 %6d  EOS 位 %6d  正文 %4d tok" % m)
PYEOF
