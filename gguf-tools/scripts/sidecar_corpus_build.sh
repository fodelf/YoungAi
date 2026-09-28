#!/bin/bash
# sidecar_corpus_build.sh — 法律 / 医疗 / 科研三个领域侧车的语料构建(Mac 侧跑; 2026-09-27 用户令"已经支持金融、编程侧车了,
#   再加入法律、医疗、科研这三个场景的侧车以及五指标")。下游与编程侧车逐字同一条链: amp_campaign.sh --profile <域> idshalf
#   切 fit/judge 两份 → amp_campaign.sh --profile <域> sidecar 解侧车 + 打五指标。
#
# 产物: gguf-tools/data/corpus/<域>/<子域>.jsonl —— 一个文件 = 一个子域, 一行 = 一篇 {"text": 原文}。每个域 6 个子域
#   (3 中文 + 3 英文, 按"法条/文献 · 案例/实务 · 试题"三种文体配齐), 每个子域都必须凑满 TARGET_BYTES, 缺一个就硬停。
#   law  law_statute_zh   中国法律法规全文              twang2218/chinese-law-and-regulations (apache-2.0)
#        law_case_zh      刑事案件公诉事实(裁判文书)    china-ai-law-challenge/cail2018
#        law_exam_zh      国家司法考试真题(JEC-QA)      hails/agieval-jec-qa-ca + -kd
#        law_case_en      美国最高法院判决书            coastalcph/lex_glue scotus (cc-by-4.0)
#        law_statute_en   欧盟法规                      coastalcph/lex_glue eurlex
#        law_contract_en  合同条款(SEC 披露文件)        coastalcph/lex_glue ledgar
#   med  med_encyc_zh     医学百科问答                  FreedomIntelligence/huatuo_encyclopedia_qa (apache-2.0)
#        med_consult_zh   在线问诊(寻医问药真实医生回答) sentence-transformers/cmedqa-v2 triplet
#        med_exam_zh      执业医师资格考试题+解析        fzkuji/CMExam
#        med_paper_en     PubMed 论文摘要               MedRAG/pubmed(8 个年代分片等距取)
#        med_case_en      PMC 病例报告患者摘要          zhengyun21/PMC-Patients
#        med_exam_en      USMLE 执照考试题              GBaker/MedQA-USMLE-4-options (cc-by-4.0)
#   sci  sci_paper_zh     中文核心期刊论文摘要·理工农医  neuclir/csl (apache-2.0)
#        sci_paper_hss_zh 同上·人文社科                 neuclir/csl
#        sci_exam_zh      大学理工科试题(C-Eval)        ceval/ceval-exam 16 个理工科目
#        sci_arxiv_en     arXiv 论文 LaTeX 原文         EleutherAI/proof-pile-2 arxiv(4 个分片头部)
#        sci_paper_en     跨学科论文全文(S2ORC)         allenai/peS2o v2 s2orc (odc-by)
#        sci_exam_en      研究生级理工科题(MMLU-Pro)    TIGER-Lab/MMLU-Pro 6 个理工类 (mit)
#
# 为什么是 .jsonl 保留换行而不是金融那样折成一行: 法条按"第X条"分行、判决书有案号/抬头分行、LaTeX 的换行和环境是语法,
#   问答题的"选项/答案"靠换行分隔 —— 折成空格分词完全不同, 等于在一种不存在的分布上解侧车(同 code_corpus_build.sh)。
# 为什么中英各半: 用户场景是中文, 但这三个领域的一手资料(判例/论文/执照考试)大量是英文; 只拟中文, 侧车在读英文文献时就是
#   裸底座水平。子域等权(amp_campaign.sh idshalf 按子域等配额抽样), 所以 6 个子域各占 1/6。
# 为什么不用 Huatuo26M-Lite / pubmed-summarization: 前者的回答经 ChatGPT 改写(拟合的是另一个模型的文风), 后者被预处理成
#   全小写 + 标点两侧加空格("one - third", "( asymptomatic )"), 都不是真实文本。
# 为什么 CAIL 只收不带"××"的事实: 这个数据集把罪名/法条挖成"××"做预测题, 09-27 实测 61% 的事实带挖空, 留着就是在拟合挖空符。
# 为什么 PubMed 摘要要去残标签: MedRAG 的摘要里 <i>P</i> 丢了 ">" 变成 "<iP</i", 09-27 抽样 37% 篇带这种残壳;
#   去壳 + HTML 反转义后还原成 "P<0.01"。
# 为什么"等距抽"不"取前 N 篇": CSL 按学科排序(文件头全是文学)、法规按效力级别排序、MedRAG 分片按年代排序,
#   取前 N 篇会全是同一类。所以要么整文件下载后等距抽, 要么从多个分片各取一段(分片之间本身就跨年代/跨主题)。
# 为什么 TARGET_BYTES 600 KB: fit/judge 两份各只从每个子域取 15360/6 = 2560 token, 600 KB ≈ 15 万 token 已是 30 倍余量
#   (同 code_corpus_build.sh)。
#
# 用法: bash gguf-tools/scripts/sidecar_corpus_build.sh <law|med|sci|all> [raw 缓存根]
#   默认缓存根 gguf/go-onebit/<域>/raw(gguf/ 在 .gitignore 内, 共约 1 GB); 已下的文件不重下, 产物目录每次整体重建。
#   Python 只做文本编排(读 parquet/json/zstd、过滤、写 jsonl), 不参与任何数值。
# 出错会怎样: 任一文件下载失败或任一子域凑不满 600 KB 就硬停(退出码 1), 不产缺子域的语料 —— 缺一个子域, idshalf 照样能切,
#   但那一块能力在侧车里就没人管, 五指标也量不到。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHICH="${1:?用法: sidecar_corpus_build.sh <law|med|sci|all> [raw 缓存根]}"
RAWROOT="${2:-$ROOT/gguf/go-onebit}"
PYENV="$ROOT/gguf/go-onebit/pyenv"   # pyarrow(读 parquet) + zstandard(读 proof-pile-2 的 .zst), 三个域共用一份
TARGET_BYTES=600000                  # 每个子域必须凑满的字节(UTF-8)
LOG(){ echo "[sidecorpus $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }
case "$WHICH" in law|med|sci) DOMS="$WHICH";; all) DOMS="law med sci";; *) DIE "不认识的域 $WHICH(law|med|sci|all)";; esac

if [ ! -x "$PYENV/bin/python" ]; then
    LOG "建虚拟环境 $PYENV(pyarrow + zstandard, 只读文件不参与数值)"
    python3 -m venv "$PYENV" && "$PYENV/bin/pip" install -q pyarrow zstandard || DIE "虚拟环境安装失败"
fi

for DOM in $DOMS; do
    RAW="$RAWROOT/$DOM/raw"; OUT="$ROOT/gguf-tools/data/corpus/$DOM"
    mkdir -p "$RAW"; rm -rf "$OUT"; mkdir -p "$OUT"
    LOG "== $DOM: 缓存 $RAW → 产物 $OUT"
    "$PYENV/bin/python" - "$DOM" "$RAW" "$OUT" "$TARGET_BYTES" <<'PY' || DIE "$DOM 抽取失败"
import ast, csv, gzip, html, io, json, os, re, subprocess, sys, unicodedata, zlib
import pyarrow.parquet as pq
import zstandard
DOM, RAW, OUT, TARGET = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
HF = "https://huggingface.co/datasets"
csv.field_size_limit(1 << 30)

def fetch(repo, path, rng=0):
    """下到 RAW(已在不重下)。rng>0 = 只取开头 rng 字节(HTTP Range), 服务端在 range 末尾断流(rc=18)是正常收尾。"""
    local = os.path.join(RAW, repo.replace("/", "__") + "__" + path.replace("/", "__"))
    if os.path.exists(local) and os.path.getsize(local) > 0: return local
    print("  下载 %s/%s%s" % (repo, path, " (前 %d MB)" % (rng >> 20) if rng else ""), flush=True)
    cmd = ["curl", "-sL", "-m", "1800", "-o", local + ".part"] + (["-r", "0-%d" % (rng - 1)] if rng else []) \
          + ["%s/%s/resolve/main/%s" % (HF, repo, path)]
    rc = subprocess.run(cmd).returncode
    if not (rc == 0 or (rng and rc == 18)) or os.path.getsize(local + ".part") == 0:
        sys.exit("★下载失败 rc=%d: %s/%s★" % (rc, repo, path))
    os.rename(local + ".part", local)
    return local

def clean(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "".join(c for c in s if c in "\n\t" or unicodedata.category(c)[0] != "C")
    s = re.sub(r"[ \t]+\n", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def cap(s, n):
    """截到 n 字以内, 优先在换行/句末断(截在半句中间 = 拟合一个被砍断的句子)。"""
    if len(s) <= n: return s
    cut = max([s.rfind("\n", 0, n), s.rfind(". ", 0, n)] + [s.rfind(p, 0, n) for p in "。！？；"])
    return s[:cut + 1].rstrip() if cut > n // 2 else s[:n]

def pq_rows(local, cols): return pq.read_table(local, columns=cols).to_pylist()

def jsonl_lines(raw_bytes):
    for ln in raw_bytes.decode("utf-8", "ignore").split("\n"):
        ln = ln.strip()
        if not ln: continue
        try: yield json.loads(ln)
        except ValueError: pass       # range 取片的最后半行

def gz_head(local):                   # 截尾的 gzip 流: 解到哪算哪, 到截断处为止的整行都完整
    return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(open(local, "rb").read())

def zst_head(local):
    out = io.BytesIO()
    try:
        with zstandard.ZstdDecompressor().stream_reader(open(local, "rb")) as r:
            while True:
                b = r.read(1 << 20)
                if not b: break
                out.write(b)
    except zstandard.ZstdError: pass  # range 截尾
    return out.getvalue()

def opts_text(pairs):                  # [(键, 文本)] → "A. …\nB. …"
    return "\n".join("%s. %s" % (k, v) for k, v in pairs)

def lit(v):                            # 有的数据集把 list/dict 存成 Python 字面量字符串, 两种都认
    return ast.literal_eval(v) if isinstance(v, str) else v

ZH, EN = 4000, 6000                    # 单篇上限(字符): 中文 4000 字 ≈ 2700 token, 英文 6000 字符 ≈ 1500 token

# ---------------- 各子域的取料与成文 ----------------
def law_statute_zh():
    f = fetch("twang2218/chinese-law-and-regulations", "data/train-00000-of-00001-8329bce6db03c820.parquet")
    for r in pq_rows(f, ["title", "content"]):
        # 源是从 markdown 转的: 行尾 "\" 是续行符、行首 "> " 是引用块标记(题注), 都不是法条原文
        t = re.sub(r"\\\n", "\n", r["content"] or "")
        yield re.sub(r"(?m)^> ?", "", t)

def law_case_zh():
    f = fetch("china-ai-law-challenge/cail2018", "data/exercise_contest_valid-00000-of-00001.parquet")
    for r in pq_rows(f, ["fact"]):
        if "××" not in r["fact"]: yield r["fact"]

def law_exam_zh():
    for rep in ("hails/agieval-jec-qa-ca", "hails/agieval-jec-qa-kd"):
        for r in pq_rows(fetch(rep, "data/test-00000-of-00001.parquet"), ["query", "gold"]):
            q = r["query"].split("\n答案：")[0]           # 去掉 AGIEval 拼的作答提示"答案：从A到D, 我们应选择"
            g = lit(r["gold"])
            yield q + "\n答案：" + "".join("ABCD"[i] for i in g)

def lexglue(cfg, cols=("text",)):
    return (r["text"] for r in pq_rows(fetch("coastalcph/lex_glue", "%s/validation-00000-of-00001.parquet" % cfg), list(cols)))
def law_case_en(): return lexglue("scotus")
def law_statute_en(): return lexglue("eurlex")
def law_contract_en(): return lexglue("ledgar")

def med_encyc_zh():
    for sp in ("validation", "test"):
        for d in jsonl_lines(open(fetch("FreedomIntelligence/huatuo_encyclopedia_qa", "%s_datasets.jsonl" % sp), "rb").read()):
            qs, ans = lit(d["questions"]), lit(d["answers"])
            if qs and qs[0] and ans: yield "问：%s\n答：%s" % (qs[0][0], ans[0])

def med_consult_zh():
    for r in pq_rows(fetch("sentence-transformers/cmedqa-v2", "triplet/train-00000-of-00001.parquet"), ["anchor", "positive"]):
        yield "问：%s\n答：%s" % (r["anchor"], r["positive"])

def med_exam_zh():
    for sp in ("valid", "test"):
        for d in jsonl_lines(open(fetch("fzkuji/CMExam", "%s.json" % sp), "rb").read()):
            o = lit(d["Options"])
            t = "%s\n%s\n答案：%s" % (d["Question"], opts_text([(x["key"], x["value"]) for x in o]), d["Answer"])
            yield t + ("\n解析：" + d["Explanation"] if (d.get("Explanation") or "").strip() else "")

TAGSHELL = re.compile(r"</?(?:sup|sub|i|b|u)(?=[^a-z])")   # "<iP</i&lt;0.01" 这种丢了 ">" 的残壳
def med_paper_en():
    # MedRAG 按 PubMed 基线分片号(≈ 入库年代)排, 从 8 个等距分片各取开头 1 MB, 年代与主题都铺开
    for k in (150, 300, 450, 600, 750, 900, 1050, 1150):
        for d in jsonl_lines(open(fetch("MedRAG/pubmed", "chunk/pubmed23n%04d.jsonl" % k, 1 << 20), "rb").read()):
            yield "%s\n\n%s" % (d["title"], html.unescape(TAGSHELL.sub("", d["content"])))

def med_case_en():
    rd = csv.DictReader(open(fetch("zhengyun21/PMC-Patients", "PMC-Patients.csv"), encoding="utf-8", newline=""))
    for r in rd: yield r["patient"]

def med_exam_en():
    for d in jsonl_lines(open(fetch("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl"), "rb").read()):
        o = lit(d["options"])
        yield "Question: %s\n%s\nAnswer: %s. %s" % (d["question"], opts_text(sorted(o.items())), d["answer_idx"], d["answer"])

CSL_STEM = {"理学", "工学", "农学", "医学"}
def csl(stem):
    for d in jsonl_lines(gzip.open(fetch("neuclir/csl", "data/csl.jsonl.gz")).read()):
        if (d["category"] in CSL_STEM) != stem: continue
        kw = lit(d["keywords"])
        yield "%s\n\n摘要：%s\n关键词：%s" % (d["title"], d["abstract"], "；".join(kw))
def sci_paper_zh(): return csl(True)
def sci_paper_hss_zh(): return csl(False)

CEVAL = ["advanced_mathematics", "probability_and_statistics", "discrete_mathematics", "college_physics", "college_chemistry",
         "college_programming", "computer_architecture", "computer_network", "operating_system", "electrical_engineer",
         "metrology_engineer", "veterinary_medicine", "high_school_mathematics", "high_school_physics",
         "high_school_chemistry", "high_school_biology"]
def sci_exam_zh():
    for s in CEVAL:
        for sp in ("dev", "val", "test"):
            for r in pq_rows(fetch("ceval/ceval-exam", "%s/%s-00000-of-00001.parquet" % (s, sp)), None):
                t = "%s\n%s" % (r["question"], opts_text([(k, r[k]) for k in "ABCD"]))
                if r.get("answer"): t += "\n答案：" + r["answer"]
                if (r.get("explanation") or "").strip(): t += "\n解析：" + r["explanation"]
                yield t

def sci_arxiv_en():
    # proof-pile-2 的 arxiv 是 LaTeX 源文(公式/环境原样), 分片内已打乱; 4 个分片各取开头 6 MB 压缩流
    for k in (0, 33, 66, 99):
        for d in jsonl_lines(zst_head(fetch("EleutherAI/proof-pile-2", "arxiv/train/arXiv_%03d.jsonl.zst" % k, 6 << 20))):
            yield d["text"]

def sci_paper_en():
    # peS2o v2 的 validation-00001 全是 s2orc 全文(09-27 抽样 406/406), 跨学科; train 前 10 片是 s2ag(只有摘要)
    for d in jsonl_lines(gz_head(fetch("allenai/peS2o", "data/v2/validation-00001-of-00002.json.gz", 8 << 20))):
        yield d["text"]

MMLUPRO_STEM = {"math", "physics", "chemistry", "biology", "computer science", "engineering"}
def sci_exam_en():
    for sp in ("validation", "test"):
        for r in pq_rows(fetch("TIGER-Lab/MMLU-Pro", "data/%s-00000-of-00001.parquet" % sp), None):
            if r["category"] not in MMLUPRO_STEM: continue
            L = "ABCDEFGHIJ"
            t = "Question: %s\n%s\n" % (r["question"], opts_text(list(zip(L, lit(r["options"])))))
            cot = (r.get("cot_content") or "").strip()
            yield t + (cot if cot else "Answer: %s" % r["answer"])

TABLE = {
    "law": [("law_statute_zh", law_statute_zh, ZH), ("law_case_zh", law_case_zh, ZH), ("law_exam_zh", law_exam_zh, ZH),
            ("law_case_en", law_case_en, EN), ("law_statute_en", law_statute_en, EN), ("law_contract_en", law_contract_en, EN)],
    "med": [("med_encyc_zh", med_encyc_zh, ZH), ("med_consult_zh", med_consult_zh, ZH), ("med_exam_zh", med_exam_zh, ZH),
            ("med_paper_en", med_paper_en, EN), ("med_case_en", med_case_en, EN), ("med_exam_en", med_exam_en, EN)],
    "sci": [("sci_paper_zh", sci_paper_zh, ZH), ("sci_paper_hss_zh", sci_paper_hss_zh, ZH), ("sci_exam_zh", sci_exam_zh, ZH),
            ("sci_arxiv_en", sci_arxiv_en, EN), ("sci_paper_en", sci_paper_en, EN), ("sci_exam_en", sci_exam_en, EN)],
}
MINC = 80                              # 太短的(空壳条目/一句话条款)不要

def pick(docs, target):
    """等距抽, 铺满整个来源(同 fin_corpus_build.sh), 字节口径与下面的满额校验同一把(正文 UTF-8, 不计换行)。
    步距先取 总字节/目标字节 的整数部分; 篇幅长短不一时整趟铺完可能差一点(09-27 实撞: 法规 61 篇 588 KB),
    就把步距缩一成从头重铺, 仍是等距, 不是"不够就从头部补"。"""
    B = [len(d.encode()) for d in docs]
    if sum(B) <= target: return docs
    step = max(1, sum(B) // target)
    while True:
        sel, acc = [], 0
        for i in range(0, len(docs), step):
            sel.append(docs[i]); acc += B[i]
            if acc >= target: return sel
        if step == 1: return sel
        step = max(1, step * 9 // 10)

print("  %-18s %8s %8s %10s" % ("子域", "可用篇", "入选篇", "字节"))
bad = []
for sub, gen, n in TABLE[DOM]:
    seen, docs = set(), []
    for t in gen():
        t = cap(clean((t or "")[:3 * n]), n)        # 先粗截再清洗: 法规/论文全文动辄几万字, 逐字清洗整篇是白花时间
        # 去重按全文不按前 64 字: 试题/病例的开头是模板句("A 45-year-old man presents…"、"问题：关于…下列说法正确的是"),
        # 09-27 按前缀去重把 MedQA 10178 题误删成 8027 —— 删的是不同的题, 不是重复
        if len(t) < MINC or t in seen: continue
        seen.add(t); docs.append(t)
    sel = pick(docs, TARGET); size = sum(len(d.encode()) for d in sel)
    with open(os.path.join(OUT, sub + ".jsonl"), "w", encoding="utf-8") as f:
        for d in sel: f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    print("  %-18s %8d %8d %10d" % (sub, len(docs), len(sel), size), flush=True)
    if size < TARGET: bad.append(sub)
if bad: sys.exit("★%s 这些子域凑不满 %d 字节★" % (" ".join(bad), TARGET))
PY
    LOG "$DOM 产物: $(ls "$OUT" | wc -l | tr -d ' ') 个子域, $(du -sh "$OUT" | cut -f1)"
done
LOG "SIDECAR_CORPUS_DONE"
