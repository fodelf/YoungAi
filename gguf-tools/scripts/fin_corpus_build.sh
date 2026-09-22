#!/bin/bash
# fin_corpus_build.sh — 金融语料构建(Mac 侧跑; 2026-09-08 用户令"使用金融语料量化/反修/看股票金融五指标")
#
# 产物: gguf-tools/data/corpus/fin/<域>.txt —— 一行 = 一篇文档(文内换行折成空格), 五个域各一个文件:
#   fin_announce  上市公司公告            FinCorpus announcement_data
#   fin_article   财经文章/研报            FinCorpus fin_articles_data
#   fin_news      财经新闻                 FinCorpus fin_news_data_final
#   fin_exam      金融/会计/证券从业试题    FinCorpus fin_exam
#   fin_flash     东方财富 7×24 快讯       quant_trading_flow 的 MongoDB news_items(真实抓取, 不是 LLM 产物)
#   (FinCorpus = huggingface Duxiaoman-DI/FinCorpus, apache-2.0, 中文金融资讯数据集)
#
# 为什么"一行一篇文档": amp_campaign.sh idshalf 的目录模式把每一行当一个独立文档簇, 整篇只进
#   量化/反修/判决三份中的一份 —— 同一篇公告的上下段绝不跨份(否则反修份见过量化份的原文, 拟合料不独立)。
#   段落若各占一行就会被拆散跨份, 所以文内换行一律折成空格。
# 为什么切片不整下: FinCorpus 压缩后 21 GB, 而三份合计只抽 3×8192 token; 每域 ~1.2 MB 文本
#   (≈ 60 万 token)已是需求的 20 倍余量, 与现役 calibration_datav5(1.7 MB)同量级。
#   HTTP Range 只取每个分片开头 45 MB 的 gzip 流, 解到截尾自然报错即停(gzip 流可从头部分解码)。
# 为什么在 Mac 下: spark 到 huggingface.co / hf-mirror 全超时(fable5 09-01 实测), Mac 直连 HTTP 200。
# 抽文档"等距"不"取前 N 篇": 分片开头往往按时间/公司排序, 取前 N 篇会全是同一家公司/同一周的稿子。
#
# 用法: bash gguf-tools/scripts/fin_corpus_build.sh [raw 缓存目录]
#   默认缓存 gguf/go-onebit/fin/raw(gguf/ 在 .gitignore 内); 缓存已在则不重下, 产物目录每次整体重建。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="${1:-$ROOT/gguf/go-onebit/fin/raw}"
OUT="$ROOT/gguf-tools/data/corpus/fin"
HFB="https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus/resolve/main/data"
RANGE_BYTES=45000000          # 每个大分片只取开头 45 MB gzip(≈150 MB 文本, 够抽 1.2 MB 等距文档)
TARGET_BYTES=1200000          # 每域产物目标字节(UTF-8)
DOC_CAP_CHARS=4000            # 单篇上限: 公告动辄 5 万字, 截到 4000 字保留篇数(多样性)而非单篇长度
MIN_CHARS=80                  # 太短的(标题党/空公告)不要
MONGO_CTN="quant_trading_flow-main-mongodb-1"
LOG(){ echo "[fincorpus $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }
mkdir -p "$RAW" "$OUT"

fetch(){   # $1=远端文件 $2=本地名 $3=range(空=整下)
    local f="$RAW/$2"
    [ -s "$f" ] && { LOG "缓存已在 $2 $(du -h "$f" | cut -f1)"; return 0; }
    LOG "下载 $1 ${3:+(range 0-$3)}"
    # rc=18 = 服务端在 range 末尾提前断流(部分传输), 对 range 取片这是正常收尾, 不算失败
    curl -sL -m 1200 ${3:+-r 0-$3} -o "$f.part" "$HFB/$1"; local rc=$?
    [ $rc -eq 0 ] || [ $rc -eq 18 ] || { rm -f "$f.part"; DIE "下载失败 rc=$rc: $1"; }
    mv "$f.part" "$f"
}
fetch fin_exam.jsonl.gz          fin_exam.jsonl.gz     ""
fetch announcement_data.jsonl.gz announcement.part.gz  $RANGE_BYTES
fetch fin_articles_data.jsonl.gz fin_articles.part.gz  $RANGE_BYTES
fetch fin_news_data_final.jsonl.gz fin_news.part.gz    $RANGE_BYTES

# 快讯: 标题常是摘要的前半句, 重复则只留摘要; 同一条快讯多源抓取会重复, 按整行去重
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$MONGO_CTN"; then
    LOG "导出 MongoDB news_items(东方财富快讯)"
    docker exec "$MONGO_CTN" mongosh quant_trading --quiet --eval '
        db.news_items.find({},{title:1,summary:1,_id:0}).forEach(d=>{
            const t=(d.title||"").trim(), s=(d.summary||"").trim();
            let x = s.startsWith(t.slice(0,12)) ? s : (t ? t+"。"+s : s);
            x = x.replace(/\s+/g," ").trim(); if (x.length>=20) print(x) })' \
        | sort -u > "$RAW/fin_flash.raw.txt" || DIE "mongosh 导出失败"
else
    [ -s "$RAW/fin_flash.raw.txt" ] || DIE "MongoDB 容器 $MONGO_CTN 未运行且缓存无 fin_flash.raw.txt(先 quant_trading_flow ./start.sh)"
    LOG "MongoDB 未运行, 用缓存快讯 $(wc -l < "$RAW/fin_flash.raw.txt") 条"
fi

# jsonl.gz → 一行一篇: 解到截尾为止, 等距抽到目标字节。Python 只做文本编排, 不参与任何数值。
rm -rf "$OUT"; mkdir -p "$OUT"
python3 - "$RAW" "$OUT" "$TARGET_BYTES" "$DOC_CAP_CHARS" "$MIN_CHARS" <<'PY' || DIE "抽取失败"
import gzip, json, re, sys, os, unicodedata
raw, out, TARGET, CAP, MINC = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
WS = re.compile(r"\s+")
def clean(s):
    s = "".join(c for c in s if unicodedata.category(c)[0] != "C" or c in "\n\t ")
    s = WS.sub(" ", s).strip()
    return s[:CAP]
def docs_from_gz(path):
    seen, docs = set(), []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                try: t = json.loads(ln).get("text", "")
                except Exception: continue
                t = clean(t)
                if len(t) < MINC: continue
                k = t[:64]
                if k in seen: continue
                seen.add(k); docs.append(t)
    except (EOFError, gzip.BadGzipFile, OSError):
        pass                     # range 取片的截尾: 到这里为止的整行都是完整的
    return docs
def pick(docs, target):          # 等距抽: 步距按"总字节/目标字节"取整, 铺满整段解出的流
    tot = sum(len(d.encode()) + 1 for d in docs)
    if tot <= target: return docs
    step = max(1, tot // target); sel, acc = [], 0
    for i in range(0, len(docs), step):
        sel.append(docs[i]); acc += len(docs[i].encode()) + 1
        if acc >= target: break
    return sel
print("  %-13s %8s %8s %10s" % ("域", "解出篇", "入选篇", "字节"))
for dom, src in [("fin_announce", "announcement.part.gz"), ("fin_article", "fin_articles.part.gz"),
                 ("fin_news", "fin_news.part.gz"), ("fin_exam", "fin_exam.jsonl.gz")]:
    docs = docs_from_gz(os.path.join(raw, src)); sel = pick(docs, TARGET)
    with open(os.path.join(out, dom + ".txt"), "w", encoding="utf-8") as f:
        for d in sel: f.write(d + "\n")
    print("  %-13s %8d %8d %10d" % (dom, len(docs), len(sel), sum(len(d.encode()) + 1 for d in sel)))
flash = [clean(x) for x in open(os.path.join(raw, "fin_flash.raw.txt"), encoding="utf-8")]
flash = [x for x in flash if len(x) >= 20]
with open(os.path.join(out, "fin_flash.txt"), "w", encoding="utf-8") as f:
    for d in flash: f.write(d + "\n")
print("  %-13s %8d %8d %10d" % ("fin_flash", len(flash), len(flash), sum(len(d.encode()) + 1 for d in flash)))
PY
LOG "产物 $OUT:"; wc -c "$OUT"/*.txt | sed 's/^/  /'
LOG "FIN_CORPUS_DONE"
