#!/bin/bash
# qtf_requests_mac.sh — Mac 侧: 从 qtf 容器里按当天口径重建真实请求(大盘 / 个股 CFO), 落到本仓 gguf/v41/night/req/,
# 再推到 spark 同路径, 给 z_nightly_spark.sh 的 sample 段用(2026-09-29, back.md §14.3 逐日回放)。
#
# 为什么在 Mac 上跑: qtf 的库与 crewAI 都在 Mac 的 OrbStack 容器里, 重建必须走 CfoCrew/MarketCrew 本身
# (拿到的是 litellm 真正发出去的字节); spark 只跑模型。请求 JSON 的 _note 里已带打分事实(次日 OHLC / 大盘真值),
# spark 侧不用再连库。
#
# 用法: qtf_requests_mac.sh market <日期>            → req/market_<日期>.json
#       qtf_requests_mac.sh cfo    <日期> <代码|all> → req/cfo_<日期>_<代码>.json(all = 这一天材料齐全的全部股票)
#       qtf_requests_mac.sh inventory                → 逐日盘点(只读)
#       qtf_requests_mac.sh corpus <料名>            → 复盘全文做成后训练料(要 qtf 跑过复盘)
#       qtf_requests_mac.sh feedback <料名> <决策日>... → 交易反馈做成后训练料(只要 cfo 请求 + 次日行情, 不要复盘), 并放进 spark 料池
# 已有的文件不重建(要重建先删)。每次结束 rsync 整个 req/ 到 spark。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CAP="$ROOT/gguf-tools/scripts/qtf_capture_request.py"
REQ="$ROOT/gguf/v41/night/req"; mkdir -p "$REQ"
CTR="quant_trading_flow-main-backend-1"
PY="/app/.venv/bin/python"
run_cap(){ docker exec -i "$CTR" "$PY" - "$@" < "$CAP"; }
# feedback 只读本机 req/ 里已重建好的请求, 不碰容器
[ "${1:-}" = feedback ] || docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "★qtf 后端容器 $CTR 没起(OrbStack/docker)★"; exit 2; }

build_one(){   # <kind> <date> [symbol]
    local kind="$1" date="$2" sym="${3:-}" out
    if [ "$kind" = market ]; then out="$REQ/market_${date}.json"; else out="$REQ/cfo_${date}_${sym}.json"; fi
    if [ -s "$out" ]; then echo "  已有 $(basename "$out")"; return 0; fi
    echo "  重建 $(basename "$out") …"
    if [ "$kind" = market ]; then run_cap market "$date" > "$out.tmp" 2> "$out.log"
    else run_cap cfo "$date" "$sym" > "$out.tmp" 2> "$out.log"; fi
    if [ -s "$out.tmp" ] && python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); assert b.get("messages") and b.get("_note")' "$out.tmp" 2>/dev/null; then
        mv "$out.tmp" "$out"; rm -f "$out.log"
        python3 - "$out" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); n = b["_note"]
extra = ""
if n["kind"] == "cfo":
    extra = " 次日OHLC=%s 早盘JSON=%s" % (n["ohlc"][:2] if n.get("ohlc") else n.get("ohlc_error", "无"), n.get("live_json"))
else:
    extra = " 真值=%s" % n.get("truth")
print("    %s: messages=%s stop=%s end_date=%s%s" % (n["kind"], [(m["role"], len(m["content"])) for m in b["messages"]], b.get("stop"), n["end_date"], extra))
PY
    else
        echo "  ★重建失败, 看 $out.log 尾:★"; tail -5 "$out.log" | cut -c1-200; rm -f "$out.tmp"; return 1
    fi
}

# chunk_docs <料目录>: docs/<篇名>.txt → chunks/<篇名>_cNN.txt 教师上下文块(corpus / feedback 共用, 切法只写这一处)。
#   每块 = 篇头(个股: 到 "二、/三、/四、" 之前 = 头部行 + 结论节, 每块都带, 核心事实在每个块里都能被问到; 大盘: 第一行) + 余下段落按 ≤ 3000 字切;
#   大盘每 3 天一块(18 天挤一块时逐日数字分到的题太少, 10-01 小子集实撞: 逐日家数/涨跌全记错)。篇类按名: market_* 大盘, 其余个股。
#   为什么切块: 教师每答一题都要把块放进上下文预填一遍; 整篇 1 万多字时一题十几秒, 块 3 千字左右 3~4 秒(KMs/Cartridges 也按 2k token 级切)。
chunk_docs(){
    python3 - "${1:?料目录}" <<'PY'
import glob, os, sys
out = sys.argv[1]
CHUNK = 3000
os.makedirs(os.path.join(out, "chunks"), exist_ok=True)
nch = nm = ns = nchar = 0
for f in sorted(glob.glob(os.path.join(out, "docs", "*.txt"))):
    name = os.path.basename(f)[:-4]
    kind = "market" if name.startswith("market_") else "stock"
    text = open(f).read()
    nchar += len(text); nm += kind == "market"; ns += kind == "stock"
    paras = text.rstrip("\n").split("\n")
    if kind == "stock":
        k = next((i for i, p in enumerate(paras) if p.startswith("二、") or p.startswith("三、") or p.startswith("四、")), len(paras))
        head, rest = paras[:k], paras[k:]
    else:
        head, rest = paras[:1], paras[1:]
    hs = "\n".join(head)
    chunks, cur = [], []
    if kind == "market":
        chunks = [rest[i:i + 3] for i in range(0, len(rest), 3)]
        rest = []
    for p in rest:
        while len(p) > CHUNK:   # 一段本身超长(报告里的大表): 硬切, 不丢字
            if cur: chunks.append(cur); cur = []
            chunks.append([p[:CHUNK]]); p = p[CHUNK:]
        if cur and sum(len(x) + 1 for x in cur) + len(p) > CHUNK:
            chunks.append(cur); cur = []
        cur.append(p)
    if cur: chunks.append(cur)
    if not chunks: chunks = [[]]
    for i, c in enumerate(chunks):
        body = hs + ("\n" + "\n".join(c) if c else "")
        open(os.path.join(out, "chunks", "%s_c%02d.txt" % (name, i)), "w").write(body + "\n")
        nch += 1
print("切块: %d 篇(大盘 %d / 个股 %d), %d 字 → %d 块" % (nm + ns, nm, ns, nchar, nch))
PY
}

# corpus <料名>: 复盘全文 → gguf-tools/data/posttrain/<料名>/{docs,chunks}/, 再推 spark(后训练 ③ 第八版 = 上下文蒸馏, 2026-10-01)。
#   docs/<篇名>.txt   一篇一份复盘(大盘逐日汇总 / 个股: 结论·逐条修正·修正后要点·当日材料·修正后报告); 切块见 chunk_docs。
corpus(){
    local name="${1:?料名}" out="$ROOT/gguf-tools/data/posttrain/${1}"
    [ -e "$out/docs" ] && { echo "★$out/docs 已存在(要重导先挪走)★"; return 1; }
    mkdir -p "$out/docs" "$out/chunks"
    run_cap corpus > "$out/corpus.json" 2> "$out/corpus.log" || { tail -5 "$out/corpus.log"; return 1; }
    python3 - "$out" <<'PY' || return 1
import json, os, sys
out = sys.argv[1]
docs = json.load(open(os.path.join(out, "corpus.json")))["docs"]
for d in docs:
    open(os.path.join(out, "docs", d["name"] + ".txt"), "w").write(d["text"])
print("复盘全文: %d 篇(大盘 %d / 个股 %d), %d 字" % (len(docs), sum(d["kind"] == "market" for d in docs),
      sum(d["kind"] == "stock" for d in docs), sum(len(d["text"]) for d in docs)))
PY
    chunk_docs "$out" || return 1
    rsync -az "$out/" "spark:ds4-main/gguf-tools/data/posttrain/$name/" && echo "  推到 spark OK"
}

# feedback <料名> <决策日>...: ★每个交易日的交易反馈直接做成料★(2026-10-06, 用户 "使用每一天的交易反馈, 加入自增训练目录")。
#   来源 = 本机已重建的真实 CFO 请求 req/cfo_<决策日>_<代码>.json: messages 里的四份上游材料(决策日材料) + _note.live_json(早盘真实给出的
#   入场/目标/止损) + _note.ohlc[1](决策日之后第一个交易日 = "次日"的实际 收盘/最高/最低)。不需要 qtf 跑过复盘(lessons): 库里复盘只到决策日
#   0904, 而 0908 的 CFO 决策 + 次日行情早就在, 只是没人把它写成料 —— 这一天也就成了模型从没见过的"第二天"。
#   产物与 corpus 同构: docs/stock_<次日>_<代码>.txt(头部行与复盘头部行逐字同格式, kd_domain/finance.sh 的 finance_parse 一行不改就能进账本;
#   "四、决策当日的输入材料" 三段标题 技术数据报告/策略报告/事件 与复盘一致, 决策提示的节选规则同样适用; 基本面报告放第五节, 节选在 "五、" 停)
#   + chunks/(chunk_docs)。★金标对拍★: 同一笔决策若本机 data/posttrain/*/docs/ 里有复盘篇(lessons 口径), 头部行六个数 + 次日 + 名称逐项比,
#   不同就停车 —— 10-06 对 0903/0904 的 17 笔全同, 这是"请求 + 行情 = 复盘头部行"的依据。
#   推 spark: 料目录照 corpus; 另把 docs/ chunks/ 放进料池 incr/pool/(--ignore-existing: 同名复盘篇不覆盖), 之后 spark 上 kdtake/kdinc 自取。
feedback(){
    local name="${1:?料名}" out="$ROOT/gguf-tools/data/posttrain/${1}"; shift
    [ $# -gt 0 ] || { echo "用法: feedback <料名> <决策日>..."; return 2; }
    [ -e "$out/docs" ] && { echo "★$out/docs 已存在(要重导先挪走)★"; return 1; }
    mkdir -p "$out/docs"
    python3 - "$out" "$REQ" "$ROOT/gguf-tools/data/posttrain" "$@" <<'PY' || { rm -rf "$out"; return 1; }
import glob, json, os, re, sys
out, req, base, dates = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
HEAD = re.compile(r"【复盘 (\d+) (\d+) ([^】]+)】决策日 (\d+) 给出: 入场价 ([0-9.]+), 目标价 ([0-9.]+), 止损位 ([0-9.]+)。次日实际: 收盘 ([0-9.]+), 最高 ([0-9.]+), 最低 ([0-9.]+)")
SEC = [("交易数据报告", "## 技术数据报告"), ("策略数据报告", "## 策略报告"), ("事件分析报告", "## 事件"), ("基本面报告", None)]
def cn(d): return "%d年%d月%d日（%s）" % (int(d[:4]), int(d[4:6]), int(d[6:8]), d)
def pct(x): return "%+.1f%%" % x
n = 0
for date in dates:
    files = sorted(glob.glob(os.path.join(req, "cfo_%s_*.json" % date)))
    if not files: sys.exit("★%s 没有 cfo 请求(先 qtf_requests_mac.sh cfo %s all)★" % (date, date))
    for f in files:
        b = json.load(open(f)); nt = b["_note"]; sym, nm, lj, o = nt["symbol"], nt["symbol_code"], nt["live_json"], nt.get("ohlc") or []
        if len(o) < 2 or o[0]["date"].replace("-", "") != date: sys.exit("★%s: 次日行情不齐(ohlc=%s)★" % (f, [x["date"] for x in o[:2]]))
        E, T, S = (float(lj.get(k) or 0) for k in ("entry_price", "target_price", "stop_loss"))
        if min(E, T, S) <= 0: sys.exit("★%s: 早盘决策数字缺(%s)★" % (f, lj))
        nx = o[1]; C, H, L = float(nx["close"]), float(nx["high"]), float(nx["low"]); nd = nx["date"].replace("-", "")
        u = b["messages"][-1]["content"]
        pos = [(re.search(r"^[ \t]*\d+\. %s[:：] ?" % t, u, re.M), t) for t, _ in SEC]
        end = re.search(r"^【重要时间说明】", u, re.M)
        if any(m is None for m, _ in pos) or end is None: sys.exit("★%s: 四份上游材料切不出(%s)★" % (f, [t for m, t in pos if m is None]))
        bounds = [m.end() for m, _ in pos] + [end.start()]
        sec = [u[bounds[i]:pos[i + 1][0].start() if i + 1 < len(pos) else bounds[-1]].strip() for i in range(len(pos))]
        head = "【复盘 %s %s %s】决策日 %s 给出: 入场价 %.2f, 目标价 %.2f, 止损位 %.2f。次日实际: 收盘 %.2f, 最高 %.2f, 最低 %.2f, 收盘相对目标价偏差 %s。" % (
            nd, sym, nm, date, E, T, S, C, H, L, pct((C - T) / T * 100))
        mine = HEAD.search(head).groups()
        for g in glob.glob(os.path.join(base, "*", "docs", "stock_%s_%s.txt" % (nd, sym))):   # 金标对拍: lessons 口径的复盘头部行
            m = HEAD.search(open(g).read())
            if m and m.groups() != mine: sys.exit("★%s 与复盘 %s 头部行不同:\n  请求 %s\n  复盘 %s★" % (f, g, mine, m.groups()))
        fact = "入场到次日最高 %s，到次日最低 %s，到收盘 %s；目标%s（目标价%s次日最高价 %.1f%%）；止损%s触发。" % (
            pct((H - E) / E * 100), pct((L - E) / E * 100), pct((C - E) / E * 100), "达成" if H >= T else "未达成",
            "高出" if T > H else "低于", abs(T - H) / H * 100, "被" if L <= S else "未被")
        parts = ["【个股交易反馈】%s %s，决策日 %s，次日 %s" % (sym, nm, cn(date), cn(nd)), "一、决策与次日实际", head, fact,
                 "四、决策当日的输入材料", "# %s %s 交易决策 %s" % (sym, nm, date)]
        for (t, h), body in zip(SEC, sec):
            parts += ([h] if h else ["五、决策当日的基本面报告"]) + [body]
        open(os.path.join(out, "docs", "stock_%s_%s.txt" % (nd, sym)), "w").write("\n".join(parts) + "\n")
        n += 1
        print("  %s %s %s: 入场 %.2f 目标 %.2f 止损 %.2f → 次日 %s 收 %.2f 高 %.2f 低 %.2f; %s" % (date, sym, nm, E, T, S, nd, C, H, L, fact))
print("交易反馈: %d 篇 ← %s" % (n, " ".join(dates)))
PY
    chunk_docs "$out" || return 1
    rsync -az "$out/" "spark:ds4-main/gguf-tools/data/posttrain/$name/" || return 1
    rsync -az --ignore-existing "$out/docs/" "spark:ds4-main/gguf/v41/posttrain/incr/pool/docs/" \
        && rsync -az --ignore-existing "$out/chunks/" "spark:ds4-main/gguf/v41/posttrain/incr/pool/chunks/" \
        && echo "  推到 spark OK: 料目录 + 料池 incr/pool/(同名篇不覆盖); 接着 spark 上 z_nightly_spark.sh kdpool 看进度"
}

case "${1:-}" in
  corpus) corpus "${2:?料名}"; exit $?;;
  feedback) shift; feedback "$@"; exit $?;;
  inventory) run_cap inventory;;
  market) build_one market "${2:?日期}";;
  cfo)
    date="${2:?日期}"; sym="${3:?代码|all}"
    if [ "$sym" = all ]; then
        for s in $(run_cap symbols "$date"); do build_one cfo "$date" "$s"; done
    else build_one cfo "$date" "$sym"; fi;;
  *) echo "用法: $0 inventory | market <日期> | cfo <日期> <代码|all> | corpus <料名> | feedback <料名> <决策日>..."; exit 2;;
esac
[ "${1:-}" = inventory ] && exit 0
echo "推到 spark: gguf/v41/night/req/"
rsync -az "$REQ/" spark:ds4-main/gguf/v41/night/req/ && echo "  OK $(ls "$REQ"/*.json 2>/dev/null | wc -l | tr -d ' ') 个请求"
