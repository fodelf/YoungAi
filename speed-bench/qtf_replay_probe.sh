#!/usr/bin/env bash
# qtf_replay_probe.sh — ★真实场景尺★(2026-09-21): 用 quant_trading_flow 自己的 CFO 决策重跑(POST /api/review/replay,
# 非思考 + 温 0, 材料 = 该股当日的技术/策略/事件/基本面报告全在提示里)对自选股逐只跑一遍, 把 cfo_report 原文拉回来, 数:
#   ①复读: 每 300 字一段"前文已出现过的 4-gram"占比(与 d1_kv_ring_gate.sh loop 模式同定义; 100 = 整段逐字抄前文)
#   ②"约？"占位符次数  ③目标价/止损/入场价/预期收益齐不齐(后端从报告尾部 JSON 解析)  ④与基准 tag 逐只对比(后端 /replay/compare: 数字差 + 报告相似度)
# 为什么用它而不是 fin_chat_prompt: 那条是测速基准(凭记忆写数, 提示里没数据), 不是产品场景(2026-09-21 实撞, 白耗一周);
# 产品的请求 = qtf 早盘/复盘流水线发给 ds4-server 的, 重跑走的就是同一套提示词与同一条服务。
# 前提: spark 上 ds4-server 已起(serve_1m_spark.sh start <模型> <反修目录> [--trace FILE]); qtf 后端在本机 docker(宿主 8001)。
#   本机 shell 若设了 http_proxy, curl 会被代理吃掉回 502 —— 这里一律 --noproxy '*'。
# 用法: ./speed-bench/qtf_replay_probe.sh start  <tag> <决策日,逗号> [symbols,逗号]   发起(后端串行跑, 立即返回)
#       ./speed-bench/qtf_replay_probe.sh status <tag>                                 已落库几只(轮询用)
#       ./speed-bench/qtf_replay_probe.sh report <tag> [基准 tag]                       拉原文 + 三率 + 对比; 原文落 /tmp/qtf-replay/<tag>_<symbol>.txt
#       ./speed-bench/qtf_replay_probe.sh direct <标签> <trace文件> <请求序号|last> [temperature] [top_p] [seed] [min_p] [dry_multiplier] [dry_allowed_length] [dry_base]
#                                                                                      ★在 spark 本机跑★: 把 trace 里捕获的**那一条真实请求**原样重放给
#                                      ds4-server(8000), 只改采样参数(不给 = 原样, 即温 0), 出同一套三率。
#   为什么要 direct 档(2026-09-22): qtf 自己的 deepseek_llm_t0 写死 temperature=0, 走 /api/review/replay 改不了口径; 而"温 0 死循环只有采样能治"
#   这个判决要拿**真实请求**验(铁律: 产品问题只认真实请求)。重放用的是 trace 里的 raw request json 原字节 ⇒ 与产品同一条提示、同一套模板。
#   出错会怎样: 服务没起 = curl 空回退 2; 采样参数给错(top_p 0)= 引擎按 1.0 处理, 日志里 "解码采样" 那行会如实打出真值, 以它为准。
# 出错会怎样: 后端不可达 = curl 空回 + 退 2; 后端跳过"无材料"的股只在容器日志里说(docker logs), 这里按落库数对账。
# (python 段写在 bash 单引号里 ⇒ 里面不能出现单引号, f-string 花括号里也不能有反斜杠引号 —— 用 % 格式化。)
set -u
cd "$(dirname "$0")/.." || exit 1
API="${QTF_API:-http://127.0.0.1:8001/api}"
CMD="${1:-}"; TAG="${2:-}"
OUT=/tmp/qtf-replay; mkdir -p "$OUT"
get(){ curl -s --noproxy '*' -m 60 "$API$1"; }
case "$CMD" in
  start)
    DATES="${3:?决策日}"; SYMS="${4:-}"
    r=$(curl -s --noproxy '*' -m 30 -X POST "$API/review/replay?tag=$TAG&added_dates=$DATES&symbols=$SYMS")
    [ -n "$r" ] || { echo "★后端不可达 $API★"; exit 2; }
    echo "$r"
    get "/watchlist" | python3 -c '
import sys, json
dates = set(sys.argv[1].split(",")); syms = set(s for s in sys.argv[2].split(",") if s)
w = json.load(sys.stdin); items = w if isinstance(w, list) else w.get("items", w)
n = [x for x in items if x.get("added_date") in dates and (not syms or x.get("symbol") in syms) and (x.get("recommendation_agent_outputs") or {}).get("data_report")]
print("预计落库 %d 只: %s" % (len(n), " ".join("%s/%s" % (x.get("symbol"), x.get("added_date")) for x in n)))' "$DATES" "$SYMS";;
  status)
    get "/review/replay?tag=$TAG" | python3 -c '
import sys, json
d = json.load(sys.stdin)["items"]
print("%d 只已落库; %s" % (len(d), " ".join("%s:%s/%ss" % (x["symbol"], x.get("target_price"), x.get("elapsed_s")) for x in d)))';;
  report)
    BASE="${3:-}"
    get "/review/replay?tag=$TAG&full=true" | python3 -c '
import sys, json
tag, out = sys.argv[1], sys.argv[2]
d = json.load(sys.stdin)["items"]
print("== %s: %d 只" % (tag, len(d)))
for x in d:
    t = x.get("cfo_report") or ""
    open("%s/%s_%s.txt" % (out, tag, x["symbol"]), "w").write(t)
    c = [ch for ch in t if not ch.isspace()]
    n = len(c) - 3; seen = set(); seg = []; cnt = 0; tot = 0
    for i in range(max(n, 0)):
        g = "".join(c[i:i+4]); tot += 1
        if g in seen: cnt += 1
        seen.add(g)
        if tot == 300: seg.append(int(100 * cnt / 300 + 0.5)); cnt = 0; tot = 0
    if tot: seg.append(int(100 * cnt / tot + 0.5))
    ph = t.count("约？") + t.count("约?")
    nums = dict((k, x.get(k)) for k in ("entry_price", "target_price", "stop_loss", "expected_return"))
    miss = [k for k, v in nums.items() if v in (None, 0, "")]
    flag = "  ★死循环段★" if any(s >= 95 for s in seg) else ""
    print("  %s %s %s: %d 字 %ss  约？%d  分段: %s  数字 %s %s%s" % (x["symbol"], x.get("symbol_code", ""), x["added_date"], len(t), x.get("elapsed_s"), ph,
          " ".join(map(str, seg)), nums, ("★缺 " + ",".join(miss)) if miss else "齐", flag))' "$TAG" "$OUT"
    if [ -n "$BASE" ]; then
      echo "== 对比 基准 $BASE → $TAG(后端 /replay/compare)"
      get "/review/replay/compare?base=$BASE&new=$TAG" | python3 -c '
import sys, json
for r in json.load(sys.stdin)["rows"]:
    keys = dict((k, v) for k, v in r.items() if k.startswith(("target", "stop", "entry", "expected")))
    print(" ", r["symbol"], r.get("symbol_code", ""), r["added_date"], keys, "相似度", r.get("report_similarity"), "差异行", r.get("diff_lines"))'
    fi;;
  direct)
    TRACE="${3:?trace 文件}"; REQ="${4:-last}"; TEMP="${5:-}"; TOPP="${6:-}"; SEED="${7:-}"; MINP="${8:-}"
    DRYM="${9:-}"; DRYA="${10:-}"; DRYB="${11:-}"
    [ -s "$TRACE" ] || { echo "★没有 trace $TRACE(direct 档要在 spark 本机跑)★"; exit 2; }
    TAG="${TAG:?标签}"
    python3 - "$TRACE" "$REQ" "$OUT/direct_$TAG" "$TEMP" "$TOPP" "$SEED" "$MINP" "$DRYM" "$DRYA" "$DRYB" <<'PYEOF'
import json, sys, time, urllib.request
trace, req, out, temp, topp, seed, minp, drym, drya, dryb = sys.argv[1:11]
t = open(trace, encoding="utf-8", errors="replace").read()
parts = t.split("===== request ")
k = len(parts) - 1 if req == "last" else int(req)
raw = parts[k].split("--- raw request json ---\n", 1)[1].split("\n--- rendered prompt ---\n", 1)[0].strip()
body = json.loads(raw)
changed = []
for key, val, cast in (("temperature", temp, float), ("top_p", topp, float), ("seed", seed, int), ("min_p", minp, float),
                       ("dry_multiplier", drym, float), ("dry_allowed_length", drya, int), ("dry_base", dryb, float)):
    if val != "":
        body[key] = cast(val); changed.append("%s=%s" % (key, body[key]))
body["stream"] = False
print("重放 trace 第 %d 条请求(%d 条 message, %d 字 json); 改了: %s" %
      (k, len(body.get("messages", [])), len(raw), ", ".join(changed) or "没改(原样口径)"), flush=True)
t0 = time.time()
rq = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                            data=json.dumps(body, ensure_ascii=False).encode(),
                            headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(rq, timeout=3600).read().decode())
el = time.time() - t0
txt = r["choices"][0]["message"].get("content") or ""
open(out + ".txt", "w", encoding="utf-8").write(txt)
c = [ch for ch in txt if not ch.isspace()]
seen, cnt, tot, seg = set(), 0, 0, []
for i in range(max(len(c) - 3, 0)):
    g = "".join(c[i:i + 4]); tot += 1
    if g in seen: cnt += 1
    seen.add(g)
    if tot == 300: seg.append(int(100 * cnt / 300 + 0.5)); cnt = tot = 0
if tot: seg.append(int(100 * cnt / tot + 0.5))
nums = {}
j = txt.rfind("{")
if j >= 0:
    try: nums = json.loads(txt[j:txt.find("}", j) + 1].replace("’", "\"").replace("'", "\""))
    except Exception: nums = {}
want = ("entry_price", "target_price", "stop_loss", "expected_return")
miss = [w for w in want if not nums.get(w)]
print("%s: finish=%s %d token %.0fs %d 字  约？%d  JSON %s  死循环段 %s\n  分段: %s" % (
    out.rsplit("/", 1)[-1], r["choices"][0].get("finish_reason"), r["usage"]["completion_tokens"], el, len(txt),
    txt.count("约？") + txt.count("约?"),
    "齐 " + str({w: nums.get(w) for w in want}) if not miss else "★缺 " + ",".join(miss),
    "★有★" if any(s >= 95 for s in seg) else "无", " ".join(map(str, seg))), flush=True)
PYEOF
    ;;
  health)
    # ★整条流水线的真实请求体检★(2026-09-22): 把 trace 里捕获的**多条**真实请求按原字节依次重放,
    # 每条出一行三率 —— 判"当前模型在产品的真实输入上出不出得了活"。串行发(服务端本来就一次跑一条图)。
    # 用法: health <标签前缀> <trace文件:请求序号> [trace文件:请求序号 ...]
    #   例: health h1 /tmp/ds4-trace-cedfix3.txt:3 /tmp/ds4-trace-cedfix3.txt:2
    # 为什么不新写脚本: 单条重放的活 direct 档已经有了, 这里只是把它按列表跑一遍(尺只留一份)。
    shift 2
    [ $# -gt 0 ] || { echo "用法: $0 health <标签前缀> <trace:请求号> [...]"; exit 2; }
    for spec in "$@"; do
        tf="${spec%:*}"; rq="${spec##*:}"
        echo "=========== $tf 第 $rq 条  $(date +%H:%M:%S)"
        "$0" direct "${TAG}_${rq}" "$tf" "$rq" || echo "★这条没跑成★"
    done
    ;;
  *) echo "用法: $0 start <tag> <决策日,逗号> [symbols] | status <tag> | report <tag> [基准tag] | direct <标签> <trace> <请求序号> [temp] [top_p] [seed] [min_p] | health <标签> <trace:请求号>..."; exit 2;;
esac
