"""qtf_capture_request.py — 把 qtf 某一天的真实请求按当天口径重建出来, 截下 crewAI 发给模型的 HTTP 原字节;
顺带把打奖励要用的事实(次日 OHLC / 大盘真值)一起放进 JSON 的 _note, spark 那边不用再连库。

用法(Mac, qtf 后端容器里跑; 只读, 不写库、不改 qtf 任何文件):
    docker exec -i quant_trading_flow-main-backend-1 /app/.venv/bin/python - <子命令> [参数] \
        < gguf-tools/scripts/qtf_capture_request.py > 输出
子命令:
    inventory                 逐日盘点: 哪些天有 事件报告 / CFO 上游材料 / 大盘预测 / 大盘真值(只报数)
    symbols  <日期>           这一天上游四份材料齐全的 CFO 股票代码(一行一个)
    market   <日期> [end_date] 重建大盘请求(end_date 不给就从当天大盘报告的日期行取); _note 带大盘真值
    cfo      <日期> <代码>     重建该股 CFO 请求(库里存的四份上游材料 + 与早盘同一套 CfoCrew 提示词); _note 带次日 OHLC
    ohlc     <代码> <日期>     打印 [日期, 日期+14 天] 的日线: date open high low close(一行一根)

为什么要它: 后训练 ③ 的料必须是产品真实请求(铁律: 产品问题只认真实请求)。09-23 起大盘请求已能逐字重建;
2026-09-29 起 CFO 请求也能重建 —— 库里 recommendations.agent_outputs 存了早盘那一趟的四份上游报告
(data_report / fundamental_analysis / event_data / strategy_report), CfoCrew 的提示词只吃这四份 + symbol + end_date。
★end_date 用当天大盘报告日期行里那个时间戳★: 早盘流水线开头 datetime.now() 算一次, 大盘 crew 与每只股票的
CFO 都用同一个值, 而大盘报告第一行"日期"就是它原样打印出来的。
截取方式: 把 crewAI 的 LLM base_url 指到本进程里的一个假服务, 收到第一条 POST 就把 body 原样写到 stdout 退出。
所以拿到的是 litellm → openai SDK 真正发出去的字节(含 stop 词、模型名), 不是手拼的。CFO 走的是早盘那个
思考档 LLM(deepseek_llm, 不带温度 ⇒ 服务端按模型卡采样), 不走重跑用的温度 0 版。
泄题规矩: 行情只取 < 日期; 次日 OHLC 只进 _note(打分用), 一个字不进提示。
"""
import datetime as dt
import http.server
import json
import os
import re
import sys
import threading

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

from pymongo import MongoClient  # noqa: E402

db = MongoClient(os.environ.get("MONGODB_URI", "mongodb://mongodb:27017"))["quant_trading"]
CMD = sys.argv[1] if len(sys.argv) > 1 else ""


def _date(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _bars(symbol, d0, days=14):
    from quant_trading_flow.datahub import gateway as gw
    end = d0 + dt.timedelta(days=days)
    bars = gw.get_daily_bars(symbol, d0.isoformat(), end.isoformat())
    return [{"date": b.date.isoformat(), "open": b.open, "high": b.high, "low": b.low, "close": b.close} for b in bars]


def _market_end_date(date):
    """当天大盘报告第一行的时间戳 = 早盘流水线那一趟的 end_date(所有 crew 共用)。"""
    d = db.market_analysis.find_one({"date": date})
    if not d:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", d.get("report") or "")
    return m.group(1) if m else None


def _truth(date):
    d = db.market_reviews.find_one({"date": date})
    return (d or {}).get("true_direction") or ""


def _cfo_complete(doc):
    ao = doc.get("agent_outputs") or {}
    return all(ao.get(k) for k in ("data_report", "fundamental_analysis", "event_data", "strategy_report"))


if CMD == "inventory":
    import collections
    days = collections.defaultdict(dict)
    for d in db.news_analysis.find({}, {"date": 1, "event_report": 1}):
        days[d["date"]]["event_report"] = 1 if d.get("event_report") else 0
    for d in db.market_analysis.find({}, {"date": 1, "prediction": 1, "report": 1}):
        days[d["date"]]["market_pred"] = (d.get("prediction") or "")[:6]
        days[d["date"]]["end_date"] = _market_end_date(d["date"]) or ""
    for d in db.market_reviews.find({}, {"date": 1, "true_direction": 1}):
        days[d["date"]]["truth"] = d.get("true_direction") or ""
    for d in db.recommendations.find({}, {"date": 1, "symbol": 1, "agent_outputs": 1, "report": 1}):
        k = "cfo_full" if (_cfo_complete(d) and d.get("report")) else "cfo_partial"
        days[d["date"]][k] = days[d["date"]].get(k, 0) + 1
    print("date      event  mkt_pred  truth  end_date             cfo_full cfo_partial   大盘可回放  个股可回放")
    for k in sorted(days):
        v = days[k]
        mk = bool(v.get("event_report")) and bool(v.get("end_date")) and bool(v.get("truth"))
        print("%s  %-5s  %-8s  %-5s  %-19s  %8d %11d   %s  %s" % (
            k, v.get("event_report", 0), v.get("market_pred", ""), v.get("truth", ""), v.get("end_date", ""),
            v.get("cfo_full", 0), v.get("cfo_partial", 0), "是" if mk else "否", "是" if v.get("cfo_full") else "否"))
    sys.exit(0)

if CMD == "symbols":
    for d in db.recommendations.find({"date": sys.argv[2]}, {"symbol": 1, "agent_outputs": 1, "report": 1}).sort("rank", 1):
        if _cfo_complete(d) and d.get("report"):
            print(d["symbol"])
    sys.exit(0)

if CMD == "ohlc":
    for b in _bars(sys.argv[2], _date(sys.argv[3])):
        print("%s %s %s %s %s" % (b["date"], b["open"], b["high"], b["low"], b["close"]))
    sys.exit(0)

if CMD not in ("market", "cfo"):
    sys.stderr.write(__doc__)
    sys.exit(2)

DATE = sys.argv[2]
CUT = _date(DATE)                       # 行情只许 < 这一天
END_DATE = _market_end_date(DATE)
if CMD == "market" and len(sys.argv) > 3:
    END_DATE = sys.argv[3]
if not END_DATE:
    sys.stderr.write(f"{DATE} 没有当天大盘报告(拿不到 end_date), 无法重建\n")
    sys.exit(2)

from quant_trading_flow.modules import deepseek  # noqa: E402

OUT_FD = os.dup(1)                 # JSON 只走这个 fd; fd 1 整个改指 stderr, crewAI 的彩色日志混不进来
os.dup2(2, 1)
NOTE = {"date": DATE, "end_date": END_DATE, "kind": CMD}


class _Grab(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        req = json.loads(body)
        NOTE["path"] = self.path
        req["_note"] = NOTE
        os.write(OUT_FD, json.dumps(req, ensure_ascii=False).encode("utf-8"))
        os._exit(0)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), _Grab)
threading.Thread(target=srv.serve_forever, daemon=True).start()
deepseek.deepseek_llm.base_url = f"http://127.0.0.1:{srv.server_port}/v1"

if CMD == "market":
    na = db.news_analysis.find_one({"date": DATE})
    if not na or not na.get("event_report"):
        sys.stderr.write(f"news_analysis {DATE} 没有 event_report, 无法重建\n")
        sys.exit(2)
    # 包里导出的 gateway 就是单例对象(data_tool 也是这么拿的), 直接在对象上换方法
    from quant_trading_flow.datahub import gateway as gw  # noqa: E402

    _orig_index = gw.get_index_daily

    def _index_before_cut(symbol, days=5):
        bars = [b for b in _orig_index(symbol, 40) if b.date < CUT]   # 40 根: 回溯两周以上的日子也凑得齐 5 根
        return bars[-days:]

    def _us_before_cut():
        # 东财日线在这台机上被风控(09-23 实测 RemoteDisconnected), 改取新浪日线历史: 收盘取 < 复盘日的最后一根,
        # 涨跌幅按前一根收盘算到两位。日期写复盘日 —— 当时新浪实时快照的时间戳是北京时间(凌晨 4 点多已是复盘日)。
        import requests
        lines = []
        for cn, sym in (("道琼斯", ".DJI"), ("纳斯达克", ".IXIC"), ("标普500", ".INX")):
            t = requests.get("https://stock.finance.sina.com.cn/usstock/api/jsonp.php/var%20_x=/"
                             f"US_MinKService.getDailyK?symbol={sym}", timeout=15,
                             headers={"Referer": "https://finance.sina.com.cn"}).text
            k = [b for b in json.loads(re.search(r"\((\[.*\])\)", t, re.S).group(1))
                 if dt.date.fromisoformat(b["d"]) < CUT]
            c, p = float(k[-1]["c"]), float(k[-2]["c"])
            lines.append(f"{cn},{k[-1]['c']},{(c - p) / p * 100:.2f},{CUT.isoformat()}")
        return "；".join(lines)

    gw.get_index_daily = _index_before_cut
    gw.get_us_indices_text = _us_before_cut
    from quant_trading_flow.crews.market.tools.data_tool import get_market_stock_data  # noqa: E402

    market_data = get_market_stock_data()
    NOTE.update({"us_source": "新浪日线历史(当天是新浪实时快照, 点数小数位不同)", "market_data": market_data,
                 "truth": _truth(DATE), "symbol": "大盘"})
    from quant_trading_flow.crews.market.market import MarketCrew  # noqa: E402

    MarketCrew().crew().kickoff(inputs={"end_date": END_DATE, "market_data": market_data, "news": na["event_report"]})
else:
    SYMBOL = sys.argv[3]
    rec = db.recommendations.find_one({"date": DATE, "symbol": SYMBOL})
    if not rec or not _cfo_complete(rec):
        sys.stderr.write(f"recommendations {DATE}/{SYMBOL} 没有齐全的上游材料(agent_outputs), 无法重建\n")
        sys.exit(2)
    outs = rec["agent_outputs"]
    # 与 finetune_service._sync_replay_one 同一套字段映射(存库时 fundamental_analysis ← state.data_analysis),
    # 区别只有两处: ①end_date 用早盘那个时间戳而不是 added_date; ②不开 set_replay_t0 —— 要的是早盘那条思考档路。
    from quant_trading_flow.main import TradingState, getData  # noqa: E402
    from quant_trading_flow.crews.cfo.cfo import CfoCrew  # noqa: E402

    st = TradingState()
    st.symbol = SYMBOL
    st.symbol_alice = f"{SYMBOL}.SH" if SYMBOL.startswith("6") else f"{SYMBOL}.SZ"
    st.end_date = END_DATE
    st.file_date = f"{DATE}capture"
    st.current_price = rec.get("latest_price") or 0.0
    st.has_flag = False
    st.data_report = outs.get("data_report", "")
    st.data_analysis = outs.get("fundamental_analysis", "")
    st.fundamental_analysis = outs.get("fundamental_data", "")
    st.event_data = outs.get("event_data", "")
    st.strategy_report = outs.get("strategy_report", "")
    st.stock_data = outs.get("stock_data", "")
    st.prediction = outs.get("prediction", "")
    st.risk_management = outs.get("risk_management", "")
    st.stock_events_analysis = outs.get("stock_events_analysis", "")
    st.historical_lessons = outs.get("historical_lessons", "")
    try:
        bars = _bars(SYMBOL, CUT)
    except Exception as e:  # 没有次日行情 = 打不了分, 但请求本身照样重建, 让调用方看 _note 决定
        bars = []
        NOTE["ohlc_error"] = str(e)[:200]
    NOTE.update({"symbol": SYMBOL, "symbol_code": rec.get("symbol_code", ""), "ohlc": bars,
                 "live_report": rec.get("report", ""),
                 "live_json": {k: rec.get(k) for k in ("entry_price", "target_price", "stop_loss", "risk_return_ratio", "expected_return")}})
    CfoCrew().crew().kickoff(inputs=getData(st))

sys.stderr.write("crew 跑完了却没发出请求 —— 截取失败\n")
sys.exit(3)
