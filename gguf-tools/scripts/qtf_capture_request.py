"""qtf_capture_request.py — 把 qtf 某一天的大盘请求按当天口径重建出来, 截下 crewAI 发给模型的 HTTP 原字节。

用法(Mac, qtf 后端容器里跑; 只读, 不写库、不改 qtf 任何文件):
    docker exec -i quant_trading_flow-main-backend-1 /app/.venv/bin/python - \
        <日期 YYYYMMDD> "<end_date 原值, 如 2026-09-23 04:43:11>" < gguf-tools/scripts/qtf_capture_request.py > req.json

为什么要它: 后训练 ③ 的料必须是产品真实请求(铁律: 产品问题只认真实请求), 而当天服务端没开 --trace,
原字节没留。能重建的原因: 大盘请求只由三样东西拼成 —— end_date、指数行情、事件报告。
  - 事件报告: 当天存进 news_analysis.event_report 的原文(逐字)。
  - A 股指数: 取 40 根日线, 只留 ≤ 前一交易日的最后 5 根(不截 = 把复盘日的行情漏进请求 = 泄题)。
  - 美股: 当时走新浪实时快照; 事后取新浪日线历史(< 复盘日最后一根收盘 + 按前收算的两位涨跌幅),
    数值同源, 但快照的点数小数位(实时是 4 位)与这里(日线 2 位)写法不同。
    这是唯一已知的非逐字处, 记在输出的 _note 里。
  - end_date: 服务里是 datetime.now(), 当天报告第一行"报告日期"就是它, 由调用方照抄传入。
截取方式: 把 crewAI 的 LLM base_url 指到本进程里的一个假服务, 收到第一条 POST 就把 body 原样写到 stdout 退出。
所以拿到的是 litellm → openai SDK 真正发出去的字节(含 stop 词、reasoning_effort、模型名), 不是手拼的。
"""
import datetime as dt
import http.server
import json
import os
import sys
import threading

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

DATE = sys.argv[1]                 # 复盘日, 如 20260923(= 这条请求要预测的那一天)
END_DATE = sys.argv[2]             # 当天请求里的 end_date 原值
CUT = dt.date(int(DATE[:4]), int(DATE[4:6]), int(DATE[6:]))   # 行情只许 < 这一天

from pymongo import MongoClient  # noqa: E402

db = MongoClient(os.environ.get("MONGODB_URI", "mongodb://mongodb:27017"))["quant_trading"]
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
    import re
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


class _Grab(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        req = json.loads(body)
        req["_note"] = {"date": DATE, "end_date": END_DATE, "path": self.path,
                        "us_source": "新浪日线历史(当天是新浪实时快照, 点数小数位不同)",
                        "market_data": market_data}
        os.write(OUT_FD, json.dumps(req, ensure_ascii=False).encode("utf-8"))
        os._exit(0)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), _Grab)
threading.Thread(target=srv.serve_forever, daemon=True).start()

from quant_trading_flow.modules import deepseek  # noqa: E402

deepseek.deepseek_llm.base_url = f"http://127.0.0.1:{srv.server_port}/v1"
from quant_trading_flow.crews.market.market import MarketCrew  # noqa: E402

OUT_FD = os.dup(1)                 # JSON 只走这个 fd; fd 1 整个改指 stderr, crewAI 的彩色日志混不进来
os.dup2(2, 1)
MarketCrew().crew().kickoff(inputs={"end_date": END_DATE, "market_data": market_data,
                                    "news": na["event_report"]})
sys.stderr.write("crew 跑完了却没发出请求 —— 截取失败\n")
sys.exit(3)
