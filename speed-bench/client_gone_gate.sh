#!/bin/bash
# client_gone_gate.sh — 服务端两处协议修复的门(2026-09-22, fable5 同日上午节 S1/S2)。
#
# 跑什么(都要一个已经起着的 ds4-server, 默认 127.0.0.1:8000):
#   G1 解码期客户端挂断  → 服务端 ≤2 秒发现并停生成, 日志出现 "client disconnected after N generated tokens"
#   G2 预填期客户端挂断  → 服务端在块间发现并中止预填, 日志出现 "client disconnected during prefill"
#   G3 思考没闭合的回包  → content 为空 / reasoning_content 非空 / finish_reason = length(官方 deepseek-reasoner 语义)
#   G4 温 0 回归        → 无 system、非思考的那条冒烟请求, 输出与给定的基线 md5 逐字节相同
#
# 为什么要 G1/G2: 非流式请求整个生成期间服务端不写一个字节, 不主动探就发现不了对端已经走了。
# 2026-09-22 早盘实撞: 一条 132k token 的请求被客户端超时重发 4 遍(贪心 ⇒ 四次输出逐字节相同),
# 102 分钟 GPU 全算给已经断开的连接; 而服务串行跑图, 这些僵尸请求把后面排队的也一起拖超时。
# 判据不看 curl 的返回(它已经被我们杀了), 只看服务端日志 —— 客户端视角本来就什么都看不到。
#
# 用法: client_gone_gate.sh [host:port] [服务日志] [G4基线md5]
#   例: bash speed-bench/client_gone_gate.sh 127.0.0.1:8000 ~/ds4-server-1m.log 2e3f...
#   不给基线 md5 就只打印本次的 md5(第一次跑用它来取基线)。
set -uo pipefail
HOSTPORT="${1:-127.0.0.1:8000}"
LOGF="${2:-$HOME/ds4-server-1m.log}"
BASE_MD5="${3:-}"
URL="http://$HOSTPORT/v1/chat/completions"
PASS=0; FAIL=0
say(){ echo "[gate $(date +%H:%M:%S)] $*"; }
ok(){ PASS=$((PASS+1)); echo "  ✓ $1"; }
bad(){ FAIL=$((FAIL+1)); echo "  ✗ $1"; }

# 日志尾部从这一行之后开始看(避免看到上一轮的记录)
log_mark(){ wc -l < "$LOGF" 2>/dev/null || echo 0; }
log_since(){ tail -n +"$(( $1 + 1 ))" "$LOGF" 2>/dev/null; }

# 发一条请求然后在 N 秒后杀掉客户端。python3 只做编排(发字节/关连接), 不参与任何数值。
kill_after(){   # $1 = 秒, $2 = 请求 json 文件
    python3 - "$URL" "$1" "$2" <<'PY'
import json, socket, sys, time
from urllib.parse import urlparse
url, secs, path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
u = urlparse(url); body = open(path, "rb").read()
s = socket.create_connection((u.hostname, u.port or 80), timeout=30)
req = (f"POST {u.path} HTTP/1.1\r\nHost: {u.hostname}\r\nContent-Type: application/json\r\n"
       f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
s.sendall(req + body)
time.sleep(secs)
s.close()   # ★真挂断★: 不读回包直接关, 与客户端超时放弃/进程被杀是同一个形态
print(f"client closed after {secs}s", flush=True)
PY
}

mkprompt(){   # $1 = 重复多少段中文(每段约 40 token), $2 = 输出文件, $3 = max_tokens
    python3 - "$1" "$2" "$3" <<'PY'
import json, sys
n, path, mt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
seg = "第{}段：请逐条阅读下面这段行情记录，并在最后给出一份完整的中文分析报告，要求分点、详实、不少于两千字。\n"
text = "".join(seg.format(i) for i in range(n))
req = {"model": "deepseek-chat", "temperature": 0, "max_tokens": mt,
       "messages": [{"role": "user", "content": text + "\n请开始你的分析。"}]}
open(path, "w").write(json.dumps(req, ensure_ascii=False))
PY
}

say "G1 解码期挂断"
mkprompt 20 /tmp/gate_g1.json 4096
M=$(log_mark); kill_after 12 /tmp/gate_g1.json
sleep 4
G1=$(log_since "$M" | grep -a "client disconnected after" | tail -1)
if [ -n "$G1" ]; then ok "$G1"; else bad "没等到 'client disconnected after'(下面是这段日志)"; log_since "$M" | tail -6; fi
# 掐断之后服务必须立刻能接下一条: 对时间敏感, 所以量它
T0=$(date +%s)
curl -s -m 120 "$URL" -H 'Content-Type: application/json' \
     -d '{"model":"deepseek-chat","temperature":0,"max_tokens":8,"messages":[{"role":"user","content":"说一个字"}]}' >/dev/null
T1=$(date +%s)
if [ $((T1-T0)) -le 60 ]; then ok "掐断后队列立刻可用($((T1-T0)) 秒响应)"; else bad "掐断后下一条等了 $((T1-T0)) 秒(生成没真停?)"; fi

say "G2 预填期挂断"
mkprompt 1200 /tmp/gate_g2.json 256   # 约 5 万 token, 预填几十秒, 够在中途挂断
M=$(log_mark); kill_after 8 /tmp/gate_g2.json
sleep 6
G2=$(log_since "$M" | grep -a "client disconnected during prefill" | tail -1)
if [ -n "$G2" ]; then ok "$G2"; else bad "没等到 'client disconnected during prefill'"; log_since "$M" | tail -6; fi

say "G3 思考没闭合 = 没有正文"
curl -s -m 300 "$URL" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-local","reasoning_effort":"high","max_tokens":64,"messages":[{"role":"user","content":"请从宏观、行业、资金、技术四个角度详细分析当前A股市场，并给出明确结论。"}]}' \
  | python3 -c '
import json, sys
r = json.loads(sys.stdin.read())
m = r["choices"][0]["message"]; fin = r["choices"][0].get("finish_reason")
c = m.get("content") or ""; g = m.get("reasoning_content") or ""
print(f"  finish_reason={fin} content_len={len(c)} reasoning_len={len(g)}")
print("  reasoning 前 80 字:", g[:80].replace("\n", " "))
print("  content 原样:", repr(c[:80]))
sys.exit(0 if (fin == "length" and not c and g) else 1)'
if [ $? -eq 0 ]; then ok "content 空 / reasoning 非空 / finish=length"; else bad "回包不符合官方 deepseek-reasoner 语义"; fi

say "G4 温 0 回归(无 system 非思考)"
OUT=$(curl -s -m 300 "$URL" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"用一句话解释什么是市盈率(PE), 并说明它偏高通常意味着什么。"}]}' \
  | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["choices"][0]["message"]["content"])')
MD5=$(printf '%s' "$OUT" | md5sum | cut -d' ' -f1)
echo "  输出: $OUT"
echo "  md5: $MD5"
if [ -z "$BASE_MD5" ]; then say "(没给基线 md5, 只记录)"; elif [ "$MD5" = "$BASE_MD5" ]; then ok "与基线逐字节相同"; else bad "与基线不同(基线 $BASE_MD5)"; fi

say "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
