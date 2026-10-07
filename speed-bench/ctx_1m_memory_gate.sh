#!/bin/bash
# ctx_1m_memory_gate.sh — 1M 上下文档的内存门(2026-09-22, 用户令"上下文设置为 1m") + 并发档(2026-09-30)。
#
# 为什么要这把尺: V4.1 的请求状态是**按 ctx 分配的, 与提示多长无关**(core_v41.h 的账:
# iscore[512][ctx] + cand[512][ctx] + 候选暂存 ⇒ 524288 约 2.4 GiB, 1048576 约 4.8 GiB)。
# 所以把上限从 500k 抬到 1M, 每条请求都多吃 2.4 GiB —— 121 GB 的机器上装着 113.6 GB 的模型,
# 这 2.4 GiB 是从 MemAvailable 里出的。不量就上线 = 赌看门狗不会在早盘把服务杀掉。
#
# 跑什么: 发一条真实规模的长提示(默认 13 万 token, 与产品最长的那条选股请求同量级), 解码只要几个 token,
# 期间每 10 秒记一次 MemAvailable, 最后报最低点与看门狗红线的距离。
#
# ★并发档(2026-09-30, 用户问"1M 上下文三并发的 token 输出速度")★: 第 4 个参数 > 1 时同时发 N 路, 每路记自己的壁钟, 跑完把服务日志里
# 这段时间的 "prompt done / [v41] decode … t/s / gen= finish" 原样贴出来。N 路的提示**各不相同**(第 i 路多 (i−1)×17 条消息, 答案里的
# 条数就不同) —— 这样"每路答案 == 该提示单跑"才能抓状态串台(KV 复用串台 09-08 的教训)。每路答案落 /tmp/ctx1m_ans_<模式>_<i>.txt。
# 第 7 个参数 seq = 同样的 N 路一条一条发(单请求路的基准); conc(默认) = 同时发。两次跑完 diff 答案文件就是逐字节门。
# 没有并发调度器(不带 --batch)时 conc 的壁钟应当是 T / 2T / 3T 的阶梯; 带 --batch N 时三路应当同时推进。
#
# 第 8 个参数 = 让模型列出前几条的编号(默认 20): 要量"三路同时解码"的吞吐就开大(400 ≈ 1000 多个 token 的输出), 否则 42 个 token 就 EOS,
# 三路根本没有同时在解码的时候(09-30 实撞: 短输出下并发与串行壁钟一样, 全是预填在排队)。
# 用法: ctx_1m_memory_gate.sh [host:port] [提示 token 数] [红线 MB] [并发数=1] [解码 token 数=8] [服务日志=~/ds4-server-1m.log] [conc|seq] [列出条数=20]
set -uo pipefail
HOSTPORT="${1:-127.0.0.1:8000}"
NTOK="${2:-130000}"
RED_MB="${3:-2500}"
NCONC="${4:-1}"
NGEN="${5:-8}"
SRVLOG="${6:-$HOME/ds4-server-1m.log}"
MODE="${7:-conc}"
NLIST="${8:-20}"

python3 - "$NTOK" "$NGEN" "$NCONC" "$NLIST" <<'PY'
import json, sys
n, ngen, nconc, nlist = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# 每段约 40 token 的中文, 拼到目标 token 数(按 0.9 token/字 估, 宁可多不可少); 第 i 路多 (i-1)*17 条, 答案(条数)各不相同
seg = "第{}条：市场消息，某公司发布公告称将调整生产计划并披露最新经营数据，请阅读后备用。\n"
for i in range(1, nconc + 1):
    cnt = n // 35 + (i - 1) * 17
    text = "".join(seg.format(k) for k in range(cnt))
    req = {"model": "deepseek-chat", "temperature": 0, "max_tokens": ngen,
           "messages": [{"role": "user", "content": text + f"\n上面一共有多少条消息？只回答数字，然后逐条列出前{nlist}条的编号。"}]}
    open(f"/tmp/ctx1m_req_{i}.json", "w").write(json.dumps(req, ensure_ascii=False))
    print(f"[gate] 第 {i} 路提示 {cnt} 条约 {len(text)} 字, 解码上限 {ngen} token")
PY

( while :; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$A" >> /tmp/ctx1m_mem.txt
    sleep 10
  done ) &
SAMPLER=$!
: > /tmp/ctx1m_mem.txt
LOG0=$(wc -l < "$SRVLOG" 2>/dev/null || echo 0)
echo "[gate $(date +%H:%M:%S)] $MODE: $NCONC 路请求(每路解码 $NGEN 个 token)"
T0=$(date +%s)
one_route() {   # $1 = 路号: 发请求, 记壁钟
  local i="$1" t0 t1
  t0=$(date +%s.%N)
  curl -s -m 12000 "http://$HOSTPORT/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @"/tmp/ctx1m_req_$i.json" > "/tmp/ctx1m_out_$i.json"
  t1=$(date +%s.%N)
  echo "$t0 $t1" > "/tmp/ctx1m_t_$i.txt"
}
if [ "$MODE" = seq ]; then
  for i in $(seq 1 "$NCONC"); do one_route "$i"; done
else
  PIDS=()
  for i in $(seq 1 "$NCONC"); do one_route "$i" & PIDS+=($!); done
  for p in "${PIDS[@]}"; do wait "$p"; done
fi
T1=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "[gate] 全部返回, 总用时 $((T1-T0)) 秒"
RC=0
for i in $(seq 1 "$NCONC"); do
  read -r t0 t1 < "/tmp/ctx1m_t_$i.txt"
  printf "── 第 %d 路: 壁钟 %.1f s\n" "$i" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
  python3 -c '
import json, sys
r = json.loads(open(sys.argv[1]).read())
u = r.get("usage", {})
c = r["choices"][0]["message"].get("content") or ""
open(sys.argv[2], "w").write(c)
print("  prompt_tokens =", u.get("prompt_tokens"), "completion_tokens =", u.get("completion_tokens"),
      "finish =", r["choices"][0].get("finish_reason"))
print("  回答:", repr(c[:120]))' "/tmp/ctx1m_out_$i.json" "/tmp/ctx1m_ans_${MODE}_$i.txt" \
    || { echo "  ★服务没给正常 JSON★: $(head -c 300 "/tmp/ctx1m_out_$i.json")"; RC=1; }
done
if [ -f "$SRVLOG" ]; then
  echo "── 服务日志(这段时间的预填/解码/收尾行, 原样):"
  tail -n +"$((LOG0+1))" "$SRVLOG" | grep -a "prompt start\|prompt done\|\[v41\] decode\|decoding chunk\|gen=.*finish\|watchdog\|MemAvailable\|装不下\|失败\|并发调度器" | cut -c1-220
fi
LOW=$(sort -n /tmp/ctx1m_mem.txt | head -1)
echo "[gate] MemAvailable 最低 ${LOW} MB (采样 $(wc -l < /tmp/ctx1m_mem.txt) 次, 看门狗红线 ${RED_MB} MB)"
if [ "$LOW" -le "$RED_MB" ]; then echo "  ✗ 摸到红线, 这一档在这台机器上不安全"; exit 1; fi
echo "  ✓ 距红线还有 $((LOW-RED_MB)) MB"
exit $RC
