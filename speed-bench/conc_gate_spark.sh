#!/bin/bash
# conc_gate_spark.sh — V4.1 并发调度器的三道门(2026-09-30, batch.md §4; 在 spark 本机跑)。
#
# ① 基线: 不带 --batch 起服(单 worker 老路), 3 条**各不相同**的 ~12k 提示一条一条发 → 每路答案落 /tmp/ctx1m_ans_seq_<i>.txt;
# ② 并发: --batch 3 起服, 同样 3 条提示同时发 → /tmp/ctx1m_ans_conc_<i>.txt, 逐路 cmp ① —— 温 0 下必须逐字节同(抓状态串台);
# ③ 速度: --batch 3 下 3 条 ~106k 提示同时发, 每路解码 64 token, 看每路壁钟与服务日志里的 decoding t/s(与 09-30 午后老路的 125/250/375 s 阶梯比)。
# 起服/停服走 serve_1m_spark.sh(自带起跑清单 / 冒烟 / 看门狗); 请求走 ctx_1m_memory_gate.sh 的并发档。
# 出错会怎样: 哪一步起服失败就停在哪一步(日志尾贴出来), 不带着坏服务往下跑; cmp 不同会打 ★, 但仍跑完 ③(速度数照量, 门另判)。
#
# 第 5 个参数 = 让模型列出前几条(默认 20 ⇒ 42 个 token 就 EOS, 只够量逐字节门); 量"三路同时解码"的吞吐要开大(400 ⇒ 1000 多 token)并把
# 解码上限抬到 1024。第 3 个参数给 0 = 不跑 ③(长提示测速)。
# 用法: conc_gate_spark.sh [并发数=3] [短提示 token 数=15000] [长提示 token 数=130000] [解码 token=64] [列出条数=20]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
N="${1:-3}"; SHORT="${2:-15000}"; LONG="${3:-130000}"; NGEN="${4:-64}"; NLIST="${5:-20}"
S=gguf-tools/scripts/serve_1m_spark.sh; G=speed-bench/ctx_1m_memory_gate.sh; LOG="$HOME/ds4-server-1m.log"
LOGP(){ echo "[conc_gate $(date +%H:%M:%S)] $*"; }

LOGP "① 老路起服(不带 --batch)"
$S start > /tmp/conc_serve_old.log 2>&1
grep -q SERVE1M_UP /tmp/conc_serve_old.log || { LOGP "★老路起服失败★"; tail -8 /tmp/conc_serve_old.log | cut -c1-200; exit 1; }
LOGP "① 基线: $N 路 ~${SHORT} token 串行"
$G 127.0.0.1:8000 "$SHORT" 2500 "$N" "$NGEN" "$LOG" seq "$NLIST" > /tmp/conc_gate_seq.log 2>&1
grep -a "壁钟\|prompt_tokens\|回答\|decoding chunk\|\[v41\] decode\|gen=.*finish\|MemAvailable" /tmp/conc_gate_seq.log | cut -c1-200
$S stop > /dev/null 2>&1; sleep 3

LOGP "② --batch $N 起服"
$S start "" "" --batch "$N" > /tmp/conc_serve_b.log 2>&1
grep -q SERVE1M_UP /tmp/conc_serve_b.log || { LOGP "★--batch $N 起服失败★"; tail -12 /tmp/conc_serve_b.log | cut -c1-200; tail -20 "$LOG" | cut -c1-200; exit 1; }
LOGP "② 并发: 同样 $N 路同时发"
$G 127.0.0.1:8000 "$SHORT" 2500 "$N" "$NGEN" "$LOG" conc "$NLIST" > /tmp/conc_gate_conc.log 2>&1
grep -a "壁钟\|prompt_tokens\|回答\|prompt done\|decoding chunk\|gen=.*finish\|MemAvailable\|并发调度器\|装不下\|失败" /tmp/conc_gate_conc.log | cut -c1-200
GATE=0
for i in $(seq 1 "$N"); do
  if cmp -s "/tmp/ctx1m_ans_seq_$i.txt" "/tmp/ctx1m_ans_conc_$i.txt"; then echo "  路 $i: 并发 == 串行 逐字节同 ✓ ($(wc -c < /tmp/ctx1m_ans_conc_$i.txt) 字节)"
  else echo "  ★路 $i: 并发 ≠ 串行★"; GATE=1; fi
done
pgrep -x ds4-server >/dev/null || { LOGP "★并发服务在 ② 里死了★"; tail -20 "$LOG" | cut -c1-200; exit 1; }

if [ "$LONG" -gt 0 ]; then
  LOGP "③ 速度: $N 路 ~${LONG} token 同时发(--batch $N)"
  $G 127.0.0.1:8000 "$LONG" 2500 "$N" "$NGEN" "$LOG" conc "$NLIST" > /tmp/conc_gate_long.log 2>&1
  grep -a "壁钟\|prompt_tokens\|回答\|prompt done\|decoding chunk\|gen=.*finish\|MemAvailable\|装不下\|失败" /tmp/conc_gate_long.log | cut -c1-200
fi
$S stop > /dev/null 2>&1
LOGP "逐字节门: $([ $GATE = 0 ] && echo 过 || echo ★不过★); 日志 /tmp/conc_gate_{seq,conc,long}.log, 服务日志 $LOG"
echo CONC_GATE_DONE
exit $GATE
