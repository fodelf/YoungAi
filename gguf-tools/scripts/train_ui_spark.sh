#!/bin/bash
# train_ui_spark.sh — 起/停工作台主进程 ds4-train(2026-10-10)。
#   用法: train_ui_spark.sh start [端口 8000] | stop | status
#   主进程永远不退: 页面 http://<主机>:<端口>/ 出聊天/模型/训练/语料/记录; 模型(ds4-server)它起在 127.0.0.1:<端口+1> 当子进程并转发 /v1,
#   训练是它自己的作业线程(停模型 → 训 → 门 → 模型回来, 全在 C 里)。stop 只停主进程, 模型子进程留着(下次 start 直接接上; 要停模型用 serve_1m_spark.sh stop)。
#   日志 /tmp/ds4_train_ui.log。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
PORT="${2:-8000}"
case "${1:-status}" in
  start)
    [ -x ./ds4-train ] || { echo "没有 ./ds4-train, 先 make ds4-train"; exit 1; }
    pgrep -x ds4-train >/dev/null && { echo "已在跑: $(pgrep -a -x ds4-train)"; exit 0; }
    nohup ./ds4-train --host 0.0.0.0 --port "$PORT" > /tmp/ds4_train_ui.log 2>&1 </dev/null &
    sleep 1
    pgrep -x ds4-train >/dev/null && echo "起了: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/" || { cat /tmp/ds4_train_ui.log; exit 1; }
    ;;
  stop) pkill -x ds4-train && echo "停了(模型子进程没动)" || echo "没在跑";;
  status) pgrep -a -x ds4-train || echo "没在跑"; curl -s --noproxy '*' "http://127.0.0.1:$PORT/api/train/status" 2>/dev/null; echo;;
  *) echo "用法: $0 start [端口] | stop | status"; exit 1;;
esac
