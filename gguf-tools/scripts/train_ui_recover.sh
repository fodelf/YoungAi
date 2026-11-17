#!/bin/bash
# train_ui_recover.sh — 停/重起工作台主进程 ds4-train 时的安全收口(2026-11-17 用户令: "停 ds4-train 要留脚本再起, 不然编码 agent 断模型").
# 主进程是编码 agent 的模型入口(/v1 由 ds4-train 转给模型子进程 127.0.0.1:<port+1>), 它停 = 模型入口断;
# 模型子进程本身自成会话, ds4-train 被 SIGTERM 时留着(见 src/model/model_child.c), 所以主进程重起直接接上旧模型, 不用再装。
#
# 用法: train_ui_recover.sh start [端口 8888] | stop | restart [端口 8888] | model-restart [端口 8888]
#   start        没有 ds4-train 就 nohup 起(默认 8888, 与当前 spark 工作台一致); 有就不动
#   stop         只 SIGTERM 主进程; 模型子进程留着(要停模型用 serve_1m_spark.sh stop)
#   restart      停主进程后立刻按同一端口起回来 = 换了 ds4-train 二进制时的收口; 模型子进程不动
#   model-restart 换了 ds4-server 二进制后: 等助理/训练闲, 停模型子进程, 再 POST /api/train/serve 按 gguf/serve_pick.txt 装回来(主进程不动)
#
# 先等助理手上没有 running/waiting 的会话、也没在训练才动手: 助理循环与训练线程住主进程里, 一停就把它们正在做的掐断。
# 口令 = ~/.ds4/assistant/token(0600), 日志 /tmp/ds4_train_ui.log。全部行为走 CLI/API, 不读 DS4_* env。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
PORT="${2:-8888}"
LOG="/tmp/ds4_train_ui.log"
TOK=""
tok_read() {
    TOK=$(cat ~/.ds4/assistant/token 2>/dev/null) || true
    [ -n "$TOK" ] || { echo "没有 ~/.ds4/assistant/token"; exit 1; }
}
api() { curl -s --noproxy '*' -m 8 -H "Authorization: Bearer $TOK" "$@" 2>/dev/null; }
wait_idle() {
    tok_read; local s b i
    for i in $(seq 1 100000); do
        s=$(api "http://127.0.0.1:$PORT/api/assistant/sessions" | grep -o '"state":"[a-z_]*"' | grep -vc '"idle"')
        b=$(api "http://127.0.0.1:$PORT/api/train/status" | grep -c '"busy":true')
        [ "${s:-0}" = 0 ] && [ "${b:-0}" = 0 ] && return 0
        [ $((i % 6)) = 1 ] && echo "$(date +%T) 等: 助理 ${s} 个会话在跑/排队, 训练中 ${b}"
        sleep 10
    done
}
start_train() {
    [ -x ./ds4-train ] || { echo "没有 ./ds4-train, 先 make ds4-train"; exit 1; }
    if pgrep -x ds4-train >/dev/null; then echo "ds4-train 已在跑: $(pgrep -a -x ds4-train)"; return 0; fi
    nohup ./ds4-train --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 </dev/null &
    sleep 1
    pgrep -x ds4-train >/dev/null && echo "起了 ds4-train: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/" || { cat "$LOG"; exit 1; }
}
stop_train() {
    pkill -x ds4-train && echo "停了 ds4-train(模型子进程没动)" || echo "ds4-train 没在跑"
}
wait_train_gone() {
    for i in $(seq 1 30); do pgrep -x ds4-train >/dev/null || return 0; sleep 1; done
    echo "★30 s 还停不掉 ds4-train★"; exit 1
}
start_model() {
    tok_read
    local r
    r=$(api -X POST "http://127.0.0.1:$PORT/api/train/serve")
    echo "serve: $r"
    for i in $(seq 1 150); do
        local st
        st=$(api "http://127.0.0.1:$PORT/api/train/status")
        echo "$st" | grep -q '"mode":"serving"' && { echo "装好($((i * 4)) s 内): $(pgrep -a -x ds4-server | cut -c1-80)"; return 0; }
        sleep 4
    done
    echo "★10 分钟没装好: $(api "http://127.0.0.1:$PORT/api/train/status" | cut -c1-300)★"; exit 1
}
stop_model() {
    tok_read
    if pkill -x ds4-server; then
        echo "SIGTERM ds4-server, 等它卸映射…"
        for i in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
        pgrep -x ds4-server >/dev/null && { echo "★ds4-server 2 分钟停不掉★"; exit 1; }
        sleep 3   # 主进程看门狗线程每秒 waitpid 一次, 收到退出才放回 IDLE; 早一步 POST 会被 409
    else echo "ds4-server 没在跑"; fi
}
case "${1:-start}" in
  start)
    start_train
    tok_read
    st=$(api "http://127.0.0.1:$PORT/api/train/status")
    if echo "$st" | grep -q '"mode":"lobby"' && echo "$st" | grep -q '"server":false'; then
        start_model
    fi
    ;;
  stop) stop_train;;
  restart)
    wait_idle; stop_train; wait_train_gone; start_train;;
  model-restart)
    wait_idle; start_train; stop_model; start_model;;
  *) echo "用法: $0 start [端口] | stop | restart [端口] | model-restart [端口]"; exit 1;;
esac
