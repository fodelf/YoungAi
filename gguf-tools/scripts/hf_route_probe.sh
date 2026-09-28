#!/bin/bash
# hf_route_probe.sh — 同一份数据走两条出网路径各传一次到 HF, 比"连接被掐断"的次数和速度(spark 本机跑, 2026-09-25)。
#
# 为什么有这个尺: 113.6 GB 的 GGUF 连传 35 趟都卡在 1%, hf_xet 日志里 10 分钟 346 次
#   "peer closed connection without sending TLS close_notify" —— 连接在传数据途中被中间一环直接掐掉。
#   Mihomo 把 cas-server.xethub.hf.co 交给了兜底规则的代理节点。要判定是不是这个节点的病, 就拿同一份数据
#   换一条路再传一次: 掐断只出现在一条路上 = 病在那条路; 两条路都掐 = 病在更上游(本机/宽带/HF)。
#
# 用法: hf_route_probe.sh <tun|mac> [MB]
#   tun  走 spark 默认路由(Mihomo TUN → 当前节点), 与正式上传同路
#   lfs  同 tun, 但关掉 Xet 走老的 LFS 通道(HF_HUB_DISABLE_XET=1, 数据直传 S3) —— 目的地不同, 用来判"掐断是不是只针对 xethub"
#   mac  走 127.0.0.1:17897(ssh 隧道到 Mac 的代理, 出口不同), hf_xet 按 HTTPS_PROXY 连它
#   MB   随机测试文件大小, 默认 512(≈ 8 个 64 MB 的 xorb, 足够撞上掐断)
#   第三参数 N: 把 hf_xet 的上传并发钉死成 N(HF_XET_FIXED_UPLOAD_CONCURRENCY, 只给 hf 这个第三方工具);
#          不给 = hf_xet 自适应并发。09-25 实测: 单连接 256 MB 2 分多钟不断, 8 路并发 6~17 s 全被掐 ⇒ 病在并发
# 传到私有仓库 wenzhouwu/youngai-route-probe(不存在就建), 测完由人决定删不删; 测试文件放 gguf/hf_upload/probe/。
#   sweep MB 节点1 节点2 ...   逐个把 spark 上 Mihomo 的 Proxy 组切到这些节点各测一次(tun 路), 最后切回原节点
#          (两边机器订阅的是同一家机场, mac 路不是独立对照, 要判"是不是这个节点"只能换节点比)
set -uo pipefail
if [ "${1:-}" = sweep ]; then
    MB="$2"; shift 2
    SOCK=$(ls /tmp/mihomo-party-*.sock | head -1)
    api(){ curl -s --unix-socket "$SOCK" "$@"; }
    orig=$(api http://localhost/proxies/Proxy | python3 -c 'import sys,json; print(json.load(sys.stdin)["now"])')
    restore(){ api -X PUT http://localhost/proxies/Proxy -d "{\"name\":\"$orig\"}" >/dev/null; echo "已切回 $orig"; }
    trap restore EXIT
    for node in "$@"; do
        api -X PUT http://localhost/proxies/Proxy -d "{\"name\":\"$node\"}" >/dev/null
        now=$(api http://localhost/proxies/Proxy | python3 -c 'import sys,json; print(json.load(sys.stdin)["now"])')
        printf '[%s] ' "$now"; "$0" tun "$MB"
    done
    exit 0
fi
ROUTE="${1:?用法: $0 <tun|mac> [MB] [并发]}"; MB="${2:-512}"; CONC="${3:-}"
HF="$HOME/v41env/bin/hf"; REPO="wenzhouwu/youngai-route-probe"
DIR="$HOME/ds4-main/gguf/hf_upload/probe"; mkdir -p "$DIR"
F="$DIR/probe_${ROUTE}_${MB}M.bin"
# 每条路一份新的随机数据: 用同一份的话第二趟会被 Xet 按块去重, 一个字节都不传, 比了等于没比
head -c $((MB * 1024 * 1024)) /dev/urandom >"$F"
case "$ROUTE" in
  tun) PX=(env);;
  lfs) PX=(env HF_HUB_DISABLE_XET=1);;
  mac) PX=(env HTTPS_PROXY=http://127.0.0.1:17897 HTTP_PROXY=http://127.0.0.1:17897);;
  *) echo "路由只认 tun|lfs|mac"; exit 2;;
esac
[ -n "$CONC" ] && PX+=(HF_XET_FIXED_UPLOAD_CONCURRENCY="$CONC")
LOGDIR="$HOME/.cache/huggingface/xet/logs"
before=$(ls "$LOGDIR" 2>/dev/null | wc -l)
t0=$(date +%s)
"${PX[@]}" "$HF" upload "$REPO" "$F" "$(basename "$F")" --repo-type model --private >/tmp/hf_route_probe_$ROUTE.out 2>&1
rc=$?; dt=$(( $(date +%s) - t0 ))
xl=$(ls -t "$LOGDIR"/*.log | head -1)
[ "$(ls "$LOGDIR" | wc -l)" -gt "$before" ] || xl=""
cut=$( [ -n "$xl" ] && grep -a -c "close_notify\|tls handshake eof" "$xl" || echo "?")
echo "路由 $ROUTE | 并发 ${CONC:-自适应} | ${MB} MB | 结果 $([ $rc = 0 ] && echo 成功 || echo "失败 rc=$rc") | 用时 ${dt}s | 平均 $(( MB / (dt > 0 ? dt : 1) )) MB/s | 掐断 $cut 次 | xet 日志 ${xl:-无}"
rm -f "$F"
