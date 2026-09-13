#!/bin/bash
# kv_stale_tail_probe.sh — 定罪探针(2026-09-08): 新请求比活缓存短、又共享一段前缀时,
# 超出新长度的【陈旧 KV 行】是否还被注意到 ⇒ 模型照抄上一条请求的内容。
#
# 实撞现场: 决策重跑第二只 603978, 提示词确认是它自己的材料(14600 token), 服务端日志
#   `live kv cache miss live=22449 prompt=14600 common=521` → `ctx=521..14600:14079`,
#   输出却是上一条 002278 的完整报告(连价格都是 002278 的原值)。
#   第一只反而正确: 它的日志是 `ctx=0..13216`(零前缀复用), 且提示词比当时的活缓存长。
# ⇒ 假设: 活缓存回退到公共前缀后, [新长度, 旧长度) 这段陈旧行没作废, 生成时仍被注意到。
#
# 探针设计(两条请求, 分钟级):
#   A 长提示(约 2000 token) + 生成 300 ⇒ 活缓存 ≈ 2300 行, 内容带独一无二的暗号。
#   B 短提示(约 600 token), 与 A 共享同一段开头(公共前缀非零), 问一件完全无关的事。
#   判据: B 的回答里出现 A 的暗号 = 陈旧尾巴泄漏(定罪); 不出现 = 假设不成立, 另找。
#   对照 C: 重复 B 一次(此时活缓存 = B 自己, 无更长尾巴) —— C 干净而 B 脏 = 只与尾巴有关。
#
# 用法: kv_stale_tail_probe.sh [host=127.0.0.1] [port=8000]
set -uo pipefail
H="${1:-127.0.0.1}"; P="${2:-8000}"
URL="http://$H:$P/v1/chat/completions"
MARK="紫貂七号"          # 暗号: 正常语料里不会出现, 出现即来自 A
# ★公共前缀必须超过一个复用块★(2026-09-08 第一版探针没复现的原因): 引擎按 ~512 token 的块
# 粒度复用活缓存(实撞日志 common=512/521 就是一个块), 前缀只有 20 token 时 cached_tokens=0,
# 根本没进复用路径 —— 条件没搭上, 自然测不出东西。这里把共同开头撑到 700+ token。
SHARED=$(python3 - <<'PY'
print("请阅读下面的分析框架后再回答问题。" + "分析框架要求：先看趋势与波动，再看关键位置与催化剂，最后评估风险与仓位；每一步都要给出可核对的依据，不得凭印象下结论。" * 30)
PY
)

# A 的资料段: 撑到约 2000 token 的重复段落 + 暗号
A_BODY=$(python3 - "$MARK" <<'PY'
import sys
mark = sys.argv[1]
para = "本机构对该标的的跟踪记录显示，其经营节奏平稳，产能利用率维持在正常区间，渠道库存未见异常堆积，管理层未有变动。"
print("项目代号：%s。以下为跟踪记录正文。" % mark + para * 60)
PY
)
B_BODY="请解释什么是市盈率，并说明它偏高通常意味着什么。"

ask(){   # $1=正文 $2=生成上限; 温 0, 关思考(deepseek-chat)
    python3 - "$URL" "$SHARED$1" "$2" <<'PY'
import json, sys, urllib.request
url, content, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({"model": "deepseek-chat", "temperature": 0, "max_tokens": n,
                   "messages": [{"role": "user", "content": content}]}).encode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
print(r["choices"][0]["message"].get("content", "").replace("\n", " "))
print("USAGE", json.dumps(r.get("usage", {}), ensure_ascii=False))
PY
}

echo "== A(长提示 + 长生成, 建立长活缓存; 内容带暗号 $MARK) =="
A_OUT=$(ask "$A_BODY 请复述项目代号, 并用不少于 200 字复述跟踪记录要点。" 400)
echo "$A_OUT" | head -2 | cut -c1-200

echo
echo "== B(短提示, 与 A 共享开头, 问无关问题) =="
B_OUT=$(ask "$B_BODY" 120)
echo "$B_OUT" | head -2 | cut -c1-300

echo
echo "== C(再问一次 B: 此时活缓存就是 B 自己, 没有更长的尾巴) =="
C_OUT=$(ask "$B_BODY" 120)
echo "$C_OUT" | head -2 | cut -c1-300

echo
b=$(echo "$B_OUT" | grep -c "$MARK"); c=$(echo "$C_OUT" | grep -c "$MARK")
echo "暗号出现次数: B=$b C=$c"
if [ "$b" -gt 0 ] && [ "$c" -eq 0 ]; then
    echo "★定罪: 陈旧尾巴泄漏(B 脏 C 净)★"
elif [ "$b" -gt 0 ]; then
    echo "★B 与 C 都脏 —— 不是尾巴, 是更前面的东西(前缀复用本身)★"
else
    echo "★暗号未泄漏 —— 本探针没复现, 换更贴近实撞的规模(A 提示 13k / 生成 9k, B 提示 14.6k)★"
fi
