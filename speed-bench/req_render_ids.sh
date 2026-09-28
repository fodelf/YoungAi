#!/bin/bash
# req_render_ids.sh — 把截下来的真实请求 body(qtf_capture_request.py 的输出, OpenAI chat 格式)渲染成引擎吃的 token id(2026-09-24)。
#
# 怎么用: speed-bench/req_render_ids.sh <请求 JSON> <输出 ids>
#   例: speed-bench/req_render_ids.sh gguf/v41/night/review0923/req_20260922.json /tmp/req0922.ids
#   产物给 d1_kv_ring_gate.sh 的 sim 档当 <提示 ids>(--gen-ids 续写)。
# 为什么要它: 测速要换一条**没拿来调过参数**的真实请求当验证集(用户 09-24: "按着 cfo 文件生成是不是作弊了")。
#   CFO 那条被拿来定过草稿器子词表的门槛, 在它上面量的接受率偏乐观; 真实请求才是产品口径(铁律: 产品问题只认真实请求)。
# 渲染规则: 与服务端思考档(high)同一串字节 —— BOS <｜System｜> effort 前缀 system <｜User｜> 最后一条 user <｜Assistant｜><think>,
#   与 z_nightly_spark.sh stage_review 同一口径(09-22 修过漏 <｜System｜> 的坑, bug.md §1)。只接 system + user 的单轮请求。
# 出错会怎样: 渲染漏一个特殊 token 不报错, 模型照样续写, 但那已经不是产品那条请求 —— 所以 id 数与 dump 的首尾都打出来看一眼。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
REQ="${1:?用法: $0 <请求 JSON> <输出 ids>}"
OUT="${2:?用法: $0 <请求 JSON> <输出 ids>}"
MODEL=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
[ -f "$REQ" ] || { echo "★没有请求 $REQ★"; exit 1; }
command -v jq >/dev/null || { echo "★没有 jq★"; exit 1; }
EFF=$'Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n'
jq -j --arg eff "$EFF" '"<｜begin▁of▁sentence｜><｜System｜>" + $eff
    + ([.messages[] | select(.role == "system") | .content] | join("\n\n"))
    + "<｜User｜>" + ([.messages[] | select(.role == "user") | .content] | last)
    + "<｜Assistant｜><think>"' "$REQ" > "$OUT.txt" || { echo "★渲染失败★"; exit 2; }
# 先落盘再取第一行(直接 | head 会让 ds4 吃 SIGPIPE, pipefail 下判失败)
./ds4 -m "$MODEL" --dump-tokens --prompt-file "$OUT.txt" > "$OUT.dump" 2>/dev/null || { echo "★分词失败★"; exit 2; }
head -1 "$OUT.dump" | tr -d '[] ' | tr ',' '\n' | grep -v '^$' > "$OUT"
echo "$OUT: $(wc -l < "$OUT") 个 id; 渲染文本 $OUT.txt($(wc -c < "$OUT.txt") 字节); 首 3 个 id: $(head -3 "$OUT" | tr '\n' ' ')末 3 个: $(tail -3 "$OUT" | tr '\n' ' ')"
rm -f "$OUT.dump"
