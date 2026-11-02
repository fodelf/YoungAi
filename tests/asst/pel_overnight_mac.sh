#!/bin/bash
# pel_overnight_mac.sh — 夜跑的 Mac 半边(2026-11-01 深夜): 等 spark 的 entity_overnight_spark.sh 写出 /tmp/bugmd/ent_overnight.done,
#   把选中的 ③ 挂到 spark 的 serve_pick(--posttrain 两行) 装回, 用 dsh 原句画两趟鹈鹕(pel-ent04a/b, 每趟上限 65 分钟), 画完【一定】卸 ③ 装回纯 c5f2。
# 用法: nohup bash tests/asst/pel_overnight_mac.sh [标签=ent04] [③目录] > /tmp/bugmd/pel_overnight.log 2>&1 &   (标签与 spark 侧 entity_overnight_spark.sh 的第一个参数同)
#   第 2 参(11-02): 直接给 spark 上的 ③/vqpatch 目录(仓库相对路径, 如 gguf/v41/posttrain/vstep-v11/ckpt_v07), 不等 done 文件 —— 白天手动点名一个候选画鹈鹕
#   第 3 参(11-02 晚): 画几趟(缺省 2 = a b; 1 = 只画 a)
# 判读(早上): gguf/dsh/runs/local-pel-<标签>{a,b}/ws/index.html(或 /private/tmp/project.*/ 没挪回的), 记分卡 tests/asst/dsh_scorecard.py --plan <run 目录>;
#   spark 的 serve_pick 应回到 2 行(本脚本 trap 保证: 中途被杀也卸)。
# 出错会怎样: done 文件第一行为空 = spark 没训出可挂的 ③, 这里不画直接退; 挂不上(装回失败)也直接卸掉退出, 不让 ③ 留在线上。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
TAG="${1:-ent04}"; DONE=/tmp/bugmd/ent_overnight.done; Q="创建一个HTML，内容是SVG绘制一个鹈鹕骑自行车的2D动画"
MAN='gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-restore_fit_n15360-engine/manifest.txt'
M(){ echo "[pel-night $(date '+%m-%d %H:%M:%S')] $*"; }
unmount(){   # 删 serve_pick 的 --posttrain 两行, 按 serve_pick 装回; 幂等
    ssh spark "cd ~/ds4-main; awk 'skip{skip=0;next} \$0==\"--posttrain\"{skip=1;next} {print}' gguf/serve_pick.txt > /tmp/sp.new && mv /tmp/sp.new gguf/serve_pick.txt; wc -l < gguf/serve_pick.txt; bash gguf-tools/scripts/with_model_stopped_spark.sh true 2>&1 | tail -1; echo '# 夜跑卸 ③ (pel_overnight_mac.sh) $(date +%m-%d\ %H:%M)' >> $MAN"
}
trap 'M "收尾: 卸 ③"; unmount; M "PEL_NIGHT_EXIT"' EXIT
CK="${2:-}"
if [ -z "$CK" ]; then
    M "等 spark 的 $DONE"
    until ssh spark "test -f $DONE" 2>/dev/null; do sleep 120; done
    CK=$(ssh spark "head -1 $DONE"); M "spark 选中 ③: '${CK:-空}'"
else
    ssh spark "test -d ~/ds4-main/$CK" || { M "★spark 上没有 $CK★"; exit 1; }; M "点名 ③: $CK"
fi
[ -n "$CK" ] || { M "没有 ③ 可挂, 不画"; exit 0; }
M "挂 ③ 装回"
ssh spark "cd ~/ds4-main; printf -- '--posttrain\n%s\n' \"\$HOME/ds4-main/$CK\" >> gguf/serve_pick.txt; echo '# 夜跑挂 ③ $CK 画鹈鹕 (pel_overnight_mac.sh) $(date +%m-%d\ %H:%M)' >> $MAN; bash gguf-tools/scripts/with_model_stopped_spark.sh true 2>&1 | tail -1"
for i in $(seq 40); do s=$(ssh spark 'curl -s --noproxy "*" -m 5 http://127.0.0.1:8001/metrics | grep -o "\"state\":\"idle\""'); [ -n "$s" ] && break; sleep 15; done
[ -n "${s:-}" ] || { M "★模型没就绪, 不画★"; exit 1; }
ARMS="a b"; [ "${3:-2}" = 1 ] && ARMS="a"
for ab in $ARMS; do
    R="pel-$TAG$ab"; M "dsh 原句 $R(上限 65 分钟)"
    # macOS 没有 timeout/gtimeout: 自己盯 65 分钟, 到点 TERM 掉 dsh_compare 和它起的 node(dsh 自审环不收手时是常态, 11-01 定位趟 1 跑了 6413 s)
    tests/asst/dsh_compare.sh local "$Q" "$R" & pid=$!
    for ((t = 0; t < 3900; t += 15)); do kill -0 "$pid" 2>/dev/null || break; sleep 15; done
    if kill -0 "$pid" 2>/dev/null; then M "$R 到 65 分钟上限, TERM"; pkill -TERM -P "$pid"; kill -TERM "$pid"; sleep 20; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; fi
    wait "$pid" 2>/dev/null; M "$R rc=$? 工具 $(grep -c '"tool_call"' "gguf/dsh/runs/local-$R/events.jsonl" 2>/dev/null)"
done
