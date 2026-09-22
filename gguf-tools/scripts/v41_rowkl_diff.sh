#!/bin/bash
# v41_rowkl_diff.sh — 两个学生态的【逐 token KL 之差】交叉表(2026-09-13, plan.md 针 1)。
#
# 【干什么】anchor_metrics 的 --row-out 会把每个位置的 KL/Σmin/是否同 top 落一行。本脚本拿
# 两份这样的文件(基线态 / 待判态)逐行相减, 按四个切面分组报均值:
#   ①全段 ②窗内位置(窗首/窗尾 —— 窗与窗之间是上下文断点, 窗首=冷启动)
#   ③子域(按 .layout) ④按基线 KL 的分位(模型本来就懵的位置 vs 本来很确定的位置)
#
# 【读它干什么】判"伤害是均匀相干的, 还是集中在少数位置":
#   均匀上移(各切面 ΔKL 同号同量级, 变差行占比 ~50%以上且分布平) ⇒ 注进去的是确定性偏差,
#     每个 token 都吃一点 —— 这是"修正是 x 的确定函数, 跨层相干叠加"的形态。
#   集中(某个切面数量级更大) ⇒ 是分布外行为(冷启动/某子域/某类 token), 机理要另写。
#
# 用法: v41_rowkl_diff.sh <基线 row-out> <待判 row-out> [layout 文件]
#   产 row-out: anchor_metrics --ref-raw <锚> --ids <ids> --student <stu> --row-out <文件>
set -uo pipefail
BASE="${1:?基线 row-out 文件}"; CAND="${2:?待判 row-out 文件}"; LAYOUT="${3:-}"
[ -s "$BASE" ] && [ -s "$CAND" ] || { echo "★row-out 文件缺或空★"; exit 2; }

# layout: 前三行是注释/win, 其后每行 "<域名> <起始行> <行数>"。缺省则只报不分域的三个切面。
DOMS=""
if [ -n "$LAYOUT" ] && [ -s "$LAYOUT" ]; then
    DOMS="$(awk 'NF==3 && $2 ~ /^[0-9]+$/ {printf "%s:%s:%s ", $1, $2, $3}' "$LAYOUT")"
fi

paste "$BASE" "$CAND" | awk -v doms="$DOMS" '
BEGIN { nd = split(doms, D, " ") }
{
    r = $1; kb = $2; ka = $6; d = ka - kb;
    n++; s += d; if (d > 0) pos++;
    wp = r % 128;                                  # 窗宽 128 = 各 ids 的 win(layout 第二行)
    if (wp < 8)   { hb += d; hn++ }
    if (wp >= 120){ tl += d; tn++ }
    for (i = 1; i <= nd; i++) {
        split(D[i], a, ":");
        if (r >= a[2] && r < a[2] + a[3]) { ds[a[1]] += d; dn[a[1]]++ }
    }
    # 按基线 KL 分位: 模型本来就懵的行(KL 大)与本来很确定的行(KL 小)分开看
    if      (kb < 0.1) { q1 += d; q1n++ }
    else if (kb < 0.4) { q2 += d; q2n++ }
    else if (kb < 1.2) { q3 += d; q3n++ }
    else               { q4 += d; q4n++ }
}
END {
    printf "全段 ΔKL 均值 %+.4f   变差行占比 %.1f%%   (n=%d)\n", s/n, pos/n*100, n;
    if (hn) printf "窗首 8 行 %+.4f (n=%d)", hb/hn, hn;
    if (tn) printf "   窗尾 8 行 %+.4f (n=%d)", tl/tn, tn;
    if (hn || tn) printf "\n";
    for (k in ds) printf "域 %-14s %+.4f (n=%d)\n", k, ds[k]/dn[k], dn[k];
    printf "按基线 KL 分位: <0.1 %+.4f(n=%d) | 0.1-0.4 %+.4f(n=%d) | 0.4-1.2 %+.4f(n=%d) | >1.2 %+.4f(n=%d)\n",
           q1n?q1/q1n:0, q1n, q2n?q2/q2n:0, q2n, q3n?q3/q3n:0, q3n, q4n?q4/q4n:0, q4n;
}'
