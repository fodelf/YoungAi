#!/bin/bash
# kd_domain/finance.sh —— 金融(个股复盘)领域适配器: 校准回路里随领域变的三件 —— 解析器 / 规则与料 / 打分。契约与为什么见 _template.sh。
# 10-04 从 z_nightly_spark.sh 的 kd_ledger + kd_decide_score 原样拆出(用户 "我不是说了泛化的吗"); 拆后产物与拆前逐字节相同(用第 0003 次的产物对拍过)。
#
# 这个领域: 一笔决策 = 复盘头部一行 "【复盘 <复盘日> <代码> <名称>】决策日 <日期> 给出: 入场价 E, 目标价 T, 止损位 S。次日实际: 收盘 C, 最高 H, 最低 L"(+ 信号标签);
# 规则 = 目标 = 入场 × (1 + 历史"入场到次日最高"涨幅中位), 止损 = 入场 × (1 + 历史"入场到次日最低"跌幅中位); 打分 = |目标−最高|/最高、目标高出最高多少、
# 目标达成(目标 ≤ 最高)、|止损−最低|/最低、止损会被触发(止损 ≥ 最低)。
# 账本 tsv 列: 复盘日 代码 名称 决策日 E T S C H L 信号标签 整篇路径 序号(最后一列 = 序号是契约)。stats 一行: n mr ml mc hit shit gap pt。

# finance_parse <docs目录> <序号> <账本tsv>: docs/stock_*.txt 头部一行 → 逐笔。短版复盘(只有 错误类型/预测目标价/实际收盘/教训, 没有 入场/止损/次日最高最低,
#   如 002490、601975 那种 1000 字的)没有头部行, 进不了账本: 记数, 不停车。
#   ★必须 UTF-8 locale★: C locale 下 [^】] 是"不含这三个字节"的字节类, 名字里带 0x91 字节的(湖南黄"金")在这儿就断了 —— 10-03 实撞: 002155 "没有复盘头部行"
finance_parse(){
    local DOCS="$1" N="$2" TSV="$3" f SKIP=0 NREC=0
    ls "$DOCS"/stock_*.txt >/dev/null 2>&1 || { echo "没有 stock_*.txt 整篇复盘"; return 0; }
    for f in "$DOCS"/stock_*.txt; do
        LC_ALL=C.UTF-8 gawk -v F="$f" -v N="$N" 'match($0, /【复盘 ([0-9]+) ([0-9]+) ([^】]+)】决策日 ([0-9]+) 给出: 入场价 ([0-9.]+), 目标价 ([0-9.]+), 止损位 ([0-9.]+)。次日实际: 收盘 ([0-9.]+), 最高 ([0-9.]+), 最低 ([0-9.]+)/, a) && !hd { hd = 1; for (i = 1; i <= 10; i++) h[i] = a[i] }
             match($0, /信号标签：([^。]*)。/, t) && !tg { tg = t[1]; gsub(/、/, ",", tg) }
             END { if (!hd) exit 1
                   printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], (tg == "" ? "-" : tg), F, N }' "$f" >> "$TSV" && NREC=$((NREC + 1)) || SKIP=$((SKIP + 1))
    done
    echo "$NREC 篇进账$([ "$SKIP" = 0 ] || echo ", $SKIP 篇没有逐笔头部行(短版复盘, 无入场/止损/次日最高最低)不进账本")"
}

# finance_material <累计tsv> <序号> <stats文件> <材料块> <训练问答> <留出问答> <决策提示目录>: 累计统计 + 规则 + 材料块 + 问答 + 决策提示, 一个 gawk 全出
#   (数字只算一遍, 块/问答/真值用同一份)。文字里的"截至"用账本里最新的复盘日(LBL), 不用序号: 模型记日期比记第几次自然。
finance_material(){
    local ALL="$1" N="$2" SF="$3" CF="$4" TF="$5" EF="$6" DEC="$7" LBL
    LBL="$(cut -f1 "$ALL" | sort | tail -1)"; LBL="复盘日 ${LBL:-$N}"
    LC_ALL=C.UTF-8 gawk -v N="$N" -v LBL="$LBL" -v SF="$SF" -v CF="$CF" -v TF="$TF" -v EF="$EF" -v DEC="$DEC" -F'\t' '
        function pct(x) { return sprintf("%+.1f%%", x) }
        function med(arr, n,   b, i) { for (i = 1; i <= n; i++) b[i] = arr[i]; asort(b); return n % 2 ? b[(n + 1) / 2] : (b[n / 2] + b[n / 2 + 1]) / 2 }
        function qa(file, q, a) { printf "#Q\n%s\n#A\n%s\n", q, a >> file }
        # 决策日材料节选: 整篇里 "四、决策当日的输入材料" 之后三段各取开头几行(技术 14 行 / 策略 10 行 / 事件 8 行), 不含一~三(结果与教训)与五(修正后报告)
        function excerpt(doc,   ln, sec, out, n1, n2, n3, got) { out = ""; sec = ""; got = 0
            while ((getline ln < doc) > 0) {
                if (ln ~ /^四、决策当日的输入材料/) { got = 1; continue }
                if (!got) continue
                if (ln ~ /^五、/ || ln ~ /^## 决策报告/) break
                if (ln ~ /^## 技术数据报告/) { sec = "t"; n1 = 0; out = out "\n【技术数据(节选)】\n"; continue }
                if (ln ~ /^## 策略报告/) { sec = "s"; n2 = 0; out = out "\n【策略报告(节选)】\n"; continue }
                if (ln ~ /^## 事件/) { sec = "e"; n3 = 0; out = out "\n【事件(节选)】\n"; continue }
                if (ln ~ /^[ \t]*$/ || ln ~ /^---/ || ln ~ /^[ \t]*\|/) continue   # 跳过 markdown 表格行: 20 日明细表会把额度吃光, 决策要的是总览/极端波动/关键价位/事件
                if (sec == "t" && n1++ < 14) out = out ln "\n"
                else if (sec == "s" && n2++ < 10) out = out ln "\n"
                else if (sec == "e" && n3++ < 8) out = out ln "\n"
            }
            close(doc); return substr(out, 1, 750) }   # 学生序列上限 1024 行: 节选 750 字 ≈ 500~650 token, 加问题与答案 ~150 留有余量
        { n++; rd[n] = $1; code[n] = $2; name[n] = $3; dd[n] = $4; E[n] = $5 + 0; T[n] = $6 + 0; S[n] = $7 + 0; C[n] = $8 + 0; H[n] = $9 + 0; L[n] = $10 + 0; tags[n] = $11; doc[n] = $12
          rh[n] = (H[n] - E[n]) / E[n] * 100; rl[n] = (L[n] - E[n]) / E[n] * 100; rc[n] = (C[n] - E[n]) / E[n] * 100
          pt[n] = (T[n] - E[n]) / E[n] * 100; gap[n] = (T[n] - H[n]) / H[n] * 100; hit[n] = H[n] >= T[n]; shit[n] = L[n] <= S[n]
          if ($13 == N) today[++nt] = n }
        END {
          if (!n) exit 1
          for (i = 1; i <= n; i++) { nh += hit[i]; ns += shit[i]; sg += gap[i]; sp += pt[i]; srh += rh[i]; nd[rd[i]] = 1
              m = split(tags[i], tt, ","); for (k = 1; k <= m; k++) if (tt[k] != "-") { tn[tt[k]]++; th[tt[k]] += hit[i]; trh[tt[k]] = trh[tt[k]] " " rh[i] } }
          mr = med(rh, n); ml = med(rl, n); mc = med(rc, n)
          for (i = 1; i <= n; i++) if (rh[i] >= mr) nhr++   # 按规则(目标 = 入场 × (1 + mr%))历史上会达成几笔
          sf = SF; printf "n %d mr %.2f ml %.2f mc %.2f hit %d shit %d gap %.2f pt %.2f\n", n, mr, ml, mc, nh, ns, sg / n, sp / n > sf; close(sf)
          # ---- 材料块 ----
          cf = CF; printf "" > cf
          printf "【决策账本】截至%s 的交易决策记录与校准规则（每笔数字由代码从当日复盘头部核定，可复算）\n一、本次（第 %s 次增量）新增的逐笔决策（决策日 → 次日实际）\n", LBL, N >> cf
          for (j = 1; j <= nt; j++) { i = today[j]
              printf "%d. %s %s：决策日 %s 给出 入场价 %.2f、目标价 %.2f、止损位 %.2f；次日实际 收盘 %.2f、最高 %.2f、最低 %.2f。目标%s（目标价%s次日最高价 %s）；止损%s触发。入场到次日最高 %s，到次日最低 %s，到收盘 %s。信号标签：%s。\n", \
                  j, code[i], name[i], dd[i], E[i], T[i], S[i], C[i], H[i], L[i], (hit[i] ? "达成" : "未达成"), (gap[i] > 0 ? "高出" : "低于"), sprintf("%.1f%%", gap[i] < 0 ? -gap[i] : gap[i]), (shit[i] ? "被" : "未"), pct(rh[i]), pct(rl[i]), pct(rc[i]), tags[i] >> cf }   # printf 参数里的 > 要括起来, 不然 gawk 当成重定向
          printf "二、累计统计（共 %d 笔，覆盖 %d 个复盘日）：目标达成 %d/%d = %.0f%%；目标价平均高出次日最高价 %.1f%%；计划收益率均值 %s，而入场到次日最高的涨幅中位数只有 %s（到收盘中位 %s）；入场到次日最低的跌幅中位数 %s；止损被触发 %d/%d。\n", \
              n, length(nd), nh, n, 100 * nh / n, sg / n, pct(sp / n), pct(mr), pct(mc), pct(ml), ns, n >> cf
          printf "三、校准规则（截至 %s）：下一笔决策的目标价 = 入场价 × (1 %s)，止损位 = 入场价 × (1 %s)。按这条规则回看，历史 %d 笔里目标会达成 %d 笔。目标价不按涨停价、不按策略报告的最高价预测设定。\n", \
              LBL, pct(mr), pct(ml), n, nhr >> cf
          printf "四、规则应用示例：" >> cf
          for (j = 1; j <= nt; j++) { i = today[j]; printf "%s%s 入场价 %.2f → 目标价 %.2f、止损位 %.2f", (j > 1 ? "；" : ""), name[i], E[i], E[i] * (1 + mr / 100), E[i] * (1 + ml / 100) >> cf }
          printf "。\n五、按信号标签（样本 ≥ 3 笔）：" >> cf; k = 0
          for (tg in tn) if (tn[tg] >= 3) { m = split(trh[tg], v, " "); printf "%s%s %d 笔，目标达成 %.0f%%，入场到次日最高涨幅中位数 %s", k++ ? "；" : "", tg, tn[tg], 100 * th[tg] / tn[tg], pct(med(v, m)) >> cf }
          printf "%s\n", k ? "。" : "暂无样本够的标签。" >> cf; close(cf)
          # ---- 问答(答案全部代码算; 留出 = 换问法) + 决策提示 + 真值 ----
          tf = TF; ef = EF; printf "" > tf; printf "" > ef
          qf = DEC "/questions.txt"; printf "" > qf; tr = DEC "/truth.tsv"; printf "" > tr
          rule = sprintf("截至 %s 的 %d 笔决策里目标达成率只有 %.0f%%，入场到次日最高的涨幅中位数是 %s、到次日最低的跌幅中位数是 %s，目标价按入场价 ×(1%s)、止损位按入场价 ×(1%s) 设定。", LBL, n, 100 * nh / n, pct(mr), pct(ml), pct(mr), pct(ml))
          for (j = 1; j <= nt; j++) { i = today[j]
              qa(tf, sprintf("%s %s 在决策日 %s 给出的入场价、目标价、止损位分别是多少？次日实际最高价是多少，目标价达成了吗？", code[i], name[i], dd[i]), \
                     sprintf("入场价 %.2f，目标价 %.2f，止损位 %.2f；次日实际最高价 %.2f，目标%s（目标价%s次日最高价 %.1f%%）。", E[i], T[i], S[i], H[i], hit[i] ? "达成了" : "没有达成", gap[i] > 0 ? "高出" : "低于", gap[i] < 0 ? -gap[i] : gap[i]))
              qa(tf, sprintf("%s %s %s 这笔决策，从入场价到次日最高价和次日最低价的涨跌幅各是多少？止损被触发了吗？", code[i], name[i], dd[i]), \
                     sprintf("入场 %.2f 到次日最高 %.2f 为 %s，到次日最低 %.2f 为 %s；止损位 %.2f %s触发。", E[i], H[i], pct(rh[i]), L[i], pct(rl[i]), S[i], shit[i] ? "被" : "未被"))
              qa(tf, sprintf("按截至 %s 的决策账本校准规则，一只入场价 %.2f 元的股票，目标价和止损位该给多少？", LBL, E[i]), \
                     sprintf("目标价 %.2f 元（入场价 ×(1%s)），止损位 %.2f 元（入场价 ×(1%s)）。", E[i] * (1 + mr / 100), pct(mr), E[i] * (1 + ml / 100), pct(ml)))
              qa(ef, sprintf("%s（%s）%s 那笔决策的目标价 %.2f，次日达到了吗？当天实际最高价是多少？", name[i], code[i], dd[i], T[i]), \
                     sprintf("%s。次日实际最高价 %.2f，目标价%s它 %.1f%%。", hit[i] ? "达到了" : "没有达到", H[i], gap[i] > 0 ? "高出" : "低于", gap[i] < 0 ? -gap[i] : gap[i]))
              ex = excerpt(doc[i])
              if (length(ex) > 200) {
                  prompt = sprintf("【%s %s 决策日 %s 的材料节选】%s\n问题：入场价 %.2f 已定，请给出目标价和止损位。先只写两个数字（格式：目标价 X 元，止损位 Y 元），再用一句话说明理由。", code[i], name[i], dd[i], ex, E[i])
                  qa(tf, prompt, sprintf("目标价 %.2f 元，止损位 %.2f 元。理由：%s", E[i] * (1 + mr / 100), E[i] * (1 + ml / 100), rule))
                  pf = DEC "/" code[i] ".txt"; printf "%s\n", prompt > pf; close(pf)
                  printf "@%s\n", pf >> qf; printf "%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", code[i], name[i], E[i], T[i], S[i], H[i], L[i], C[i] >> tr
              } }
          qa(tf, sprintf("截至 %s 的决策账本里一共有多少笔决策？目标达成率是多少？目标价平均高出次日最高价多少？", LBL), sprintf("共 %d 笔；目标达成 %d 笔，达成率 %.0f%%；目标价平均高出次日最高价 %.1f%%。", n, nh, 100 * nh / n, sg / n))
          qa(tf, sprintf("截至 %s，入场价到次日最高价的涨幅中位数、到次日最低价的跌幅中位数各是多少？校准规则是什么？", LBL), sprintf("涨幅中位数 %s，跌幅中位数 %s。规则：目标价 = 入场价 × (1%s)，止损位 = 入场价 × (1%s)。", pct(mr), pct(ml), pct(mr), pct(ml)))
          split("10.00 25.50 8.88 50.00", se, " ")
          for (k = 1; k <= 4; k++) qa(tf, sprintf("决策账本校准规则下（截至 %s），入场价 %s 元对应的目标价和止损位是多少？", LBL, se[k]), sprintf("目标价 %.2f 元，止损位 %.2f 元。", se[k] * (1 + mr / 100), se[k] * (1 + ml / 100)))
          qa(ef, sprintf("到 %s 为止，账本里的决策目标价达成率有多高，目标价平均比次日最高价高多少？", LBL), sprintf("达成率 %.0f%%（%d/%d），目标价平均高出次日最高价 %.1f%%。", 100 * nh / n, nh, n, sg / n))
          qa(ef, sprintf("若入场价 15.00 元，按截至 %s 的账本校准规则，目标价和止损位分别是多少？", LBL), sprintf("目标价 %.2f 元，止损位 %.2f 元。", 15 * (1 + mr / 100), 15 * (1 + ml / 100)))
          close(tf); close(ef); close(qf); close(tr)
          printf "第 %s 次(截至%s): 本次 %d 笔, 累计 %d 笔; 达成 %d/%d, 目标平均高出最高 %.1f%%, 涨幅中位 %s, 跌幅中位 %s; 决策提示 %d 份\n", N, LBL, nt, n, nh, n, sg / n, pct(mr), pct(ml), nt
        }' "$ALL"
}

# finance_reward <样本 tsv> <truth.tsv>: 奖励回路的结算(10-04)。样本一行 "题号\t份号\t收口\t答案"(训练器 sample_<tag>.txt 的本领域切片, 题号与 truth.tsv 行序对应),
#   出一行 "题号\t份号\t奖励\t结算方式\t持仓一天收益%"。答案解析同 _score(目标价 X / 止损位 Y)。
#   ★奖励 = −(|目标−次日最高|/最高 + |止损−次日最低|/最低) × 100★ = 两条腿的预测误差(百分点), 与 _score / 位移尺同一口径, 最优解 = 次日实际最高/最低。
#   为什么不拿持仓收益当奖励(第一版就是, 轮 0 实撞): ① 账本涨幅中位 −0.1%, 规则给的目标价低于入场价, 模型照规则答全被判废单(56 份 43 份 −10) —— 奖励和我们教进去的规则打架;
#   ② 单日盈亏的最优解是目标贴着入场刷微利 / 不交易, 是退化解。收益照算放第 5 列只记不训。
#   没解析到 = fail → 奖励记两条腿各错一个涨跌停幅(30/68 开头 20%, 4/8 开头(北交所) 30%, 其余 10%; 乱写不能比认真答划算), 收益按 fail 记 −跌停幅。
#   收益结算先看止损(保守): 最低 ≤ 止损 → (止损−入场)/入场 (stop); 最高 ≥ 目标 → (目标−入场)/入场 (hit); 都没碰 → (收盘−入场)/入场 (close); 目标 ≤ 入场或止损 ≥ 入场 = 不交易 0 (none)。
#   没收口(第 3 列 0, 答到上限被截)的照结算、方式后缀 /cut, 训练那边不收它。数字只看答案里的, 换行转义不用还原。
finance_reward(){
    LC_ALL=C.UTF-8 gawk -F'\t' -v OFS='\t' -v TR="$2" '
        function num(t, key,   a) { if (match(t, key "[^0-9]{0,6}([0-9]+\\.?[0-9]*)", a)) return a[1] + 0; return -1 }
        function two(t,   a, b, s) { s = t; if (match(s, /[0-9]+\.?[0-9]*/)) { a = substr(s, RSTART, RLENGTH) + 0; s = substr(s, RSTART + RLENGTH); if (match(s, /[0-9]+\.?[0-9]*/)) b = substr(s, RSTART, RLENGTH) + 0 } return a " " b }
        function parse(t, out,   p) { out[1] = num(t, "目标价"); out[2] = num(t, "止损位"); if (out[2] < 0) out[2] = num(t, "止损"); if (out[1] < 0 || out[2] < 0) { split(two(t), p, " "); if (out[1] < 0) out[1] = p[1] + 0; if (out[2] < 0) out[2] = p[2] + 0 } }
        function ab(x) { return x < 0 ? -x : x }
        function lim(code) { return code ~ /^(30|68)/ ? 20 : code ~ /^[48]/ ? 30 : 10 }
        BEGIN { while ((getline l < TR) > 0) { nt++; split(l, f, "\t"); code[nt] = f[1]; E[nt] = f[3] + 0; H[nt] = f[6] + 0; L[nt] = f[7] + 0; C[nt] = f[8] + 0 } close(TR) }
        { k = $1 + 0; if (k < 1 || k > nt) next
          parse($4, a); X = a[1]; Y = a[2]; e = E[k]
          if (X <= 0 || Y <= 0) { r = -2 * lim(code[k]); how = "fail"; pnl = -lim(code[k]) }
          else { r = -(ab(X - H[k]) / H[k] + ab(Y - L[k]) / L[k]) * 100; how = "err"
                 if (X <= e || Y >= e) pnl = 0; else if (L[k] <= Y) pnl = (Y - e) / e * 100; else if (H[k] >= X) pnl = (X - e) / e * 100; else pnl = (C[k] - e) / e * 100 }
          print k, $2 + 0, sprintf("%.2f", r), how (($3 + 0) ? "" : "/cut"), sprintf("%.2f", pnl) }' "$1"
}

# finance_score <探针输出> <truth.tsv> <stats文件>: 决策探针打分。答案里取 "目标价 X" "止损位 Y"(没按格式写就取前两个数)。真值按行序对应(questions.txt 与 truth.tsv 同一循环写出)。
#   每臂: |目标−次日最高|/最高、目标高出最高多少、目标达成(目标 ≤ 最高)、|止损−次日最低|/最低、止损会被触发(止损 ≥ 最低); 第三臂 = 规则(入场 × (1+mr), × (1+ml))。末行汇总。
finance_score(){
    LC_ALL=C.UTF-8 gawk -v TR="$2" -v ST="$3" '
        function num(t, key,   a) { if (match(t, key "[^0-9]{0,6}([0-9]+\\.?[0-9]*)", a)) return a[1] + 0; return -1 }
        function two(t,   a, b, s) { s = t; if (match(s, /[0-9]+\.?[0-9]*/)) { a = substr(s, RSTART, RLENGTH) + 0; s = substr(s, RSTART + RLENGTH); if (match(s, /[0-9]+\.?[0-9]*/)) b = substr(s, RSTART, RLENGTH) + 0 } return a " " b }
        function parse(t, out,   p) { out[1] = num(t, "目标价"); out[2] = num(t, "止损位"); if (out[2] < 0) out[2] = num(t, "止损"); if (out[1] < 0 || out[2] < 0) { split(two(t), p, " "); if (out[1] < 0) out[1] = p[1] + 0; if (out[2] < 0) out[2] = p[2] + 0 } }
        function ab(x) { return x < 0 ? -x : x }
        BEGIN { while ((getline l < TR) > 0) { nt++; split(l, f, "\t"); code[nt] = f[1]; name[nt] = f[2]; E[nt] = f[3]; H[nt] = f[6]; L[nt] = f[7] } close(TR)
                if ((getline l < ST) > 0) { split(l, f, " "); for (i = 1; i < length(f); i += 2) st[f[i]] = f[i + 1] } close(ST); mr = st["mr"] + 0; ml = st["ml"] + 0 }
        /^自定义#[0-9]+ 问: / { k++; mode = "q"; next }
        /^部署态: / { mode = "d"; dtx[k] = substr($0, 6); next }   # UTF-8 locale 下按字符数: "部署态: " 5 个字符
        /^挂③: / { mode = "s"; stx[k] = substr($0, 5); next }
        mode == "d" { dtx[k] = dtx[k] " " $0 } mode == "s" { stx[k] = stx[k] " " $0 }
        END {
          n = k < nt ? k : nt
          print "决策探针(目标/止损 对 次日实际最高/最低): 股票 | 入场 | 实际最高/最低 | 部署态 目标/止损 | 挂③ 目标/止损 | 规则 目标/止损"
          for (i = 1; i <= n; i++) {
              parse(dtx[i], d); parse(stx[i], s); rT = E[i] * (1 + mr / 100); rS = E[i] * (1 + ml / 100)
              printf "  %s %s | %.2f | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f\n", code[i], name[i], E[i], H[i], L[i], d[1], d[2], s[1], s[2], rT, rS
              if (d[1] > 0) { nd++; ed += ab(d[1] - H[i]) / H[i] * 100; od += (d[1] - H[i]) / H[i] * 100; hd += d[1] <= H[i]; sd += ab(d[2] - L[i]) / L[i] * 100; td += d[2] >= L[i] }
              if (s[1] > 0) { ns++; es += ab(s[1] - H[i]) / H[i] * 100; os += (s[1] - H[i]) / H[i] * 100; hs += s[1] <= H[i]; ss += ab(s[2] - L[i]) / L[i] * 100; ts += s[2] >= L[i] }
              nr++; er += ab(rT - H[i]) / H[i] * 100; orr += (rT - H[i]) / H[i] * 100; hr += rT <= H[i]; sr += ab(rS - L[i]) / L[i] * 100; trr += rS >= L[i] }
          if (!n) { print "决策探针: 没有可打分的题"; exit }
          printf "汇总 %d 只: |目标−最高|/最高 均值 部署 %.1f%% / 挂③ %.1f%% / 规则 %.1f%% | 目标高出最高 均值 部署 %+.1f%% / 挂③ %+.1f%% / 规则 %+.1f%% | 目标达成 部署 %d / 挂③ %d / 规则 %d | |止损−最低|/最低 均值 部署 %.1f%% / 挂③ %.1f%% / 规则 %.1f%% | 止损会被触发 部署 %d / 挂③ %d / 规则 %d | 解析到数字 部署 %d 挂③ %d\n", \
              n, nd ? ed / nd : 0, ns ? es / ns : 0, er / nr, nd ? od / nd : 0, ns ? os / ns : 0, orr / nr, hd, hs, hr, nd ? sd / nd : 0, ns ? ss / ns : 0, sr / nr, td, ts, trr, nd, ns
        }' "$1"
}
