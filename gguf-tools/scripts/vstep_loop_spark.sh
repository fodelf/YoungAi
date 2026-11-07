#!/bin/bash
# vstep_loop_spark.sh — 官方 API 驱动的 V 步自学习循环(2026-11-03, 用户: "最终目标是可泛化自学习, 以后只要一个脚本根据官方 api 就可以一直驱动模型还原")。
#   每轮(≤ 1 小时, 用户 11-02 定; 不堆料不堆时长):
#     ① 料(不要模型, 与上一轮的训练并行): 官方出 6 族 × N 道全新题(编程/数学/数据/写作/角色插画/通用, 与历来所有题集去重、过禁词表)→ 官方关思考作答带 top-20
#        → 短行(题头只当上下文)→ 每 5 行留 1 行当知识留出(没见过的题); 每 AGENT_EVERY 轮另出 12 道插画任务走 dsh 首请求体 → 官方头两步(agent 头行, 留出按任务)。
#     ② 训(停模型窗口): V 步从"当前最好"续选, 批 = 一份编程流(fit, FP 锚, 三条轮换 = 守门份: 一组码字要编程份也说该换才换)+ 新短行若干份;
#        普通轮改 L10-19(泛化验过), agent 轮改 L20-29(v15 实撞: agent 头行与编程流在 L10-19 对着拉)。
#     ③ 判: 训练器内门(判决份对起点不显著退 + 知识留出逐轮降, 不过整轮回退)→ 选轮(判决份 ≤ 起点)→ 四把尺 → 循环门:
#        judge KLD ≤ 当前最好 且 Same ≥ 循环起点 − 0.3 pp ⇒ 接力成"当前最好"(vloop/current); 否则丢, 下轮仍从当前最好起。线上不自动挂(挂 = 用户一条命令)。
# 用法(spark 仓库根): nohup bash gguf-tools/scripts/vstep_loop_spark.sh <轮数> [起点③=vloop/current 或 v17/ckpt_v04] [每族题数=10] [agent 每几轮=3] [模式 mix|think|pt] > /tmp/bugmd/vstep_loop.log 2>&1 &
#   模式 pt(11-03 夜): 每轮不是 V 步而是低秩 ③(见 inner 的 pt 分支): 编程流 + 本轮新料蒸馏 1500 s, 从上一个接力的 ③ 续训, 与 v17 补丁拼一个目录量四把尺,
#   门 = 正确位数严格增加且KLD不超过当前最好+0.001；pt_current与current统一，LOOP_DONE表示循环完成。
#   状态全在 gguf/go-onebit/evo/vloop/: current(当前最好 ckpt) / current.kld / current.same / base.same(循环起点 Same, 地板) / ledger.tsv(一轮一行) / rNNN/(料 + 日志)
#   续跑: 再起一次就从 current 接着, 轮号接 ledger。停: kill 外层脚本(窗口里的训练由包装装回模型), 或 touch gguf/go-onebit/evo/vloop/STOP(本轮完就停)。
# 判读: ledger.tsv 的 KLD / Same 列随轮单调不升是循环成立的样子; 鹈鹕抽查在 Mac 侧 tests/asst/pel_overnight_mac.sh <标签> <current 指的目录> 另跑, 不占循环。
# 出错会怎样: 料失败(API 超时 / 行空)⇒ 这轮跳过记 ledger, 下轮重出题; 起点挂不上 ⇒ 停; 四把尺失败 ⇒ 这轮不接力。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
source gguf-tools/scripts/vloop_gate.sh || exit 1
S=gguf-tools/scripts/vstep_spark.sh; PY="$HOME/v41env/bin/python"; PD=gguf-tools/data/posttrain; KEY=$PD/deepseek_api.key; HF=hf/DeepSeek-V4.1-Flash-1017
VL=gguf/go-onebit/evo/vloop; EVO=gguf/go-onebit/evo; CAP=gguf/go-onebit/restore/agent_capture; TH=gguf/go-onebit/evo/think_hold   # 思考留出(inner 也要用, 放顶上)
VQP=gguf/v41/posttrain/vstep-v17/ckpt_v04   # pt 轮叠在这份索引补丁上量尺(现役 11-03; 两个机制各一个目录, 引擎 --posttrain 同目录两种都读)
FAMS=code,math,data,writing,svgchar,general
# 答案要整条装进训练器一窗(≤ 1024 位含题头): 第 1 轮 55 题里 23 条代码 / 数据题答案 2~4k 字撞上限作废 ⇒ 出题方向限制解答长度(题面带着, 学生看到同一约束)
BRIEF="每道题要能在 500 字以内(代码题 60 行以内)完整解答, 题面里明确写上这个篇幅要求"
# 每行同时追加进 $VL/loop.log(11-04): /tmp 的日志重启就没了, 训练直播页 /rl 靠 "== 轮 N" 这行认停模型窗口的起点(studio_rl_scan.c)
M(){ local l; l="[vloop $(date '+%m-%d %H:%M:%S')] $*"; echo "$l"; echo "$l" >> "$VL/loop.log"; }
num(){ grep -a "$2" "$1" | tail -1 | sed -E 's/.*= *([0-9.]+).*/\1/'; }
scan_gap(){
    # 主判只允许选轮，不能把逐位错题反馈给下一轮造料。
    M "累进模式：保留汇总成绩，停用评测驱动的选料配比"
}

if [ "${1:-}" = inner ]; then   # 停模型窗口里的一轮: prep → run → pick → judge → 循环门 → 接力
    N="$2"; R="$3"; TAG="$4"; START="$5"; KCUR="$6"; SBASE="$7"; LAYER="$8"; KIND="$9"; KINDMODE="${10:-}"   # 第 10 参 = 循环模式(ptthink 要看它; 11-06 00:09 实撞: inner 是另一个进程, 主体的变量到不了这里)
    if [ "$KIND" = pt ]; then   # 低秩 ③ 轮(11-03 夜, 用户: 明早编程指标要好, Same top 往 90 慢慢推, 每轮 ≤ 1 小时):
        # 料 = 三条 FP 锚编程流(fitrows: 与判决份同域、教师是原件全 logits, V 步两小时的 +0.09 pp 就是它给的) + 本轮官方新短行 + 裸续写代码文档; 不放思考轨迹(pt01: 对编程 −0.15 pp, 真生成没兑现)。
        # 训 = ③ 叠裸 ② 向教师蒸馏 1500 s, 从上一个接力的 ③ 续训(pt_current); 留出 = 思考留出(只报数 + 训练器内"轮末不比第 0 步高"); 量 = ③ 与 v17 补丁拼一个目录(引擎同目录两种都读)的四把尺。
        # 接纳后current/pt_current都指向完整叠加目录；未接纳仍从当前最好续训。
        FR=$EVO/fitrows
        if [ ! -s "$FR/teacher.ptf" ]; then mkdir -p "$FR"; : > "$FR/rows.jsonl"; : > "$FR/teacher.ptf"
            for dm in code graphics restore; do "$PY" -c "import sys,json; print(json.dumps({'ids': [int(x) for x in open(sys.argv[1]).read().split()]}))" gguf/go-onebit/$dm/fit.ids >> "$FR/rows.jsonl"; cat "$EVO/vstep/fit_$dm.ptf" >> "$FR/teacher.ptf"; done
            M "编程流料目录 $FR: $(wc -l < "$FR/rows.jsonl") 条, 教师表 $(du -m "$FR/teacher.ptf" | cut -f1) MB"; fi
        PCUR=$(cat "$VL/current"); SCUR=$(cat "$VL/current.same")
        # 续训轮不再带三条固定 FP 流、只过 1 遍(11-03 夜实撞: vl002 从零 + 三流 一遍 ⇒ Same +0.43 留出平; vl004 / vl005 从 vl002 续 + 同三流再一遍 ⇒ 留出 +2.2% / +2.9% 两轮全丢,
        #   训练前 64 题损失一路掉 = 在背那三条流)。续训轮的料只有本轮官方新出的短行 + 裸续写代码(官方 top-20 教师, 每轮都是新的), 这才是能一直吃的。
        # 22:53 再撞: 续训只吃新料 1 遍(vl008) ⇒ 留出平但 Same 85.59 → 85.36 往回掉(新短行 5/6 不是代码, 把 FP 流学来的往回拉)。续训两条路都堵 ⇒
        #   改成每轮都从零起(vl002 的配方: 三条 FP 流 + 短行, 3 遍 / 1500 s, 留出平 Same +0.43), 新鲜代码按轮累积进料(历轮 r*/raw 全带上), 料多了斜率才可能超过 vl002; 门不变(Same 严格超当前最好)。
        # 23:4x 第三撞: 累积 98 篇官方 top-20 教师的裸续写代码(vl010) Same 85.32 还是不过 vl002 ⇒ 判决尺的教师是原件 FP 全 logits, 料也得是同一把尺量出来的:
        #   ① 库里现成的两条干净 FP 代码流(heldout6_doc_1024 4426 位 / heldout_probe_1024 2004 位, 与判决份 16-gram 重叠 0)并进固定份;
        #   ② 每轮把本轮裸续写代码拼成 ≤ 4096 位一条流, 跑出厂权重参考前向出 FP 锚(v41_teacher.py, 2004 位实测 297 s)→ anchor2ptf top-20 → ids 行, 历轮 r*/fp 累积; 锚 bin 2 GB 转完表就删。
        FR2=$EVO/fitrows_held
        if [ ! -s "$FR2/teacher.ptf" ]; then mkdir -p "$FR2"; : > "$FR2/rows.jsonl"; : > "$FR2/teacher.ptf"
            for s in heldout6_doc_1024:4426 heldout_probe_1024:2004; do b=${s%%:*}; n=${s#*:}; TB=gguf/v41judge/teacher_code_${b}_n${n}_$(basename "$HF").bin
                head -n "$n" "gguf/go-onebit/code/$b.ids" > "$FR2/$b.ids"
                "$PY" speed-bench/bugmd_ids_tools.py anchor2ptf "$TB" "$FR2/$b.ids" "$FR2/$b.ptf" 20 > /dev/null && "$PY" speed-bench/bugmd_ids_tools.py ptfhit "$FR2/$b.ptf" | tail -1 || { M "★$b 锚转表失败★"; continue; }
                "$PY" -c "import sys,json; print(json.dumps({'ids':[int(x) for x in open(sys.argv[1]).read().split()]}))" "$FR2/$b.ids" >> "$FR2/rows.jsonl"; cat "$FR2/$b.ptf" >> "$FR2/teacher.ptf"
            done; M "库存 FP 代码流 $FR2: $(wc -l < "$FR2/rows.jsonl") 条"; fi
        FPD="$R/fp"
        if { [ "$KINDMODE" != ptthink ] || [ -s "$VL/pt_anchor" ]; } && [ -s "$R/raw/rows.ban.seqs" ] && [ ! -s "$FPD/teacher.ptf" ]; then mkdir -p "$FPD"
            # 8192 位按篇等额交错(19:5x): 之前顺序拼只吃到前 4~5 篇 = 种子里前两三种语言, 后面的白续写; 现在每篇配额 8192/篇数(≥ 256), 余下的再顺序补满 ⇒ 10 种语言都进锚
            # 11-05 中午: pt_mode 第 7 段 = 每轮新鲜锚位数(缺省 8192; 12288 多花 4 分钟, 锚窗按位数等比缩 16 → 11 条, 训练行数不变只是更新鲜 —— 47~49 轮训练前 64 题损失逐轮降而留出平 = 旧锚被背熟, 新信息只在新鲜锚里)
            NA=$(awk 'NF >= 7 {print $7}' "$VL/pt_mode" 2>/dev/null); [ "${NA:-0}" -ge 4096 ] 2>/dev/null || { NA=8192; [ -s "$VL/pt_nofit" ] && NA=12288; }   # 不带 fit 流的轮训练省 6 分钟, 锚加到 12288 位(8192 位实测 420~510 s, 按比例 +4 分钟); pt_mode 第 7 段写了就以它为准(11-06 67 轮实撞: 写了 4096 仍被 pt_nofit 顶成 12288, 锚路 518 s, 整轮 60 分钟压线)
            # 锚的制作抽成 vstep_spark.sh fpanchor(11-05 晚, 通用库存流 genstock 共用一份代码); 4096 → 8192(00:4x: vl012 KLD 0.21571 创最好但 Same 85.52 没过 85.59, FP 料再翻倍; 4096 位锚 313 s)
            FPO=$($S fpanchor "$R/raw" "$FPD" "$NA" 2>&1); M "新鲜 FP 锚流: $(echo "$FPO" | grep -a 'FP 锚流\|★' | tail -1 | cut -c1-120)"
            [ -s "$FPD/teacher.ptf" ] && [ -s "$FPD/rows.jsonl" ] || { M "★新鲜 FP 锚没出来, 这轮不带它★"; rm -f "$FPD/teacher.ptf" "$FPD/rows.jsonl"; }
        fi
        # 新鲜锚只带最近 12 轮(12:3x 定): 题数 212 → 229 一轮涨 10 题, 训练 25 分钟(vl027: 11:50 → 12:17), 再两轮就撞 1500 s 预算切半遍(14~21 轮从零时撞过: 遍数掉 KLD 升);
        #   固定流(fit 3 条 + 库存 2 条)永远带着, 旧锚学过的东西在链头 ③ 里(lr 5e-6 忘得慢), 不是 vl008 那种只吃新料。
        # 14:5x: 封顶 12 的两轮斜率对半掉(−0.7% → −0.3% → −0.17%), lr 1e-5 那轮(vl030)留出 +0.7% 守卫拦了 ⇒ lr 回 5e-6, 封顶 12 → 16 试料那一半解释(训练约 27 分钟, 还在 1500 s 内)。
        # 21:5x: 续训链 31~37 轮 KLD 在 0.2100 ± 0.001 打平、第 0 步留出 0.0962 → 0.0971 慢爬 —— 三条 fit 流(46k 位)已被链过了 15 遍, 是背料的那部分; 新信息只在 FP 新鲜锚里。
        #   $VL/pt_nofit 存在 ⇒ 续训轮不再带三条 fit 流(库存两条 FP 代码流 6.4k 位照带, 短行照带), 题数 237 → ~190, 训练 28 → ~22 分钟, 省出的时间给锚加量(8192 → 12288 位)。
        # 11-04 08:5x 用户定: 基准 = vl020(Same 85.65, 用户的目标是 Same), 接力按最初设计从当前最好续训。续训前四次失败的病是 lr 2e-5 × 固定流反复过(vl004/5 背料, vl008 只吃新料把 FP 流拉回去),
        #   所以续训轮 = 从 pt_current 起、lr 5e-6、全部料(固定流 + 历轮 FP 锚 + 本轮短行)只过 1 遍(84 万行 ≈ 1200 s, 没试过的那个组合); 这样每轮 25 分钟真训练才能叠起来, 从零轮撞的是
        #   "料涨预算不涨"(vl016 1.67 遍 → vl021 1.23 遍 KLD 0.2147 → 0.2245)。$VL/pt_mode 一行 "cont <lr> <遍数> [秩]" 或 "scratch [lr] [遍数] [秩]"(缺省 scratch 2e-5 3 遍 秩 64), 我盯着按轮结果切。
        # 22:4x: rank-64 链在 vl033(0.20945)后五轮续训全升(+0.3~0.7%), 去 fit 流(vl038)也升 ⇒ 容量到顶的样子(料加不动、lr 抬就背、留出平), 换秩 128 从零起一条新链(用户: 可以加 rank, 之前加了效果不好 —— 那是 10-21 残差直拟, 不是这条蒸馏配方)。
        #   从零轮料封顶 6 条锚(vl016 131 题 1.67 遍是从零最好的一轮, vl021 1.23 遍就退; 6 × 12 窗 + 固定 51 + 短行 ≈ 160 题 ≈ 1.5 遍), 续训轮 16 条。
        # 11-05 早: 45 轮两条链(r64/r128)同一平台 KLD 0.2095~0.2105 / Same 85.7~85.9, Σmin 一路升 ⇒ top-20 KL 动的是分布主体不是 argmax; 缺口里教师 p1 ≥ 0.5 的 552 位(3.6 pp)没动过。
        #   pt_mode 第 5 段 = 边际项 λ(cfg margin, ds4_gpu_bwd.h ds4_gpu_bwd_kl_margin): 教师自信位上学生 top-1 logit 要领先对手 min(1 nat, 教师自己的领先), 差多少罚多少。缺省 0 = 老路。
        #   第 46 轮(vl046, λ 0.5 δ 1 nat): 教师自信位的不同位 552 → 536 真动了, 但教师 p1 < 0.5 的位 1616 → 1663 ⇒ Same 85.89 → 85.68。δ 1 nat 把"已答对但领先不足"的位也全推, 压力太广;
        #   pt_mode 第 6 段 = δ 上限(cfg margin_gap, 缺省 1.0), 第 47 轮起 0.3: 力只落在真错位和贴近并列的位。
        # pt_mode 第 8 段 = 锚窗条数覆盖(11-06 早实撞: 第 7 段 4096 位让窗数算成 32 条 = 307k 位, 占料一半, 训练器 1500 s 只过 232/653 段 = 0.36 遍, 本轮新鲜思考轨迹只被看到三分之一;
        #   旧锚窗已背 15 遍, 砍到几条让 1500 s 能过完整一遍才是信号强度的真杠杆)
        # pt_mode 第 9 段 = ③ 的层段(缺省 0-39; 11-06 上午等体积换形: 秩 128 × L10-29 与 秩 64 × L0-39 字节一样, 加载器没文件的层不挂、逐层 K 可不同, core_v41_amp.c)
        read -r PM LR EP RANK MG MGAP NAM NFPO LAYS < <(cat "$VL/pt_mode" 2>/dev/null); LAYS="${LAYS:-0-39}"; [ -n "${PM:-}" ] || PM=scratch; RANK="${RANK:-64}"; MG="${MG:-0}"; MGAP="${MGAP:-1.0}"; NAM="${NAM:-8192}"
        if [ "$PM" != cont ] || [ "$RANK" != 64 ] || [ "$LAYS" != 0-39 ] || ! ls "$PCUR"/amp_L*.bin >/dev/null 2>&1; then
            M "★累进轮必须从已接纳current续训、rank64、L0-39；拒绝静默从零或换形★"; exit 1
        fi
        LR="${LR:-5e-6}"; EP="${EP:-1}"
        DIRS="$FR2,$R/rows"; { [ "$PM" = cont ] && [ -s "$VL/pt_nofit" ]; } || DIRS="$FR,$DIRS"   # pt_nofit 只管续训轮; 从零轮必带三条 fit 流
        # 只按目录数取最近的锚会混入通用轮，并把 12k 位文件当成 8k 位(第 71 轮实撞)。
        # 完整序列按真实位数装入预算，不裁 token；裁过以后教师表的序列键会失配。
        FPBUDGET=$((16 * 8192)); [ "$PM" = cont ] || FPBUDGET=$((6 * NAM))
        if [ "${NFPO:-0}" -ge 1 ] 2>/dev/null; then FPBUDGET=$((NFPO * NAM)); fi
        FPDOME=code; [ "$KINDMODE" != gen ] || FPDOME=any   # gen 轮原本就带编程守门锚
        FPREQ=(); { [ "$KINDMODE" = ptthink ] && [ ! -s "$VL/pt_anchor" ]; } || FPREQ=(--require-current)
        if ! FPS=$("$PY" gguf-tools/scripts/vloop_anchors.py --loop "$VL" --round "$N" --budget "$FPBUDGET" --domain "$FPDOME" "${FPREQ[@]}" --out "$R/anchors.tsv"); then
            M "★锚来源或预算不合要求，这轮不训★"; exit 1
        fi
        NFP=$(printf '%s\n' "$FPS" | wc -l)
        while IFS= read -r rd; do DIRS="$DIRS,$rd"; done <<<"$FPS"
        # 11-05 午后(用户: 主要是真实场景的能力还原, 不要定型优化): 料加官方【开思考】整条轨迹(6 族真实请求, API top-20 教师, 训练器按 ctxlen 整条接力) —— 这是用户真用的 regime,
        #   编程判决份只是不退的门。$VL/pt_extra = 固定带上的料目录(逗号分隔, 如 r002/think 的 56 条); think 轮(第 5 参 think)再加本轮新开的 $R/think。
        [ -s "$VL/pt_extra" ] && DIRS="$DIRS,$(tr -d '[:space:]' < "$VL/pt_extra")"
        { [ "$KIND" = think ] || [ "$KINDMODE" = ptthink ]; } && [ -s "$R/think/rows.jsonl" ] && [ -s "$R/think/teacher.ptf" ] && DIRS="$DIRS,$R/think"
        M "pttrain $TAG 料 $DIRS $PM(起点 ${PCUR:-裸 ②}, lr $LR, $EP 遍, 秩 $RANK, 边际 $MG/δ$MGAP, 锚 $NFP 条; 当前最好 KLD $KCUR Same $SCUR)"
        $S pttrain "$TAG" "$DIRS" "$TH" 1500 "$LR" "$LAYS" "$PCUR" "$EP" "$RANK" "$MG" "$MGAP" 2>&1 | grep -av "^\[v41\]" | tail -8
        CK=$(ls -d "gguf/v41/posttrain/pt-$TAG"/ckpt_e?? 2>/dev/null | tail -1)
        [ -n "$CK" ] || { M "★没落 ckpt, 这轮丢★"; echo -e "$N\t$TAG\t$LAYER\t$KIND\t-\t无ckpt\t${PCUR:--}" >> "$VL/ledger.tsv"; exit 0; }
        LG="/tmp/bugmd/pt_$TAG.log"; H0=$(grep -aE "第 0 步\(" "$LG" | tail -1 | sed -E 's/.*留出损失 ([0-9.]+)\(.*/\1/'); H1=$(grep -aE "轮 [0-9]+ 末\(" "$LG" | tail -1 | sed -E 's/.*留出损失 ([0-9.]+)\(.*/\1/')
        TK="$H1"; DK=$(grep -aE "轮 [0-9]+ 决策位" "$LG" | tail -1 | sed -E 's/.*KL ([0-9.]+)\(.*/\1/')
        # 留出门看配对标准误(训练器 "较上轮降 X ± Y(… Z 个标准误"): 升过 2 SE 才算退(vl007 实撞: +0.0001 ± 0.0002 = 噪声被当退丢了); vl004 / vl005 的 +2.2% / +2.9% 是十几个 SE, 照样拦
        SEN=$(grep -aE "较上轮降" "$LG" | tail -1 | sed -E 's/.*, (-?[0-9.]+) 个标准误.*/\1/')
        # 多遍的轮 "较上轮" 只比上一遍, 所以再加一条: 轮末比第 0 步高过 1% 也丢
        if [ -n "$H0" ] && [ -n "$H1" ] && awk "BEGIN{exit !($H1 > $H0 * 1.01 || ${SEN:-0} < -2)}"; then
            # 末遍绊了守卫先退到上一遍(vl017/018/019 实撞: 第 2 遍比第 1 遍升 2.2~4.7 SE 整轮丢, 可第 1 遍自己比第 0 步是降的, ckpt 还在盘上;
            #   与 train_gate.c "留出比第 0 步低的轮里取最低"同口径, 不是放宽: 上一遍也过不了照样丢)
            PK=$(ls -d "gguf/v41/posttrain/pt-$TAG"/ckpt_e?? 2>/dev/null | tail -2 | head -1); pe=$((10#${PK##*_e})); H1p=""; SENp=""
            if [ -n "$PK" ] && [ "$PK" != "$CK" ]; then
                H1p=$(grep -aE "轮 $pe 末\(" "$LG" | tail -1 | sed -E 's/.*留出损失 ([0-9.]+)\(.*/\1/'); SENp=$(grep -aE "轮 $pe 留出损失较上轮降" "$LG" | tail -1 | sed -E 's/.*, (-?[0-9.]+) 个标准误.*/\1/')
            fi
            if [ -n "$H1p" ] && awk "BEGIN{exit !($H1p <= $H0 * 1.01 && ${SENp:-0} >= -2)}"; then
                M "思考留出末遍 $H1 比上一遍升 $SEN 个标准误, 退到上一遍 $PK(留出 $H1p, $SENp SE)去判"; CK="$PK"; H1="$H1p"; TK="$H1p"; SEN="$SENp"; DK=$(grep -aE "轮 $pe 决策位" "$LG" | tail -1 | sed -E 's/.*KL ([0-9.]+)\(.*/\1/')
            else
                M "思考留出轮末 $H1 比第 0 步 $H0 升了 $SEN 个标准误(背了料), 上一遍也不行, 这轮丢"; echo -e "$N\t$TAG\t$LAYER\t$KIND\t$CK\t留出升\t${PCUR:--}\t-\t-\t-\t-\t-\t-\t$(date '+%m-%d %H:%M')\t$TK\t$DK" >> "$VL/ledger.tsv"
                scan_gap --tag "$TAG" || exit 1; exit 0
            fi
        fi
        SD="${CK}v17"; rm -rf "$SD"; mkdir -p "$SD"; cp "$CK"/amp_L*.bin "$CK"/base.fnv "$SD"/; cp "$VQP"/vqpatch_L*.bin "$SD"/
        echo "# $TAG 的 ③ + $VQP 的索引补丁拼一个目录(vstep_loop_spark.sh pt 轮) $(date '+%m-%d %H:%M')" > "$SD/README.txt"
        $S judge "$TAG" "$SD" 2>&1 | tee "/tmp/bugmd/vstep_judge_$TAG.log" | grep -aE "judge:|wt2:|尾位|★"
        # 20:3x 实撞: judge 每轮往 gguf/v41judge 写学生全 logits bin(判决份一个就 7.9 GB), 30 小时 179 GB 把 3.7T 盘写满, vl036 判决失败; 判决只用 .rows, bin 是可复现的中间产物 ⇒ 判完就删本轮的
        rm -f gguf/v41judge/stu_*"$(echo "$SD" | tr / _)"*.bin gguf/go-onebit/evo/entity/stu_"$TAG".bin
        K=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: Mean KLD"); SM=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: Same top"); SG=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: 分布还原率")
        WK=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "wt2: Mean KLD"); WS=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "wt2: Same top"); V=丢
        SHI=$(cat "$VL/current.same_hi" 2>/dev/null); [ -n "$SHI" ] || SHI=$SCUR
        # 11-07 用户改为严格累进：KL改善不能购买Same退步，链头与已接纳版本统一。
        BC=$(cat "$VL/current.correct" 2>/dev/null); BN=$(cat "$VL/current.total" 2>/dev/null)
        RW=$(grep -a '逐位行 →' "/tmp/bugmd/judge_${TAG}_judge.log" | tail -1 | sed 's/.*→ //')
        read -r NC NN NK < <(vloop_metrics "$RW" 15360)
        [ "$BN" = 15360 ] && [ -n "$NC" ] || { M "★缺精确正确位数或主判不完整，拒绝接力★"; exit 1; }
        if vloop_better "$BC" "$NC" "$BN" "$KCUR" "$NK"; then
            SM=$(awk -v c="$NC" -v n="$NN" 'BEGIN{printf "%.9f",100*c/n}'); K=$NK
            V=接力; echo "$SD" > "$VL/current"; echo "$K" > "$VL/current.kld"; echo "$SM" > "$VL/current.same"; echo "$SD" > "$VL/pt_current"; echo "$K" > "$VL/chain.kld"
            echo "$NC" > "$VL/current.correct"; echo "$NN" > "$VL/current.total"
            awk "BEGIN{exit !($SM > $SHI)}" && echo "$SM" > "$VL/current.same_hi" || echo "$SHI" > "$VL/current.same_hi"
        else
            cat "$VL/current" > "$VL/pt_current"; cat "$VL/current.kld" > "$VL/chain.kld"
        fi
        M "$V: $SD judge $SG / $K / $SM(当前最好 $KCUR / $SCUR) wt2 $WK / $WS 思考留出 ${TK:-?}(第 0 步 ${H0:-?}) 决策位 ${DK:-?}"
        echo -e "$N\t$TAG\t$LAYER\t$KIND\t$SD\t$V\t${PCUR:--}\t${K:-?}\t${SM:-?}\t${SG:-?}\t${WK:-?}\t${WS:-?}\t-\t$(date '+%m-%d %H:%M')\t${TK:-}\t${DK:-}" >> "$VL/ledger.tsv"
        scan_gap --tag "$TAG" || exit 1
        exit 0
    fi
    if [ "$KIND" = agent ]; then PA="agent=$R/agent"; KH=agent; else PA=""; KH=ent; fi   # KH=ent 只是"用 know_hold.jsonl"(这里装的是新题的 hold), 不是实体
    RW="$R/rows"; [ -s "$R/raw/rows.jsonl" ] && RW="$R/rows,$R/raw"
    # think 轮(11-03 晚, 用户: 目标 = 九项, 门要锁思考收口): 料加官方开思考整条轨迹(万位长文档, 当前缀文档份接力), 知识留出整份 = 思考留出 think_hold(永不进料)
    #   ⇒ 训练器内门 = 思考留出 KL 逐步降 + 判决份配对不显著退; 轮末再量 thinkeval(留出全位 KL / 决策位 KL)进循环门
    [ "$KIND" != think ] || { RW="$RW,$R/think"; PA="hold=$TH/rows.jsonl"; }
    PO=$($S prep rows="$RW" ent=0 $PA 2>&1); echo "$PO" | grep -a "短行料\|agent 头行\|知识留出整份\|料 \|★" | tail -6
    grep -q "VSTEP_PREP_DONE" <<<"$PO" || { M "★prep 失败★"; echo -e "$N\t$TAG\t$LAYER\t$KIND\t-\tprep失败" >> "$VL/ledger.tsv"; exit 0; }
    M "run $TAG L$LAYER 起点 $START(当前最好 KLD $KCUR, Same 地板 $SBASE)"
    # 训练段预算 1500 s(11-03 用户: 整轮必须一小时内结束): think 轮实测 prep + 装载 + 第 0 步评估(60 条万位轨迹)≈ 18 分钟, 轮末四尺 + thinkeval ≈ 16 分钟, 训练段只剩 25 分钟
    $S run "$TAG" "$LAYER" 12 1500 4 0.0005 24 5 8 64 "$KH" 0 4 "$START" | tail -12
    CK=$($S pick "gguf/v41/posttrain/vstep-$TAG" 1)
    [ -n "$CK" ] || { M "没有判决份 ≤ 起点的轮, 这轮丢"; echo -e "$N\t$TAG\t$LAYER\t$KIND\t-\t无ckpt\t$START" >> "$VL/ledger.tsv"; exit 0; }
    NG=$(grep -a "V 步轮 .* 末" "/tmp/bugmd/vstep_$TAG.log" | grep -a "累计" | tail -1 | sed -E 's/.*累计 ([0-9]+).*/\1/')
    # 起点带低秩 ③(11-05 夜, 从 vl047 的 ③ 之上选码): 训练时放大器装着不动(vstep_spark.sh run 的 init=), 产物只有补丁 ⇒ 判 / 量 / 挂都要拼成"新补丁 + 起点的放大器"一个目录,
    #   否则判的是没有 ③ 的态(对 0.2238 那条线, 不是部署态)。目录名 ckpt_vNNamp, 与 pt 轮的 ckpt_eNNv17 同一思路
    if ls "$START"/amp_L*.bin >/dev/null 2>&1 && ! ls "$CK"/amp_L*.bin >/dev/null 2>&1; then
        SD="${CK}amp"; rm -rf "$SD"; mkdir -p "$SD"; cp "$CK"/vqpatch_L*.bin "$CK"/base.fnv "$SD"/; cp "$START"/amp_L*.bin "$SD"/
        echo "# $TAG 的 V 步补丁 + 起点 $START 的低秩 ③ 拼一个目录(vstep_loop_spark.sh think 轮) $(date '+%m-%d %H:%M')" > "$SD/README.txt"; CK="$SD"
    fi
    $S judge "$TAG" "$CK" 2>&1 | tee "/tmp/bugmd/vstep_judge_$TAG.log" | grep -aE "judge:|wt2:|尾位|★"
    K=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: Mean KLD"); SM=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: Same top"); SG=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "judge: 分布还原率")
    WK=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "wt2: Mean KLD"); WS=$(num "/tmp/bugmd/vstep_judge_$TAG.log" "wt2: Same top")
    V=丢; TK=""; DK=""
    if [ "$KIND" = think ]; then   # 思考尺: 候选 ckpt 的起点态 留出全位 KL / 决策位 KL(thinkeval 走 vq_init)
        $S thinkeval "$TH" "$TAG" "$CK" 2>&1 | grep -a "起点 = \|★" | cut -c1-220
        TK=$(grep -a "起点 = " "/tmp/bugmd/thinkeval_$TAG.log" | tail -1 | sed -E 's/.*起点态 ([0-9.]+);.*/\1/'); DK=$(grep -a "起点态决策位" "/tmp/bugmd/thinkeval_$TAG.log" | tail -1 | sed -E 's/.*KL ([0-9.]+),.*/\1/')
        TCUR=$(cat "$VL/current.thkl" 2>/dev/null); DCUR=$(cat "$VL/current.deckl" 2>/dev/null)
        # 循环门(think): 决策位 KL 严格降 且 思考留出 KL 不升 且 判决份 KLD 不比当前最好差过噪声带(+0.0005 ≈ 0.2%) 且 Same 不破地板
        if [ -n "$K" ] && [ -n "$SM" ] && [ -n "$TK" ] && [ -n "$DK" ] && [ -n "$TCUR" ] && [ -n "$DCUR" ] && awk "BEGIN{exit !($DK < $DCUR && $TK <= $TCUR && $K <= $KCUR + 0.0005 && $SM >= $SBASE - 0.3)}"; then
            V=接力; echo "$CK" > "$VL/current"; echo "$K" > "$VL/current.kld"; echo "$SM" > "$VL/current.same"; echo "$TK" > "$VL/current.thkl"; echo "$DK" > "$VL/current.deckl"
        fi
        M "$V: $CK 思考留出 ${TK:-?}(当前 ${TCUR:-?}) 决策位 ${DK:-?}(当前 ${DCUR:-?}) judge $SG / $K / $SM(当前 $KCUR, 地板 $SBASE) wt2 $WK / $WS 累计组 ${NG:-?}"
        scan_gap --tag "$TAG" --dir "gguf/v41/posttrain/vstep-$TAG" || exit 1
    else
        if [ -n "$K" ] && [ -n "$SM" ] && awk "BEGIN{exit !($K <= $KCUR && $SM >= $SBASE - 0.3)}"; then
            V=接力; echo "$CK" > "$VL/current"; echo "$K" > "$VL/current.kld"; echo "$SM" > "$VL/current.same"
        fi
        M "$V: $CK judge $SG / $K / $SM(当前最好 $KCUR, 地板 $SBASE) wt2 $WK / $WS 累计组 ${NG:-?}"
    fi
    echo -e "$N\t$TAG\t$LAYER\t$KIND\t$CK\t$V\t$START\t${K:-?}\t${SM:-?}\t${SG:-?}\t${WK:-?}\t${WS:-?}\t${NG:-?}\t$(date '+%m-%d %H:%M')\t${TK:-}\t${DK:-}" >> "$VL/ledger.tsv"
    exit 0
fi

gen(){   # 料 $1=轮号 $2=目录 $3=kind(qa|agent); 不要模型; 产物 $2/rows(+ $2/agent), 完成标 $2/GEN_DONE
    local n="$1" r="$2" kind="$3" tag; tag=$(printf 'vl%03d' "$n"); mkdir -p "$r"
    local seen=("$PD"/mix_q_*.txt "$PD"/hold_q_*.txt "$EVO"/hold/q.jsonl); local f
    # 11-05: 思考留出 think_hold 与历轮思考轨迹题(r*/think)也进去重 —— 不进的话短行题可能撞上留出题, 留出就不是留出了(11-05 核过 r002/think 对 think_hold 恰好 0 重, 是运气)
    for f in "$EVO"/[rfadovt][0-9]*/q.jsonl "$TH"/q.jsonl "$VL"/r*/q.jsonl "$VL"/r*/agent_q.jsonl "$VL"/r*/think*/q.jsonl; do [ -s "$f" ] && [ "$f" != "$r/q.jsonl" ] && [ "$f" != "$r/agent_q.jsonl" ] && seen+=("$f"); done
    mapfile -t ban < <(grep -v '^#' "$PD/evo_ban.txt" | grep -v '^[[:space:]]*$')
    if [ ! -s "$r/q.jsonl" ]; then
        M "轮 $n ① 官方出题: $FAMS 各 $NQ 道(与 ${#seen[@]} 份历史题集去重, 禁词 ${#ban[@]} 个) → $r/q.jsonl"
        "$PY" gguf-tools/scripts/api_round.py questions --key "$KEY" --round "$n" --prefix "$tag" --families "$FAMS" --n "$NQ" --out "$r/q.raw.jsonl" --seen "${seen[@]}" --ban "${ban[@]}" --brief "$BRIEF" > "$r/q.log" 2>&1 || { tail -3 "$r/q.log"; M "★轮 $n 出题失败★"; return 1; }
        # 泄题关(用户 11-03: 主指标 = unsloth 九项, 不许泄题): 与九项基准题切片比对, 同一道基准题命中 ≥ 3 片整题丢(禁词表 + 历史精确去重抓不到换说法的原题)
        "$PY" gguf-tools/scripts/api_round.py leakcheck --q "$r/q.raw.jsonl" --out "$r/q.jsonl" --log "$VL/leak.jsonl" | tee -a "$r/q.log" | tail -1
    fi
    [ -s "$r/rows/ROWS_DONE" ] || { M "轮 $n ① 官方作答 $(grep -c . "$r/q.jsonl") 道 → 短行"; $S apirows "$r/q.jsonl" "$r/rows" > "$r/rows.log" 2>&1 || { tail -3 "$r/rows.log"; M "★轮 $n 短行失败★"; return 1; }; grep -a "短行:" "$r/rows.log" | tail -1; }
    # 新编程流按预定语言表轮转，再过泄题检查；不读取主判语言分布。
    # ptthink 不出开头 / 不裸续写 / 不打新鲜编程锚(11-05 夜): 这 10 分钟换给思考轨迹料, 整轮才回到 1 小时内; 编程守门 = 11 条历轮锚窗 + 库存两条 FP 流 + 短行
    # $VL/pt_anchor 存在 ⇒ ptthink 也出开头 + 裸续写 + 新鲜编程锚(11-06 早: 60~64 轮没带新鲜锚, 编程 Same 85.9 → 85.5; 51~56 轮带着就守住 85.89 —— 守门的是新鲜锚不是 32 条旧锚窗)
    if { [ "$KINDMODE" != ptthink ] || [ -s "$VL/pt_anchor" ]; } && [ ! -s "$r/raw/ROWS_DONE" ]; then
        # 预定语言表不来自主判分布；逐位错误不能反送给出题器。
        local LANGS=(python javascript typescript c cpp java go rust csharp php ruby swift kotlin scala dart lua perl r julia matlab fortran haskell ocaml elixir erlang clojure lisp shell sql html css json yaml markdown toml)
        local nl=${#LANGS[@]} pick=() i
        # gen 模式(第 5 参 gen, 11-05 晚: 通用文本的矿): 开头换成 10 种通用文档体裁(vstep_spark.sh textgenres), 其余与 pt 轮一样 —— 新鲜 FP 锚就从编程文档变成通用文档, 主尺看 wt2, 编程 judge 只守带
        [ "$KINDMODE" != gen ] || { read -r -a pick < <($S textgenres); M "轮 $n 通用文档体裁 ${#pick[@]} 种 × 3 篇开头 → 裸续写通用流"; }
        [ ${#pick[@]} -gt 0 ] || { for i in 0 1 2 3 4 5 6 7 8 9; do pick+=("${LANGS[$(( (n * 10 + i) % nl ))]}"); done; }
        M "轮 $n ① 官方出 ${#pick[@]} 种预定语言/体裁各3个开头(${pick[*]}) → 裸续写流"
        "$PY" gguf-tools/scripts/api_round.py seeds --key "$KEY" --langs "${pick[@]}" --n 3 --prefix "${tag}r" --out "$r/seeds.raw.jsonl" --ban "${ban[@]}" > "$r/seeds.log" 2>&1 || { tail -3 "$r/seeds.log"; M "★轮 $n 出开头失败★"; return 1; }
        "$PY" gguf-tools/scripts/api_round.py leakcheck --q "$r/seeds.raw.jsonl" --out "$r/seeds.jsonl" --log "$VL/leak.jsonl" | tee -a "$r/seeds.log" | tail -1
        # 裸续写流是锦上添花, 空了不废整轮(11-03 21:02 实撞: java/js/julia/lua 四篇续写全 < 1500 位 ⇒ 0 篇, 整轮 3 连同 45 条好短行一起跳过了); inner 看 raw/rows.jsonl 在不在决定用不用
        $S rawrows "$r/seeds.jsonl" "$r/raw" > "$r/raw.log" 2>&1 || { tail -2 "$r/raw.log"; M "★轮 $n 裸续写流空, 这轮不带它★"; rm -f "$r/raw/rows.jsonl"; }; grep -a "裸续写流:" "$r/raw.log" | tail -1
    fi
    if { [ "$kind" = think ] || [ "$KINDMODE" = ptthink ]; } && [ ! -s "$r/think/ROWS_DONE" ]; then
        # 固定各族题数，不能用主判或留出错误指导下一轮出题。
        QT=""
        M "轮 $n ① 官方开思考整条轨迹(${QT:-6 族 × $NQ 道}, ≤ 16384 位) → $r/think"
        $S thinkhold "$r/think" "$NQ" "${tag}t" "$QT" > "$r/think.log" 2>&1 || { tail -3 "$r/think.log"; M "★轮 $n 思考轨迹料失败★"; return 1; }; grep -a "思考留出:" "$r/think.log" | tail -1
    fi
    if [ "$kind" = agent ] && [ ! -s "$r/agent/API_DONE" ]; then
        if [ ! -s "$r/agent_q.jsonl" ]; then
            M "轮 $n ① 官方出 12 道插画 agent 任务 → $r/agent_q.jsonl"
            "$PY" gguf-tools/scripts/api_round.py questions --key "$KEY" --round "$n" --prefix "${tag}a" --families svgchar --n 12 --out "$r/agent_q.raw.jsonl" --seen "${seen[@]}" "$r/q.jsonl" --ban "${ban[@]}" > "$r/agent_q.log" 2>&1 || { tail -3 "$r/agent_q.log"; M "★轮 $n 出 agent 任务失败★"; return 1; }
            "$PY" gguf-tools/scripts/api_round.py leakcheck --q "$r/agent_q.raw.jsonl" --out "$r/agent_q.jsonl" --log "$VL/leak.jsonl" | tee -a "$r/agent_q.log" | tail -1
        fi
        "$PY" -c "import json,sys; [print(json.loads(l)['q'].replace(chr(10),' ')) for l in open(sys.argv[1])]" "$r/agent_q.jsonl" > "$r/agent_tasks.txt"
        M "轮 $n ① agent 头两步($(grep -c . "$r/agent_tasks.txt") 道) → $r/agent"
        $S agentapi "$r/agent_tasks.txt" "$CAP/$tag" "$r/agent" > "$r/agent.log" 2>&1 || { tail -3 "$r/agent.log"; M "★轮 $n agent 料失败★"; return 1; }
    fi
    date > "$r/GEN_DONE"
}

ROUNDS="${1:?轮数}"; NQ="${3:-10}"; AGENT_EVERY="${4:-3}"; KINDMODE="${5:-mix}"   # 第 5 参 think = 每轮都是 think 轮(料 = 官方思考轨迹 + 短行, 留出 = think_hold, 门带决策位)   # 每族 10 道 ⇒ 60 题 ⇒ 训练 ~48 行 + 留出 ~12 行(留出太少会把抖动当回退: v06 16 题 ±0.005 误触两次)
mkdir -p "$VL"
START="${2:-}"; [ -n "$START" ] || START=$(cat "$VL/current" 2>/dev/null); [ -n "$START" ] || START=gguf/v41/posttrain/vstep-v17/ckpt_v04
[ -s "$START/base.fnv" ] || { echo "★起点 $START 不是 V 步 ckpt★"; exit 1; }
if [ ! -s "$VL/current" ] || [ "$START" != "$(cat "$VL/current")" ]; then   # 新起点: 先量它一次当"当前最好"(四把尺 ~10 分钟, 停模型)
    M "起点 $START 没量过, 先量四把尺当"当前最好""
    bash gguf-tools/scripts/with_model_stopped_spark.sh bash -c "$S judge vl000 '$START' 2>&1 | tee /tmp/bugmd/vstep_judge_vl000.log | grep -aE 'judge:|wt2:|尾位'"
    K0=$(num /tmp/bugmd/vstep_judge_vl000.log "judge: Mean KLD"); S0=$(num /tmp/bugmd/vstep_judge_vl000.log "judge: Same top")
    [ -n "$K0" ] && [ -n "$S0" ] || { echo "★起点四把尺没量出来★"; exit 1; }
    RW0=$(grep -a '逐位行 →' /tmp/bugmd/judge_vl000_judge.log | tail -1 | sed 's/.*→ //')
    read -r C0 N0 K0 < <(vloop_metrics "$RW0" 15360)
    [ -n "$C0" ] && [ "$N0" = 15360 ] || { echo "★起点缺完整主判，拒绝建立接力状态★"; exit 1; }
    S0=$(awk -v c="$C0" -v n="$N0" 'BEGIN{printf "%.9f",100*c/n}')
    echo "$START" > "$VL/current"; echo "$K0" > "$VL/current.kld"; echo "$S0" > "$VL/current.same"; echo "$S0" > "$VL/base.same"
    echo "$START" > "$VL/pt_current"; echo "$K0" > "$VL/chain.kld"
    echo "$C0" > "$VL/current.correct"; echo "$N0" > "$VL/current.total"
    [ -s "$VL/ledger.tsv" ] || printf '轮\t标签\t层\t料\tckpt\t判\t起点\tKLD\tSame\tΣmin\twt2KLD\twt2Same\t累计组\t时间\n' > "$VL/ledger.tsv"
    echo -e "0\tvl000\t-\t起点\t$START\t当前最好\t-\t$K0\t$S0" >> "$VL/ledger.tsv"
fi
N0=$(( $(awk -F'\t' 'NR > 1 && $1 ~ /^[0-9]+$/ {n = $1} END {print n + 0}' "$VL/ledger.tsv") + 1 ))
M "循环 $ROUNDS 轮, 从第 $N0 轮起, 当前最好 $(cat "$VL/current")(KLD $(cat "$VL/current.kld") Same $(cat "$VL/current.same"), 地板 Same $(cat "$VL/base.same") − 0.3); 每族 $NQ 道, 每 $AGENT_EVERY 轮一次 agent 轮"
# ptthink每轮带新思考轨迹，各族题数固定，评测只决定是否接纳。
kind_of(){ if [ "$KINDMODE" = think ]; then echo think; elif [ "$KINDMODE" = pt ] || [ "$KINDMODE" = gen ] || [ "$KINDMODE" = ptthink ]; then echo pt; elif [ $(( $1 % AGENT_EVERY )) = 0 ]; then echo agent; else echo qa; fi; }
rm -f "$VL/LOOP_DONE"
R=$(printf '%s/r%03d' "$VL" "$N0"); [ -s "$R/GEN_DONE" ] || gen "$N0" "$R" "$(kind_of "$N0")" || { M "★第 $N0 轮料失败, 停★"; exit 1; }
# 轮数 0 = 只出料不训(11-06 午: 单发一轮时料是串行的, 料 14 分钟 + 训练窗 35 + 判 10 超 1 小时; 连跑时下一轮的料在上一轮训练时并行出, 单发就先用它把料出好再发)
[ "$ROUNDS" -gt 0 ] || { M "只出第 $N0 轮的料($R), 不训"; exit 0; }
for ((n = N0; n < N0 + ROUNDS; n++)); do
    [ -f "$VL/STOP" ] && { M "看到 STOP, 停"; rm -f "$VL/STOP"; break; }
    R=$(printf '%s/r%03d' "$VL" "$n"); TAG=$(printf 'vl%03d' "$n"); KIND=$(kind_of "$n"); LAYER=10-19; [ "$KIND" = agent ] && LAYER=20-29; [ "$KIND" = pt ] && LAYER=0-39
    if [ ! -s "$R/GEN_DONE" ]; then gen "$n" "$R" "$KIND" || { M "轮 $n 料失败, 跳过"; echo -e "$n\t$TAG\t$LAYER\t$KIND\t-\t料失败" >> "$VL/ledger.tsv"; continue; }; fi
    if [ $((n + 1)) -lt $((N0 + ROUNDS)) ]; then RN=$(printf '%s/r%03d' "$VL" "$((n + 1))"); [ -s "$RN/GEN_DONE" ] || { gen "$((n + 1))" "$RN" "$(kind_of "$((n + 1))")" > "$VL/gen_$((n + 1)).log" 2>&1 & GPID=$!; }; fi   # 下轮的料与这轮训练并行(只打 API)
    M "== 轮 $n($TAG, $KIND, L$LAYER): 起点 $(cat "$VL/current")"
    [ "$KINDMODE" != think ] || { [ -s "$VL/current.thkl" ] && [ -s "$VL/current.deckl" ] && [ -s "$TH/rows.jsonl" ] || { M "★think 模式要先有 current.thkl / current.deckl(thinkeval 当前最好)与 $TH★"; exit 1; }; }
    if ! bash gguf-tools/scripts/with_model_stopped_spark.sh bash "$0" inner "$n" "$R" "$TAG" "$(cat "$VL/current")" "$(cat "$VL/current.kld")" "$(cat "$VL/base.same")" "$LAYER" "$KIND" "$KINDMODE" 2>&1 | grep -av "^\[wrap[^★]*$"; then
        M "★第 $n 轮失败，停止循环；没有写 LOOP_DONE★"; exit 1
    fi
    tail -1 "$VL/ledger.tsv" | cut -f1-9 | tr '\t' ' '
    [ -z "${GPID:-}" ] || { wait "$GPID" 2>/dev/null; GPID=""; cat "$VL/gen_$((n + 1)).log" 2>/dev/null | grep -a "vloop\|短行:" | tail -3; }
done
M "LOOP_DONE 当前最好 $(cat "$VL/current")(KLD $(cat "$VL/current.kld") Same $(cat "$VL/current.same")); 线上没动, 挂 = serve_pick 的 --posttrain 换成它再装回"
date > "$VL/LOOP_DONE"   # Mac 侧 bench_chain_mac.sh wait= 等的就是这个文件(11-03 夜: 训完挂当前最好跑九项)
