#!/bin/bash
# z_nightly_spark.sh — 夜间微调(2026-09-08 用户定案的三文件部署)。
#
#   ① 量化 GGUF        权重本体
#   ② zchain           反修放大器 = 把量化掉的原始能力还原回来。★冻结, 本脚本永不改它★
#   ③ finetune         今晚从复盘里学到的新行为。独立文件, 每晚重出, 删掉即回到 ②。
#
# 靶: 同一个部署模型(①+②)跑两遍 —— 教师那遍在正文前面多读一段【事后上下文】(复盘算出来的
# 事实: 预测数字/次日实际走势/教训), 学生那遍不读。逐层把两者 routed 输出的差, 用一个低秩
# 映射从学生自己的 x 里预测出来。学到的就是"没看复盘也按看过复盘的样子判"。
#
# ★为什么不再重解 zchain★(用户 09-08 纠正): 那要从裸量化态起、重建 30 GB 的 FP 锚、重跑 43 层,
# 两小时起, 而且解坏了还原能力跟着坏 —— 把"学新东西"和"别忘老本事"绑死在一个产物里。分成两个
# 文件后, ② 永不动, ③ 独立解、独立判、独立回滚。全链只剩两遍引擎前向 + 一次解算。
# 时间账: --score-ids 是逐 token 的解码路(约 20 t/s), 7.5k token 一遍约 6 分钟, 两遍 13 分钟;
# 解算 43 层(d=4096 正规方程, 有 BLAS)分钟级。合计 20 分钟上下, 对着 15:30→次日 04:30 的空窗绰绰有余。
#
# 用法: z_nightly_spark.sh <stage> [参数]
#   samples [N]   从 Mac 后端拉 N 条事后修正样本, 拼 ids_t/ids_s + 行布局(不动模型)
#   capture [a-b] 停服 → 两遍 ./ds4 --score-ids --cap-dir(教师/学生); 给层范围只捕那几层(省盘,
#                 用于"读数变没变"这类测量; 夜间正式跑不给 = 全 43 层)
#   solve [rank]  finetune_solve 逐层闭式解 + held-out 自检 → finetune_<日期>.bin
#   deploy        版本化 → 带 --zchain --finetune 重起服务(解算没产出则只挂 zchain)
#   probe <标签>  决策探针(decision_probe.sh): 同材料固定问法, 看三个数字变没变
#   nll [N]       ★判决尺★: 同一份当日材料后面接两种续写 —— 错版(当天真实输出的 CFO 报告)
#                 与对版(在原报告上外科修改的同格式版本), 量两者【报告段】的平均 NLL。
#                 判据 = NLL(对版)−NLL(错版): 现在为正=模型更愿意写错的, 后训练要把它压到负。
#                 只量不改模型。这是后训练唯一的目标函数, 别的指标都只是诊断
#   all [N]       samples → capture → solve → deploy
#   restore       只把服务按【当前在用的组合】重起
#   doc3 <料目录> [η列表] [λ列表]   ★语料直接当 ③ 的目标★(10-01): 目录下每篇 *.txt 按 BOS 正文 EOS 分词, 走第七版
#                 --adv-list 取料+解(每篇一行, A=1, 提示长 1 ⇒ 整篇每个 token 一条"抬 η nat"的方程), 没有 FP 教师
#   docprobe <③目录|-> <标签> <温度|-> <问题>...  挂/不挂 ③ 走 CLI 聊天模板逐题问(- = 温 0 贪心; 数字 = 采样种子 1), 原始输出落盘
#   docgate <③目录> [ntok]   守门: 同趟 v41_judge 出 ② 与 ②+③ 的 wt2 五指标(默认 512 token)
#   docflip <料目录> <η> <λ> <K> <conf-min> [标签]   ★一篮子★: doc3 → 基线/挂 ③ 逐题探针(温 0 + 两粒种子) → docgate;
#                 问题在 <料目录>/probe/questions.txt(10-01 晚, 复盘文本当 ③ 目标的翻转验证)
#   ★③ 后训练(2026-10-10 收口: 一份 jsonl 一条命令; 料两种 = messages 问答 / text 原文, 与 unsloth 同)★
#   gen <料目录> [块名正则] [轮数]   可选数据工具: 模型读 chunks/*.txt 每块出问答(synthetic-data-kit 那句提示) → <料目录>/<料名>.jsonl(带 context)
#   gensplit <料目录>   只重拆问答(gen/*.txt → jsonl), 不起服务
#   train <数据.jsonl> [轮 3] [层 0-39] [lr 2e-4] [k=v,...]   ★训 ③★: ./ds4 --ptrain → $FTD/kd-<料名>-<标签>/ → kdpick 选轮(过 wt2 门里留出损失最低)
#   kdpick <③根目录>   只重选轮;  kdprof <数据.jsonl> [步数 5] [nsys|-] [k=v,...] [二进制] [额外引擎参数]   训练计时(要现成教师表)
#   kddiag <③根目录> <ckpt_eNN|->   逐位诊断(不训练): 留出题答案逐位并排 教师 / 挂 ③ / 部署态 前 5 名 + 分叉位汇总
#   ptgate <训练器输出目录(epochs=0)> [标签] [k=v]   生成侧改动的逐字节门;  ptcheck <训练器输出目录> <k=v,...> [标签]   同配置再跑一趟开检查项(packcheck/gradcheck/计时)
#   docgate <③目录> [ntok]   守门: 同趟 v41_judge 出 ② 与 ②+③ 的 wt2 五指标(默认 512 token)
#   docprobe <③目录|-> <标签> <温度|-> <问题>...  挂/不挂 ③ 走 CLI 聊天模板逐题问, 原始输出落盘
#   (10-10 砍掉: kdinc/kdtake/kdpool/kdledger/kdfwd/kdrft/kdscore/kdrun/jsonl/kdeval/kdgengate/kdtrain 与 kd_domain/ 领域适配器 —— 账本/奖励回路/决策探针全是
#    金融 CFO 专属的定制件, 用户判为跑偏; 盘上 incr/ 产物不删只是没有命令再读它们)
set -uo pipefail
# 根 = 本脚本往上两级(源码仓库根, 或解压出来的发布包根), 不写死 ~/ds4-main: 发布包里的训练页也走这条链(10-10)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
SC="$ROOT/gguf-tools/scripts"; AMP="$ROOT/gguf-tools/amp"
# ★2026-09-13 起整条链走 DeepSeek V4.1★(用户令"不要考虑 V4 了"): 三个文件依次加载 ——
#   ① 量化 GGUF(1.519 bpw VQ + FP4 骨架)  ② 反修插件目录(逐专家逐通道增益, 冻结)
#   ③ 后训练目录(同构增益, 每晚重出; 与 ② 的表逐元素相乘, 见 core_v41_amp.c)
# V4 时代那套(ds4-fin86q8ve.gguf + zchain.bin 单文件 + --finetune 低秩拼秩)已停用, 见 back.md §1.2。
D2="$ROOT/gguf/v41/night"
# ★①② = 现役部署对★(2026-09-22 换成 v3: vq8sh14-q4k + grrb; 老的 fp4 那对已按用户令删除, 见 gguf/deleted_0922_manifest.txt)。
# 指着旧对解出的 ③ 指纹(base.fnv)对不上现役, 挂上去引擎直接拒。改 ①② 先改这里, 与 serve_1m_spark.sh 的默认值保持一致。
MDL="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"   # ① 量化模型(不改)
ZCH="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine"   # ② 反修(金融 j Same top 74.57%)
# ★工作台选了哪套就训哪套★(10-10): gguf/serve_pick.txt 在(模型页"加载"写的, 格式归 serve_1m_spark.sh pick 管)就按它 ——
#   ③ 是对着 ①② 这个态解的(base.fnv), 训练用的对与服务装的对不一致, 训完挂上去引擎直接拒。第 3 行起只拿 --engram-dir
#   (发布包里模型在 gguf/hub, engram 表不在转换机的路径上, 训练器/判决器同样要它); 侧车 none = 对着裸 ① 训; ③ 不继承(起点恒为 ①+②)。
ENG=()
if [ -s "$ROOT/gguf/serve_pick.txt" ]; then
    mapfile -t _PK < "$ROOT/gguf/serve_pick.txt"; MDL="${_PK[0]}"; ZCH="${_PK[1]:-none}"
    for ((_i = 2; _i < ${#_PK[@]}; _i++)); do [ "${_PK[_i]}" = --engram-dir ] && ENG=(--engram-dir "${_PK[_i + 1]}"); done
fi
[ "$ZCH" = none ] && ZCH=""
ZARGS=(); [ -z "$ZCH" ] || ZARGS=(--zchain "$ZCH")
FTD="$ROOT/gguf/v41/posttrain"            # ③ 后训练件版本库(pt-<日期>/)
HFDIR="$ROOT/hf/DeepSeek-V4.1-Flash"   # HF 出厂目录(后训练只用它读形状, 不读权重)
BEN="$ROOT/gguf-tools/bench"
# 守门 3"不忘老本事"的两把判决料: 金融五域 j 与通用 wt2。挂 ③ 之后在这两把尺上不许退 ——
# 后训练是为了学当天的教训, 不是为了把模型带偏。
FINJ="$ROOT/gguf/go-onebit/vqfin41/vqhalf_j.ids"
WT2="$ROOT/gguf/go-onebit/g7/wt2.ids"
BACKEND="http://192.168.2.203:8001"
SAMPLES_N=8                               # 一晚最多取几条样本(每天产 ~2 条 ≈ 四个交易日滚动窗)
RANK=16                                   # 微调秩: 学的是行为增量, 不是还原, 十几维够; 上限 1024
LAMBDA=0.05                               # 岭正则(样本少, 宁欠拟合不过拟合)
MIN_GAIN=0.02                             # 层门槛: held-out 残差能量下降比, 不到就不注入该层
LOGF="$D2/nightly.log"
LOG(){ echo "[znight $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGF"; }
DIE(){ LOG "★$*★"; exit 1; }
mkdir -p "$D2" "$FTD"
need_idle(){ local b; b=$(for p in ds4 ds4-bench ds4quant_run zlayer vq_merge_v4 finetune_solve v41_amp_run; do pgrep -x "$p"; done)
             [ -z "$b" ] || DIE "机器非空(实例锁): $b"; }

# ---------------- ① 样本 → ids ----------------
# ids 两条: 教师 [上下文 ‖ 正文], 学生 [正文]。上下文全拼在前面 = 只需要一个偏移区间
# (row_layout 的 xshift 只支持一段), 语义上也顺: 教师先读完当日复盘, 再看这几份材料。
# ★训练/判决必须隔离★(2026-09-10 用户点破): 此前 samples 与 nll 两段各自拉同一个 API,
# 于是解算用的样本和判决用的样本是同一批 —— 判据里"真泛化"和"记住了这几篇报告"分不开。
# 现在一次拉全量, ★按交易日切★: 最晚那一个交易日的全部样本留作判决, 其余日子训练。
# 为什么从"最近 N 条"改成"最晚一天"(2026-09-14, back.md 第四版 §2.4): 12 条样本只有两个
# 交易日(09-04 三条 / 09-07 九条), 按条切出来的 4 条判决集全是 09-07, 而训练集里也有 5 条
# 09-07 —— "没见过的那天"其实见过了(同一天不同股票, 同一批复盘教训), 判决尺自带泄题。
# 按天切还顺带把判决集从 4 条变成 9 条, 决策点多一倍, 尺的分辨率从 1.5pp 提到 ~0.7pp。
# 切分单独一段: 改切法(按条 → 按天)不必重拉 API, 也不必重跑分词(那要加载一次模型)。
# samples 段拉完料就调它; 手上已经有 samples_all.json 时直接 `z_nightly_spark.sh split`。
stage_split(){
    [ -s "$D2/samples_all.json" ] || DIE "没有 $D2/samples_all.json — 先跑 samples"
    python3 - "$D2" <<'PY' 2>&1 | tee -a "$LOGF" || DIE "训练/判决切分失败"
import json, os, sys
d = sys.argv[1]
items = json.load(open(os.path.join(d, "samples_all.json")))["items"]
# 按日期升序: 早的在前。切点 = 最晚那个交易日的第一条 ⇒ 判决集 = 整整一天。
items.sort(key=lambda x: (x.get("date") or "", x.get("symbol") or ""))
days = sorted({(x.get("date") or "") for x in items})
if len(days) < 2:
    print("只有 %d 个交易日(%s), 切不出'没见过的一天'" % (len(days), ",".join(days))); sys.exit(1)
last = days[-1]
tr = [x for x in items if (x.get("date") or "") != last]
te = [x for x in items if (x.get("date") or "") == last]
json.dump({"items": tr}, open(os.path.join(d, "samples.json"), "w"), ensure_ascii=False)
json.dump({"items": te}, open(os.path.join(d, "eval_samples.json"), "w"), ensure_ascii=False)
print("★按交易日切★: 训练 %d 条 / %d 天(%s) | 判决 %d 条 / 1 天(%s), 解算一行都不看"
      % (len(tr), len(days) - 1, ",".join(days[:-1]), len(te), last))
PY
    return ${PIPESTATUS[0]}
}

stage_samples(){
    local N="${1:-$SAMPLES_N}"
    LOG "① 拉样本(最多 $N 条) ← $BACKEND"
    curl -sf -m 60 "$BACKEND/api/review/finetune-samples?limit=$N&full=true" -o "$D2/samples_all.json" \
        || DIE "拉样本失败(Mac 后端没起? ./start.sh --backend)"
    stage_split || DIE "切分失败"
    python3 - "$D2" <<'PY' || DIE "样本切文本失败"
import json, os, sys
d = sys.argv[1]
items = json.load(open(os.path.join(d, "samples.json")))["items"]
ctx, body = [], []
for it in items:
    c = (it.get("context") or "").strip()
    p = (it.get("body_prompt") or "").strip()
    r = (it.get("body_report") or "").strip()
    if c and r:
        ctx.append(c); body.append(p + "\n" + r)
if not body:
    print("样本为空或都缺字段"); sys.exit(1)
open(os.path.join(d, "ctx.txt"), "w").write("\n".join(ctx) + "\n")
open(os.path.join(d, "body.txt"), "w").write("\n\n".join(body) + "\n")
print("样本 %d 条: 上下文 %d 字, 正文 %d 字" % (len(body), sum(map(len, ctx)), sum(map(len, body))))
PY
    # 分词走引擎自己的 tokenizer(与部署同一份)。两段分开分词再拼, 段边界才精确可控。
    # ★只取方括号里那串★: --dump-tokens 之后还会打一张"id 原文"对照表, 按行 grep 数字会收两遍。
    for seg in ctx body; do
        ./ds4 --cuda -m "$MDL" --dump-tokens --prompt-file "$D2/$seg.txt" 2>/dev/null > "$D2/$seg.dump"
        python3 - "$D2/$seg.dump" "$D2/$seg.ids" <<'PY' || DIE "$seg 分词失败"
import re, sys
m = re.search(r"\[([0-9,\s]+)\]", open(sys.argv[1], encoding="utf-8", errors="replace").read())
if not m:
    print("dump-tokens 没有 id 列表"); sys.exit(1)
ids = [x.strip() for x in m.group(1).split(",") if x.strip()]
open(sys.argv[2], "w").write("\n".join(ids) + "\n")
print("%s → %d token" % (sys.argv[2].split("/")[-1], len(ids)))
PY
        [ -s "$D2/$seg.ids" ] || DIE "$seg 分词结果为空"
    done
    python3 - "$D2" <<'PY' || DIE "ids 拼装失败"
import os, sys
d = sys.argv[1]
ctx = [l.strip() for l in open(os.path.join(d, "ctx.ids")) if l.strip()]
body = [l.strip() for l in open(os.path.join(d, "body.ids")) if l.strip()]
open(os.path.join(d, "ids_t.txt"), "w").write("\n".join(ctx + body) + "\n")
open(os.path.join(d, "ids_s.txt"), "w").write("\n".join(body) + "\n")
# 布局: 解算器只要 xshift(行偏移) + hindsight(正文块, 教师行号)。
open(os.path.join(d, "ids_t.txt.layout"), "w").write(
    "# 夜间微调行布局 — z_nightly_spark.sh samples 产出, 勿手改\n"
    "# xshift <S0> <N>: 锚行 [S0,S0+N) 是教师独有的事后上下文, 学生没有\n"
    "win 128\nhindsight %d %d\nxshift 0 %d\n" % (len(ctx), len(body), len(ctx)))
open(os.path.join(d, "rows.env"), "w").write(
    "S_T=%d\nS_S=%d\nXN=%d\nBODY_LO=%d\nBODY_N=%d\n"
    % (len(ctx) + len(body), len(body), len(ctx), len(ctx), len(body)))
print("教师 %d 行 = 上下文 %d + 正文 %d; 学生 %d 行" % (len(ctx) + len(body), len(ctx), len(body), len(body)))
PY
    . "$D2/rows.env"
    LOG "① ids 就绪: 教师 $S_T 行 / 学生 $S_S 行(上下文 $XN 行只给教师)"
}

# ---------------- 统一打分入口(V4.1) ----------------
# V4.1 引擎只有 --score-ids 这一条 teacher-forced 打分路(分块 512 的 prefill, 与部署同路);
# V4 的 --eval-ids 批路属于另一套会话实现, V4.1 没接, 别照抄老命令。
# 三个小出口(2026-09-13 落地): --score-nll 逐位 NLL f32[S] / --score-topk K 逐位 top-K /
# --score-no-logits 不写全词表 logits。★必须带最后这个★: 每位置 517 KB, 2 万行就是 10 GB,
# 而 GB10 是统一内存 —— 写大文件 = 掏 GPU 内存(09-08 夜就是这么把机器写崩的)。
# --score-rms 第四个小出口(09-13 夜): 每位置一个 inv = 出口 RMSNorm 的 rsqrt 标量。后训练第二版
# 要它把"增益改动"换算成"logit 差改动"(back.md §4.1); 判决路用不上, 传空就不写。
#   $1 ids 文件  $2 nll 输出  $3 topk 输出  $4 日志  $5 后训练目录(空=只挂 ①+②)  $6 rms 输出(空=不写)
score_pass(){
    # $7 = 部署同路切分点(提示长度; 空 = 老口径)。后训练的训练表与验证表必须同一口径(v41_sft_run.inc.c 取料按它切)
    local ids="$1" nll="$2" top="$3" log="$4" pt="${5:-}" rms="${6:-}" split="${7:-}"
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" ${pt:+--posttrain "$pt"} --mem-budget-mb 110000 \
        --score-ids "$ids" --score-no-logits --score-nll "$nll" --score-topk 64 "$top" ${split:+--score-split "$split"} \
        ${rms:+--score-rms "$rms"} > "$log" 2>&1 </dev/null || { tail -8 "$log"; DIE "打分失败(见 $log)"; }
    # ★查产出, 不只查退出码★: 实例锁拒启动那次进程也"正常"返回, 产物却是空的(09-10 实撞)
    [ -s "$nll" ] && [ -s "$top" ] || DIE "打分没产出($nll / $top)"
    [ -z "$rms" ] || [ -s "$rms" ] || DIE "打分没产出 rms($rms)"
}

# ---------------- ② 两遍捕获(部署同路) ----------------
# 引擎 --cap-dir 逐层落 raw_ffn_in(x) 与 raw_ffn_out(routed, ★zchain 之后★),
# 正是"量化+zchain"那个部署态的真值 —— 不需要 FP 教师, 不需要建锚。
stage_capture(){
    DIE "教师−学生中间激活差的老靶(09-08 定罪: 靶错/尺错/类型错) —— V4 口径, 已停用(back.md §1.2)"
    local CAPL="${1:-}"          # 层范围 lo-hi; 空=全 43 层
    [ -s "$D2/ids_t.txt" ] || DIE "先跑 samples"
    . "$D2/rows.env"
    LOG "② 停服(清理=关进程, 不删文件)"; bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    for role in t s; do
        local ids="$D2/ids_$role.txt" dir="$D2/cap_$role"
        rm -rf "$dir"; mkdir -p "$dir"
        # score-out 丢 /dev/null: 本段只要逐层捕获, 那份 logits(每遍约 4 GB)没有消费者。
        LOG "② ${role} 遍捕获($(wc -l < "$ids") 行, 解码路约 20 t/s) → $dir"
        # 上下文没有参数(2026-09-22): 以前按 ids 行数 ×1.1 传 --ctx 是 V4 会话的事(09-08 实撞: 会话默认 32768, 喂到
        # 第 32779 行静默截断); V4.1 打分路按 ids 行数分配状态, 上下文只有 1M 一个取值, --ctx 已不存在。
        ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 \
            --score-ids "$ids" --score-out /dev/null --cap-dir "$dir" ${CAPL:+--cap-layers "$CAPL"} \
            > "$D2/cap_$role.log" 2>&1 </dev/null || { tail -5 "$D2/cap_$role.log"; DIE "${role} 遍失败"; }
        local n want; n=$(ls "$dir"/raw_ffn_out_L* 2>/dev/null | wc -l)
        want=43; [ -n "$CAPL" ] && want=$(( ${CAPL#*-} - ${CAPL%-*} + 1 ))
        [ "$n" -eq "$want" ] || DIE "${role} 遍捕获不齐 $n/$want"
        LOG "② ${role} 遍 ✓ $n/$want 层 ($(du -sh "$dir" | cut -f1))"
    done
}

# ---------------- ③ 解微调 ----------------
stage_solve(){
    DIE "同上, 低秩 z^L 拼秩形态(V4 专用) —— V4 口径, 已停用(back.md §1.2)"
    [ -d "$D2/cap_t" ] && [ -d "$D2/cap_s" ] || DIE "先跑 capture"
    need_idle
    [ -x "$AMP/finetune_solve" ] || make -C "$ROOT/gguf-tools" finetune_solve >>"$LOGF" 2>&1
    local rank="${1:-$RANK}" tag; tag=$(date +%Y%m%d)
    LOG "③ 解微调(rank $rank λ $LAMBDA 门槛 $MIN_GAIN)"
    # ★判成败看解算器的退出码, 不看文件在不在★(2026-09-08 实撞): 一层都没过门槛时它照样
    # 写了个只有 43 个空层头的 352 字节壳子, `[ -s ]` 会当成功, 于是把空壳软链成 current ——
    # 下游就"部署了一个什么都不做的微调"。管道里还有 tee, 所以用 PIPESTATUS 取解算器本身的码。
    set -o pipefail
    "$AMP/finetune_solve" --teacher "$D2/cap_t" --student "$D2/cap_s" \
        --layout "$D2/ids_t.txt.layout" --out "$FTD/finetune_$tag.bin" \
        --rank "$rank" --lambda "$LAMBDA" --min-gain "$MIN_GAIN" 2>&1 | tee -a "$LOGF"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || { rm -f "$FTD/finetune_$tag.bin"; DIE "解算未产出可用微调(退出码 $rc) — 今晚不上线, 保持 zchain 原样"; }
    ln -sfn "$FTD/finetune_$tag.bin" "$FTD/current.bin"
    LOG "③ 微调文件 ✓ $(ls -l "$FTD/finetune_$tag.bin" | awk '{printf "%.1f MB", $5/1e6}')"
}

# ---------------- ★段 1: 尺 C 上下文天花板(back.md 第四版 §3.2)★ ----------------
# 问的是一件很朴素的事: 把复盘里人写的教训【原文摆进上下文】, 模型在没见过的那一天的
# 错误判断上, 改判几个? 这个数是任何后训练文件 ③ 的上界 —— ③ 无非是把"记忆段在上下文里"
# 这件事压进权重, 压得再好也超不过原本摆着看的效果。
# ★为什么必须先跑它★(09-14 第三版判死之后): 第二/三版让解算器拟合的是"这一位该写 13 不是 14",
# 而复盘真正教的是四五条跨股票复用的规则(ADX>50 别按涨停价定目标 / 震荡市黄金分割不可信 ...),
# 那些规则写在 edits[].why 里, 前三版一次都没喂进去。C 不过门 ⇒ 要改的是复盘写法或数据量,
# 不是解算器, 那时候写一行解算代码都是浪费。
# 两档教师: rules = 只放 why 原文; cases = why + 原判/修正首行。谁高谁当教师(§3.1)。
# 顺带一针 C′(自泄题上界): 训练日样本读【含自己那条】的记忆段 —— 它到不了 100% 的那部分,
# 是模型根本表达不出来的(人算出来的具体数字), 那些位置进"只报不判"桶(§4.1)。
# 全部走 --verify-tabs: 一次加载跑完所有序列, 不是每条起一次 ./ds4(那样光加载就 107s×N)。
#   $1 = 只取前 N 条判决样本(0=全部; 冒烟传 2)
CEIL_GATE=4.5      # 可判桶的 ΔC 门(= 3 个决策点; back.md §4.1)
# 记忆段拼接单独一段: 尺 C(teacher-forced)与自由生成尺都要它, 两处必须是同一份文本,
# 否则两把尺量的不是同一个教师。
#   $1 = 落盘目录
stage_mem(){
    local W="${1:?目录}"
    [ -s "$D2/samples.json" ] || DIE "先跑 samples/split"
    mkdir -p "$W"
    python3 - "$D2" "$W" <<'PYEOF' 2>&1 | tee -a "$LOGF" || DIE "记忆段拼接失败"
import json, os, sys
d, w = sys.argv[1], sys.argv[2]
def one(s, n=70):
    s = " ".join((s or "").split())
    return s[:n] + ("…" if len(s) > n else "")
items = json.load(open(os.path.join(d, "samples.json")))["items"]
seen, rules, cases = set(), [], []
for it in items:
    dt, sym = it.get("date") or "?", it.get("symbol") or "?"
    for e in (it.get("edits") or []):
        why = " ".join((e.get("why") or "").split())
        if not why or why in seen:      # 同一条教训在一份报告里会被每处派生编辑重复写一遍
            continue
        seen.add(why)
        rules.append("- [%s %s] %s" % (dt, sym, why))
        cases.append("- [%s %s] 原判「%s」→ 修正「%s」。原因：%s" % (dt, sym, one(e.get("old")), one(e.get("new")), why))
if not rules:
    print("训练日样本里没有 edits[].why —— 记忆段无从拼起"); sys.exit(1)
HEAD = "【复盘记忆】以下是此前交易日复盘中人工纠正的判断，撰写本报告时必须遵守：\n"
open(os.path.join(w, "mem_rules.txt"), "w").write(HEAD + "\n".join(rules) + "\n")
open(os.path.join(w, "mem_cases.txt"), "w").write(HEAD + "\n".join(cases) + "\n")
print("  记忆段: %d 条教训(去重后) | rules %d 字 / cases %d 字"
      % (len(rules), len(HEAD) + sum(len(x) + 1 for x in rules), len(HEAD) + sum(len(x) + 1 for x in cases)))
PYEOF
    return ${PIPESTATUS[0]}
}

stage_ceiling(){
    local NMAX="${1:-0}" W="$D2/ceil"
    [ -s "$D2/eval_samples.json" ] || DIE "先跑 samples(判决集还没切出来)"
    [ -x "$AMP/v41_amp_run" ] || make -C "$ROOT/gguf-tools" v41_amp_run >>"$LOGF" 2>&1 || DIE "取表器编译失败"
    mkdir -p "$W/out"
    LOG "C① 拼记忆段(训练日复盘 why 原文, 确定性拼接, 零改写)"
    stage_mem "$W" || DIE "记忆段失败"
    LOG "C② 一次分词(记忆段两档 + 判决日样本 + 训练日样本)"
    python3 - "$D2" "$W" "$NMAX" <<'PYEOF' || DIE "分词输入拼装失败"
import json, os, sys
d, w, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
SEP = "<｜end▁of▁sentence｜>"
ev = json.load(open(os.path.join(d, "eval_samples.json")))["items"]
tr = json.load(open(os.path.join(d, "samples.json")))["items"]
if nmax > 0: ev, tr = ev[:nmax], tr[:nmax]
segs = [open(os.path.join(w, "mem_rules.txt")).read(), open(os.path.join(w, "mem_cases.txt")).read()]
keep = []
for tag, items in (("e", ev), ("t", tr)):
    for k, it in enumerate(items):
        p = (it.get("body_prompt") or "").strip()
        r = (it.get("body_report") or "").strip()
        g = (it.get("orig_report") or "").strip()
        if p and r and g: keep.append((tag, k, p, r, g))
if not keep: print("样本缺 body_prompt/body_report/orig_report"); sys.exit(1)
for _, _, p, _, _ in keep: segs.append(p)
for _, _, _, r, _ in keep: segs.append(r)
for _, _, _, _, g in keep: segs.append(g)
open(os.path.join(w, "ceil_all.txt"), "w").write(SEP + SEP.join(segs))
json.dump([[t, k] for t, k, _, _, _ in keep], open(os.path.join(w, "keep.json"), "w"))
print("  分词输入: 记忆段 2 段 + 样本 %d 条(判决 %d / 训练 %d)"
      % (len(keep), sum(1 for x in keep if x[0] == "e"), sum(1 for x in keep if x[0] == "t")))
PYEOF
    ./ds4 --cuda -m "$MDL" --dump-tokens --prompt-file "$W/ceil_all.txt" 2>/dev/null > "$W/ceil_all.dump" || DIE "分词失败"
    LOG "C③ 组装三态序列(学生 / 教师-rules / 教师-cases) + 决策点分桶"
    python3 - "$W" <<'PYEOF' 2>&1 | tee -a "$LOGF" || DIE "序列组装失败"
import json, os, re, sys, difflib
w = sys.argv[1]
raw = open(os.path.join(w, "ceil_all.dump"), encoding="utf-8", errors="replace").read()
m = re.search(r"\[([0-9,\s]+)\]", raw)
if not m: print("dump-tokens 没有 id 列表"); sys.exit(1)
ids = [int(x) for x in m.group(1).split(",") if x.strip()]
# id → 原文: dump 在 id 列表之后还打一张"id 原文"对照表, 分桶要靠它认"这个 token 是不是数字"
txt = {}
for ln in raw[m.end():].splitlines():
    mm = re.match(r"^\s*(\d+)\s\s(.*)$", ln)
    if mm and int(mm.group(1)) not in txt: txt[int(mm.group(1))] = mm.group(2)
sep = ids[0]
chunks, cur = [], []
for t in ids[1:]:
    if t == sep: chunks.append(cur); cur = []
    else: cur.append(t)
chunks.append(cur)
keep = json.load(open(os.path.join(w, "keep.json")))
N = len(keep)
if len(chunks) != 2 + 3 * N: print("切分段数 %d ≠ 2+3×%d" % (len(chunks), N)); sys.exit(1)
mem = {"rules": chunks[0], "cases": chunks[1]}
pro, right, wrong = chunks[2:2+N], chunks[2+N:2+2*N], chunks[2+2*N:]
CTXMAX = 32768
def s(t): return txt.get(t, "")
def dig(x): return any(c.isdigit() for c in x)
NUM = re.compile(r"\d+\.\d+|\d+")
lines, items, buckets, maxlen, stats = [], [], [], 0, {}
for i, (tag, k) in enumerate(keep):
    a = len(pro[i])
    sm = difflib.SequenceMatcher(None, right[i], wrong[i], autojunk=False)
    # ★对手序列按 diff 对齐, 不按行号硬取★(09-14 修的口径): 判决器读的是 alt[row+1], 即"错版在
    # 这一位写的是谁"。原先直接把整条错版序列当 alt, 只有第一处改动的行号是对齐的 —— 从第二处起
    # 两版长度已经不同, alt[row+1] 取到的是错版里毫不相干的一个 token。于是第二处之后的决策点,
    # "是不是决策点"和"离翻转还有多远"两个数都是错的。这里改成: alt = 对版序列的副本, 只在每处
    # 改动的块首放上【错版在该处真正写的那个 token】, 其余位置与对版逐位相同(=判决器自动跳过)。
    alt_body, blocks = list(right[i]), []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal" and i2 > i1:
            if j1 < len(wrong[i]): alt_body[i1] = wrong[i][j1]
            blocks.append((a + i1 - 1, "".join(s(t) for t in right[i][i1:i2]), s(right[i][i1])))
    if not blocks:
        print("  样本 %s%d 两版一样, 跳过" % (tag, k)); continue
    # ★分桶★(back.md §4.2): 根决策 = 这份报告的第一处改动(目标价/基准值本身); 派生 = 后面那些
    # 把根决策的新数字代进算术的位置(收益率/风报比/JSON), 它量的是模型的算术不是教训, 只报不判。
    root_nums = set(NUM.findall(blocks[0][1]))
    bk = []
    for q, (row, body, first) in enumerate(blocks):
        if q == 0: b = "root_num" if dig(first) else "root_txt"
        elif root_nums & set(NUM.findall(body)): b = "derived"
        elif dig(body): b = "num_other"
        else: b = "wording"
        bk.append((row, b))
        stats[b] = stats.get(b, 0) + 1
    # 三态: 学生(什么都不读) / 教师-rules / 教师-cases。教师序列 = 记忆段拼在最前面, 行号整体右移。
    vs = [("stu", [])] + ([("rules", mem["rules"]), ("cases", mem["cases"])] if tag == "e" else [("cases", mem["cases"])])
    for vn, pre in vs:
        sh = len(pre)
        seq_r, seq_w = pre + pro[i] + right[i] + [sep], pre + pro[i] + alt_body + [sep]
        if max(len(seq_r), len(seq_w)) > CTXMAX:
            print("  样本 %s%d/%s 太长(%d > %d), 跳过" % (tag, k, vn, len(seq_r), CTXMAX)); continue
        key = "%s_%s%d" % (vn, tag, k)
        fr = os.path.join(w, "ids_%s_right.txt" % key); fw = os.path.join(w, "ids_%s_wrong.txt" % key)
        open(fr, "w").write("\n".join(map(str, seq_r)) + "\n")
        open(fw, "w").write("\n".join(map(str, seq_w)) + "\n")
        allr = os.path.join(w, "rows_%s_all.txt" % key)
        open(allr, "w").write("\n".join(str(r + sh) for r, _ in bk) + "\n")
        for b in sorted({x for _, x in bk}):
            rf = os.path.join(w, "rows_%s_%s.txt" % (key, b))
            open(rf, "w").write("\n".join(str(r + sh) for r, bb in bk if bb == b) + "\n")
            buckets.append("%s %s %s %d" % (key, b, rf, sum(1 for _, bb in bk if bb == b)))
        tab = os.path.join(w, "c_tab_%s.bin" % key)      # --verify-tabs c_ 会在文件名前加前缀
        lines.append("%s %s %s %s 0:1 @%s" % (fr, os.path.join(w, "tab_%s.bin" % key),
                                              os.path.join(w, "rms_%s.bin" % key), fw, allr))
        items.append("%s %s %s %s %s" % (key, vn, tag + str(k), tab, fw))
        maxlen = max(maxlen, len(seq_r))
    print("  样本 %s%d: 报告 %d token | 决策点 %d(%s)"
          % (tag, k, len(right[i]), len(bk), " ".join("%s=%d" % (b, sum(1 for _, x in bk if x == b)) for b in sorted({x for _, x in bk}))))
if not lines: print("没有可用序列"); sys.exit(1)
open(os.path.join(w, "list.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(w, "items.txt"), "w").write("\n".join(items) + "\n")
open(os.path.join(w, "buckets.txt"), "w").write("\n".join(buckets) + "\n")
open(os.path.join(w, "meta.env"), "w").write('MAXLEN=%d\nFIRST="%s"\nNSEQ=%d\n'
                                             % (maxlen + 8, lines[0].split()[0], len(lines)))
print("  ★共 %d 条序列(最长 %d token), 决策点分桶: %s★"
      % (len(lines), maxlen, " ".join("%s=%d" % (b, n) for b, n in sorted(stats.items()))))
PYEOF
    . "$W/meta.env"
    LOG "C④ 停服 → 一次加载跑完 $NSEQ 条序列(只出 top-K 表, 不取料)"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    set -o pipefail
    "$AMP/v41_amp_run" "$MDL" "$HFDIR" "$FIRST" "$MAXLEN" "$W/out" \
        --only-layer 39 --sft-list "$W/list.txt" --verify-tabs c_ \
        --base-amp "$ZCH" --mem-budget-mb 110000 2>&1 | tee -a "$LOGF" | tail -30
    [ "${PIPESTATUS[0]}" = 0 ] || DIE "尺 C 出表失败(见 $LOGF)"
    LOG "C⑤ 判决: 逐样本逐桶比 argmax(不再跑前向)"
    local V="$W/verdict.txt"; : > "$V"
    local key vn smp tab alt b rf n
    while read -r key vn smp tab alt; do
        [ -s "$tab" ] || DIE "缺表 $tab(出表那一步没写?)"
        while read -r k2 b rf n; do
            [ "$k2" = "$key" ] || continue
            [ "$n" -gt 0 ] || continue
            "$BEN/anchor_metrics" --ref-topk "$tab" --rows "@$rf" --alt "$alt" > "$V.tmp" 2>&1 \
                || { cat "$V.tmp"; DIE "判决器失败: $key $b"; }
            grep -h "决策点" "$V.tmp" | sed "s|^|$vn $smp $b |" >> "$V"
        done < "$W/buckets.txt"
    done < "$W/items.txt"
    python3 - "$V" "$CEIL_GATE" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import re, sys
rows = {}
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    p = ln.split()
    m = re.search(r"決?决策点\(两版不同的位置\) (\d+) 个: argmax 已站到目标一侧 (\d+) 个 = [\d.]+%; "
                  r"平均 p\(目标\)−p\(另一版\) = ([+-][\d.]+)", ln)
    if len(p) < 3 or not m: continue
    vn, smp, b = p[0], p[1], p[2]
    n, h, dp = int(m.group(1)), int(m.group(2)), float(m.group(3))
    a = rows.setdefault((vn, "eval" if smp.startswith("e") else "train", b), [0, 0, 0.0])
    a[0] += h; a[1] += n; a[2] += dp * n
def g(v, s, b):
    a = rows.get((v, s, b)); return a if a else None
BK = ["root_txt", "root_num", "derived", "num_other", "wording"]
def pct(a): return 100.0 * a[0] / a[1] if a[1] else 0.0
print("\n★尺 C(判决日, 解算一行都没跑): 记忆段摆进上下文之后, 错误判断改判了几个★")
print("  桶            | 点数 |  学生   | 教师rules | 教师cases |  Δ(最好档)")
best = {}
for b in BK:
    st = g("stu", "eval", b)
    if not st: continue
    r_, c_ = g("rules", "eval", b), g("cases", "eval", b)
    dr = pct(r_) - pct(st) if r_ else float("nan")
    dc = pct(c_) - pct(st) if c_ else float("nan")
    d = max([x for x in (dr, dc) if x == x] or [0.0])
    best[b] = d
    print("  %-13s | %4d | %6.2f%% | %8.2f%% | %8.2f%% | %+6.2fpp"
          % (b, st[1], pct(st), pct(r_) if r_ else 0.0, pct(c_) if c_ else 0.0, d))
    print("      平均 p(对版)−p(错版): 学生 %+.4f | rules %+.4f | cases %+.4f"
          % (st[2]/st[1] if st[1] else 0, (r_[2]/r_[1]) if r_ and r_[1] else 0, (c_[2]/c_[1]) if c_ and c_[1] else 0))
# 可判桶 = 根决策(文字/数字) + 其余独立数字改动(止损位这类, 有自己的教训, 不是根决策的算术派生)。
# 派生位量的是模型的算术, 措辞位量的是行文习惯, 两者只报不判。
jn = [b for b in ("root_txt", "root_num", "num_other") if b in best]
js = [0, 0]; jt = {"rules": [0, 0], "cases": [0, 0]}
for b in jn:
    a = g("stu", "eval", b); js[0] += a[0]; js[1] += a[1]
    for v in jt:
        x = g(v, "eval", b)
        if x: jt[v][0] += x[0]; jt[v][1] += x[1]
if not js[1]:
    print("\n★可判桶(根决策)一个点都没有 — 分桶器或判决集有问题, 停车★"); sys.exit(2)
base = 100.0 * js[0] / js[1]
dd = {v: (100.0 * a[0] / a[1] - base) if a[1] else float("nan") for v, a in jt.items()}
bv = max(dd, key=lambda v: dd[v] if dd[v] == dd[v] else -99)
print("\n★可判桶(%s) %d 个点: 学生 %.2f%% → 最好档 %s %.2f%% = ΔC %+.2fpp (门 +%.1fpp)★"
      % ("+".join(jn), js[1], base, bv, base + dd[bv], dd[bv], float(sys.argv[2])))
for b in ("derived", "wording"):
    if b in best: print("  (只报不判) %s Δ %+.2fpp" % (b, best[b]))
tn = [0, 0]; tc = [0, 0]
for b in jn:
    a, c = g("stu", "train", b), g("cases", "train", b)
    if a: tn[0] += a[0]; tn[1] += a[1]
    if c: tc[0] += c[0]; tc[1] += c[1]
if tn[1]:
    print("★C′ 自泄题上界(训练日读含自己那条的记忆段, 可判桶 %d 点): %.2f%% → %.2f%%★"
          % (tn[1], 100.0 * tn[0] / tn[1], 100.0 * tc[0] / tc[1] if tc[1] else 0.0))
    print("  它到不了 100% 的那部分 = 模型根本表达不出来的位置(人算出来的具体数字), 尺 A 的分母按它重定")
ok = dd[bv] >= float(sys.argv[2])
print("\n★段 1 判决: %s★" % ("ΔC 过门 ⇒ 进段 2(解 ③, 教师档 = %s)" % bv if ok else
      "ΔC 不过门 ⇒ 停车。规则摆在眼前都不改判, 要改的是复盘写法(why 要写成'条件→动作')或数据量, 不是解算器 —— 段 2 一行代码都别写"))
sys.exit(0 if ok else 1)
PYEOF
    return ${PIPESTATUS[0]}
}

# ---------------- ★段 1′: 自由生成尺(09-14 尺 C 判死之后的新口径)★ ----------------
# 尺 C 为什么不够用: 它是 teacher-forced 的, 决策点 = 人改动的第一个 token, 而人几乎只改【结论】。
# 实测 002104: 结论 "最终目标价：13.51元" 的前文是十条子项论证, 里面写了五遍 14.36, 人一个字没改,
# 新理由还放在结论【后面】。于是那一位上模型看到的是"14.36×5", 要它写 13.51 等于要它跟自己刚写的
# 十行数字矛盾 —— 那不是权重学不学得会的问题, 学会了也只是"无视前文", 迁移不出去。
# 实测代价: 尺 C 可判桶只 +1.92pp, 根决策桶 0.00%→0.00%(连把自己那天的答案摆眼前都不改判)。
#
# 这一段换成部署里真实发生的事: 前缀只喂到【目标价那一小节的标题之前】, 让模型★自己写★这一节的
# 论证与结论, 贪心解码, 再用同一个抽取器从三处(生成/对版/错版)抽目标价, 比数值离谁近。
# 论证是模型自己写的 ⇒ 结论与论证天然一致, 没有前文锁死; 记忆段要是真起作用, 它会改的是论证。
#   $1 = 只取前 N 条判决样本(0=全部; 第一针传 3)   $2 = 生成上限 token(默认 500)
# ★为什么上限只给 500★(09-14 实测): 这条 1.5bpw VQ 态的解码只有 1.60 t/s(日志里跟着一行
# "VQ 码本 4096×16 B 进不了(回全局 gather)"), 放开写模型会一路写到"执行纪律"727 token 才自己停,
# 光解码就 7.6 分钟。判决只看目标价那一小节(250~350 token), 写到下一个小节标题就够了。
stage_freegen(){
    local NMAX="${1:-0}" GEN="${2:-500}" W="$D2/free"
    [ -s "$D2/eval_samples.json" ] || DIE "先跑 samples/split"
    mkdir -p "$W"
    LOG "G① 记忆段(与尺 C 同一份文本)"
    stage_mem "$W" || DIE "记忆段失败"
    LOG "G② 造前缀(截到目标价小节标题之前) + 记下两版小节的边界"
    python3 - "$D2" "$W" "$NMAX" <<'PYEOF' 2>&1 | tee -a "$LOGF" || DIE "前缀构造失败"
import json, os, re, sys
d, w, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
ev = json.load(open(os.path.join(d, "eval_samples.json")))["items"]
if nmax > 0: ev = ev[:nmax]
mem = {v: open(os.path.join(w, "mem_%s.txt" % v)).read().rstrip() + "\n\n" for v in ("rules", "cases")}
rows = []
for i, it in enumerate(ev):
    p = (it.get("body_prompt") or "").strip()
    r = (it.get("body_report") or "").strip()
    g = (it.get("orig_report") or "").strip()
    if not (p and r and g): continue
    n = min(len(r), len(g)); k = 0
    while k < n and r[k] == g[k]: k += 1          # 两版第一个分歧字符
    # 截断点 = 第一分歧之前最后一个 "### " 级小节标题的行首。为什么取三井号而不是最近的任意标题:
    # 有的报告在 "#### 子项分析" 里就把论证写完了, 截在四井号标题上等于把论证留在前缀里(又锁死)。
    def cut_of(t, kk):
        hs = [m.start() for m in re.finditer(r"(?m)^### ", t[:kk])] or [m.start() for m in re.finditer(r"(?m)^## ", t[:kk])]
        return hs[-1] if hs else -1
    cr, cg = cut_of(r, k), cut_of(g, k)
    if cr < 0 or cg < 0:
        print("  样本 %d(%s) 找不到小节标题, 跳过" % (i, it.get("symbol"))); continue
    # ★必须是 completions 续写口径, 不是 chat★(09-14 实撞): 直接喂文本时引擎走 ds4_encode_chat_prompt,
    # 把整段当成用户消息, 模型会【从头另写一份报告】而不是接着这半份往下写 —— 那量的根本不是同一件事。
    # cli_gen.c:is_rendered_chat_prompt: 提示以 BOS 开头就原样分词、直接续写。前缀布局与尺 C 的
    # ids 一致(材料 + 报告前半, 无 chat 标记), 两把尺才可比。
    # ★前缀要【含小节标题那一行】★(09-14 实撞): 切在标题之前时, 模型续写直接跳到 "### 4.2 止损价",
    # 根本不写目标价这一节, 抽不出数来。标题行本身不含结论数字, 留着不泄题。
    nl = r.find("\n", cr)
    head_end = (nl + 1) if nl >= 0 else cr
    for v in ("stu", "rules", "cases"):
        open(os.path.join(w, "prompt_%s_e%d.txt" % (v, i)), "w").write(
            "<｜begin▁of▁sentence｜>" + ("" if v == "stu" else mem[v]) + p + "\n" + r[:head_end])
    rows.append("e%d %s %d %d" % (i, it.get("symbol"), cr, cg))
    print("  样本 e%d %s: 前缀 %d 字(报告截去后 %d 字) | 截断处: %s"
          % (i, it.get("symbol"), len(p) + cr, len(r) - cr, r[cr:cr+18].replace("\n", " ")))
if not rows: print("没有可用判决样本"); sys.exit(1)
open(os.path.join(w, "cuts.txt"), "w").write("\n".join(rows) + "\n")
print("  ★%d 条样本 × 3 态 = %d 次自由生成★" % (len(rows), 3 * len(rows)))
PYEOF
    # ★清掉上一轮的生成★: 本轮某条失败时脚本会停车, 但残留的旧 gen_* 会被判决段当成本轮产物读走
    rm -f "$W"/gen_*.txt "$W"/gen_*.err
    LOG "G③ 停服 → 逐条贪心生成(--temp 0 --seed 1 -n $GEN, 挂 ①+②)"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    local smp sym rest v n=0
    while read -r smp sym rest; do
        for v in stu rules cases; do
            local out="$W/gen_${v}_${smp}.txt"
            n=$((n+1))
            LOG "G③ 生成 $n: $smp($sym) 态 $v"
            ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --temp 0 --seed 1 \
                -n "$GEN" --prompt-file "$W/prompt_${v}_${smp}.txt" > "$out" 2>"$out.err" </dev/null \
                || { tail -5 "$out.err"; DIE "生成失败 $smp/$v"; }
            [ -s "$out" ] || DIE "生成没产出 $out"
        done
    done < "$W/cuts.txt"
    stage_freejudge
}

# 判决单独一段: 改抽取口径不必重烧生成(一趟生成 3~4 分钟, 9 趟就是半小时)。
stage_freejudge(){
    local W="$D2/free"
    [ -s "$W/cuts.txt" ] || DIE "先跑 freegen"
    LOG "G④ 判决: 同一个抽取器抽三处的目标价, 比离谁近"
    python3 - "$D2" "$W" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import json, os, re, sys
d, w = sys.argv[1], sys.argv[2]
# ★抽取器只此一份★(对版/错版/三态生成文本都走它, 否则几个数不可比)。
# 规则从 9 份真报告实测出来: 目标价小节里【最后一个加粗数字】就是这一节的结论(9 条里 8 条命中);
# 没有加粗的那一份走兜底(最后一个"目标价/基准值…数字")。为什么是"最后一个"而不是第一个:
# 小节前半是十来条子项论证, 每条都带一个候选数字, 结论写在最后。
BOLD = re.compile(r"\*\*[^*\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*元?\*\*")
ANCH = re.compile(r"(?:最终目标价|执行目标|目标价|基准值|策略基准|目标)[^0-9\n]{0,30}([0-9]+(?:\.[0-9]+)?)\s*元")
YUAN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*元")
def sect(t, cut):          # 从小节标题起, 到下一个 "### " 标题为止
    m = re.search(r"(?m)^### ", t[cut + 4:])
    return t[cut:cut + 4 + m.start()] if m else t[cut:]
def target(s):
    # 三级, 都取【最后一个】(小节前半是子项论证, 结论写在最后):
    # ①加粗数字 —— 9 份真报告命中 8 份, 对版/错版真值全对;
    # ②"目标价/执行目标/基准值…X元" —— 生成文本常不加粗, 靠它; 真值侧 18 个数里对 15 个, 只作兜底;
    # ③任何"X元" —— 最后的退路, 已知会被结论后面的执行条件句带偏, 只在前两级都空时用。
    ms = BOLD.findall(s) or ANCH.findall(s) or YUAN.findall(s)
    return float(ms[-1]) if ms else None
def written(t):            # 生成把这一节写完了没: 见到下一个标题, 或已经写到止损位
    return bool(re.search(r"(?m)^### ", t)) or ("止损" in t)
ev = json.load(open(os.path.join(d, "eval_samples.json")))["items"]
res = {v: [0, 0, 0.0, 0] for v in ("stu", "rules", "cases")}   # 站对版侧 / 可判 / 相对位置和 / 没写完
print("\n★自由生成尺(判决日, 模型自己写论证与结论; 前缀止于目标价小节标题)★")
print("  样本    对版    错版  |  裸底座          教师rules        教师cases")
for ln in open(os.path.join(w, "cuts.txt")):
    p = ln.split()
    if len(p) < 4: continue
    smp, sym, cr, cg = p[0], p[1], int(p[2]), int(p[3])
    it = ev[int(smp[1:])]
    rt = target(sect(it["body_report"].strip(), cr))
    wt = target(sect(it["orig_report"].strip(), cg))
    if rt is None or wt is None or rt == wt:
        print("  %-6s 两版真值抽不出或相同(%s/%s), 跳过" % (smp, rt, wt)); continue
    cells = []
    for v in ("stu", "rules", "cases"):
        fp = os.path.join(w, "gen_%s_%s.txt" % (v, smp))
        t = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
        # 生成文本要先切到【下一个小节标题之前】: 模型会一路写到止损位, 那里也有加粗数字(**13.04元**),
        # 不切就会把止损位当成目标价抽走。
        m = re.search(r"(?m)^#{2,3} ", t)   # 只切二/三井号: 四井号是本小节内部的子标题(子项分析/最终目标价修正)
        seg = t[:m.start()] if m else t
        x = target(seg) if written(t) else None
        if x is None:
            res[v][3] += 1; cells.append("   (没写完)    "); continue
        res[v][1] += 1
        # 站到对版一侧 = 离对版比离错版近; 相对位置 = (x−错版)/(对版−错版), 1=完全按对版, 0=完全按错版
        if abs(x - rt) < abs(x - wt): res[v][0] += 1
        rel = (x - wt) / (rt - wt)
        res[v][2] += rel
        cells.append("%8.2f(%+.2f)" % (x, rel))
    print("  %-6s %6.2f %6.2f | %s" % (smp, rt, wt, "  ".join(cells)))
# ★主尺 = 平均相对位置★(用户 09-14 定: "只要能撬动就可以了, 能不能翻转再说"):
# 相对位置 = (模型写的数 − 错版)/(对版 − 错版), 0 = 跟错版一模一样, 1 = 跟对版一模一样。
# 它量的是"朝对版方向挪了多少", 连续、不设台阶; 站对版侧率(要挪过中点才算)降为辅助读数。
print("\n  态        可判  ★平均相对位置★(0=照错版写 1=照对版写)   站对版侧          没写完")
b_rel = b_pct = None
for v in ("stu", "rules", "cases"):
    h, n, sm, miss = res[v]
    if not n: print("  %-8s   0" % v); continue
    rel, pct = sm / n, 100.0 * h / n
    if v == "stu": b_rel, b_pct = rel, pct
    print("  %-8s %4d   %+.4f (%+.4f)                    %6.2f%% (%+.2fpp)   %d"
          % (v, n, rel, rel - b_rel if b_rel is not None else 0.0, pct,
             pct - b_pct if b_pct is not None else 0.0, miss))
if b_rel is not None:
    best = max(("rules", "cases"), key=lambda v: (res[v][2] / res[v][1]) if res[v][1] else -9)
    d = (res[best][2] / res[best][1] - b_rel) if res[best][1] else 0.0
    print("\n★自由生成 撬动量 Δ相对位置 = %+.4f(最好档 %s)★" % (d, best))
    print("  明显 >0 ⇒ 记忆段在【模型自己写】的口径下真的把判断往对版推了 ⇒ 前三版败在 teacher-forced 尺上,")
    print("           ③ 的路重开, 靶换成教师自己生成的那份报告(论证与结论一起, 没有前文锁死)")
    print("  ≈0 或 <0 ⇒ 记忆段在自由生成下也推不动, 病在复盘素材(论证段没改), 要改人工流程")
PYEOF
}

# 只用盘上已有的 top-K 表重算主尺(改了行口径就跑它, 不必重跑前向 —— 一趟 3 分钟, 白烧没意义)
stage_argmax(){
    local i role
    [ -s "$D2/nll_all.dump" ] || DIE "先跑 nll(要分词产物)"
    stage_rows          # 行段重算一遍: 改了口径就是改这里, 不重跑前向
    . "$D2/nll_rows.env"
    : > "$D2/argmax_verdict.txt"
    for i in $KEPT; do
        for role in right wrong; do
            local top="$D2/nll_${role}_$i.top" frw alt
            [ -s "$top" ] || DIE "缺 $top(先跑 nll)"
            eval "frw=\$FROWS_$(echo $role | tr a-z A-Z)_$i"
            alt="$D2/nll_ids_wrong_$i.txt"; [ "$role" = wrong ] && alt="$D2/nll_ids_right_$i.txt"
            "$BEN/anchor_metrics" --ref-topk "$top" --rows "$frw" --alt "$alt" \
                2>&1 | sed "s/^/[${role}-argmax] /" | tee -a "$D2/argmax_verdict.txt" >> "$LOGF" || true
        done
    done
    python3 - "$D2" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import os, re, sys
d = sys.argv[1]
arg = {}
for ln in open(os.path.join(d, "argmax_verdict.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"\[(right|wrong)-argmax\] ★决策点\(两版不同的位置\) (\d+) 个: argmax 已站到目标一侧 (\d+) 个 = [\d.]+%; 平均 p\(目标\)−p\(另一版\) = ([+-][\d.]+)★", ln)
    if m:
        k, n, h, dp = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        a = arg.setdefault(k, [0, 0, 0.0]); a[0] += h; a[1] += n; a[2] += dp * n
if "right" not in arg:
    print("没读到决策点读数"); sys.exit(1)
h, n, dp = arg["right"]
print("  ★主尺 决策点(每处改动的第一个分歧位置, 两版前缀逐字相同): %d/%d = %.2f%% 上 argmax 选对版★"
      % (h, n, 100.0 * h / n if n else 0.0))
print("  平均 p(对版)−p(错版) = %+.4f  ⇒ %s" % (dp / n if n else 0.0,
      "还没撬动的那些位置要靠后训练把这个差推过零" if h < n else "全部决策点已站到对版一侧"))
PYEOF
}

# ---------------- ★尺 A: 训练日自己的错误判断撬动了没有★(back.md §2) ----------------
# 挂上候选, 把训练样本【重打一趟分】, 在决策点上看 argmax 站到了哪一边。
# 为什么这是必过项: 学不会当天的教训, 这个文件就没有存在的理由。线性上它应该接近 100%
# (147 条方程对 1.97M 个未知数, 极度欠定), 所以 <90% 不是"方法不行", 是取料行号/α/inv/
# W 行/bf16 五处之一错位 —— 停车去查, 别接着调参数。
# 顺带两件: 自检 2(解算器的预测 vs 这一趟的真前向)与守门 1(约束行的 argmax 变了多少)。
#   $1 = 候选目录
stage_gatea(){
    local CAND="${1:?候选目录}" SKIP="${2:-0}" W="$D2/sft"
    [ -s "$CAND/gr_L39.bin" ] || DIE "候选 $CAND 里没有 gr_L39.bin"
    [ -s "$CAND/predict.txt" ] || DIE "候选 $CAND 里没有 predict.txt(解算器没写?)"
    [ -s "$W/meta.env" ] || DIE "先跑 sft(训练样本 ids 还没有)"
    . "$W/meta.env"
    [ "$SKIP" = 1 ] || { bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3; need_idle; }
    local A="$D2/gatea_$(basename "$CAND").txt"; : > "$A"
    local i
    for i in $KEPT; do
        # ★改判决口径不必重烧前向★: 挂候选那一趟的 top-K 表在盘上, 第 2 参数传 1 就直接用它
        if [ "$SKIP" = 1 ]; then LOG "尺A 样本 $i 用盘上已有的候选态表"
        else LOG "尺A 样本 $i 挂候选重打分"
             # 切分点 = 最小约束/决策行 + 1(= 提示长度), 与解算器取料同一口径
             local sp; sp=$(cat "$W/rows_ctr_$i.txt" "$W/rows_dec_$i.txt" | sort -n | head -1); sp=$((sp + 1))
             score_pass "$W/ids_right_$i.txt" "$W/g_nll_$i.bin" "$W/g_top_$i.bin" "$W/g_score_$i.log" "$CAND" "" "$sp"; fi
        # ★判决器失败必须停车★(09-13 实撞): 原来这三行都带 `|| true`, 于是 --rows 不认 "@文件"
        # 那次报的是"尺A 0/0 = 0.00%" —— 看着像"一个决策点都没有", 其实是参数根本没被接受。
        # 判决器出错和判决为零是两回事, 不许混成一个读数。
        am(){ "$BEN/anchor_metrics" "$@" > "$A.tmp" 2>&1 || { cat "$A.tmp"; DIE "判决器失败: $*"; }; }
        # 决策点 argmax: --alt 给错版序列, --rows 给决策点行 —— 与段 0 基线同一把尺同一套行号
        am --ref-topk "$W/g_top_$i.bin" --rows "@$W/rows_dec_$i.txt" --alt "$W/ids_wrong_$i.txt"
        sed "s/^/[尺A-$i] /" "$A.tmp" | tee -a "$A" >> "$LOGF"
        # ★守门 1 = 两态对比★: 基线态(sft/top_i)与挂了候选那一趟(g_top_i)在同一批约束行上
        # argmax 是不是同一个 token。09-13 实撞: 原来拿"argmax 命中教师强制的下一个 token"当这把尺,
        # 挂候选前 62.70%、挂后 62.64% —— 那是基线自身的属性(报告段普通位置, 模型最想说的词
        # 未必就是实际写出来的那个), 根本不是"被改坏了多少"。
        am --ref-topk "$W/top_$i.bin" --vs-topk "$W/g_top_$i.bin" --rows "@$W/rows_ctr_$i.txt"
        sed "s/^/[守门1-$i] /" "$A.tmp" | tee -a "$A" >> "$LOGF"
        # 决策行同样看两态对比(诊断: 该动的动了多少)
        am --ref-topk "$W/top_$i.bin" --vs-topk "$W/g_top_$i.bin" --rows "@$W/rows_dec_$i.txt"
        sed "s/^/[决策行两态-$i] /" "$A.tmp" | tee -a "$A" >> "$LOGF"
        # 自检 2: 解算器预测的 logit 差 vs 这一趟真前向的
        am --ref-topk "$W/g_top_$i.bin" --predict "$CAND/predict.txt" --predict-sample "$i"
        sed "s/^/[自检2-$i] /" "$A.tmp" | tee -a "$A" >> "$LOGF"
    done
    python3 - "$A" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ah = an = 0
for m in re.finditer(r"\[尺A-\d+\] ★决策点\(两版不同的位置\) (\d+) 个: argmax 已站到目标一侧 (\d+) 个", txt):
    an += int(m.group(1)); ah += int(m.group(2))
ch = cn = 0
for m in re.finditer(r"\[守门1-\d+\] ★两态对比\[行段\]: argmax 没变 (\d+)/(\d+)", txt):
    ch += int(m.group(1)); cn += int(m.group(2))
dh = dn2 = 0
for m in re.finditer(r"\[决策行两态-\d+\] ★两态对比\[行段\]: argmax 没变 (\d+)/(\d+)", txt):
    dh += int(m.group(1)); dn2 += int(m.group(2))
cors = [float(m.group(1)) for m in re.finditer(r"\[自检2-\d+\].*?相关 ([-\d.]+)", txt)]
agr = [float(m.group(1)) for m in re.finditer(r"\[自检2-\d+\].*?翻/不翻一致 ([\d.]+)%", txt)]
fa = 100.0 * ah / an if an else 0.0
print("  ★尺A 训练日决策点翻转: %d/%d = %.2f%%★ (门 ≥90%%)" % (ah, an, fa))
print("  守门1 约束行 argmax 与基线相同: %d/%d = %.2f%% (门 ≥99.5%%)" % (ch, cn, 100.0 * ch / cn if cn else 0.0))
if dn2: print("  诊断 决策行 argmax 被改动: %d/%d = %.2f%%" % (dn2 - dh, dn2, 100.0 * (dn2 - dh) / dn2))
if cors:
    print("  自检2 预测 vs 真前向: 相关 %.4f, 翻/不翻一致 %.2f%% (门 相关≥0.95 且一致≥95%%)"
          % (sum(cors) / len(cors), sum(agr) / len(agr) if agr else 0.0))
    print("  ★自检2 不过就先查错位, 别调参数★" if (sum(cors) / len(cors) < 0.95) else "")
sys.exit(0 if fa >= 90.0 else 1)
PYEOF
    return ${PIPESTATUS[0]}
}

# ---------------- ★尺 B + 守门 2/3: 举一反三与不忘老本事★ ----------------
# 判决集(最晚 4 条, 解算一行没见过)的决策点翻转率, 与段 0 基线 50.75% 比; 全序列 NLL 不许涨 >1%;
# 再用 金融 j / wt2 两把判决料跑五指标, 看挂了 ③ 之后老本事退了多少。
#   $1 = 候选目录  $2 = 标签(默认按目录名)
stage_gate(){
    local CAND="${1:?候选目录}" TAG="${2:-$(basename "$CAND")}"
    [ -s "$CAND/gr_L39.bin" ] || DIE "候选 $CAND 里没有 gr_L39.bin"
    [ -s "$D2/verdict_base.txt" ] || DIE "缺基线读数 — 先跑 nll(不挂候选)"
    stage_nll "$CAND" "$TAG"
    python3 - "$D2" "$TAG" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import os, re, sys
d, tag = sys.argv[1], sys.argv[2]
def rd(p):
    v = {}
    txt = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"★分歧段.*?对版 n=(\d+)\s+平均NLL=([\d.]+).*?错版 n=(\d+)\s+平均NLL=([\d.]+)", txt, re.S)
    if m: v["diff_r"], v["diff_w"] = float(m.group(2)), float(m.group(4))
    m = re.search(r"全报告段.*?对版 n=(\d+)\s+平均NLL=([\d.]+)", txt, re.S)
    if m: v["all_r"] = float(m.group(2))
    m = re.search(r"(\d+)/(\d+) = ([\d.]+)% 上 argmax 选对版", txt)
    if not m: m = re.search(r"决策点翻转: (\d+)/(\d+) = ([\d.]+)%", txt)
    if m: v["flip"] = float(m.group(3)); v["flip_h"], v["flip_n"] = int(m.group(1)), int(m.group(2))
    return v
b = rd(os.path.join(d, "verdict_base.txt"))
c = rd(os.path.join(d, "verdict_%s.txt" % tag))
if "flip" not in b or "flip" not in c or "all_r" not in b or "all_r" not in c:
    print("★读数不全, 判不了(基线 %s / 候选 %s)★" % (sorted(b), sorted(c))); sys.exit(1)
# ★尺 B 是主尺★: 判决集决策点翻转率。一个决策点 = 1/67 = 1.5pp, 门定 +4.5pp(3 个点) —— 
# 比这小的变化读不出来是真进步还是抖动。NLL 那几条现在只当诊断(V4 四轮教训: 判据降了 argmax 没翻)。
db = c["flip"] - b["flip"]
print("  ★尺B 判决集决策点翻转★  基线 %.2f%% → 候选 %.2f%%  %+.2fpp (%d→%d / %d)  门 ≥+4.5pp"
      % (b["flip"], c["flip"], db, b["flip_h"], c["flip_h"], c["flip_n"]))
print("  守门2 全序列 NLL(对版)   %.4f → %.4f  %+.2f%%  门 ≤+1%%"
      % (b["all_r"], c["all_r"], 100.0 * (c["all_r"] - b["all_r"]) / b["all_r"]))
if "diff_r" in b and "diff_r" in c:
    print("  诊断 判据(对−错, 分歧段) %+.4f → %+.4f" % (b["diff_r"] - b["diff_w"], c["diff_r"] - c["diff_w"]))
ok = db >= 4.5 and c["all_r"] <= b["all_r"] * 1.01
print("  ★%s★" % ("尺B 与守门2 都过" if ok else "尺B/守门2 没过 — 不上线, 保持 ①+②"))
sys.exit(0 if ok else 1)
PYEOF
    local rc=${PIPESTATUS[0]}
    # 守门 3: 不忘老本事。同一把判决料, ②态 vs ②+③态的五指标(engine 档第五字段 = 后训练目录)。
    LOG "守门3 五指标(金融 j / wt2) ②+③ 态"
    local j
    for j in "$FINJ" "$WT2"; do
        [ -s "$j" ] || { LOG "守门3 跳过: $j 不在"; continue; }
        bash "$SC/v41_judge.sh" "$j" 8192 "engine::$ZCH::$CAND" 2>&1 | tee -a "$LOGF" | tail -8
    done
    LOG "守门3 读数见上: Same top 退 ≤0.5pp / KLD 涨 ≤3% 才算过(与 ②态的历史读数比)"
    return $rc
}

# ---------------- ★整晚一条龙★ ----------------
# 解出候选网格 → 按【预测】排序取前 3 → 每个先过尺 A(训练日自己的教训学会了没, 必过),
# 过了再过尺 B/守门 2/守门 3 → 第一个全过的上线。一个都不过 = 今晚保持 ①+②, 退出码非 0。
# 为什么只真跑前 3: 真跑一个候选 = 8 条训练样本重打分 + 4 条判决样本 + 两把五指标, 不是秒级的事。
stage_all_sft(){
    local OUT; OUT="$(stage_sft "${1:-1}" "${2:-1,10}" "${3:-}" 0 "${4:-1,10}" 0 0 "${5:-1e9}" "${6:-0}" "${7:-1,4}" | tail -1)"
    [ -d "$OUT" ] || DIE "解算没出候选目录"
    # ★排序键 = 第 7 列尺 L(按样本条留一的折外翻转率)★, 不是拟合率 —— 按拟合率排就是专挑最会背题的
    # 那个(09-14 实测: 拟合 76.92% 与 57.26% 的两个候选, 判决点翻转一模一样)。注释行一律 grep 掉。
    # ★上线排序键 = 尺L 为主、尺A 为辅★(09-14 定): 纯按尺A 排会挑中"最会背题"的那个 ——
    # 实测拟合 91.5% 的候选对没见过的一天是 −4.27pp, 而拟合 64.1% 的只有 −0.85pp。
    # 复合键 = 尺L(第8列) + 尺A(第9列)/1000: 尺L 拉开差距时它说了算, 尺L 打平才看尺A。
    local top3; top3=$(grep -v '^#' "$OUT/candidates.txt" | awk '{printf "%.6f %s\n", $8 + $9/1000, $1}' | sort -k1 -gr | head -3 | awk '{print $2}')
    [ -n "$top3" ] || DIE "候选表是空的"
    local cd_ win=""
    for cd_ in $top3; do
        LOG "★候选 $cd_ 过尺 A(训练日)★"
        if ! stage_gatea "$OUT/$cd_"; then LOG "候选 $cd_ 尺 A 没过, 换下一个"; continue; fi
        LOG "★候选 $cd_ 过尺 B + 守门★"
        if stage_gate "$OUT/$cd_" "$(basename "$OUT")_$cd_"; then win="$OUT/$cd_"; break; fi
        LOG "候选 $cd_ 尺 B/守门没过, 换下一个"
    done
    [ -n "$win" ] || { mkdir -p "$FTD/rejected"; mv "$OUT" "$FTD/rejected/$(basename "$OUT").$(date +%H%M%S)"; DIE "三个候选都没过门 — 今晚保持 ①+②(候选留在 rejected/ 可事后看)"; }
    # ★上线 = 与上一版逐元素相乘★: ③ₖ = ③ₖ₋₁ ⊙ Δₖ。这里落的是【本轮解出来的那一份】——
    # 相乘由引擎装载时做(多个 --posttrain 目录还没接, 所以 current 只指一个)。
    ln -sfn "$win" "$FTD/current"
    LOG "★上线: $(basename "$OUT")/$(basename "$win") → $FTD/current★"
    # ★长期台账★(每晚一行, 这就是"复盘天数 → 举一反三"那条曲线的原始数据):
    # 样本条数 / 诊断投影占比(泛化上限的刻度) / 胜出候选的整行(含尺L、尺A、守门1、|s−1|)。
    # 曲线要能回答的问题: 天数涨上去之后, 投影占比与尺L 到底涨不涨。
    local HIST="$FTD/history.txt"
    [ -s "$HIST" ] || echo "# 日期 样本条数 折外投影占比 专家重合% 词对交集% | 胜出候选整行(目录 R tau rho lam kappa nu 尺L 尺A val 守门1 |s-1| CG 方程 轮 残余)" > "$HIST"
    {
        printf "%s %s %s | " "$(date +%Y%m%d)" "$(grep -c '^[^#]' "$D2/sft/list.txt" 2>/dev/null || echo '?')" \
               "$(awk '$1=="ALL"{print $3, $5, $6}' "$OUT/diag.txt" 2>/dev/null || echo '- - -')"
        grep -v '^#' "$OUT/candidates.txt" | awk -v w="$(basename "$win")" '$1==w'
    } >> "$HIST"
    LOG "★台账 $HIST 记一行(复盘天数→举一反三 曲线的原始数据)★"
    stage_deploy
    LOG "ZNIGHT_ALL_DONE"
}

# ---------------- ③ 后训练(三文件的第三件, 2026-09-13) ----------------
# 靶 = 目标 token 的损失对末层 MoE 输出的梯度(闭式, 见 v41_sft_run.inc.c); 解出来的是与 ② 同构的
# 逐专家逐通道增益, 引擎里两张表逐元素相乘。① 与 ② 一个字节不动。
#
# 流程: ①训练样本逐条分词(对版/错版两条序列 + 两类行段) → ②当前态逐条打一趟分拿 top-K + rms
#       → ③写清单 → ④v41_amp_run --sft-list 一次取料、τ×ρ×λ 网格各出一个候选 → pt-<日期>/cand_*/
# ★解算器只出【预测】★(而且是 fp4 落地态算的); 真前向判决(尺 A/B + 三把守门尺)在 gate 段。
#   $1 = τ 网格(目标 logit 差, 默认 1)  $2 = ρ 网格(约束行权重, 默认 0.1,1,10)
#   $3 = 上一版后训练目录(空=从 ②起)  $4 = 1 跳过打分  $5 = λ 网格(默认 0.3,1,3,10)
#   $6 = 只取前 N 条训练样本(0=全部; 段 1′ 冒烟传 3)  $7 = 1 走共享通道模式(--share)
#   $8 = κ 网格(专家频率岭, 默认 1e9 = 不加)  $9 = ν 网格(词对重复度权重, 默认 0 = 不看)
#   $10 = R 网格(x 键控门数, 默认 1 = 旧形态; 传 "1,4" 一次取料把自检基准与新形态一起跑完)
stage_sft(){
    # ★不走环境变量★(铁律): 跳过打分是第 4 个位置参数, 由入口 sftsolve 传 1
    local TAU="${1:-1}" RHO="${2:-0.1,1,10}" PREV="${3:-}" SKIP="${4:-0}" LAM="${5:-0.3,1,3,10}" NMAX="${6:-0}" SHR="${7:-0}"
    local KAP="${8:-1e9}" NUW="${9:-0}" GAT="${10:-1}"
    local TAG="pt-$(date +%Y%m%d)"
    local OUT="$FTD/$TAG" W="$D2/sft"
    [ -s "$D2/samples.json" ] || DIE "先跑 samples"
    [ -x "$AMP/v41_amp_run" ] || make -C "$ROOT/gguf-tools" v41_amp_run >>"$LOGF" 2>&1 || DIE "解算器编译失败"
    mkdir -p "$W"
    LOG "③① 训练样本分词 + 行段"
    python3 - "$D2" "$NMAX" <<'PYEOF' || DIE "训练样本拼分词输入失败"
import json, os, sys
d = sys.argv[1]
# ★冒烟用★(段 1′): 只取前 N 条训练样本。取料一条一趟前向(实测 161s), 8 条 21 分钟 ——
# 机制还没验过就烧 21 分钟不合算, 先用 3 条把管子跑通(10 分钟律)。0 = 全部。
NMAX = int(sys.argv[2])
SEP = "<｜end▁of▁sentence｜>"
items = json.load(open(os.path.join(d, "samples.json")))["items"]
pro, right, wrong = [], [], []
for it in items:
    p = (it.get("body_prompt") or "").strip()
    r = (it.get("body_report") or "").strip()
    w = (it.get("orig_report") or "").strip()
    if p and r and w:
        pro.append(p); right.append(r); wrong.append(w)
if NMAX > 0:
    pro, right, wrong = pro[:NMAX], right[:NMAX], wrong[:NMAX]
if not right:
    print("训练样本缺 body_prompt/body_report/orig_report"); sys.exit(1)
open(os.path.join(d, "sft_all.txt"), "w").write(SEP + SEP.join(pro + right + wrong))
open(os.path.join(d, "sft_n.txt"), "w").write(str(len(right)))
print("训练样本 %d 条: 材料 %d 字 / 对版 %d 字 / 错版 %d 字"
      % (len(right), sum(map(len, pro)), sum(map(len, right)), sum(map(len, wrong))))
PYEOF
    ./ds4 --cuda -m "$MDL" --dump-tokens --prompt-file "$D2/sft_all.txt" 2>/dev/null > "$D2/sft_all.dump" || DIE "分词失败"
    python3 - "$D2" "$W" <<'PYEOF' || DIE "训练 ids 组装失败"
import os, re, sys, difflib
d, w = sys.argv[1], sys.argv[2]
N = int(open(os.path.join(d, "sft_n.txt")).read().strip())
m = re.search(r"\[([0-9,\s]+)\]", open(os.path.join(d, "sft_all.dump"), encoding="utf-8", errors="replace").read())
if not m:
    print("dump-tokens 没有 id 列表"); sys.exit(1)
ids = [int(x) for x in m.group(1).split(",") if x.strip()]
sep = ids[0]
chunks, cur = [], []
for t in ids[1:]:
    if t == sep: chunks.append(cur); cur = []
    else: cur.append(t)
chunks.append(cur)
if len(chunks) != 3 * N:
    print("切分段数 %d ≠ 3×%d" % (len(chunks), N)); sys.exit(1)
pro, right, wrong = chunks[:N], chunks[N:2*N], chunks[2*N:]
CTXMAX, WIN = 32768, 128
lines, kept, maxlen = [], [], 0
for i in range(N):
    seq_r = pro[i] + right[i] + [sep]
    seq_w = pro[i] + wrong[i] + [sep]
    if len(seq_r) > CTXMAX or len(seq_w) > CTXMAX:
        print("  样本 %d 太长(%d/%d > %d), 跳过" % (i, len(seq_r), len(seq_w), CTXMAX)); continue
    # 决策点 = 两版 token 序列的非公共块(在对版这一侧)。★行整体左移一格★: 位置 i 预测 ids[i+1]。
    sm = difflib.SequenceMatcher(None, right[i], wrong[i], autojunk=False)
    a = len(pro[i])
    dec, first = set(), []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and i2 > i1:
            for r in range(a + i1 - 1, a + i2): dec.add(r)
            # ★成对项只认块首★: 那里两版前缀逐字相同、行号对齐, "该写谁"才可比(判决尺同口径)
            first.append(a + i1 - 1)
    # ★两类行(第二版)★: 决策行 = 每处改动的第一个分歧位置(出方程, 要撬的就是它们);
    # 约束行 = 报告段其余位置每 4 行取 1(出"别处别动"的方程: 这一行原本最想说的 top1 与 top2
    # 的 logit 差不许变)。全收下是 3.4 万行, 光 ye 取料就 4 GB 主机 + 4 GB 设备 —— GB10 是统一
    # 内存, 那 8 GB 就是 GPU 少 8 GB(引擎权重已占 103 GB)。
    # 决策行的 3:1 拟合/val 切分在解算器里做(按到达顺序, 不随机 —— 同一份清单解两次必须一样)。
    ctr = []
    for k, r in enumerate(range(a - 1, a + len(right[i]))):
        if r in dec or r in first: continue
        if k % 4 == 0: ctr.append(r)
    if not dec:
        print("  样本 %d 两版一样, 没有决策点, 跳过" % i); continue
    # 采样后的行号不连续(几千段), 区间写法会撑爆清单字段 ⇒ 落成文件, 清单里写 @路径
    # (解算器的 rows_spec_parse 认这个前缀)。
    def spec(rows, tag):
        fp = os.path.join(w, "rows_%s_%d.txt" % (tag, i))
        open(fp, "w").write("\n".join(map(str, sorted(set(rows)))) + "\n")
        return "@" + fp
    for nm, seq in (("right", seq_r), ("wrong", seq_w)):
        open(os.path.join(w, "ids_%s_%d.txt" % (nm, i)), "w").write("\n".join(map(str, seq)) + "\n")
    maxlen = max(maxlen, len(seq_r), len(seq_w))
    lines.append("%s/ids_right_%d.txt %s/top_%d.bin %s/rms_%d.bin %s/ids_wrong_%d.txt %s %s"
                 % (w, i, w, i, w, i, w, i, spec(ctr, "ctr"), spec(first, "dec")))
    kept.append(i)
    print("  样本 %d: 序列 %d 行 | 报告段 %d | 决策点(块首) %d | 约束行 %d | 分歧行合计 %d"
          % (i, len(seq_r), len(right[i]), len(first), len(ctr), len(dec)))
if not kept:
    print("没有可用训练样本"); sys.exit(1)
open(os.path.join(w, "list.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(w, "meta.env"), "w").write('KEPT="%s"\nMAXLEN="%d"\n' % (" ".join(map(str, kept)), maxlen))
PYEOF
    . "$W/meta.env"
    # ★打分与取料是同一趟前向★(09-13 夜改): 榜单/NLL/rms 在出口那一步算, 逐专家输出在钩子那一步取,
    # 本来就在一次前向里。早先分两趟: 同一批 token 前向两遍(8 条白烧 22 分钟), 外加 8 次独立进程
    # 各加载一遍模型(107s×8 ≈ 14 分钟)。现在解算器自己出表, 这一段只在【复用盘上的表】时才做事。
    if [ "$SKIP" = 1 ]; then LOG "③② 复用盘上已有的 top-K/rms 表(改的是行口径, 不重烧前向)"
    else LOG "③② 停服 → 榜单与逐专家输出同一趟前向出(解算器内做)"
         bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
         need_idle
    fi
    local REUSE=""; [ "$SKIP" = 1 ] && REUSE="--reuse-tables"
    # ★--share★(第 7 个位置参数, ★不走环境变量★ —— 本仓禁新增 env, ${VAR:-默认} 直喂二进制同样算):
    # 全体专家共用一组通道增益(未知数 D 个而不是 384×D)。每专家各自独立时最小范数解拟合的是
    # 逐 token 的专家输出, 那东西在 5120 维里近似正交 ⇒ 泛化恒为 0(实测样本 3→8 条, val 纹丝不动)。
    local SHARE=""; [ "$SHR" = 1 ] && SHARE="--share"
    LOG "③③ 解算(网格 R=$GAT τ=$TAU ρ=$RHO λ=$LAM κ=$KAP ν=$NUW, 层 L39) → $OUT"
    mkdir -p "$OUT"
    set -o pipefail
    "$AMP/v41_amp_run" "$MDL" "$HFDIR" "$W/ids_right_${KEPT%% *}.txt" "$MAXLEN" "$OUT" \
        --only-layer 39 --capture-ye --sft-list "$W/list.txt" $REUSE $SHARE \
        --tau-list "$TAU" --rho-list "$RHO" --lam-list "$LAM" --kappa-list "$KAP" --nu-list "$NUW" --gates "$GAT" \
        --base-amp "$ZCH" ${PREV:+--base-pt "$PREV"} --mem-budget-mb 110000 2>&1 | tee -a "$LOGF" | tail -40
    # ★不过门不删目录★(09-13 实撞: 第一版把解删了, 事后连缩放因子长什么样都看不到)
    [ "${PIPESTATUS[0]}" = 0 ] || { mkdir -p "$FTD/rejected"; mv "$OUT" "$FTD/rejected/$TAG.$(date +%H%M%S)"; DIE "后训练解算失败 — 今晚不上线, 保持 ①+②(残留在 rejected/)"; }
    [ -s "$OUT/candidates.txt" ] || { DIE "候选表没产出"; }
    LOG "③④ 候选 $(grep -vc '^#' "$OUT/candidates.txt") 个落在 $OUT ($(du -sh "$OUT" | cut -f1))"
    [ -s "$OUT/diag.txt" ] && tail -1 "$OUT/diag.txt" | sed 's/^/  [诊断·折外投影占比 专家重合 词对交集] /' | tee -a "$LOGF"
    grep -v '^#' "$OUT/candidates.txt" | sort -k8 -gr | head -5 | sed 's/^/  [候选·按★尺L★排] /' | tee -a "$LOGF"
    echo "$OUT"
}

# ---------------- ④ 上线 ----------------
# ★V4.1 还没有服务(引擎 P5: 会话/采样/HTTP 未接, src/server 里零处 V4.1)★
# 所以这一段只做"版本化 + 报告当前在用的组合", 不起服务。quant_trading_flow 接入要等 P5,
# 那之前行为复核走 CLI 贪心探针(decision_probe 的问法)。不假装起了服务 —— 假成功比失败更坏。
stage_deploy(){
    local cur=""
    [ -e "$FTD/current" ] && cur="$(readlink -f "$FTD/current")"
    if [ -n "$cur" ]; then LOG "④ 当前部署组合: ①量化 + ②反修($(basename "$ZCH")) + ③后训练($(basename "$cur"))"
    else LOG "④ 当前部署组合: ①量化 + ②反修($(basename "$ZCH")), 没有后训练件"; fi
    LOG "④ V4.1 服务未接(引擎 P5), 本段只落盘不起服务"
}
stage_probe(){ bash "$SC/decision_probe.sh" "${1:?标签(z0/z1)}" "$BACKEND"; }

# ---------------- ★判决尺: 目标 token 的 NLL★ ----------------
# 2026-09-08 定罪后新增, 当晚重做口径。前三轮拿"专家重合率/残差能量"当判决 —— 那不是任何
# 后训练的目标函数。后训练要降的就是【本应写出的报告】那些 token 的 NLL。
#
# ★口径(09-08 夜重做)★: 同一份当日材料后面接两种续写 ——
#   错版 = 当天真实输出的 CFO 报告(模型现在的偏好)
#   对版 = 在原报告上做外科修改的同格式同篇幅版本(只改被教训推翻的那几处)
# 判据 = 平均NLL(对版) − 平均NLL(错版)。
# 为什么换掉"教师读事后上下文"那个口径: 教师能直接抄上下文里的次日价格和教训措辞, 那个差是
# 上限不是可学量; 错版/对版是同前缀、同格式、同篇幅的两条续写, 差多少就是行为差多少。
# ★材料段不算数★: 每条样本 = [材料][报告], 只判报告段 —— N 条样本 = N 段不连续行,
# anchor_metrics 的 --rows 收多段合成一个池子。
# ★走 prefill 批路(--eval-ids)★: 报告 8000 字 ≈ 5k token, 逐 token 的 --score-ids(约 20 t/s)
# 一条就要 4 分钟; 批路做同样的 teacher-forced 打分快一个量级。--eval-ids 在 zchain 上传之后
# 才跑(core_engine_open.c:544), 口径与部署一致。
# 判决集的分词 + 行段组装(纯 CPU, 幂等): nll 与 argmax 都从这里拿 ids 与行号,
# 所以改了行口径只要重跑它, 不必重跑一趟前向(一趟 3 分钟, 白烧没意义)。
stage_rows(){
    LOG "尺① 判决集(解算未见): $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(len(d[\"items\"]),'条:', ','.join(x.get('symbol','?') for x in d['items']))" "$D2/eval_samples.json")"
    # 一次分词拿到所有段的精确边界: 用特殊 token 当分隔符 —— tokenizer 遇到特殊 token 会
    # 断开 span(core_bpe.c), 所以每段各自成词, 边界零歧义。分隔符放最前面 ⇒ ids[0] 就是它自己。
    python3 - "$D2" <<'PYEOF' || DIE "拼分词输入失败"
import json, os, sys
d = sys.argv[1]
SEP = "<｜end▁of▁sentence｜>"
items = json.load(open(os.path.join(d, "nll_samples.json")))["items"]
pro, right, wrong = [], [], []
for it in items:
    p = (it.get("body_prompt") or "").strip()
    r = (it.get("body_report") or "").strip()     # 对版: 外科修正后的同一份报告
    w = (it.get("orig_report") or "").strip()     # 错版: 当天真实输出的 CFO 报告
    if p and r and w:
        pro.append(p); right.append(r); wrong.append(w)
if not right:
    print("样本缺 body_prompt/body_report/orig_report — 后端重跑过素材了吗?"); sys.exit(1)
pieces = pro + right + wrong                       # 顺序固定: N 份材料, N 份对版, N 份错版
open(os.path.join(d, "nll_all.txt"), "w").write(SEP + SEP.join(pieces))
open(os.path.join(d, "nll_n.txt"), "w").write(str(len(right)))
print("样本 %d 条: 材料 %d 字 / 对版 %d 字 / 错版 %d 字"
      % (len(right), sum(map(len, pro)), sum(map(len, right)), sum(map(len, wrong))))
PYEOF
    ./ds4 --cuda -m "$MDL" --dump-tokens --prompt-file "$D2/nll_all.txt" 2>/dev/null > "$D2/nll_all.dump" \
        || DIE "分词失败"
    # ★一条样本一趟★(2026-09-13, V4.1): 当时引擎 V4.1 的上下文是 32768(今天从模型元数据
    # deepseek4.context_length 读, 1M; 候选块核用 shared 存整段组数), 而 4 条样本串起来就两万多行,
    # 再多几条必然撞墙 —— 撞了不会明着报"上下文满", 只会给一串安静的错数。所以每条样本
    # 各写各的 ids, 各跑一趟, 最后按行数加权合并(合并在收口那段, 全是计数, 不碰数值)。
    python3 - "$D2" <<'PYEOF' || DIE "ids 组装失败"
import os, re, sys, difflib
d = sys.argv[1]
N = int(open(os.path.join(d, "nll_n.txt")).read().strip())
m = re.search(r"\[([0-9,\s]+)\]", open(os.path.join(d, "nll_all.dump"), encoding="utf-8", errors="replace").read())
if not m:
    print("dump-tokens 没有 id 列表"); sys.exit(1)
ids = [int(x) for x in m.group(1).split(",") if x.strip()]
sep = ids[0]
chunks, cur = [], []
for t in ids[1:]:
    if t == sep: chunks.append(cur); cur = []
    else: cur.append(t)
chunks.append(cur)
if len(chunks) != 3 * N:
    print("切分段数 %d ≠ 3×%d — 分隔符被并进正文了?" % (len(chunks), N)); sys.exit(1)
pro, right, wrong = chunks[:N], chunks[N:2*N], chunks[2*N:]
CTXMAX = 32768
env = ['N="%d"' % N]
kept = []
for i in range(N):
    # ★判决行只取两版真正不同的那些 token★(09-08 夜收紧): 两版报告 97.3% 逐字相同,
    # 相同 token 的 NLL 差恒为 0 却全进分母 —— 首轮 +0.0178 就是这么被稀释了 35 倍。
    # 在【token id 序列】上求最长公共子序列, 取各自的非公共块。只挑"判哪几行", 不碰数值。
    sm = difflib.SequenceMatcher(None, right[i], wrong[i], autojunk=False)
    blk = {"right": [], "wrong": []}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        if i2 > i1: blk["right"].append((i1, i2))
        if j2 > j1: blk["wrong"].append((j1, j2))
    too_long = False
    lines = []
    for name, rep in (("right", right), ("wrong", wrong)):
        seq = pro[i] + rep[i] + [sep]
        if len(seq) > CTXMAX:
            print("  样本 %d 的 %s 序列 %d 行 > 上限 %d, 跳过这条" % (i, name, len(seq), CTXMAX))
            too_long = True
            break
        a = len(pro[i])                                  # 报告第一个 token 的行号
        # ★位置 i 预测 ids[i+1]★ ⇒ 判决行整体左移一格
        segs_all = [(a - 1, a + len(rep[i]))]
        segs_diff = [(a + b - 1, a + e) for (b, e) in blk[name]]
        # ★真正的决策点 = 每块的第一个分歧位置★(2026-09-13 修口径): 在块首, 两版的前缀逐字相同、
        # 行号也对齐, 所以"模型在这里会写哪个 token"是可比的。块内往后的位置, 对版序列的前缀已经
        # 是对版自己写的了 —— 那里 argmax 当然偏向对版, 量出来的是 teacher-forcing 的自洽偏置,
        # 不是行为。首轮就踩了这个: 分歧段整体命中率 57~74%, 看着像"已经会了", 其实是假账。
        segs_first = [(a + b - 1, a + b) for (b, e) in blk[name]]
        open(os.path.join(d, "nll_ids_%s_%d.txt" % (name, i)), "w").write("\n".join(map(str, seq)) + "\n")
        # ★值一律加引号★: KEPT 里是空格分隔的样本号, 不加引号时 `. env` 会把它当命令跑
        # (实撞: "行 34: 1: 未找到命令", 然后 KEPT 未绑定整段停车)。
        lines.append('S_%s_%d="%d"' % (name.upper(), i, len(seq)))
        lines.append('ROWS_%s_%d="%s"' % (name.upper(), i, ",".join("%d:%d" % t for t in segs_all)))
        lines.append('DROWS_%s_%d="%s"' % (name.upper(), i, ",".join("%d:%d" % t for t in segs_diff)))
        lines.append('FROWS_%s_%d="%s"' % (name.upper(), i, ",".join("%d:%d" % t for t in segs_first)))
        lines.append('NDIFF_%s_%d="%d"' % (name.upper(), i, sum(hi - lo for lo, hi in segs_diff)))
        print("  样本 %d %-5s: 序列 %5d 行 | 报告段 %5d token | ★分歧段 %4d token (%d 块)★"
              % (i, name, len(seq), len(rep[i]), sum(hi - lo for lo, hi in segs_diff), len(segs_diff)))
    if too_long: continue
    if not any(l.startswith('NDIFF_RIGHT_%d=' % i) and not l.endswith('"0"') for l in lines):
        print("  样本 %d 两版一模一样, 没有决策点, 跳过" % i); continue
    env += lines
    kept.append(i)
if not kept:
    print("一条可判的样本都没有"); sys.exit(1)
env.append('KEPT="%s"' % " ".join(map(str, kept)))
open(os.path.join(d, "nll_rows.env"), "w").write("\n".join(env) + "\n")
PYEOF
}

stage_nll(){
    local FT="${1:-}" TAG="${2:-base}"   # FT: 判决时挂的后训练目录(空=基线, 只有 ①+②); TAG: 读数存档名
    local BEN="$ROOT/gguf-tools/bench"
    [ -x "$BEN/anchor_metrics" ] || make -C "$ROOT/gguf-tools" anchor_metrics >>"$LOGF" 2>&1
    # ★只用判决集★: samples 段切出来的 eval_samples.json(解算一行都没看过)。
    # 它不在就硬停 —— 回退去拉 API 会悄悄把训练样本混进判决, 那正是这次要堵的漏。
    [ -s "$D2/eval_samples.json" ] || DIE "缺 $D2/eval_samples.json — 先跑 samples 做训练/判决切分"
    cp "$D2/eval_samples.json" "$D2/nll_samples.json"
    stage_rows
    . "$D2/nll_rows.env"
    LOG "尺② 停服(清理=关进程, 不删文件)"; bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    : > "$D2/nll_verdict.txt"
    local i role
    for i in $KEPT; do
        for role in right wrong; do
            local ids="$D2/nll_ids_${role}_$i.txt" out="$D2/nll_${role}_$i.nll" top="$D2/nll_${role}_$i.top"
            local rws drw frw alt
            eval "rws=\$ROWS_$(echo $role | tr a-z A-Z)_$i"
            eval "drw=\$DROWS_$(echo $role | tr a-z A-Z)_$i"
            eval "frw=\$FROWS_$(echo $role | tr a-z A-Z)_$i"
            [ -n "$rws" ] && [ -n "$drw" ] || DIE "样本 $i $role 的行段为空(nll_rows.env 没生成对?)"
            alt="$D2/nll_ids_wrong_$i.txt"; [ "$role" = wrong ] && alt="$D2/nll_ids_right_$i.txt"
            LOG "尺③ 样本 $i ${role} 打分($(wc -l < "$ids") 行, --score-ids 与部署同路)"
            score_pass "$ids" "$out" "$top" "$D2/nll_${role}_$i.log" "$FT"
            # 两个行集合各判一次: 全报告段(看整体) + ★分歧段(真正做决策的那些 token)★
            "$BEN/anchor_metrics" --ref-nll "$out" --ids "$ids" --rows "$rws" \
                2>&1 | sed "s/^/[$role] /" | tee -a "$D2/nll_verdict.txt" >> "$LOGF" || true
            "$BEN/anchor_metrics" --ref-nll "$out" --ids "$ids" --rows "$drw" \
                2>&1 | sed "s/^/[${role}-diff] /" | tee -a "$D2/nll_verdict.txt" >> "$LOGF" || true
            # ★主尺: 决策点上 argmax 到底选了谁★。NLL 是连续量而部署是贪心 —— V4 那轮把判据
            # 压掉 61.9% 却一个数字没动, 就是因为没人量这个。--alt 给另一版 ids, 决策点由判决器自己认。
            "$BEN/anchor_metrics" --ref-topk "$top" --rows "$frw" --alt "$alt" \
                2>&1 | sed "s/^/[${role}-argmax] /" | tee -a "$D2/nll_verdict.txt" >> "$LOGF" || true
        done
    done
    LOG "尺④ 收口(按行数加权合并各样本) → $D2/verdict_$TAG.txt"
    cp "$D2/nll_verdict.txt" "$D2/raw_verdict_$TAG.txt"
    python3 - "$D2" <<'PYEOF' 2>&1 | tee "$D2/verdict_$TAG.txt" | tee -a "$LOGF"
import os, re, sys
d = sys.argv[1]
acc = {}          # 标签 → [Σ nll·n, Σ n]
arg = {}          # 标签 → [命中, 总数, Σ(p目标−p另一版)·n]
for ln in open(os.path.join(d, "nll_verdict.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"\[(right|wrong|right-diff|wrong-diff)\] 参考 PPL\[行段 n=(\d+)\] = [\d.]+\s+\(平均 NLL ([\d.]+)\)", ln)
    if m:
        k, n, v = m.group(1), int(m.group(2)), float(m.group(3))
        a = acc.setdefault(k, [0.0, 0]); a[0] += v * n; a[1] += n
        continue
    m = re.match(r"\[(right|wrong)-argmax\] ★决策点\(两版不同的位置\) (\d+) 个: argmax 已站到目标一侧 (\d+) 个 = [\d.]+%; 平均 p\(目标\)−p\(另一版\) = ([+-][\d.]+)★", ln)
    if m:
        k, n, h, dp = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        a = arg.setdefault(k, [0, 0, 0.0]); a[0] += h; a[1] += n; a[2] += dp * n
if not {"right", "wrong"} <= set(acc):
    print("读数不全, 拿不到判决"); sys.exit(1)
for tag, title in (("", "全报告段(含两版逐字相同的部分)"), ("-diff", "★分歧段(两版真正不同的 token)★")):
    if ("right" + tag) not in acc or ("wrong" + tag) not in acc: continue
    sr, nr = acc["right" + tag]; sw, nw = acc["wrong" + tag]
    lr, lw = sr / nr, sw / nw
    print("  %s" % title)
    print("    对版 n=%-6d 平均NLL=%.4f" % (nr, lr))
    print("    错版 n=%-6d 平均NLL=%.4f" % (nw, lw))
    print("    判据 NLL(对版)−NLL(错版) = %+.4f  (总 log 差 %+.1f nats)" % (lr - lw, sr - sw))
if "right" in arg:
    h, n, dp = arg["right"]
    print("  ★主尺 决策点翻转: %d/%d = %.2f%% 的决策点上 argmax 已经选对版; 平均 p(对版)−p(错版) = %+.4f★"
          % (h, n, 100.0 * h / n if n else 0.0, dp / n if n else 0.0))
    print("  ★%s★" % ("还没撬动: 贪心部署下这些位置仍会写出错版" if h < n
                      else "全部决策点都已站到对版一侧"))
PYEOF
    LOG "尺⑤ 收口完成"
}

# ★换工具前的机制审计★: prefill 批路(--eval-ids)与逐 token 解码路(--score-ids)在同一批 ids 上
# 必须给出同一个 NLL。判决尺从解码路换到批路是为了速度, 但速度不能换口径 —— 两条路要是不等,
# 后面所有读数都不能和历史比。用上一轮 nll 段留下的解码路 logits 做基准, 不重跑它。
stage_evalaudit(){
    DIE "V4 批路 vs 解码路的口径审计(V4.1 只有 --score-ids 一条路) —— V4 口径, 已停用(back.md §1.2)"
    local BEN="$ROOT/gguf-tools/bench"
    local ids="$D2/nll_ids_s.txt" dec="$D2/nll_logits_s.bin" bat="$D2/nll_logits_audit.bin"
    [ -s "$ids" ] && [ -s "$dec" ] || DIE "缺解码路基准($ids / $dec)"
    local rows; rows=$(wc -l < "$ids")
    LOG "审计① 停服"; bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    LOG "审计② 批路重打同一批 ids($rows 行)"
    # ★--eval-no-bos 必须给★: 批路默认在流首插 BOS, 于是每个 token 的位置都往后挪一格
    # (RoPE 位置全变) 且上下文多一个 token —— 首轮审计就栽在这, 两条路差了 0.178 nat,
    # 比要测的效应(0.223)还大。要对账就必须喂同一条 token 流。
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --eval-no-bos \
        --eval-ids "$ids" --eval-logits "$bat" > "$D2/nll_audit.log" 2>&1 </dev/null \
        || { tail -5 "$D2/nll_audit.log"; DIE "批路打分失败"; }
    LOG "审计③ 两条路对读数(全段)"
    # ★先落盘再 grep★: 直接 `... 2>&1 | grep 关键字` 会把崩溃信息一起过滤掉 —— 上一版就是这么
    # 把一个段错误看成"没输出"的(anchor_metrics 段错误退 139, grep 照样返回 0)。
    for way in dec bat; do
        local f lbl
        if [ "$way" = dec ]; then f="$D2/audit_dec.txt"; lbl="解码路(--score-ids)"
                                  "$BEN/anchor_metrics" --ref-raw  "$dec" --ids "$ids" > "$f" 2>&1
        else                      f="$D2/audit_bat.txt"; lbl="批路(--eval-ids)"
                                  "$BEN/anchor_metrics" --ref-eval "$bat" --ids "$ids" > "$f" 2>&1
        fi
        local rc=$?
        echo "--- $lbl (退出码 $rc)" | tee -a "$LOGF"
        grep -E "参考 PPL|top-1" "$f" | tee -a "$LOGF" \
            || { echo "  ★没读到指标, 原始输出尾部:★"; tail -4 "$f"; } | tee -a "$LOGF"
    done
    rm -f "$bat"                      # 本段自己造的临时件, 量完即删(盘只剩十几 G)
    LOG "审计④ 恢复服务"; stage_deploy
}

# ★--eval-nll 口径审计(2026-09-10 新出口上线前必跑)★
# 新出口只有在【与既有出口给出同一个数】时才能用 —— 否则以后所有判决读数都不能与历史比,
# 这条纪律上一轮救过一次场(批路默认插 BOS, 差 0.178 nat 比要测的效应还大)。
# 做法: 同一趟前向同时写全词表 logits 与逐位 NLL, 两条路各自算平均, 必须相等。
# ★为什么只取几百行★: logits 每位置 517 KB, 300 token 才 155 MB; 全量 7.3k 行就是 8.1 GB,
# 那正是 09-08 夜把统一内存写崩的量。审计要的是口径一致, 不是长度。
stage_nllaudit(){
    DIE "V4 --eval-nll 口径审计(V4.1 的小出口审计走 nllaudit41) —— V4 口径, 已停用(back.md §1.2)"
    local N="${1:-300}"
    local BEN="$ROOT/gguf-tools/bench"
    local src="$D2/nll_ids_right.txt"
    [ -s "$src" ] || DIE "缺 ids 源 $src(先跑 nll)"
    [ -x "$BEN/anchor_metrics" ] || make -C "$ROOT/gguf-tools" anchor_metrics >>"$LOGF" 2>&1
    local ids="$D2/nllaudit_ids.txt" lg="$D2/nllaudit_logits.bin" nl="$D2/nllaudit.nll"
    head -n "$N" "$src" > "$ids"
    LOG "审计① 停服(清理=关进程, 不删文件)"; bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    # ★无论成败都把服务放回去★: 中途 DIE 会跳过末尾的恢复, 而这台机器同时是 qtf 的 LLM
    # (LOCAL_LLM_BASE_URL 指着它), 停着不恢复等于把另一条线一起掐了。
    trap 'bash "$SC/serve_1m_spark.sh" start "$MDL" "$ZCH" plain >>"$LOGF" 2>&1 || true' EXIT
    need_idle
    LOG "审计② 一趟前向同时出 logits 与 nll($N 行)"
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --eval-no-bos \
        --eval-ids "$ids" --eval-logits "$lg" --eval-nll "$nl" \
        > "$D2/nllaudit.log" 2>&1 </dev/null || { tail -5 "$D2/nllaudit.log"; DIE "打分失败"; }
    [ -s "$lg" ] && [ -s "$nl" ] || DIE "两个出口没都产出"
    LOG "审计③ 两条路各自算平均 NLL"
    "$BEN/anchor_metrics" --ref-eval "$lg" --ids "$ids" > "$D2/nllaudit_am.txt" 2>&1 || true
    python3 - "$D2" "$ids" <<'PY' | tee -a "$LOGF"
import os, re, struct, sys
d, idsp = sys.argv[1], sys.argv[2]
nll = open(os.path.join(d, "nllaudit.nll"), "rb").read()
v = struct.unpack("<%df" % (len(nll) // 4), nll)
nids = sum(1 for l in open(idsp) if l.strip())
# 末位是 NaN(没有下一个 token), 不进平均 —— 与引擎内部那句冒烟读数同口径
fin = [x for x in v if x == x]
mine = sum(fin) / len(fin)
am = None
for ln in open(os.path.join(d, "nllaudit_am.txt"), encoding="utf-8", errors="replace"):
    m = re.search(r"平均 NLL ([\d.]+)", ln)
    if m: am = float(m.group(1)); break
print("  --eval-nll   : n=%d 平均 NLL %.6f  (文件 %d 字节, ids %d 行)" % (len(fin), mine, len(nll), nids))
if am is None:
    print("  ★anchor_metrics 没给出平均 NLL, 拿不到对照★"); sys.exit(1)
print("  --eval-logits: 平均 NLL %.6f  (anchor_metrics 同一趟)" % am)
diff = abs(mine - am)
print("  差 %.2e ⇒ %s" % (diff, "★口径一致, --eval-nll 可用★" if diff < 1e-4
                          else "★口径不一致, 不许上线★"))
sys.exit(0 if diff < 1e-4 else 1)
PY
    local rc=${PIPESTATUS[0]}
    rm -f "$lg"                      # 审计自造的临时件, 量完即删(统一内存机器上它就是压力源)
    trap - EXIT                      # 交回给下面的显式恢复, 免得起两遍服务
    LOG "审计④ 恢复服务"; stage_deploy
    [ "$rc" = 0 ] || DIE "口径审计未通过 — --eval-nll 不许上线"
}

# 把 anchor_metrics 的"罪犯 token"榜翻译成人话: 光有 id 看不出是【次日价格数字】(教师在抄,
# 不可学)还是【下修/放弃交易 这类行为词】(可学)。★只读现有产物, 不重跑打分★。
stage_nlltext(){
    [ -s "$D2/nll_verdict.txt" ] && [ -s "$D2/nll_all.dump" ] || DIE "先跑 nll"
    python3 - "$D2" <<'PY'
import os, re, sys
d = sys.argv[1]
# 词表里存的是字节级 BPE 形态(GPT-2 bytes_to_unicode: 空格=Ġ, 中文=一串拉丁怪字),
# 直接打出来是乱码 ⇒ 先按那张表反解回字节, 再 UTF-8 解码。只影响可读性, 不参与任何数值。
_bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
_cs = _bs[:]; _n = 0
for _b in range(256):
    if _b not in _bs: _bs.append(_b); _cs.append(256 + _n); _n += 1
_U2B = {chr(c): b for b, c in zip(_bs, _cs)}
def bdec(s):
    try: return bytes(_U2B[ch] for ch in s).decode("utf-8", "replace")
    except KeyError: return s
# id→原文: dump-tokens 的对照表每行 "%6d  <原文>"。含换行的 token 只取首行, 诊断够用。
txt = {}
for ln in open(os.path.join(d, "nll_all.dump"), encoding="utf-8", errors="replace"):
    m = re.match(r"^ *(\d+)  (.*)$", ln.rstrip("\n"))
    if m and int(m.group(1)) not in txt: txt[int(m.group(1))] = bdec(m.group(2))
ids = [int(x.strip()) for x in open(os.path.join(d, "nll_ids_t.txt")) if x.strip()]
def s(i): return txt.get(ids[i], "?") if 0 <= i < len(ids) else ""
print("\n== 罪犯 token 的人话版(左=学生已看到的上文, ★=学生猜不中的那个目标 token) ==")
for ln in open(os.path.join(d, "nll_verdict.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"\s*pos=(\d+) tgt=(\d+) dNLL=\+([\d.]+)", ln)
    if not m: continue
    p, tgt, dn = int(m.group(1)), int(m.group(2)), m.group(3)
    left = "".join(s(i) for i in range(max(0, p - 24), p + 1))
    right = "".join(s(i) for i in range(p + 2, p + 8))
    print("  dNLL+%-7s …%s ★%s★ %s" % (dn, left.replace("\n", "⏎")[-46:],
                                        txt.get(tgt, "?").replace("\n", "⏎"),
                                        right.replace("\n", "⏎")[:24]))
PY
}

# ---------------- 当天复盘 → ③ 的训练语料(2026-09-23, 用户令"拿复盘的语料再训练, 再跑早上的看结论能不能反转") ----------------
# 大盘复盘只给一个事实: 早盘预测方向 vs 实际方向。错版 = 早上那条请求的模型原输出(思考 + 正文),
# 对版 = 同一份输出里把预测方向换成复盘真值(其余一字不动 ⇒ 两版逐 token 对齐, 决策点就是被换掉的那几个 token)。
# 写成 sft 段现成认的 samples.json(body_prompt / body_report / orig_report), 之后直接 `sft`。
#   $1 = 请求 JSON(qtf_capture_request.py 的输出, 带 _note)  $2 = 模型输出(流式 SSE 原文或非流式 JSON)
#   $3 = 复盘真值 上涨|下跌
# 提示按服务端 render_chat_prompt_text 的字节拼(思考档 high): BOS <｜System｜> effort 前缀 system <｜User｜> user
# <｜Assistant｜><think>。拼错 = 训练的不是产品那条请求(模板漏 <｜System｜> 的旧坑, bug.md §1)。
stage_review(){
    # $4 = add: 追加到已有 samples.json(迭代轮用: 挂 ③ 的新轨迹上又拍回错方向的那一处, 作为新决策点)。
    # 退出码: 0 = 拼好; 1 = 这条输出里没有错方向的拍板(没东西可学); 3 = 与已有样本重复(同一处, 再解也不会多学)。
    local REQ="${1:?请求 JSON}" OUTF="${2:?模型输出}" TRUTH="${3:?复盘真值 上涨|下跌}" MODE="${4:-new}"
    [ -s "$D2/samples.json" ] && cp -n "$D2/samples.json" "$D2/samples.json.bak-$(date +%Y%m%d%H%M%S)"
    python3 - "$REQ" "$OUTF" "$TRUTH" "$D2/samples.json" "$MODE" <<'PYEOF'
import json, os, sys
req, outf, truth, dst, mode = sys.argv[1:6]
wrong_dir = {"上涨": "下跌", "下跌": "上涨"}[truth]
r = json.load(open(req, encoding="utf-8"))
sysm = "\n\n".join(m["content"] for m in r["messages"] if m["role"] == "system")
user = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
effort = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
prompt = "<｜begin▁of▁sentence｜><｜System｜>" + effort + sysm + "<｜User｜>" + user + "<｜Assistant｜><think>"
raw = open(outf, encoding="utf-8", errors="replace").read()
think = text = ""
if raw.lstrip().startswith("{"):
    m = json.loads(raw)["choices"][0]["message"]; think, text = m.get("reasoning_content") or "", m.get("content") or ""
else:
    for ln in raw.splitlines():
        if not ln.startswith("data: {"): continue
        ch = json.loads(ln[6:]).get("choices") or []
        if ch:
            de = ch[0].get("delta", {}); think += de.get("reasoning_content") or ""; text += de.get("content") or ""
import re
# ★决策点 = 思考段里第一次拍板的那个方向词★(思考档的结论在思考里就定了, 正文只是誊写)。
# 跳过"是 "大盘上涨" 还是 "大盘下跌""这种列选项的句子(后面紧跟 or / 或 / /)。
# 序列截在这个词后面: 之后的思考(09-23 实测贪心在这里掉进 2000 字周期复读 24 遍)与正文都不进训练 ——
# 对版只许改结论, 论证一字不动, 截断后两版逐 token 对齐, 决策行就是被换的那一两个 token。
wd, td = "大盘" + wrong_dir, "大盘" + truth
cut = None
# 列选项的两种写法都跳过: `"大盘上涨" or …`(看后面) 与 `… or "大盘上涨"?`(看前面, 09-24 挂 ③ 的轨迹里实撞)
def listing(t, m):
    return (re.match(r'["”」]?\s*(or|或|/|还是)', t[m.end():m.end() + 8]) is not None or
            re.search(r'(or|或|/|还是)\s*["“「]?$', t[max(0, m.start() - 8):m.start()]) is not None)
for m in re.finditer(re.escape(wd), think):
    if listing(think, m): continue
    cut = m; break
if cut is not None:
    orig = think[:cut.end()]
    fixed = think[:cut.start()] + td
    where = "思考段第 %d 字: …%s" % (cut.start(), think[max(0, cut.start() - 60):cut.end()].replace("\n", "⏎"))
elif wd in text:
    orig = think + "</think>" + text
    fixed = orig.replace(wd, td)
    where = "正文(思考段没拍板), 全部 %d 处" % text.count(wd)
else:
    print("模型原输出里没有 '%s' —— 早上本来就没判错, 没有可学的" % wd); sys.exit(1)
print("决策点: " + where)
items = []
if mode == "add" and os.path.exists(dst):
    items = json.load(open(dst, encoding="utf-8"))["items"]
    if any(it["orig_report"] == orig for it in items):
        print("这一处已在语料里(同一段前文同一个拍板), 不重复加"); sys.exit(3)
items.append({"body_prompt": prompt, "body_report": fixed, "orig_report": orig, "date": r["_note"]["date"], "symbol": "大盘"})
json.dump({"items": items}, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
print("复盘语料: 提示 %d 字 / 训练段 %d 字 (%s → %s) | 语料共 %d 条" % (len(prompt), len(orig), wd, td, len(items)))
PYEOF
}

# 挂上 ③ 重跑早上那条请求, 看结论翻没翻。部署同路: ds4-server(现役 ①②) + --posttrain, 请求字节 = 早上那条。
#   $1 = 请求 JSON  $2 = ③ 目录(传 - = 不挂, 当对照)  $3 = 输出 SSE 落盘路径
# 读数: 思考段第一次拍板的方向 + 正文里的"大盘上涨/下跌"。★复读判据★: 思考尾部 2000 字在全文出现 ≥3 次 =
# 贪心掉进周期(09-23 基座这条请求实测在 6k 字处拍板后 2000 字周期 24 遍) —— 这时停流, 如实记"复读", 不当结论。
stage_reviewrun(){
    # 第 4 参(可选) = 温度: 给了就覆盖请求里的(真实请求不带温度 = 服务端默认温 1 采样); 挂/不挂 ③ 的 A/B 给 0, 两臂都确定性, 差别只来自 ③
    local REQ="${1:?请求 JSON}" PT="${2:?③目录或 -}" OUTF="${3:?输出路径}" TEMP="${4:-}"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3; need_idle
    local extra=(); [ "$PT" = - ] || extra=(--posttrain "$PT")
    bash "$SC/serve_1m_spark.sh" start "" "" ${extra[@]+"${extra[@]}"} >>"$LOGF" 2>&1 || DIE "服务没起来(看 $LOGF)"
    LOG "重跑 $(basename "$REQ") 挂 ③=$PT"
    python3 - "$REQ" "$OUTF" "$TEMP" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import json, re, sys, urllib.request
req, outf, temp = sys.argv[1], sys.argv[2], sys.argv[3]
b = json.load(open(req, encoding="utf-8")); b.pop("_note", None); b["stream"] = True
if temp != "": b["temperature"] = float(temp)
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
rq = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(b, ensure_ascii=False).encode(),
                            {"Content-Type": "application/json"})
think = text = ""; loop = False; n = 0
with op.open(rq, timeout=7200) as r, open(outf, "w", encoding="utf-8") as fo:
    for ln in r:
        ln = ln.decode("utf-8", "replace"); fo.write(ln)
        if not ln.startswith("data: {"): continue
        ch = json.loads(ln[6:]).get("choices") or []
        if not ch: continue
        de = ch[0].get("delta", {}); think += de.get("reasoning_content") or ""; text += de.get("content") or ""; n += 1
        # 复读判据: 尾部 2000 字在全文里出现 ≥3 遍 = 逐字周期, 停流。正文开写后查正文(10-02 实撞: 不挂 ③ 温 0 跑 601069,
        #   正文第 12217 字起以 17576 字为周期逐字重复整篇报告 + 决策 JSON, 只查思考段时这一趟永远不结束)
        body = text if text else think
        if n % 500 == 0 and len(body) > 8000 and body.count(body[-2000:]) >= 3:
            loop = True; break
def first(t):
    for m in re.finditer(r"大盘(上涨|下跌)", t):
        if re.match(r'["”」]?\s*(or|或|/|还是)', t[m.end():m.end() + 8]): continue
        if re.search(r'(or|或|/|还是)\s*["“「]?$', t[max(0, m.start() - 8):m.start()]): continue
        return m.group(0), m.start(), t[max(0, m.start() - 80):m.end()].replace("\n", "⏎")
    return None
f = first(think)
print("思考 %d 字 / 正文 %d 字%s" % (len(think), len(text), (" / ★%s复读(尾部 2000 字周期 ≥3 遍), 已停流★" % ("正文" if text else "思考段")) if loop else ""))
print("思考段第一次拍板: %s" % ("%s @%d …%s" % f if f else "无"))
c = re.findall(r"大盘(?:上涨|下跌)", text)
print("正文结论: %s" % (c[0] if c else "(正文没写到结论)"))
print("REVIEW_VERDICT content=%s loop=%d" % (c[0][2:] if c else "none", int(loop)))   # 迭代段按这一行判停
js = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)   # CFO 报告末尾的决策 JSON(第八版 ③ 的产品级对照: 目标价/止损/风报比挂没挂 ③ 变没变)
if js:
    try:
        j = json.loads(js[-1])
        print("CFO_JSON " + json.dumps({k: j.get(k) for k in ("symbol", "entry_price", "target_price", "stop_loss", "risk_return_ratio", "expected_return")}, ensure_ascii=False))
    except Exception as ex:
        print("CFO_JSON 解析失败: %s | %s" % (ex, js[-1][:200]))
PYEOF
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1
}

# 自我迭代(2026-09-24, 用户令"继续"): 单点 ③ 能翻第一次拍板, 但模型会在训练没见过的新位置拍回去(fable5 09-24)。
# 每轮: 在上一轮挂 ③ 的轨迹上找第一个错方向的拍板 → 追加为新决策点 → 全部样本从 ② 起按部署同路重解 ③ → 挂上重跑。
# 停: 正文结论 = 真值(翻了) / 新轨迹里没有错方向拍板 / 与已有样本重复 / 到轮数上限。结束(含中途 DIE)一律把现役服务起回来。
#   $1 = 请求 JSON  $2 = 复盘真值  $3 = 轮数上限  $4.. = 种子输出(第一条是基座输出, 其余是已有的挂 ③ 轨迹)
stage_reviewiter(){
    local REQ="${1:?请求 JSON}" TRUTH="${2:?真值}" MAXR="${3:?轮数上限}"; shift 3
    local IT="$D2/review_iter_$(date +%Y%m%d_%H%M)"; mkdir -p "$IT"
    local TAG="pt-$(date +%Y%m%d)"
    trap 'bash "$SC/serve_1m_spark.sh" start >>"$LOGF" 2>&1' EXIT
    [ -e "$FTD/$TAG" ] && mv "$FTD/$TAG" "$IT/prev_$TAG"
    local first=1 src rc
    for src in "$@"; do
        if [ $first = 1 ]; then stage_review "$REQ" "$src" "$TRUTH" || DIE "种子 $src 拼不出样本"; first=0
        else stage_review "$REQ" "$src" "$TRUTH" add; rc=$?; [ $rc = 0 ] || [ $rc = 3 ] || DIE "种子 $src 拼装失败"; fi
    done
    local r OUT CAND V
    for r in $(seq 1 "$MAXR"); do
        LOG "★迭代第 $r 轮★ 语料 $(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["items"]))' "$D2/samples.json") 条"
        OUT="$(stage_sft 6 1,10 "" 0 1,10 | tail -1)"
        [ -d "$OUT" ] || DIE "第 $r 轮解算没出候选"
        # 候选固定取 τ=6 ρ=10 λ=1: τ 按部署同路的余量定(09-24 实测 τ=6 翻得动), ρ=10 = 约束行权重高档(别处少动)
        CAND=$(ls -d "$OUT"/cand_g1_t6_r10_l1_* 2>/dev/null | head -1); [ -n "$CAND" ] || DIE "第 $r 轮没有 τ6/ρ10/λ1 候选"
        mv "$OUT" "$IT/r${r}_pt"; CAND="$IT/r${r}_pt/$(basename "$CAND")"
        stage_reviewrun "$REQ" "$CAND" "$IT/r${r}.sse"
        V=$(grep "REVIEW_VERDICT" "$LOGF" | tail -1)
        LOG "第 $r 轮: $V"
        case "$V" in *"content=$TRUTH "*) LOG "★第 $r 轮正文结论翻成 $TRUTH, 停★ ③ = $CAND"; break;; esac
        stage_review "$REQ" "$IT/r${r}.sse" "$TRUTH" add; rc=$?
        [ $rc = 0 ] || { LOG "第 $r 轮轨迹里没有新的错方向拍板(rc=$rc), 停"; break; }
    done
    LOG "REVIEWITER_DONE 产物 $IT"
}

# ================= ★第七版(2026-09-29, back.md §14)★: 真实请求 → 模型卡采样 N 份 → 次日行情打分 =================
# 料在 $D2/req/<kind>_<日期>[_<代码>].json(Mac 侧 qtf_requests_mac.sh 重建后推过来, _note 里带次日 OHLC / 大盘真值)。
# 一份样本一个目录 $D2/samp/<请求名>/<③标签>/: s<k>.sse(原始流) s<k>.think.txt s<k>.content.txt
#   s<k>.prompt.ids / s<k>.gen.ids(引擎真吃/真吐的 id, 从服务端 --trace 取, 不重新分词) s<k>.meta reward.txt
TRACE_CUR="$D2/trace/current"      # serve_1m_spark.sh 起服时写的本趟 --trace 路径
REWARD="$BEN/posttrain_reward"

# sample <req.json> <N> [③目录|-] [标签] [temperature] [top_p] [②反修目录]: 停服 → 按参数重起(挂/不挂 ③, 换/不换 ②) → N 条请求(seed 1..N) → 打分。
# 为什么每次重起服务: 现役服务不知道挂的是哪份 ③, 采样必须与判决同一个"服务态"; 实例锁只许一个大模型进程。
# 第 7 参数换 ②(2026-09-30, 用户令"编程域语料带英文, 试一试有没有同样的问题"): 同一条真实请求挂另一份侧车再采 N 份, 与 base 并排;
#   不给 = serve_1m_spark.sh 的现役默认(金融 grrb)。挂的哪份写进样本目录 zchain.txt, 事后能对上。
# 为什么 seed 显式给: 模型卡采样默认路下服务端不带 seed 是随机的, N 份要可复现且互不相同。
# ★采样参数显式写进请求, 值 = 服务端默认(模型卡配方 温 1.0 / top_p 1.0 / min_p 0)★: 采样必须与产品路同一条分布, 显式写只是
#   为了 seed 可复现与日志自说明。09-29 实撞: 这条 CFO 请求两份样本全烂(一份 20 万 token 才停, 一份 5.1 万 token 被看门狗杀),
#   当晚误判为"纯采样尾巴脏"改走 top_p 0.95 —— 真因是设备采样核随机数撞 u == 1.0f 时 Gumbel 键 +∞, 每位以 0.77%/流的概率
#   硬塞一个均匀随机的词(逐位复算 200,119 个 token 里 1,880 个), top_p 0.95 只是把保留集缩小到撞不到, 是遮丑不是修。
#   核已修(cuda_v41_sample.inc.cu v41_u01), 这里退回服务端默认; 再改 top_p 是产品决定, 不在这个脚本里替用户定。
# 单条总时长上限 1700 s = qtf 里 LLM 的 timeout(modules/deepseek.py _LOCAL_TIMEOUT): 产品在那一刻挂断, 采样照抄
#   (挂断 = 关连接, 服务端每 16 token 探一次对端就停; 这种样本没有 JSON, 奖励器判 INVALID, 与产品里"这次 CFO 失败"同义)。
stage_sample(){
    local REQ="${1:?请求 JSON}" N="${2:?份数}" PT="${3:--}" TAG="${4:-}" TEMP="${5:-1.0}" TOPP="${6:-1.0}" ZC2="${7:-}"
    [ -s "$REQ" ] || DIE "没有 $REQ"
    [ -x "$REWARD" ] || make -C "$ROOT/gguf-tools" posttrain_reward >>"$LOGF" 2>&1 || DIE "奖励器编译失败"
    [ -z "$ZC2" ] || ls "$ZC2"/gr_L*.bin >/dev/null 2>&1 || DIE "② 目录缺或没有 gr_L*.bin: $ZC2"
    [ -n "$TAG" ] || { if [ "$PT" = - ]; then TAG=base; else TAG="$(basename "$PT")"; fi; }
    local NAME; NAME="$(basename "$REQ" .json)"
    local OUT="$D2/samp/$NAME/$TAG"; mkdir -p "$OUT"
    cp -n "$REQ" "$OUT/req.json" 2>/dev/null || cp "$REQ" "$OUT/req.json"
    echo "$PT" > "$OUT/posttrain.txt"
    echo "${ZC2:-$ZCH}" > "$OUT/zchain.txt"
    local need=0 k
    for k in $(seq 1 "$N"); do [ -s "$OUT/s$k.gen.ids" ] || need=1; done
    if [ "$need" = 1 ]; then
        local extra=(); [ "$PT" = - ] || extra=(--posttrain "$PT")
        bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3; need_idle
        bash "$SC/serve_1m_spark.sh" start "" "$ZC2" ${extra[@]+"${extra[@]}"} >>"$LOGF" 2>&1 || DIE "服务没起来(看 $LOGF)"
        local TR; TR=$(cat "$TRACE_CUR" 2>/dev/null); [ -n "$TR" ] && [ -e "$TR" ] || DIE "服务没开 --trace($TRACE_CUR 空)"
        LOG "采样 $NAME ×$N 挂 ③=$PT ②=${ZC2:-现役默认} 温 $TEMP top_p $TOPP → $OUT (trace $TR)"
        for k in $(seq 1 "$N"); do
            [ -s "$OUT/s$k.gen.ids" ] && { LOG "  s$k 已有, 跳过"; continue; }
            python3 - "$OUT" "$k" "$TR" "$TEMP" "$TOPP" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import json, os, re, sys, time, urllib.request
out, k, tr = sys.argv[1], int(sys.argv[2]), sys.argv[3]
b = json.load(open(os.path.join(out, "req.json"), encoding="utf-8")); note = b.pop("_note", {})
b["stream"] = True; b["seed"] = k
b["temperature"] = float(sys.argv[4]); b["top_p"] = float(sys.argv[5]); b["min_p"] = 0.0   # 其余字段原样(stop 词照旧)
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
rq = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(b, ensure_ascii=False).encode(),
                            {"Content-Type": "application/json"})
think = text = ""; finish = "?"; usage = None; t0 = time.time(); ntok = 0
WALL = 1700.0                                # qtf 的 LLM timeout: 总时长, 不是单次读的间隔
tr_pos = os.path.getsize(tr)                 # 只看本条请求之后新写的 trace
with op.open(rq, timeout=600) as r, open(os.path.join(out, "s%d.sse" % k), "w", encoding="utf-8") as fo:
    try:
        for ln in r:
            ln = ln.decode("utf-8", "replace"); fo.write(ln)
            if time.time() - t0 > WALL: finish = "client_abort:wall%.0fs" % WALL; break   # 关连接 = 产品那一刻的挂断
            if not ln.startswith("data: {"): continue
            j = json.loads(ln[6:]); usage = j.get("usage") or usage
            for ch in j.get("choices") or []:
                de = ch.get("delta", {}); think += de.get("reasoning_content") or ""; text += de.get("content") or ""; ntok += 1
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    except Exception as e:                   # 断流: 如实记, 不重试
        finish = "client_abort:%s" % type(e).__name__
el = time.time() - t0
try: avail = [int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0]
except Exception: avail = -1
open(os.path.join(out, "s%d.think.txt" % k), "w", encoding="utf-8").write(think)
open(os.path.join(out, "s%d.content.txt" % k), "w", encoding="utf-8").write(text)
# 引擎真吃/真吐的 id: 本条请求结束后 trace 里最后一块 "--- token ids: prompt P, generated G ---"。
# 服务端在发完 [DONE] 之后才写这一块, 所以要等它落盘(最多 30 s), 不能一读到流尾就去翻。
# ★按字节切, 再解码★(09-29 实撞): tr_pos 是 getsize 的字节数, 而 trace 里有整段中文提示(一字 3 字节), 拿字节偏移去切
#   .read() 出来的字符串, 四条请求之后差了十几万个位置, 第 4 份的整块被切掉 —— 报"trace 里没有本条的 token id 块",
#   其实块 22:25:38 就写好了。前三份只是差得还不够多。
m = None
for _ in range(60):
    tail = open(tr, "rb").read()[tr_pos:].decode("utf-8", "replace")
    for m in re.finditer(r"--- token ids: prompt (\d+), generated (\d+) ---\nprompt:([ \d]*)\ngenerated:([ \d]*)\n", tail): pass
    if m is not None: break
    time.sleep(0.5)
if m is None:
    print("  ★s%d: trace 里没有本条的 token id 块(finish=%s), 这份不可取料★" % (k, finish)); sys.exit(1)
P, G = int(m.group(1)), int(m.group(2))
pids = m.group(3).split(); gids = m.group(4).split()
if len(pids) != P or len(gids) != G:
    print("  ★s%d: trace id 块计数不符 P %d/%d G %d/%d★" % (k, len(pids), P, len(gids), G)); sys.exit(1)
open(os.path.join(out, "s%d.prompt.ids" % k), "w").write("\n".join(pids) + "\n")
open(os.path.join(out, "s%d.gen.ids" % k), "w").write("\n".join(gids) + "\n")
open(os.path.join(out, "s%d.meta" % k), "w").write("seed=%d temperature=%s top_p=%s min_p=0 finish=%s prompt=%d generated=%d think_chars=%d content_chars=%d elapsed_s=%.0f mem_avail_mb=%d usage=%s\n"
                                                    % (k, sys.argv[4], sys.argv[5], finish, P, G, len(think), len(text), el, avail, json.dumps(usage)))
print("  s%d: finish=%s 提示 %d / 生成 %d token, 思考 %d 字 / 正文 %d 字, %.0fs, 采完 MemAvailable %d MB" % (k, finish, P, G, len(think), len(text), el, avail))
PYEOF
            pgrep -x ds4-server >/dev/null || { LOG "★服务在第 $k 份中途没了(看门狗/崩溃, 看 $HOME/ds4-server-watchdog.log), 停采★"; break; }
        done
        bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1
    fi
    stage_reward "$OUT"
}

# reward <样本目录>: 每份正文 → 奖励器 → reward.txt(一行一份: k rc 奖励行)。个股: 次日 OHLC 来自 req.json 的 _note;
# 大盘: 真值来自 _note.truth。早盘真跑那份(个股 _note.live_report)也打一行 "live"(只报; 它是另一台模型写的, 不进组)。
stage_reward(){
    local OUT="${1:?样本目录}"
    python3 - "$OUT" "$REWARD" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import glob, json, os, subprocess, sys
out, tool = sys.argv[1], sys.argv[2]
note = json.load(open(os.path.join(out, "req.json"), encoding="utf-8")).get("_note", {})
kind, date = note.get("kind"), note.get("date")
rows = []
def score(path):
    if kind == "cfo":
        ohlc = os.path.join(out, "ohlc.txt")
        open(ohlc, "w").write("".join("%s %s %s %s %s\n" % (b["date"], b["open"], b["high"], b["low"], b["close"]) for b in note.get("ohlc") or []))
        r = subprocess.run([tool, "stock", path, ohlc, date], capture_output=True, text=True)
    else:
        r = subprocess.run([tool, "market", path, note.get("truth") or "?"], capture_output=True, text=True)
    return r.returncode, (r.stdout.strip() or r.stderr.strip())
files = sorted(glob.glob(os.path.join(out, "s*.content.txt")), key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))
for p in files:
    k = int(os.path.basename(p)[1:].split(".")[0]); rc, line = score(p); rows.append((str(k), rc, line))
if kind == "cfo" and note.get("live_report"):
    lp = os.path.join(out, "live.content.txt"); open(lp, "w", encoding="utf-8").write(note["live_report"])
    rc, line = score(lp); rows.append(("live", rc, line))
with open(os.path.join(out, "reward.txt"), "w") as f:
    f.write("# k rc " + ("R_pnl R_bin trade entry target stop rr next_date hit R_pnl_t1 R_bin_t1" if kind == "cfo" else "R_bin pred") + "  (rc 0=打出分 2=INVALID 3=还没次日行情)\n")
    for k, rc, line in rows: f.write("%s %d %s\n" % (k, rc, line))
valid = [(k, l) for k, rc, l in rows if rc == 0 and k != "live"]
print("[奖励] %s %s %s: %d 份可判 / %d 份; %s" % (kind, date, note.get("symbol", ""), len(valid), len([r for r in rows if r[0] != "live"]),
      " | ".join("s%s→%s" % (k, " ".join(l.split()[:2])) for k, l in valid) or "无可判样本"))
for k, rc, line in rows:
    if rc != 0 or k == "live": print("   %s: rc=%d %s" % (k, rc, line[:120]))
PYEOF
}

# solve3 <样本目录> [η列表] [λ列表]: reward.txt → 组内减均值得优势 → 清单 → v41_amp_run --adv-list(部署同路取料 + 解) → 候选;
# 最后一行打印 J_out(留一预测)最好的候选目录。个股奖励取带幅度列(R_pnl), 大盘取 R_bin; 只用 rc=0 的样本; 早盘那份(live)不进组。
# ★行预算 ROWS_CAP(09-29 内存账, 详见 v41_adv_run.inc.c 头注释)★: 113.6 GB 模型驻留后整机余 ~10 GB, 再扣出口头 1.3 GB 与
#   取料窗(一份生成段 2.1 万行 bf16 = 1.3 GB); 设备 ye 缓冲 = 行数 × 6 × 5120 × 4 B: 12000 行 = 1.47 GB + 方向表 0.25 GB,
#   合计 ~4.4 GB 落在余量里, 看门狗红线 2.5 GB 之上还留 3 GB。行按每份样本系统抽样(等距), 不是截头。
# ★折号 = 样本序号★: demo 只有一个请求, 留一只能按样本分折(同请求另一条轨迹算"没解过的"); 多请求夜跑时这里改按请求分折。
ROWS_CAP=12000
# 解算趟自带看门狗(铁律: 大内存运行必须有 RSS 预算 + 看门狗): 与 serve_1m_spark.sh 同口径, MemAvailable 连续两次 < 红线就杀解算器。
solve_guard(){
    local a bad=0
    while pgrep -x v41_amp_run >/dev/null; do
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$a" -lt 2500 ]; then bad=$((bad+1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then LOG "★解算看门狗: MemAvailable ${a} MB < 2500 连续两次, 杀 v41_amp_run★"; pkill -x v41_amp_run; sleep 3; pkill -9 -x v41_amp_run 2>/dev/null; return 1; fi
        sleep 5
    done
}
stage_solve3(){
    local SD="${1:?样本目录}" ETA="${2:-0.1,0.3,1}" LAM="${3:-0.3,1,3,10}"
    [ -s "$SD/reward.txt" ] || DIE "没有 $SD/reward.txt(先跑 sample)"
    [ -x "$AMP/v41_amp_run" ] || make -C "$ROOT/gguf-tools" v41_amp_run >>"$LOGF" 2>&1 || DIE "解算器编译失败"
    local W="$SD/solve"; mkdir -p "$W"
    python3 - "$SD" "$W" <<'PYEOF' 2>&1 | tee -a "$LOGF" || DIE "清单拼装失败"
import json, os, sys
sd, w = sys.argv[1], sys.argv[2]
rows = []
for ln in open(os.path.join(sd, "reward.txt"), encoding="utf-8"):
    f = ln.split()
    if not f or f[0].startswith("#") or f[0] == "live" or len(f) < 3 or f[1] != "0": continue
    k = int(f[0])
    if not (os.path.exists(os.path.join(sd, "s%d.prompt.ids" % k)) and os.path.exists(os.path.join(sd, "s%d.gen.ids" % k))):
        print("  s%d 有分没有 id(trace 没截到), 不进组" % k); continue
    rows.append((k, float(f[2])))                   # cfo: R_pnl(带幅度) / market: R_bin —— 都在第 3 列
if len(rows) < 2: print("可判样本 %d 份 < 2, 组内比不出高低" % len(rows)); sys.exit(1)
mean = sum(r for _, r in rows) / len(rows)
adv = [(k, R - mean) for k, R in rows]
if all(abs(a) < 1e-12 for _, a in adv): print("N 份奖励全同(%.5f): 这份请求没有裁量, 无可重分" % mean); sys.exit(1)
lines, maxlen = [], 0
for fold, (k, a) in enumerate(adv):
    pids = open(os.path.join(sd, "s%d.prompt.ids" % k)).read().split()
    gids = open(os.path.join(sd, "s%d.gen.ids" % k)).read().split()
    ids = os.path.join(w, "ids_%d.txt" % k)
    open(ids, "w").write("\n".join(pids + gids) + "\n")
    maxlen = max(maxlen, len(pids) + len(gids))
    lines.append("%s %s/top_%d.bin %s/rms_%d.bin %.6f %d %d" % (ids, w, k, w, k, a, len(pids), fold))   # 末列 = 留一折号(按样本)
open(os.path.join(w, "list.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(w, "meta.env"), "w").write('MAXLEN="%d"\nFIRST="%s"\n' % (maxlen + 8, lines[0].split()[0]))
print("[优势] " + " ".join("s%d=%+.5f" % (k, a) for k, a in adv) + "  (组均值 %.5f; 1 组 %d 份; 留一按样本 %d 折)" % (mean, len(adv), len(adv)))
PYEOF
    [ "${PIPESTATUS[0]}" = 0 ] || return 1
    . "$W/meta.env"
    local OUT="$W/pt"; rm -rf "$OUT"; mkdir -p "$OUT"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3; need_idle
    LOG "③ 解算(第七版, η=$ETA λ=$LAM, L39, 行预算 $ROWS_CAP) → $OUT; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
    ( sleep 20; solve_guard ) &
    local GUARD=$!
    # --score-chunk 512(09-30 实撞): 引擎默认分块 2048 的打分状态(logits 主机+设备各 1.06 GB, hc/q/o 按 2048 行)在模型驻留后的 9 GB 里
    #   装不下第七版的料, 30 s 就被解算看门狗杀; 分块不改输出(预填尺四档逐字节同), 09-24 那次取料也是 512。
    "$AMP/v41_amp_run" "$MDL" "$HFDIR" "$FIRST" "$MAXLEN" "$OUT" --only-layer 39 --capture-ye --adv-list "$W/list.txt" \
        --eta-list "$ETA" --lam-list "$LAM" --base-amp "$ZCH" --mem-budget-mb 110000 --rows-cap "$ROWS_CAP" --score-chunk 512 2>&1 | tee -a "$LOGF" | grep -v "^ds4:" | tail -30
    local SRC="${PIPESTATUS[0]}"
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    [ "$SRC" = 0 ] || DIE "解算失败(rc=$SRC)"
    local BEST; BEST=$(grep -v '^#' "$OUT/candidates.txt" | sort -k4 -gr | head -1 | cut -d' ' -f1)
    [ -n "$BEST" ] || DIE "没有候选"
    LOG "★J_out 最好的候选: $OUT/$BEST★"
    echo "$OUT/$BEST"
}

# demo <req.json> [N]: ★最小端到端★(用户 09-29: "最小 demo 跑通即可") —— 采 N 份 → 打分 → 解 ③ → 挂 ③ 再采 N 份 → 两份 reward.txt 并排。
stage_demo(){
    local REQ="${1:?请求 JSON}" N="${2:-4}"
    local NAME; NAME="$(basename "$REQ" .json)"
    stage_sample "$REQ" "$N" - base || DIE "基线采样失败"
    local CAND; CAND=$(stage_solve3 "$D2/samp/$NAME/base" | tail -1)
    [ -d "$CAND" ] || DIE "没有 ③ 候选($CAND)"
    stage_sample "$REQ" "$N" "$CAND" pt3 || DIE "挂 ③ 采样失败"
    LOG "★demo 读数 $NAME★"
    echo "--- 基线(无 ③) ---"; cat "$D2/samp/$NAME/base/reward.txt"
    echo "--- 挂 ③ ($CAND) ---"; cat "$D2/samp/$NAME/pt3/reward.txt"
    LOG "DEMO_DONE $NAME"
}

# doc3 <料目录> [η列表] [λ列表]: ★复盘语料直接当 ③ 的目标 —— 最小验证★(2026-10-01, 用户"用1加一试一试")。
#   料 = 目录下每篇 *.txt(纯文本, 不带特殊 token), 这里按预训练样子包成 BOS 正文 EOS 再分词(与 docend_corpus_build 同口径;
#   $(cat) 顺手吃掉文件尾的换行, 免得 EOS 前多出一个 "\n" token)。分词走 ./ds4 --dump-tokens: 文本以 BOS 起头时 CLI 按
#   "已渲染"处理, 不套聊天模板(req_render_ids.sh 同一招)。
#   解算复用第七版 --adv-list: 每篇一行, 优势 A=1、提示长 1、折号=篇号 ⇒ 整篇每个 token 都是"把 log p 抬 η nat"的方程,
#   右端只有 η, 没有 FP 教师, 没有奖励。own-pmax 默认 0.95: 模型本来就 ≥0.95 预测对的 token 不进料(方向≈0)。
#   η 网格默认 1,3,10,30: 把 "2" 翻成 "3" 要顶过极强的先验, 差多少 nat 事前不知道, 由 predict.txt 的预测 Δlog p 与 docprobe 真跑定。
#   产物: $FTD/doc-<目录名>/{ids/,list.txt,solve.log,pt/cand_adv_e*_l*/}; 候选挑哪个只认 docprobe 的原始输出, 这里全是预测。
#   第 4 参 K(10-01): 给了就走 ★低秩加性形态★(--lowrank K, 产物 cand_lr<K>_e*_l*/amp_L39.bin, λ 此时是相对岭), 不给 = 增益表。
#   料目录下的 hold/*.txt 是 ★约束料★(A=0: "这里别动"), 低秩形态里是整行向量约束; 它们不筛 p, 全部进料。
#   第 5 参 conf-min(10-01 下午, 复读环正修): 事实行只推"榜首 p ≥ 阈值且榜首 ≠ 料 token"的位置(模型笃定答错 = 真正要改的知识),
#   排版转移("三→。""=→3")不再被推。★给了 conf-min 时事实料按行拆成独立序列★(每行 = 抬头 + 该行 + EOS): 部署时问题单独来, 第二行不该
#   靠第一行的上下文(整篇时 "1+1=→3" 在抄第一句, p 0.86 ⇒ 榜首已是 3 ⇒ 不算答错, 推不到)。约束料仍整篇(它要的正是上下文)。
#   ★行内 "⇒"(10-01 晚, 复盘料): "情况⇒结论" —— ⇒ 前的前缀并入提示段只当上下文, 只推 ⇒ 后的结论段; conf-min 给 0 = 按行拆 +
#   结论段里 p(料)<0.95 的位置全推(复盘事实模型多半"不知道", 没有"笃定答错"的位置可筛)。
stage_doc3(){
    local DOCD="${1:?料目录}" ETA="${2:-1,3,10,30}" LAM="${3:-1}" K="${4:-}" CONF="${5:-}"
    [ -d "$DOCD" ] || DIE "没有料目录 $DOCD"
    local NAME; NAME="$(basename "$DOCD")"
    # 只重建 ids/ 与清单, pt/ 里的候选按 η/λ 命名不会撞, 留着 —— 第二趟补网格时不能把第一趟已经探过针的候选删掉
    local W="$FTD/doc-$NAME"; rm -rf "$W/ids" "$W/list.txt"; mkdir -p "$W/ids"
    [ -x "$AMP/v41_amp_run" ] || make -C "$ROOT/gguf-tools" v41_amp_run >>"$LOGF" 2>&1 || DIE "解算器编译失败"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    # ★提示段 = 固定抬头★(10-01 实撞): 第七版取料把"提示末块"跑满解码器, 而逐专家输出只在 prefill GEMM 路物化 ——
    #   块 ≤ DS4_V41_GEMV_MAX_TOK(8, ds4_gpu_v41.h) 个 token 走解码 GEMV 路, 钩子拿不到 ye 直接停车。提示只有 BOS 一个 token 时
    #   正是这样炸的(0 行 < 需要 23)。抬头的 token 全在提示段, 一个都不进 ③(生成段从抬头末位起), 它只是给引擎一个够大的块。
    local HDR='下面是一份需要牢牢记住的参考文档，请逐字阅读并记住其中的每一句话。'$'\n\n'
    tok_dump(){ ./ds4 -m "$MDL" --dump-tokens --prompt-file "$1" > "$2" 2>"$2.err" </dev/null || { tail -5 "$2.err"; return 1; }
                head -1 "$2" | tr -d '[] ' | tr ',' '\n' | grep -v '^$' > "$3"; [ -s "$3" ]; }
    printf '<｜begin▁of▁sentence｜>%s' "$HDR" > "$W/ids/hdr.render.txt"
    tok_dump "$W/ids/hdr.render.txt" "$W/ids/hdr.dump" "$W/ids/hdr.ids" || DIE "抬头分词失败"
    local P; P=$(wc -l < "$W/ids/hdr.ids")
    [ "$P" -gt 8 ] || DIE "抬头只有 $P 个 token, 末块会走解码 GEMV 路(要 > 8)"
    local maxlen=0 n=0 nh=0 lines=() f b cnt A src
    # 事实料按行拆(只在给了 conf-min 时): 每个非空行落成 $W/ids/<名>_l<行号>.txt 当一篇; 约束料整篇
    local -a SRCS=()
    for f in "$DOCD"/*.txt; do
        [ -f "$f" ] || continue
        if [ -n "$CONF" ]; then
            local ln=0 line lf
            while IFS= read -r line || [ -n "$line" ]; do
                [ -n "$line" ] || continue; ln=$((ln+1)); lf="$W/ids/$(basename "$f" .txt)_l$ln.txt"
                # ★行内 "⇒" = 情况 ⇒ 结论★(10-01 晚, 复盘料实撞): 复盘记录天然是"日期/主语/数据 ⇒ 结论", 而 conf-min 只按概率筛,
                #   分不开"地址"(2026-09-23 A股大盘实际走势)和"结论"(下跌) —— 两边模型都不笃定, 一起被推。推了地址的后果是探针
                #   把问题原样复述一遍就 EOS(Q2 实撞), 决策 token 根本走不到。有 ⇒ 的行: 前缀并入提示段(只当上下文, 一个 token 不推),
                #   只推 ⇒ 后面的结论段; 没 ⇒ 的行照旧整行当生成段。⇒ 本身不进文本。
                rm -f "$lf.pre" "$lf.chat"
                case "$(basename "$f")" in
                  chat_*)
                    # ★chat_*.txt: 问⇒答 按部署同路渲染★(10-01 晚, 第 6 趟实撞): 文档里写"问：…答：⇒下跌"推的是文档里"答："之后的位置,
                    #   聊天模板里 <｜Assistant｜></think> 之后的答案位是另一个 x, ③ 在那里不点火。这里直接按 CLI/服务端的不思考渲染
                    #   (core_chat_frame.c ds4_encode_chat_prompt: BOS <｜User｜>问 <｜Assistant｜> </think>)拼成已渲染文本, 提示段 = 到 </think>,
                    #   只推答案段 —— 取料位置 = 产品答题位置。没有 ⇒ 的行不合法(整行当答案就没有问题了)。
                    case "$line" in *⇒*) ;; *) DIE "$f 第 $ln 行没有 ⇒(chat_ 料必须是 问⇒答)";; esac
                    printf '<｜begin▁of▁sentence｜><｜User｜>%s<｜Assistant｜></think>' "${line%%⇒*}" > "$lf.pre"
                    printf '%s%s' "$(cat "$lf.pre")" "${line#*⇒}" > "$lf"; : > "$lf.chat";;
                  *) case "$line" in
                       *⇒*) printf '%s' "${line%%⇒*}" > "$lf.pre"; printf '%s%s' "${line%%⇒*}" "${line#*⇒}" > "$lf";;
                       *)   printf '%s' "$line" > "$lf";;
                     esac;;
                esac
                SRCS+=("$lf")
            done < "$f"
        else SRCS+=("$f"); fi
    done
    for f in "$DOCD"/hold/*.txt; do [ -f "$f" ] && SRCS+=("$f"); done
    for f in "${SRCS[@]}"; do
        b="$(basename "$f" .txt)"; A=1.0
        case "$f" in */hold/*) b="hold_$b"; A=0.0; nh=$((nh+1));; esac
        local CHAT=0; [ -e "$f.chat" ] && CHAT=1   # chat_ 料: 文件本身已是渲染好的字节(BOS 起头), 不加抬头
        if [ $CHAT = 1 ]; then printf '%s<｜end▁of▁sentence｜>' "$(cat "$f")" > "$W/ids/$b.render.txt"
        else printf '<｜begin▁of▁sentence｜>%s%s<｜end▁of▁sentence｜>' "$HDR" "$(cat "$f")" > "$W/ids/$b.render.txt"; fi
        tok_dump "$W/ids/$b.render.txt" "$W/ids/$b.dump" "$W/ids/$b.ids" || DIE "分词失败 $f"
        cnt=$(wc -l < "$W/ids/$b.ids")
        [ "$cnt" -ge $((P+3)) ] || [ $CHAT = 1 ] || DIE "$f 只分出 $cnt 个 token(抬头 $P; 看 $W/ids/$b.dump)"
        # 抬头在整篇里必须分出同一串 id(分词边界: 抬头以空行结尾), 否则 nprompt 指错位置
        [ $CHAT = 1 ] || cmp -s <(head -n "$P" "$W/ids/$b.ids") "$W/ids/hdr.ids" || DIE "$f: 抬头在整篇里分词变了(前 $P 个 id 对不上 hdr.ids)"
        # ⇒ 行: 提示段 = 抬头 + 前缀。前缀单独分一遍, 必须是整句 id 的前缀(切点落在一个 token 中间就对不上 ⇒ 换切点)
        local PL="$P"
        if [ -s "$f.pre" ]; then
            if [ $CHAT = 1 ]; then cp "$f.pre" "$W/ids/$b.pre.render.txt"
            else printf '<｜begin▁of▁sentence｜>%s%s' "$HDR" "$(cat "$f.pre")" > "$W/ids/$b.pre.render.txt"; fi
            tok_dump "$W/ids/$b.pre.render.txt" "$W/ids/$b.pre.dump" "$W/ids/$b.pre.ids" || DIE "前缀分词失败 $f"
            PL=$(wc -l < "$W/ids/$b.pre.ids")
            cmp -s <(head -n "$PL" "$W/ids/$b.ids") "$W/ids/$b.pre.ids" || DIE "$f: ⇒ 落在一个 token 中间(前缀 $PL 个 id 与整句对不上), 换个切点"
            [ "$cnt" -ge $((PL+2)) ] || DIE "$f: ⇒ 后面没有结论 token"
        fi
        [ "$cnt" -gt "$maxlen" ] && maxlen=$cnt
        lines+=("$W/ids/$b.ids $W/ids/$b.top.bin $W/ids/$b.rms.bin $A $PL $n")
        LOG "料 $b(A=$A): $cnt token(BOS+抬头 $P+正文+EOS), 提示段 $PL, 进料 $((cnt-PL)) 行; 末 3 个 id $(tail -3 "$W/ids/$b.ids" | tr '\n' ' ')"
        n=$((n+1))
    done
    [ "$n" -gt "$nh" ] || DIE "$DOCD 下没有事实料(*.txt; hold/ 只是约束)"
    printf '%s\n' "${lines[@]}" > "$W/list.txt"
    local OUT="$W/pt"; mkdir -p "$OUT"
    local LR=(); [ -z "$K" ] || LR=(--lowrank "$K"); [ -z "$CONF" ] || LR+=(--conf-min "$CONF")
    local FORM="增益表"; [ -z "$K" ] || FORM="低秩 K=$K"
    local SEL=""; [ -z "$CONF" ] || { SEL="(按行拆, 只推榜首≥$CONF 且答错的位置)"; [ "$CONF" = 0 ] && SEL="(按行拆, 推 p(料)<0.95 的位置; ⇒ 行只推结论段)"; }
    LOG "③ 解算(事实料 $((n-nh)) 篇$SEL + 约束料 $nh 篇, η=$ETA λ=$LAM, L39, 形态=$FORM, 提示段=抬头 $P token 起) → $OUT; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
    ( sleep 20; solve_guard ) &
    local GUARD=$!
    "$AMP/v41_amp_run" "$MDL" "$HFDIR" "${lines[0]%% *}" "$((maxlen+8))" "$OUT" --only-layer 39 --capture-ye --adv-list "$W/list.txt" \
        --eta-list "$ETA" --lam-list "$LAM" --base-amp "$ZCH" --mem-budget-mb 110000 --rows-cap "$ROWS_CAP" --score-chunk 512 ${LR[@]+"${LR[@]}"} \
        2>&1 | tee "$W/solve.log" | grep -v "^ds4:" | tail -40
    local SRC="${PIPESTATUS[0]}"
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    [ "$SRC" = 0 ] || DIE "解算失败(rc=$SRC, 见 $W/solve.log)"
    echo "--- candidates.txt ---"; cat "$OUT/candidates.txt"
    LOG "DOC3_DONE $W"
}

# docprobe <③目录|-> <标签> <问题>...: 挂/不挂 ③ 走 CLI 聊天模板(产品同一串字节: 默认不思考, 空 system), 温 0 贪心逐题问。
#   ★一题一次装载(~75 s)★: 几道题塞进一个提示会让前一题的答案进上下文污染后一题("1+1=3"在上下文里, 2+2 的读数就不干净)。
#   -n 96 只是不让它写散文: 判"停不停"另有尺(上限地板 128k), 这里只看内容。脚本不判对错, 原始输出原样落盘并打印, 判读归人。
#   第 3 参 = 温度: "-" = 温 0 贪心(主读数); 给数字 = 模型卡采样口径(top_p/min_p 走 CLI 默认, 种子固定 1 保证可复现), 产品路是采样,
#   贪心翻了还要看采样下翻不翻。采样趟的产物带 .tN 后缀, 与贪心趟并排。
# ③ 的挂法(10-01 下午): 两种形态都是 --zchain ② --posttrain ③ —— 引擎 core_v41_amp.c 已把 ③ 目录的 amp_Lnn.bin 与 ② 的按秩拼接
#   (gr_Lnn.bin 照旧逐元素相乘)。此前低秩候选靠"配对目录"(② 软链 + 候选 amp 当 ② 挂)过渡, 门 = 同一候选两种挂法输出逐字节同(fable5 10-01)。
#   "pair=" 前缀保留给这道门自己: pt_args pair=<③> 仍拼配对目录。
pt_args(){
    local PT="$1"
    case "$PT" in
      pair=*) PT="${PT#pair=}"
        local PAIR="$PT/pair" f
        rm -rf "$PAIR"; mkdir -p "$PAIR"
        for f in "$ZCH"/*; do ln -s "$(readlink -f "$f")" "$PAIR/$(basename "$f")"; done
        for f in "$PT"/amp_L*.bin; do
            [ -e "$ZCH/$(basename "$f")" ] && { echo "★② 自己有 $(basename "$f"), 配对目录不等价于 ②+③★" >&2; return 1; }
            ln -sf "$(readlink -f "$f")" "$PAIR/$(basename "$f")"
        done
        echo "--zchain $PAIR";;
      *) echo "--zchain $ZCH --posttrain $PT";;
    esac
}
stage_docprobe(){
    local PT="${1:?③目录或 -}" TAG="${2:?标签}" TEMP="${3:?温度或 -}"; shift 3
    [ $# -ge 1 ] || DIE "没有问题"
    local W="$FTD/probe-$TAG"; mkdir -p "$W"
    local ARGS="--zchain $ZCH"; [ "$PT" = - ] || { [ -d "${PT#pair=}" ] || DIE "没有 ③ 目录 $PT"; ARGS=$(pt_args "$PT") || DIE "配对目录失败"; }
    # 温度可带种子 "T:seed"(默认 1): 采样下的复读要看不止一粒种子才敢说"治了/没治"
    local samp=(--temp 0) suf=""
    if [ "$TEMP" != - ]; then local SEED="${TEMP#*:}"; [ "$SEED" = "$TEMP" ] && SEED=1; TEMP="${TEMP%%:*}"; samp=(--temp "$TEMP" --seed "$SEED"); suf=".t${TEMP}s$SEED"; fi
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    local i=0 q of
    for q in "$@"; do
        i=$((i+1)); of="$W/q$i$suf.txt"
        # "@文件" = 提示从文件读(10-01 晚: 判别探针要喂 BOS+抬头+料前缀 这种带换行的原始续写, 命令行参数传不干净);
        #   以 BOS 起头的文本 CLI 按"已渲染"处理不套聊天模板 —— 和 doc3 分词同一招
        case "$q" in @*) [ -s "${q#@}" ] || DIE "没有提示文件 ${q#@}"; q="$(cat "${q#@}")";; esac
        # shellcheck disable=SC2086
        ./ds4 --cuda -m "$MDL" $ARGS --mem-budget-mb 110000 "${samp[@]}" -n 96 -p "$q" \
            > "$of" 2>"$of.err" </dev/null || { tail -5 "$of.err"; DIE "探针失败 Q$i(见 $of.err)"; }
        LOG "[$TAG ③=$PT 温=$TEMP] Q$i: $q"
        echo "--- A$i ---"; cat "$of"; echo
    done
    LOG "DOCPROBE_DONE $TAG"
}

# docgate <③目录> [ntok]: 守门"不忘老本事" —— 同一趟 v41_judge 跑 ②态 与 ②+③态 的五指标(wt2, 默认 512 token, 教师有缓存),
#   两臂同趟同尺, 读数直接并排; 门与 stage_gate 守门 3 同口径: Same top 退 ≤0.5pp / KLD 涨 ≤3%。
stage_docgate(){
    # 第 3 参(可选) = 判决全文另存一份(两臂各一块: 先 ②态 后 ②+③态), kdeval 按它判过不过门; 屏幕上只留尾部 12 行(只有 ③ 那块)
    # 第 4 参(可选) = 判决料(缺省 wt2 通用料; 10-07 加: 金融 j = $FINJ —— 九次增量只守 wt2, 本域在 j 上漂到 Σmin −3.5pp / KLD +16% 没人拦, 用户 "金融域侧车指标明显坏了")
    local CAND="${1:?③目录}" N="${2:-512}" SAVE="${3:-/dev/null}" IDS="${4:-$WT2}"
    [ -d "${CAND#pair=}" ] || DIE "没有 ③ 目录 $CAND"
    [ -s "$IDS" ] || DIE "没有判决料 $IDS"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    local ARGS ARM; ARGS=$(pt_args "$CAND") || DIE "配对目录失败"
    case "$ARGS" in *--posttrain*) ARM="engine::$ZCH::$CAND";; *) ARM="engine::${ARGS#--zchain }";; esac   # 低秩候选 = 配对目录当 ②
    LOG "守门 $(basename "$IDS" .ids) $N: ② vs ②+③($CAND; 臂 $ARM)"
    bash "$SC/v41_judge.sh" "$IDS" "$N" "engine::$ZCH" "$ARM" 2>&1 | tee -a "$LOGF" | tee "$SAVE" | grep -v "^ds4:" | tail -12
    LOG "DOCGATE_DONE $CAND"
}

# gate_read <docgate 全文>: 打一行两臂对照, 退出码 0 = 过门: 主尺 Σmin 退 ≤0.5pp 且 Mean KLD 涨 ≤3%。
#   为什么不用 Same top(守门 3 原口径): top-1 一致率是还原率铁律里已退役的宽松指标 —— 10-02 实撞: e01 的 ③ Same top +1.96pp、
#   Mean KLD −3.1% 按老口径"过", 而 Σmin 0.7887 → 0.7751 退了 1.4pp、中位 KLD 翻倍。Same top / 中位 KLD 照打不判。
#   全文里每臂恰好一行 "Same top token = x%"、"Mean KLD = y (中位 …)"、"分布还原率 Σmin = z", 先 ② 后 ②+③; 少于两臂退出码 2。
gate_read(){
    awk '/Same top token/{v=$5; sub("%","",v); st[++a]=v} /Mean KLD/{kl[++b]=$4; v=$6; sub(",","",v); md[b]=v}
         /分布还原率/{sm[++c]=$4}
         END{ if (b < 2 || c < 2) { print "两臂读数不全"; exit 2 }
              d = 100 * (sm[2] - sm[1]); r = kl[2] / kl[1] - 1
              printf "Σmin %s → %s(%+.2fpp)| Mean KLD %.5f → %.5f(%+.1f%%)| 中位 KLD %s → %s | Same top %s%% → %s%%\n", \
                     sm[1], sm[2], d, kl[1], kl[2], 100 * r, md[1], md[2], st[1], st[2]
              exit !(d >= -0.5 && r <= 0.03) }' "$1"
}

# kdpick <③根目录>: ★自动选轮★(10-02 用户: "不要从结果看, 从技术指标看, 怎么从轮数里面自动最优")。根目录 = kdtrain 的输出。
#   目标 = 留出 KL: 训练没见过的问法上, 挂 ③ 不看复盘的学生对看着复盘的教师, 越低越像读过复盘(取 train.log 最后一趟的 epoch 行);
#   约束 = wt2 门(gate_read: Σmin 退 ≤0.5pp 且 Mean KLD 涨 ≤3%), 每轮都守, 判决全文落 <根>/eval/gate_ckpt_eNN.txt。
#   规则: 过门的轮里留出 KL 最低者胜。不把两类加权成一个分 —— 换算比例没有依据; 门是底线, 底线之上只比学会了多少。
#   CFO 真实请求回放不参与选轮: 一只股票的次日走势是一次抽样, 拿它挑轮次等于挑运气(10-02 按回放推了 e02, 按本规则是 e03)。
#   保持料 KL 只打不判: 只有 16 道题, 而且与正式门对不上(10-02 全量 e01 保持料 KL 最低, wt2 却没过)。
#   选中的目录放进全局 KD_PICK; 一轮都没过就停车(没过通用门的 ③ 不进任何对比)。
KD_PICK=""
kd_pick(){
    local R0="${1:?③根目录}" W="${1}/eval" c n row g r bkl="" kl hk gn se
    [ -s "$R0/train.log" ] || DIE "没有 $R0/train.log(不是 kdtrain 的输出目录?)"
    mkdir -p "$W"; KD_PICK=""
    # 同一目录重训时 train.log 是追加写的: 以最后一个 "step 0" 行为界, 只认这一趟的 epoch 行 —— 上一趟残留的 ckpt 不参选。
    # epoch 行: epoch N eval_kl X train_kl Y hold_kl Z [gain D se S](10-02 前的老日志没有后两项, 打 -)
    local TBL; TBL=$(awk '/^step 0 /{delete e} /^epoch /{e[$2 + 0] = $4 " " $8 " " ($10 == "" ? "-" : $10) " " ($12 == "" ? "-" : $12)}
                          END{for (k in e) print k, e[k]}' "$R0/train.log")
    for c in "$R0"/ckpt_e[0-9][0-9]; do
        [ -s "$c/amp_L39.bin" ] || continue
        n=$((10#${c##*ckpt_e}))
        row=$(awk -v n="$n" '$1 == n {print $2, $3, $4, $5}' <<< "$TBL")
        [ -n "$row" ] || { LOG "选轮 $(basename "$c"): train.log 最后一趟没有第 $n 轮(上一趟的残留), 不参选"; continue; }
        read -r kl hk gn se <<< "$row"
        g="$W/gate_$(basename "$c").txt"
        stage_docgate "$c" 512 "$g" >/dev/null
        if r=$(gate_read "$g"); then
            LOG "选轮 $(basename "$c"): 留出 KL $kl(较上轮降 $gn ± $se) 保持料 KL $hk | $r | 过门"
            if [ -z "$bkl" ] || awk -v a="$kl" -v b="$bkl" 'BEGIN{exit !(a < b)}'; then KD_PICK="$c"; bkl="$kl"; fi
        else LOG "选轮 $(basename "$c"): 留出 KL $kl(较上轮降 $gn ± $se) 保持料 KL $hk | $r | ★没过门★"; fi
    done
    rm -f "$R0/pick.txt"   # 重选时先清: 旧的选中不许留给训练页(ds4-train 读它当"done")
    [ -n "$KD_PICK" ] || DIE "选轮: 没有一轮过 wt2 门, 不进任何对比(读数见 $W/gate_*.txt)"
    basename "$KD_PICK" > "$R0/pick.txt"   # 训练页(ds4-train /api/runs)读: 选中的 ckpt 名
    LOG "选轮结果: $KD_PICK(过门的轮里留出 KL 最低, $bkl)"
}
stage_kdpick(){ kd_pick "${1:?③根目录}"; LOG "KDPICK_DONE $KD_PICK"; }

# docnll <ids文件> <标签> [二进制=./ds4] [chunk]: 同一份 ids 用指定引擎二进制打逐位 NLL(--score-ids 部署同路, 挂现役 ②), 逐位打印。
#   为什么要它(10-01 实撞): 解算器 13:08 重编(链 11:04 回滚后的引擎对象)后, 同一篇 44 token 文档的基线 logp 与 09-30 00:08 那版
#   解算器对不上("1+1=→3" p 0.86 → 0.29, "结果是→三" ≥0.95 → 0.13) —— 上下文抄写变差, 不是舍入。两版引擎只能有一个对, 这里拿
#   部署二进制 ds4、留档基线 ds4.base_*、以及不同分块(--v41-chunk)逐位对账; 分块改变 NLL 就是引擎 bug(09-22 块 16 vs 块 8 那条门的同款)。
stage_docnll(){
    local IDS="${1:?ids 文件}" TAG="${2:?标签}" BIN="${3:-./ds4}" CH="${4:-}"
    [ -s "$IDS" ] || DIE "没有 ids $IDS"; [ -x "$BIN" ] || DIE "没有二进制 $BIN"
    local W="$FTD/nll"; mkdir -p "$W"
    local out="$W/$TAG.nll" log="$W/$TAG.log"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    "$BIN" --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --score-ids "$IDS" --score-no-logits --score-nll "$out" ${CH:+--v41-chunk "$CH"} \
        > "$log" 2>&1 </dev/null || { tail -5 "$log"; DIE "打分失败($log)"; }
    [ -s "$out" ] || DIE "没产出 $out"
    LOG "[docnll $TAG] $BIN chunk=${CH:-默认}: $(grep -o "NLL 已写.*" "$log" | head -1)"
    od -An -f -v -w4 "$out" | awk -v t="$TAG" '{printf "%s %d %.4f\n", t, NR-1, $1}' > "$W/$TAG.txt"
    LOG "DOCNLL_DONE $TAG → $W/$TAG.txt"
}

# ---------------- docflip: 复盘文本 → ③ → 翻没翻, 一篮子(2026-10-01 晚, 用户"把复盘文本放到指定目录下面看看, 能不能实现翻转") ----------------
# docflip <料目录> <η> <λ> <K> <conf-min> [标签]: doc3(低秩 K, 按行拆, 只推笃定答错位) → 候选唯一 = pt/cand_lr<K>_e<η>_l<λ>
#   → 不挂 ③ 的基线探针(标签 <料名>-base; 已有且题数够就不重跑 —— 基线与 ③ 无关, 换 η 重来时省下七次装载)
#   → 挂 ③ 温 0 逐题 → 前两题各采样两粒种子 → wt2 512 守门。问题读 <料目录>/probe/questions.txt(一行一题;
#   doc3 只吃 <料目录>/*.txt 与 hold/*.txt, probe/ 子目录不会被当成料)。全部原始输出落 $FTD/probe-<标签>/, 判读归人。
#   ★事实行一行都没留就停车★: 复盘里"某日大盘下跌"这类事实, 模型多半是"不知道"而不是"笃定答错"(1+1=2 那种), conf-min 0.5 可能把
#   所有位置筛光 —— 空 ③ 挂上去探针只是白烧七次装载, 换阈值重来。
stage_docflip(){
    local DOCD="${1:?料目录}" ETA="${2:?η}" LAM="${3:?λ}" K="${4:?K}" CONF="${5:?conf-min}"
    local NAME; NAME="$(basename "$DOCD")"; local TAG="${6:-$NAME-e${ETA}l${LAM}c${CONF}}"
    local QF="$DOCD/probe/questions.txt"; [ -s "$QF" ] || DIE "没有问题文件 $QF"
    local -a QS=(); local q
    while IFS= read -r q || [ -n "$q" ]; do [ -n "$q" ] && QS+=("$q"); done < "$QF"
    [ "${#QS[@]}" -ge 2 ] || DIE "问题至少两条(采样趟取前两题)"
    stage_doc3 "$DOCD" "$ETA" "$LAM" "$K" "$CONF" || DIE "doc3 失败"
    local W="$FTD/doc-$NAME" CAND="$FTD/doc-$NAME/pt/cand_lr${K}_e${ETA}_l${LAM}"
    [ -s "$CAND/amp_L39.bin" ] || DIE "没有候选 $CAND(看 $W/pt/candidates.txt 里的目录名)"
    local NF; NF=$(grep -o "正A行均Δlogp [^(]*([0-9]*)" "$W/solve.log" | head -1 | grep -o "([0-9]*)" | tr -d "()")
    [ -n "$NF" ] && [ "$NF" -gt 0 ] || DIE "事实行留了 ${NF:-?} 行(conf-min $CONF 下没有'笃定答错'的位置), 不拿空 ③ 探针; 看 $W/solve.log 各份'留 N 行'"
    LOG "候选 $CAND: 事实行 $NF 行进了解算; 探针 ${#QS[@]} 题"
    # 基线只和问题有关, 和料无关 ⇒ 基线目录按问题文件内容哈希命名(10-01 实撞: 按料目录名命名, 换个料目录名就把 7 次装载的基线重跑了一遍)
    local QH; QH=$(md5sum "$QF" | cut -c1-8); local BTAG="qs$QH-base" BASE="$FTD/probe-qs$QH-base"
    if [ "$(ls "$BASE"/q*.txt 2>/dev/null | grep -vc '\.t[0-9]')" -ge "${#QS[@]}" ]; then LOG "基线探针已有($BASE), 不重跑"
    else stage_docprobe - "$BTAG" - "${QS[@]}"; fi
    stage_docprobe "$CAND" "$TAG" - "${QS[@]}"
    stage_docprobe "$CAND" "$TAG" 1:1 "${QS[0]}" "${QS[1]}"
    stage_docprobe "$CAND" "$TAG" 1:2 "${QS[0]}" "${QS[1]}"
    stage_docgate "$CAND"
    LOG "DOCFLIP_DONE $TAG 候选 $CAND"
}

# ---------------- ③ 后训练: 一份 jsonl 一条命令(2026-10-10 收口; 用户: "更简单的数据结构, 更好更快的训练效果") ----------------
# 料 = jsonl, 一行一题: {"messages":[{"role":"user","content":"问"},{"role":"assistant","content":"答"}], "context":"材料(可省)"}
#   带 context = 上下文蒸馏(教师 = 读了材料的部署态自己, 学生不看材料, 蒸 top-K 分布; 文献 KMs/PD/Cartridges 的共识); 不带 = 答案 one-hot(交叉熵)。
#   别的字段训练器直接报错(10-09 那版收 reward/mode/split, 10-10 用户判为跑偏)。留出 = 问题去空白 FNV 五取一(训练器定, 与料文件无关)。
#   保持料 = 固定的 $HOLD(通用题 + 部署态自己的回答, 教师 = 部署态自己), 训练器自动混入, 用户不用管 —— 10-01 实撞: 没有它 ③ 局部性崩。
HOLD="$ROOT/gguf-tools/data/posttrain/hold.jsonl"
# gen <料目录> [块名正则] [轮数]: 文档 → 问答料(可选的数据工具, 不是训练的一环: 文档也可以直接当 {"text"} 原文料训, unsloth 同样两条路)。
#   chunks/*.txt 每块 × 轮数, 模型读块出问答(合批 --gen-jobs, 8 路; 种子按 块名+轮 定 ⇒ 可复现) → gensplit 按"问：/答："拆对 → <料目录>/<料名>.jsonl(每行 context = 块正文)。
#   提示只有一句, 照 Meta synthetic-data-kit 默认的 qa_generation(问题只问原文里的重要事实 / 答案必须有原文直接支持 / 每块固定 N 组), 10-10 对齐时把原先
#   带复盘味的三条("有什么教训""读者请教怎么做")删了; 唯一加的要求是问题自带完整指代 —— 学生训练时不看材料, "这份材料"指不到任何东西。
#   料目录自带 seeds/qa.txt 就用它(对应 synthetic-data-kit 的自定义 prompts.qa_generation)。采样走模型卡(温 1), 不思考。
KD_SEED_QA='请根据上面这段文字写 10 组问答，用于训练大模型。规则：1. 问题只问这段文字里的重要事实；2. 答案必须能在这段文字里直接找到依据；3. 每个问题写明完整指代（具体的名称、日期、对象），不许出现“这段文字”“上文”之类的说法。严格按下面的格式输出，不要输出别的内容：
问：……
答：……'
# gensplit <料目录>: gen/*.txt(引擎原始输出; 老目录的 gen/*.json 也认) → <料目录>/<料名>.jsonl。同块同题去重; 没拆出问答的份数打出来
kd_split(){
    local DOCD="${1:?料目录}" OUT; OUT="$DOCD/$(basename "$DOCD").jsonl"
    python3 - "$DOCD" "$OUT" <<'PYSPLIT' || DIE "问答拆对失败"
import glob, json, os, re, sys
d, out = sys.argv[1], sys.argv[2]
by, bad, chunks = {}, 0, {}
def body(p):
    if p.endswith(".json"): return json.load(open(p))["choices"][0]["message"].get("content") or ""
    return open(p, encoding="utf-8").read()
seen_files = set()
for p in sorted(glob.glob(os.path.join(d, "gen", "*.txt")) + glob.glob(os.path.join(d, "gen", "*.json"))):
    b = os.path.basename(p)
    if b.startswith("hold_q") or b in ("gen.out", "jobs.tsv"): continue
    stem = re.sub(r"\.(txt|json)$", "", b)
    if stem in seen_files: continue      # 同名 .txt 与 .json 只读一份
    seen_files.add(stem)
    chunk = re.sub(r"_s\d+r\d+$", "", stem)
    cf = os.path.join(d, "chunks", chunk + ".txt")
    if not os.path.exists(cf): continue
    chunks[chunk] = open(cf, encoding="utf-8").read().rstrip()
    txt = body(p).replace("**", "")
    cur_q, cur_a, mode, pairs = [], [], None, []
    for line in txt.splitlines():
        # 标记行: "问：/答：", 前面可带序号("1. 问："), 后面也可带序号("问1：/答1：")
        m = re.match(r"^\s*(?:\d+[.、)]\s*)?(问|答)\s*\d*\s*[：:]\s*(.*)$", line)
        if m:
            if m.group(1) == "问":
                if cur_q and cur_a: pairs.append(("\n".join(cur_q).strip(), "\n".join(cur_a).strip()))
                cur_q, cur_a, mode = [m.group(2)], [], "q"
            else:
                cur_a, mode = [m.group(2)], "a"
            continue
        if mode == "q": cur_q.append(line)
        elif mode == "a": cur_a.append(line)
    if cur_q and cur_a: pairs.append(("\n".join(cur_q).strip(), "\n".join(cur_a).strip()))
    if not pairs: bad += 1
    by.setdefault(chunk, []).extend(x for x in pairs if x[0] and x[1])
n = 0
with open(out, "w", encoding="utf-8") as f:
    for chunk in sorted(by):
        seen = set()
        for q, a in by[chunk]:
            k = re.sub(r"\s+", "", q)
            if k in seen: continue
            seen.add(k); n += 1
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "context": chunks[chunk]}, ensure_ascii=False) + "\n")
print("问答拆对: %d 块, %d 题 → %s; 没拆出问答的生成 %d 份" % (len(by), n, out, bad))
PYSPLIT
}
stage_gensplit(){ kd_split "${1:?料目录}"; }
stage_gen(){
    local DOCD="${1:?料目录}" PAT="${2:-.}" ROUNDS="${3:-1}"
    [ -d "$DOCD/chunks" ] || DIE "没有块目录 $DOCD/chunks(要写进 ③ 的文本, 一块一个 .txt; Mac 上 qtf_requests_mac.sh corpus <料名> 出的就是)"
    local -a CH=(); local f b
    for f in "$DOCD"/chunks/*.txt; do b="$(basename "$f" .txt)"; [ -s "$f" ] && [[ "$b" =~ $PAT ]] && CH+=("$f"); done
    [ "${#CH[@]}" -gt 0 ] || DIE "块名正则 $PAT 一个都没选中"
    mkdir -p "$DOCD/gen/prompt"
    local JOBS="$DOCD/gen/jobs.tsv" s r seed name sd
    : > "$JOBS"
    # 一份作业 = 提示文件 + 原始输出(.txt, 引擎写; 没收口的不落盘) + 种子 + 上限; 产物已在的跳过(断点续跑)
    seed="$KD_SEED_QA"; [ -s "$DOCD/seeds/qa.txt" ] && seed="$(cat "$DOCD/seeds/qa.txt")"
    for r in $(seq 1 "$ROUNDS"); do for f in "${CH[@]}"; do
        name="$(basename "$f" .txt)_s0r${r}"   # 名字留 _s0: gensplit 按 _s<n>r<n> 去后缀认块, 老目录的 s1/s2 产物照样能拆
        { [ -s "$DOCD/gen/$name.txt" ] || [ -s "$DOCD/gen/$name.json" ]; } && continue
        { cat "$f"; printf '\n%s' "$seed"; } > "$DOCD/gen/prompt/$name.txt"
        sd=$(printf '%s' "$name" | cksum | cut -d' ' -f1); [ "$sd" = 0 ] && sd=1
        printf '%s\t%s\t%s\t%s\n' "$DOCD/gen/prompt/$name.txt" "$DOCD/gen/$name.txt" "$sd" 4096 >> "$JOBS"
    done; done
    LOG "gen: ${#CH[@]} 块 × $ROUNDS 轮, 本次要生成 $(wc -l < "$JOBS") 份 → $DOCD/gen(合批 --gen-jobs)"
    local t0; t0=$(date +%s)
    if [ -s "$JOBS" ]; then
        bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
        ( sleep 20; train_guard ) &
        local GUARD=$! RC
        ./ds4 --cuda -m "$MDL" ${ZARGS[@]+"${ZARGS[@]}"} ${ENG[@]+"${ENG[@]}"} --mem-budget-mb 110000 --no-dspark --gen-jobs "$JOBS" > "$DOCD/gen/gen.out" 2>&1 </dev/null
        RC=$?
        kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
        grep -a "★" "$DOCD/gen/gen.out" | tee -a "$LOGF" | head -20
        grep -a "gen-jobs\] 完" "$DOCD/gen/gen.out" | tee -a "$LOGF"
        [ "$RC" = 0 ] || DIE "合批出题失败(rc=$RC, 见 $DOCD/gen/gen.out)"
    fi
    kd_split "$DOCD"
    LOG "GEN_DONE $(( $(date +%s) - t0 )) s"
}

# train <数据.jsonl> [轮 3] [层 0-39] [lr 2e-4] [k=v,...]: ★训 ③★ 停服 → 看门狗 → ./ds4 --ptrain(教师 top-K 带缓存 <料>.teacher.bin → 第 0 步评估 → 训练
#   → 每轮留出损失 + 贪心探针 + ckpt_eNN) → kdpick(每轮 wt2 门, 过门里留出损失最低者) → ③ = $FTD/kd-<料名>-<标签>/ckpt_eNN。
#   一轮 = 全量过一遍(epoch_tok=0); 料大时 k=v 给 epoch_tok=N 切片(10-03 用户定的 10 分钟一轮)。续训给 init=<上一份 ③>。
pt_cfg_write(){   # <输出目录> <料> <层> <lr> <轮> [k=v,...]: 训练配置(train / kdprof 共用)
    local OUT="$1" SRC="$2" LAYERS="$3" LR="$4" EP="$5" EXTRA="${6:-}"
    [ -s "$HOLD" ] || DIE "没有保持料 $HOLD(通用题 + 部署态回答)"
    printf 'data=%s\nout=%s\nhold=%s\nlayers=%s\nrank=64\nlr=%s\nepochs=%s\nbatch=4\ntopk=64\nmaxlen=1024\nprobe_n=6\nprobe_tok=96\nepoch_tok=0\n' \
        "$SRC" "$OUT" "$HOLD" "$LAYERS" "$LR" "$EP" > "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
}
stage_train(){
    local SRC="${1:?数据 .jsonl}" EP="${2:-3}" LAYERS="${3:-0-39}" LR="${4:-2e-4}" EXTRA="${5:-}"
    [ -s "$SRC" ] || DIE "没有 $SRC"
    # 目录名带发车时间: 同料同参数再训一次是新目录, 不盖掉上一次的 ckpt / 门读数(10-10 训练页实撞: 重训写进同名目录, 旧 pick.txt 让新趟一开始就显示"完成")
    local NAME TAG OUT; NAME="$(basename "$SRC" .jsonl)"; TAG="L${LAYERS}-lr${LR}-e${EP}-$(date +%m%d%H%M)"; OUT="$FTD/kd-$NAME-$TAG"
    mkdir -p "$OUT"
    pt_cfg_write "$OUT" "$SRC" "$LAYERS" "$LR" "$EP" "$EXTRA"
    LOG "train: $SRC → 层 $LAYERS lr $LR 轮 $EP → $OUT"
    kd_train_run "$OUT"
    stage_kdpick "$OUT"
    # ★训完只产出, 不挂★(10-10 晚用户: "训练完就是训练完不要直接挂"): 不碰 serve_pick.txt, train_cycle.sh 装回来的还是训前那套。
    # ③ 关联的侧车 = 训练时装着的 $ZCH(③ 的 base.fnv 就是它的指纹); 聊天页切到这份侧车、打开"后训练"开关才挂上, 切走就卸。
    LOG "TRAIN_DONE $OUT 选中 $KD_PICK (关联侧车 ${ZCH:-none}; 没挂, 聊天页开关挂)"
}

# kdprof <数据.jsonl> [步数 5] [nsys|-] [k=v,...]: 训练计时(./ds4 --ptrain prof=1 max_steps=N, 10-02)。配置照 train 缺省(全层 0-39, lr 2e-4, batch 4)
#   + maxlen 880(复盘全量口径), 第 4 参追加/覆盖。只跑 N 步: 不做第 0 步评估/探针、不评估不存盘; 每 5 步打一张整步分段表
#   (前向 / 损失 / 出口反传 / 逐层反传 = 重算 + routed 专家反向 + 注意力半层 + 层内其余 + 层间 / Adam, 段和对每题墙钟)。
#   要现成的教师表(<料>.teacher.bin, train 跑过一趟就有), 没有就停车 —— 不让教师那遍混进计时。
#   第 3 参给 nsys = 套 nsys 采 GPU 时间线, 出逐核名合计表(cuda_gpu_kern_sum; 训练路 prof 每段同步、单流, 核时长不重叠, 直接加和可信);
#   nsys 下墙钟会虚高, 只看比例。第 5 参 = 用哪个引擎二进制(缺省 ./ds4; 改反传前 cp 一份 ds4.base_xxx, 同一题同一配置 A/B,
#   例: 第 4 参 gradcheck=2,gclayers=0/14/20/30/39 两个二进制各跑一遍, 比每层"反传"梯度与有限差分比值)。输出按二进制名分文件, 不互相覆盖。
#   第 6 参 = 额外的引擎命令行参数(原样追加), 例 --v41-prof: 训练前向里每层把逐专家 token 数落 /tmp/v41_route_Lnn_n<行数>.txt
#   (给 gguf-tools/bench/v41_vq_train_bench.cu 当训练形状的真路由; 每层要同步读回 sel, 计时作废)。
stage_kdprof(){
    local SRC="${1:?数据 .jsonl}" STEPS="${2:-5}" NS="${3:-}" EXTRA="${4:-}" BIN="${5:-./ds4}" XARGS="${6:-}"
    [ -x "$BIN" ] || DIE "没有引擎二进制 $BIN"; [ -s "$SRC" ] || DIE "没有 $SRC"
    local NAME OUT RC; NAME="$(basename "$SRC" .jsonl)"; OUT="$FTD/kd-$NAME-prof"; mkdir -p "$OUT"
    pt_cfg_write "$OUT" "$SRC" 0-39 2e-4 1 "maxlen=880,prof=1,max_steps=$STEPS${EXTRA:+,$EXTRA}"
    [ -s "$SRC.teacher.bin" ] || DIE "kdprof 要现成的教师表 $SRC.teacher.bin, 先跑一趟 train"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    local B; B="$(basename "$BIN")"
    LOG "kdprof: $NAME $STEPS 步$([ "$NS" = nsys ] && echo ', 套 nsys')${EXTRA:+, 额外 $EXTRA}, 二进制 $B → $OUT"
    ( sleep 20; train_guard "$B" ) &
    local GUARD=$!
    local CMD=("$BIN" --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --no-dspark)
    [ -z "$XARGS" ] || read -r -a XA <<< "$XARGS"; [ -z "$XARGS" ] || CMD+=("${XA[@]}")
    CMD+=(--ptrain "$OUT/ptrain.cfg")
    if [ "$NS" = nsys ]; then
        rm -f "$OUT/prof_$B.nsys-rep" "$OUT/prof_$B.sqlite" "$OUT/prof_kern_$B"*.csv
        nsys profile -o "$OUT/prof_$B" --force-overwrite true -t cuda "${CMD[@]}" > "$OUT/prof_$B.out" 2>&1 </dev/null; RC=$?
    else "${CMD[@]}" > "$OUT/prof_$B.out" 2>&1 </dev/null; RC=$?; fi
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/prof_$B.out" | grep -a "ptrain prof\|max_steps\|梯度检查\|★\|失败" | tail -24
    [ "$RC" = 0 ] || DIE "计时失败(rc=$RC, 见 $OUT/prof_$B.out)"
    if [ "$NS" = nsys ]; then
        for _ in $(seq 1 60); do pgrep -x "$B" >/dev/null || break; sleep 2; done   # 映射卸载完再导出(multi_probe 同一处实撞)
        nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT/prof_kern_$B" "$OUT/prof_$B.nsys-rep" >/dev/null 2>&1
        ls "$OUT/prof_kern_$B"*.csv >/dev/null 2>&1 && head -30 "$OUT/prof_kern_$B"*.csv | cut -c1-220 || LOG "★nsys 没导出逐核表(见 $OUT/prof_$B.out)★"
    fi
    LOG "KDPROF_DONE $OUT($B)"
}

train_guard(){
    # 第 1 参 = 盯哪个进程名(缺省 ds4; kdprof 拿 ds4.base_xxx 做 A/B 时传它 —— 写死 ds4 的话看门狗一进来就以为进程已退, 等于没开)
    local P="${1:-ds4}" a bad=0
    while pgrep -x "$P" >/dev/null; do
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$a" -lt 2500 ]; then bad=$((bad+1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then LOG "★训练看门狗: MemAvailable ${a} MB < 2500 连续两次, 杀 $P★"; pkill -x "$P"; sleep 3; pkill -9 -x "$P" 2>/dev/null; return 1; fi
        sleep 5
    done
}
# ---------------- 草稿器蒸馏(2026-10-07, src/core/core_draft_kd*.c; 件挂法 --draft-amp <目录>) ----------------
# 为什么: 三塔照原始 FP 模型训, 部署的是 ①+②, 采样下首位期望接受率 Σmin(p,q) 只有 0.74~0.76(09-29 陪审团)。让三塔改盯部署底座:
# 料 = 真实请求 + 底座按模型卡配方采样的续写(dkgen), 训两种低秩件(塔件 + 出口件), 底座/塔原权重/头全冻结, 验证侧不动 ⇒ 只动接受率。
DKD="$FTD/dkd"   # texts/(ids + .np 提示长 + .prompt.ids) cache/(底座取料缓存) dk-<标签>/(件 + train.log)
# dkgen <请求 json> [seed 1] [续写 token 数 1024] [温 1.0]: 请求按服务端思考档渲染(req_render_ids.sh) → 底座采样续写 → texts/<请求名>_t<T>_s<seed>.ids
stage_dkgen(){
    local REQ="${1:?请求 json}" SEED="${2:-1}" NGEN="${3:-1024}" T="${4:-1.0}"
    [ -s "$REQ" ] || DIE "没有 $REQ"
    need_idle; mkdir -p "$DKD/texts"
    local NAME; NAME="$(basename "$REQ" .json)"
    local PIDS="$DKD/texts/$NAME.prompt.ids" OUTI="$DKD/texts/${NAME}_t${T}_s${SEED}.ids" ERR
    [ -s "$PIDS" ] || bash "$ROOT/speed-bench/req_render_ids.sh" "$REQ" "$PIDS" >>"$LOGF" 2>&1 || DIE "渲染失败 $REQ"
    ERR="$OUTI.err"
    ( sleep 20; train_guard ) &   # 看门狗(MemAvailable < 2500 MB 连续两次就杀): 装 103 GB 模型的趟一律带
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --no-dspark --emit-trace --gen-ids "$PIDS" -n "$NGEN" \
        --temp "$T" --top-p 1.0 --min-p 0 --seed "$SEED" > "$OUTI.out" 2> "$ERR" </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    [ "$RC" = 0 ] || DIE "生成失败(rc=$RC), 见 $ERR"
    grep -a -h '^\[ptok\] \|^\[emit\] ' "$ERR" | sort -n -k2 | awk '{print $3}' > "$OUTI"
    local NP NG; NP=$(grep -ac '^\[ptok\] ' "$ERR"); NG=$(grep -ac '^\[emit\] ' "$ERR")
    [ "$NP" -gt 0 ] && [ "$NG" -gt 8 ] || DIE "生成段太短或没有 [ptok](NP=$NP NG=$NG), 看 $ERR"
    echo "$NP" > "$OUTI.np"
    LOG "dkgen: $(basename "$OUTI") = 提示 $NP + 续写 $NG token(温 $T seed $SEED; $(grep -a -h 'decode .* token' "$ERR" | tail -1))"
}
# dktrain <标签> <留出正则> [k=v,...] [文本正则 .]: texts/ 里的 ids 进清单(名字匹配留出正则的当留出, 其余训练), 写配置, 停服 + 看门狗 + ./ds4 --draft-train。
#   第 3 参原样追加进配置(如 gradcheck=2 只做梯度检查就退出; epochs=0 只做第 0 步评估 = 接线门; max_steps=5,prof=1 计时)。
stage_dktrain(){
    local TAG="${1:?标签}" EVPAT="${2:?留出正则}" EXTRA="${3:-}" PAT="${4:-.}"
    local OUT="$DKD/dk-$TAG"; mkdir -p "$OUT"
    local LIST="$OUT/texts.list" f b np kind; : > "$LIST"
    for f in "$DKD"/texts/*.ids; do
        b="$(basename "$f" .ids)"; [[ "$b" == *.prompt ]] && continue; [[ "$b" =~ $PAT ]] || continue
        [ -s "$f.np" ] || continue
        np=$(cat "$f.np"); kind=train; [[ "$b" =~ $EVPAT ]] && kind=eval
        echo "$f $kind $np" >> "$LIST"
    done
    [ -s "$LIST" ] || DIE "清单为空(先跑 dkgen)"
    grep -q ' eval ' "$LIST" || DIE "清单里没有留出文本(留出正则 $EVPAT 没匹配上)"
    printf 'texts=%s\nout=%s\ncache=%s\nrank=64\nrank_exit=64\ntopk=1024\nlr=3e-4\nepochs=3\nbatch_blocks=128\nlogits_rows=128\n' "$LIST" "$OUT" "$DKD/cache" > "$OUT/draft.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/draft.cfg"
    LOG "dktrain: $TAG, 清单 $(wc -l < "$LIST") 份文本(留出 $(grep -c ' eval ' "$LIST")), 配置: $(tr '\n' ' ' < "$OUT/draft.cfg")"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    ( sleep 20; train_guard ) &
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --no-dspark --draft-train "$OUT/draft.cfg" > "$OUT/train.out" 2>&1 </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/train.out" | grep -E "\[dk" | tail -40
    [ "$RC" = 0 ] || DIE "训练失败(rc=$RC, 见 $OUT/train.out)"
    LOG "DKTRAIN_DONE $OUT"
}
# dkrun <标签> <留出请求正则> [训练轮数 4] [种子数 3] [续写 token 数 1536]: 一条龙 —— review0923 下每条请求 × 每个种子 dkgen(已有的跳过) → dktrain → dkgate
#   (留出正则匹配到的第一份文本 seed 1)。10-07 首趟发车时这条链临时写在 spark /tmp 里(违"脚本落 repo"), 之后按这段跑。
stage_dkrun(){
    local TAG="${1:?标签}" EVPAT="${2:?留出请求正则}" EP="${3:-4}" NS="${4:-3}" NGEN="${5:-1536}" req seed f
    for f in "$D2"/review0923/req_*.json; do
        req="$(basename "$f" .json)"
        for seed in $(seq 1 "$NS"); do
            [ -s "$DKD/texts/${req}_t1.0_s${seed}.ids.np" ] && continue
            stage_dkgen "$f" "$seed" "$NGEN" 1.0
        done
    done
    stage_dktrain "$TAG" "$EVPAT" "epochs=$EP"
    local EV; EV="$(ls "$DKD"/texts/*_t1.0_s1.ids | grep -E "$EVPAT" | head -1)"
    [ -n "$EV" ] && stage_dkgate "$DKD/dk-$TAG" "$EV" 1.0 512
    LOG "DKRUN_DONE $DKD/dk-$TAG"
}
# dkspeed <件目录|none> <文本 ids(带 .np)> [seed 数 3] [生成 token 数 1024] [额外引擎参数]: 产品口径(温 1.0 / top_p 1 / min_p 0, 投机开)的在线速度 ——
#   采样路每趟是另一篇文本(k 随墙钟变 ⇒ 硬币变), 单趟 ±3 t/s 不可比(09-29 铁律), 所以同一提示按 seed 各跑一趟 不挂件 / 挂件, 报每趟与均值;
#   另跑一对贪心(确定性, 同文本可比)。第 5 参原样追加给引擎(例 "--dspark-verify 5" 钉死每轮验证位数, 量调度器那一笔)。
stage_dkspeed(){
    local AD="${1:?件目录|none}" IDS="${2:?文本 ids}" NS="${3:-3}" NGEN="${4:-1024}" EXTRA="${5:-}"
    [ "$AD" = none ] || [ -s "$AD/base.fnv" ] || DIE "$AD 不是件目录(缺 base.fnv)"
    [ -s "$IDS.np" ] || DIE "$IDS 没有 .np(要 dkgen 的产物)"
    need_idle
    local OUT="$DKD/speed-$(basename "$IDS" .ids)-$(date +%m%d%H%M)"; mkdir -p "$OUT"
    local NP; NP=$(cat "$IDS.np"); head -n "$NP" "$IDS" > "$OUT/prompt.ids"
    local base=(./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --gen-ids "$OUT/prompt.ids" -n "$NGEN")
    local amp=(); [ "$AD" = none ] || amp=(--draft-amp "$AD")
    local tag seed x
    for tag in g_spec g_amp; do
        [ "$tag" = g_amp ] && [ "$AD" = none ] && continue
        x=(); [ "$tag" = g_amp ] && x=("${amp[@]}")
        "${base[@]}" --temp 0 --seed 1 "${x[@]}" $EXTRA > "$OUT/$tag.out" 2> "$OUT/$tag.err" </dev/null || DIE "$tag 跑失败, 见 $OUT/$tag.err"
        echo "  $tag: $(grep -a -h 'decode .* token' "$OUT/$tag.err" | tail -1 | sed 's/.*decode //') $(grep -a -h 'DSpark: ' "$OUT/$tag.err" | tail -1 | sed 's/.*DSpark: //' | cut -c1-80)"
    done
    [ "$AD" = none ] || { cmp -s "$OUT/g_spec.out" "$OUT/g_amp.out" && echo "  贪心两趟逐字节同 ✓" || echo "  ★贪心两趟不同★"; }
    for seed in $(seq 1 "$NS"); do
        for tag in s_spec s_amp; do
            [ "$tag" = s_amp ] && [ "$AD" = none ] && continue
            x=(); [ "$tag" = s_amp ] && x=("${amp[@]}")
            "${base[@]}" --temp 1.0 --top-p 1.0 --min-p 0 --seed "$seed" "${x[@]}" $EXTRA > "$OUT/${tag}_$seed.out" 2> "$OUT/${tag}_$seed.err" </dev/null || DIE "$tag seed $seed 跑失败"
            echo "  $tag seed $seed: $(grep -a -h 'decode .* token' "$OUT/${tag}_$seed.err" | tail -1 | sed 's/.*decode //') $(grep -a -h 'DSpark: ' "$OUT/${tag}_$seed.err" | tail -1 | sed 's/.*DSpark: //' | cut -c1-60)"
        done
    done
    for tag in s_spec s_amp; do
        [ "$tag" = s_amp ] && [ "$AD" = none ] && continue
        grep -a -h 'decode .* token' "$OUT"/${tag}_*.err | sed 's/.*(\([0-9.]*\) t\/s).*/\1/' | awk -v t="$tag" '{s+=$1; n++} END {if (n) printf "  %s 均值 %.2f t/s(%d 趟)\n", t, s/n, n}'
        grep -a -h 'DSpark: ' "$OUT"/${tag}_*.err | sed 's/.*平均接受 \([0-9.]*\)\/5.*/\1/' | awk -v t="$tag" '{s+=$1; n++} END {if (n) printf "  %s 均接受 %.2f/5\n", t, s/n}'
    done
    LOG "DKSPEED_DONE $OUT"
}
# dkloop <件目录> <文本 ids(带 .np)> [seed 数 3] [上限 4096]: 复读门(用户 10-07: "别最后变成贪心 bug, 重复文本") —— 产品口径(温 1.0 投机开)同一提示按 seed
#   各跑 不挂件 / 挂件 到 EOS 或上限, 每趟过 bugmd_ids_tools.py loopstat(锁死窗数 / 入环位 / 末 8192 位去重 / 复读率曲线); 两边统计同档才算过。
#   采样路没有逐字节门: 件只改草稿 q, 验证核的拒绝采样让吐出的边缘恰是 p, 所以判据是分布与复读统计, 不是逐字节(09-29 铁律)。
#   带上限的趟只能说"到上限没停"(09-22 铁律), 这里比的是两边同档不同档, 不判模型停不停。
stage_dkloop(){
    local AD="${1:?件目录}" IDS="${2:?文本 ids}" NS="${3:-3}" NCAP="${4:-4096}"
    [ -s "$AD/base.fnv" ] || DIE "$AD 不是件目录(缺 base.fnv)"
    [ -s "$IDS.np" ] || DIE "$IDS 没有 .np"
    need_idle
    local OUT="$AD/loop-$(basename "$IDS" .ids)"; mkdir -p "$OUT"
    local NP; NP=$(cat "$IDS.np"); head -n "$NP" "$IDS" > "$OUT/prompt.ids"
    local seed tag x
    for seed in $(seq 1 "$NS"); do
        for tag in none amp; do
            x=(); [ "$tag" = amp ] && x=(--draft-amp "$AD")
            ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --gen-ids "$OUT/prompt.ids" -n "$NCAP" --temp 1.0 --top-p 1.0 --min-p 0 --seed "$seed" \
                --emit-trace "${x[@]}" > "$OUT/${tag}_$seed.out" 2> "$OUT/${tag}_$seed.err" </dev/null || DIE "$tag seed $seed 跑失败"
            grep -a -h '^\[emit\] ' "$OUT/${tag}_$seed.err" | sort -n -k2 | awk '{print $3}' > "$OUT/${tag}_$seed.gen.ids"
            echo "  $tag seed $seed: $(grep -a -h 'decode .* token' "$OUT/${tag}_$seed.err" | tail -1 | sed 's/.*decode //') 末 id $(tail -1 "$OUT/${tag}_$seed.gen.ids")"
            python3 "$ROOT/speed-bench/bugmd_ids_tools.py" loopstat "$OUT/${tag}_$seed.gen.ids" | sed 's/^/     /' | head -3
        done
    done
    LOG "DKLOOP_DONE $OUT"
}
# dkab <文本 ids(带 .np)> <引擎旗标> [生成 token 数 512]: 同一二进制、同一提示, 贪心不带/带旗标各一趟 —— 逐字节 cmp + t/s + 一轮分账(诊断发法用, 如 --no-side-stream)
stage_dkab(){
    local IDS="${1:?文本 ids}" FLAG="${2:?旗标}" NGEN="${3:-512}"
    [ -s "$IDS.np" ] || DIE "$IDS 没有 .np"
    need_idle
    local OUT="$DKD/ab-$(basename "$IDS" .ids)-$(echo "$FLAG" | tr -d ' -')"; mkdir -p "$OUT"
    local NP; NP=$(cat "$IDS.np"); head -n "$NP" "$IDS" > "$OUT/prompt.ids"
    local tag x
    for tag in base flag; do
        x=(); [ "$tag" = flag ] && x=($FLAG)
        ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --gen-ids "$OUT/prompt.ids" -n "$NGEN" --temp 0 --seed 1 "${x[@]}" > "$OUT/$tag.out" 2> "$OUT/$tag.err" </dev/null || DIE "$tag 跑失败, 见 $OUT/$tag.err"
        echo "  $tag: $(grep -a -h 'decode .* token' "$OUT/$tag.err" | tail -1 | sed 's/.*decode //') $(grep -a -h 'DSpark: ' "$OUT/$tag.err" | tail -1 | sed 's/.*DSpark: //' | cut -c1-60)"
        echo "     $(grep -a -h '一轮 .* ms = 草稿' "$OUT/$tag.err" | tail -1 | sed 's/.*\[v41\] //' | cut -c1-110)"
    done
    cmp -s "$OUT/base.out" "$OUT/flag.out" && echo "  两趟逐字节同 ✓" || echo "  ★两趟不同★"
    LOG "DKAB_DONE $OUT"
}
# dkgate <件目录> <文本 ids(dkgen 的产物, 带 .np)> [温 1.0] [生成 token 数 512]: 部署门三件 ——
#   ①贪心: 纯解码 / 投机不挂件 / 投机挂件 三趟输出逐字节同(投机只提议, 件不许动最终文本) + 各自接受率
#   ②温 T(模型卡配方)投机 挂件 vs 不挂: 接受率与 t/s(单趟只作参考, 采样路每趟是另一篇文本)
#   ③取料路陪审团(--dspark-capture, 同一份文本同温度)挂件 vs 不挂: 挂件那行应与训练器留出评估的首位同数 —— 证明部署路把件用上了
stage_dkgate(){
    local AD="${1:?件目录}" IDS="${2:?文本 ids}" T="${3:-1.0}" NGEN="${4:-512}"
    [ -s "$AD/base.fnv" ] || DIE "$AD 不是件目录(缺 base.fnv)"
    [ -s "$IDS.np" ] || DIE "$IDS 没有 .np(要 dkgen 的产物)"
    need_idle
    local OUT="$AD/gate-$(basename "$IDS" .ids)-t$T"; mkdir -p "$OUT"
    local NP; NP=$(cat "$IDS.np"); head -n "$NP" "$IDS" > "$OUT/prompt.ids"
    local base=(./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --gen-ids "$OUT/prompt.ids" -n "$NGEN" --seed 1)
    local tag; for tag in g_pure g_spec g_amp s_spec s_amp; do
        local a=(); case $tag in g_pure) a=(--no-dspark --temp 0);; g_spec) a=(--temp 0);; g_amp) a=(--temp 0 --draft-amp "$AD");;
                                 s_spec) a=(--temp "$T" --top-p 1.0 --min-p 0);; s_amp) a=(--temp "$T" --top-p 1.0 --min-p 0 --draft-amp "$AD");; esac
        "${base[@]}" "${a[@]}" > "$OUT/$tag.out" 2> "$OUT/$tag.err" </dev/null || DIE "$tag 跑失败, 见 $OUT/$tag.err"
        echo "  $tag: $(grep -a -h 'decode .* token' "$OUT/$tag.err" | tail -1) $(grep -a -h 'DSpark: ' "$OUT/$tag.err" | tail -1 | cut -c1-120)"
    done
    if cmp -s "$OUT/g_pure.out" "$OUT/g_spec.out" && cmp -s "$OUT/g_pure.out" "$OUT/g_amp.out"; then echo "  ①贪心三趟逐字节同 ✓ ($(wc -c < "$OUT/g_pure.out") 字节)"
    else echo "  ★①贪心三趟不同: 件动了最终文本(或投机本来就不同轨), 停下查★"; cmp "$OUT/g_pure.out" "$OUT/g_amp.out" | head -2; fi
    local cap; for cap in none amp; do
        local a=(); [ $cap = amp ] && a=(--draft-amp "$AD")
        ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --score-ids "$IDS" --dspark-capture "$OUT/cap_$cap.dcap" --decoder-full \
            --temp "$T" --dspark-capture-prompt "$NP" "${a[@]}" > /dev/null 2> "$OUT/cap_$cap.err" </dev/null || DIE "取料 $cap 失败, 见 $OUT/cap_$cap.err"
        echo "  ③陪审团($cap): $(grep -a -h '陪审团' "$OUT/cap_$cap.err" | sed 's/.*温/温/')"
    done
    LOG "DKGATE_DONE $OUT"
}

# 训练发车(train / ptgate / ptcheck 共用): 停服务 → 看门狗(train_guard) → ./ds4 --ptrain <输出目录>/ptrain.cfg → 日志尾部上屏, 失败停车
kd_train_run(){
    local OUT="$1"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    LOG "训练: $(awk -F= '$1=="data"{v=$2} END{print v}' "$OUT/ptrain.cfg") → $OUT; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
    ( sleep 20; train_guard ) &
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" ${ZARGS[@]+"${ZARGS[@]}"} ${ENG[@]+"${ENG[@]}"} --mem-budget-mb 110000 --no-dspark --ptrain "$OUT/ptrain.cfg" > "$OUT/train.out" 2>&1 </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/train.out" | grep -E "ptrain|探针|部署态|挂 ③|教师参考|问:" | tail -80
    [ "$RC" = 0 ] || DIE "训练失败(rc=$RC, 见 $OUT/train.out)"
}

# kddiag <③根目录> <ckpt_eNN|->: 逐位诊断(./ds4 --ptrain diag=1, 10-02)。不训练, 一次装载打完训练那趟的全部留出题(不含保持料):
#   每道题的答案逐位并排 教师(读材料) / 挂 ③ / 部署态(③ 置零) 三份前 5 名 + 参考 token 的概率, 末尾汇总"分叉位"
#   (教师榜首 ≠ 部署态榜首 = 材料改了模型选择的位置: 教师选的是不是参考答案、挂 ③ 跟上没有) → <③根目录>/diag-<ckpt>/diag.txt。
#   为什么: 选轮用的留出损失是整段平均, 量不到"答二还是答三"那一位(10-02 1+1=3: 留出 KL 降 80%, 探针照样答二)。料/层/秩/maxlen 照 ③ 的 ptrain.cfg。
stage_kddiag(){
    local R0="${1:?③根目录}" CK="${2:-}"
    [ -s "$R0/ptrain.cfg" ] || DIE "$R0 不是 train 的输出目录(缺 ptrain.cfg)"
    local PT="$R0"; [ -z "$CK" ] || [ "$CK" = - ] || PT="$R0/$CK"
    [ -s "$PT/amp_L39.bin" ] || DIE "没有 $PT/amp_L39.bin"
    local OUT="$R0/diag-$(basename "$PT")"; mkdir -p "$OUT"
    grep -v '^out=\|^init=\|^diag=\|^epochs=\|^prof=\|^max_steps=\|^gradcheck=\|^packcheck=' "$R0/ptrain.cfg" > "$OUT/ptrain.cfg"
    printf 'out=%s\ninit=%s\ndiag=1\n' "$OUT" "$PT" >> "$OUT/ptrain.cfg"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    LOG "kddiag: 挂 $PT → $OUT"
    ( sleep 20; train_guard ) &
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" ${ZARGS[@]+"${ZARGS[@]}"} ${ENG[@]+"${ENG[@]}"} --mem-budget-mb 110000 --no-dspark --ptrain "$OUT/ptrain.cfg" > "$OUT/diag.out" 2>&1 </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/diag.out" | grep -E "ptrain" | tail -40
    [ "$RC" = 0 ] && [ -s "$OUT/diag.txt" ] || DIE "诊断失败(rc=$RC, 见 $OUT/diag.out)"
    LOG "KDDIAG_DONE $OUT/diag.txt"
}

# ptgate <训练器输出目录(含 ptrain.cfg, epochs=0)> [标签] [k=v]: ★生成侧改动的逐字节门★ —— 同配置再跑一趟, probe_e00.txt 与原目录逐字节比(温 0 探针 1 路一批与单请求路同轨)
stage_ptgate(){
    local SRC="${1:?训练器输出目录(含 ptrain.cfg, epochs=0)}" TAG="${2:-gate}" EXTRA="${3:-}" OUT="${1%/}_${2:-gate}" f
    [ -s "$SRC/ptrain.cfg" ] && [ -s "$SRC/probe_e00.txt" ] || DIE "ptgate: $SRC 缺 ptrain.cfg 或 probe_e00.txt"
    grep -q '^epochs=0' "$SRC/ptrain.cfg" || DIE "ptgate: 只对 epochs=0 的目录(第 0 步就是产物)"
    mkdir -p "$OUT"
    grep -v '^out=' "$SRC/ptrain.cfg" > "$OUT/ptrain.cfg"
    printf 'out=%s\n' "$OUT" >> "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    LOG "ptgate: 重跑 $SRC 的配置 → $OUT${EXTRA:+ (+ $EXTRA)}"
    kd_train_run "$OUT"
    local bad=0
    for f in probe_e00.txt; do
        [ -s "$SRC/$f" ] || continue
        if cmp -s "$SRC/$f" "$OUT/$f"; then LOG "ptgate: $f 逐字节同($(wc -c < "$OUT/$f") B)"
        else bad=1; LOG "ptgate: ★$f 不同★ $(diff "$SRC/$f" "$OUT/$f" | grep -c '^[<>]') 行有差; 前 6 行差异:"; diff "$SRC/$f" "$OUT/$f" | head -6 | cut -c1-200 | tee -a "$LOGF"; fi
    done
    LOG "ptgate 用时 旧: $(tr '\r' '\n' < "$SRC/train.out" | grep -a -E 'ptrain 探针 e00\]' | sed -E 's/.*\] //' | tr '\n' ' ')"
    LOG "ptgate 用时 新: $(tr '\r' '\n' < "$OUT/train.out" | grep -a -E 'ptrain 探针 e00\]' | sed -E 's/.*\] //' | tr '\n' ' ')"
    [ "$bad" = 0 ] && LOG "PTGATE_DONE 过门" || DIE "PTGATE 没过门(产物在 $OUT)"
}

# ptcheck <训练器输出目录> <k=v,...> [标签 check]: 拿该目录的 ptrain.cfg 原样再跑一趟、追加几项(只换 out=<目录>_<标签>), 打检查/计时行。
#   训练器自带的检查与计时都靠配置键开: packcheck=1 / gradcheck=2 / max_steps=N,prof=1(计时) —— 不动原目录。
stage_ptcheck(){
    local SRC="${1:?训练器输出目录(含 ptrain.cfg)}" EXTRA="${2:?追加的 k=v,...}" TAG="${3:-check}" OUT
    OUT="${1%/}_${TAG}"
    [ -s "$SRC/ptrain.cfg" ] || DIE "ptcheck: $SRC 缺 ptrain.cfg"
    mkdir -p "$OUT"
    grep -v '^out=' "$SRC/ptrain.cfg" > "$OUT/ptrain.cfg"
    printf 'out=%s\n' "$OUT" >> "$OUT/ptrain.cfg"
    tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    LOG "ptcheck: $SRC + $EXTRA → $OUT"
    kd_train_run "$OUT"
    tr '\r' '\n' < "$OUT/train.out" | grep -aE "packcheck|梯度检查|ptrain prof 终|max_steps=.*到了|训练料" | tee -a "$LOGF"
    LOG "PTCHECK_DONE $OUT"
}

case "${1:-all}" in
  dkgen)   stage_dkgen "${2:-}" "${3:-1}" "${4:-1024}" "${5:-1.0}";;
  dktrain) stage_dktrain "${2:-}" "${3:-}" "${4:-}" "${5:-.}";;
  dkgate)  stage_dkgate "${2:-}" "${3:-}" "${4:-1.0}" "${5:-512}";;
  dkrun)   stage_dkrun "${2:-}" "${3:-}" "${4:-4}" "${5:-3}" "${6:-1536}";;
  dkspeed) stage_dkspeed "${2:-}" "${3:-}" "${4:-3}" "${5:-1024}" "${6:-}";;
  dkloop)  stage_dkloop "${2:-}" "${3:-}" "${4:-3}" "${5:-4096}";;
  dkab)    stage_dkab "${2:-}" "${3:-}" "${4:-512}";;
  ptgate)  stage_ptgate "${2:-}" "${3:-}" "${4:-}";;
  gen)     stage_gen "${2:-}" "${3:-}" "${4:-}";;
  gensplit) stage_gensplit "${2:-}";;
  train)   stage_train "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}";;
  ptcheck) stage_ptcheck "${2:-}" "${3:-}" "${4:-}";;
  docflip) stage_docflip "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}";;
  docnll)  stage_docnll "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  doc3)    stage_doc3 "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}";;
  docprobe) shift; stage_docprobe "$@";;
  docgate) stage_docgate "${2:-}" "${3:-}";;
  sample)  stage_sample "${2:-}" "${3:-4}" "${4:--}" "${5:-}" "${6:-1.0}" "${7:-1.0}" "${8:-}";;
  reward)  stage_reward "${2:-}";;
  solve3)  stage_solve3 "${2:-}" "${3:-}" "${4:-}";;
  demo)    stage_demo "${2:-}" "${3:-4}";;
  reviewiter) shift; stage_reviewiter "$@";;
  review)  stage_review "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  reviewrun) stage_reviewrun "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  kdpick)  stage_kdpick "${2:-}";;
  kddiag)  stage_kddiag "${2:-}" "${3:-}";;
  kdprof)  stage_kdprof "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}";;
  samples) stage_samples "${2:-}";;
  split)   stage_split;;
  capture) stage_capture "${2:-}";;
  solve)   stage_solve "${2:-}";;
  deploy)  stage_deploy;;
  probe)   shift; stage_probe "${1:-}";;
  nll)     stage_nll "${2:-}" "${3:-base}";;
  rows)    stage_rows;;
  gate)    stage_gate "${2:-}" "${3:-}";;
  gatea)   stage_gatea "${2:-}" "${3:-0}";;
  argmax)  stage_argmax;;
  # ★段 1: 尺 C 天花板★(back.md 第四版 §3.2) —— 解 ③ 之前的发车闸, 不过门不许进 sft
  ceiling) stage_ceiling "${2:-0}";;
  # ★段 1′: 自由生成尺★(尺 C 判死后的新口径, 与部署同轨)
  freegen) stage_freegen "${2:-0}" "${3:-500}";;
  freejudge) stage_freejudge;;
  sft)       stage_sft "${2:-}" "${3:-}" "${4:-}" 0 "${5:-}" "${6:-}" 0 "${7:-}" "${8:-}" "${9:-}";;
  sftsolve)  stage_sft "${2:-}" "${3:-}" "${4:-}" 1 "${5:-}" "${6:-}" 0 "${7:-}" "${8:-}" "${9:-}";;
  # 共享通道模式(泛化针): 同样两个入口, 只是解的未知数从 384×5120 压到 5120
  sftshare)  stage_sft "${2:-}" "${3:-}" "${4:-}" 1 "${5:-}" "${6:-}" 1 "${7:-}" "${8:-}" "${9:-}";;
  nlltext) stage_nlltext;;
  nllaudit) stage_nllaudit "${2:-}";;
  evalaudit) stage_evalaudit;;
  restore) stage_deploy;;
  # ★整晚一条龙★见下面 all) 分支; stage_all_sft 在 stage_gate 之后定义
  # ★整晚一条龙★(第二版): 取样本 → 解出候选网格 → 逐候选过尺 A(必过) → 过了的再过尺 B/守门 → 上线。
  # 候选按【预测】排序, 只真跑前 3 个: 真跑一个候选 = 8 条训练样本重打分 + 4 条判决样本 + 两把五指标。
  all)     stage_samples "${2:-}" && stage_rows && stage_nll "" base && stage_all_sft;;
  *) echo "用法: $0 [samples|capture|solve|deploy|probe <标签>|nll [N] [微调]|sft [上轮微调] [η] [名]|nllaudit [N]|evalaudit|restore|all]"; exit 2;;
esac
