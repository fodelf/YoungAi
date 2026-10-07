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
#   kdgen <料目录> [块名正则] [轮数] [并发]   ★第八版 上下文蒸馏的料★: 起服务让模型读每个复盘块自出问答(三种种子) → qa/<块>.{train,eval}.qa
#   kdsplit <料目录>   只重拆问答(gen/*.json → qa/), 不起服务; 改了拆对规则后用
#   kdtrain <料目录> [层] [lr] [轮数] [块名正则] [标签] [k=v,...]   ★第八版训练★: ./ds4 --ptrain(教师 top-K → 反传训 ③ → 留出 KL + 贪心探针)
#                 → $FTD/kd-<料名>-<标签>/(每轮 ckpt_eNN + 末轮本目录; 轮数照训满, 挑轮交给 kdpick);
#                 第 7 参追加配置(gradcheck=2 只查梯度就退, probe_n=12 等)
#   reviewrun <请求 JSON> <③目录|-> <输出 .sse> [温度]   真实请求重跑一趟(起服务挂/不挂 ③), 打思考/正文结论 + CFO 决策 JSON; A/B 给温度 0
#   kdpick <③根目录>   自动选轮: 每轮 ckpt 过 wt2 门(Σmin/KLD) → 过门的轮里留出 KL 最低者, 打各轮对照表
#   kdtable <③根目录>  只重打逐轮对照表(留出 KL + 各轮 kddiag 的分叉位跟上率), kdrun 中途停车后补齐 kddiag 再看表时用
#   kdrun <料目录> <训练轮数> [层] [lr] [出题轮数] [标签] [k=v,...]   ★一条龙★: kdgen → kdtrain → kdpick → 逐轮 kddiag + 对照表; 出题轮数 0 = 用现有问答; 料目录 = chunks/ 文本 + hold/questions.txt
#                 [+ probe/questions.txt 自定义探针] [+ seeds/s0~s2.txt 出题提示]
#   kdeval <③根目录> <请求 JSON>...   第八版训完验收: kdpick 选轮 → 每条请求温 0 不挂/挂选中的 ③(原样输出, 不参与选轮) → 解码图门
#   kdprof <料目录> [步数 5] [nsys|-] [k=v,...] [二进制] [额外引擎参数]   训练计时(只跑 N 步, 整步分段表; nsys = 再出逐核合计表), 要现成教师表; 二进制给 ds4.base_xxx = A/B; 额外参数例 --v41-prof(落训练路由)
#   kddiag <③根目录> <ckpt_eNN|-> <问答.qa|eval> [块.txt]   逐位诊断(不训练): 每道题答案逐位并排 教师 / 挂 ③ / 部署态 前 5 名 + 分叉位汇总 → <根>/diag-<ckpt>…/diag.txt
#   kdinc [篇数上限 4] [轮数 3] [lr 1e-4] [k=v,...]   ★一次增量后训练★(10-03/04): 不用给料 —— 从料池 $FTD/incr/pool/ 取还没训过的前 K 篇(名字序), 分配下一个序号
#                 → $FTD/incr/<序号>/ → (没问答就 kdgen) → 账本 + 校准料 → 训练前 kdfwd(转移 + 决策探针) → init=上一次 ACCEPTED 的 ③ 叠加训练 → wt2 门 + 套件门
#                 → ACCEPTED(current 指过去)/REJECTED → kddiag → incr/ledger.txt。进度 = incr/SEQ(序号) + incr/current(上线的 ③) + 各序号 chunks/(训过的料); kdinc @序号 [轮] [lr] [k=v] = 重跑该文件夹
#   kdpool        只看进度: SEQ / current / 料池里还没训过的篇(kdinc 下一次会取哪些), 不动模型。喂料 = 把 qtf corpus 产物的 chunks/ docs/ [qa/] [hold/] 同步进 $FTD/incr/pool/
#   kdtake [篇数上限 4]   只取料不训练(分配序号 + 账本 + 决策提示), 之后 kdinc @序号 训; kdscore <序号> <probe 文件> [stats 序号] 对训练器任一份 probe 文件按领域打三臂分
#                 账本料在清单里带第 4 列 hard: 训练器用答案 one-hot 当目标(交叉熵) + 数字位 ×hard_num(4) 其余 ×hard_txt(0.25), 不过教师信念(10-04)
#   kdfwd <序号> [③目录] [标签]   时间向前的转移测试: 用该序号之前训出的 ③ 对该次全部问答(它没见过)跑 kddiag + 决策探针(各股决策日材料 → 目标/止损 对次日实际), 挂③ 对部署态 对规则
#   kdledger <序号>   决策账本 + 校准料(kdinc 自动跑): 逐领域适配器(gguf-tools/scripts/kd_domain/<域>.sh, 契约见 _template.sh) docs/ 整篇 → ledger/<域>/<序号>.tsv 逐笔
#                 → 累计统计与规则 stats_<序号>.txt → chunks/ledger_<域>_<序号>.txt + 代码产问答 + decide/<域>/ 决策提示; 加领域 = 加一个文件, 主脚本不改
#   kdrft [轮数 3] [每题份数 8] [留出序号 = 最新有决策提示的序号] [lr 1e-4] [锚 β 1] [k=v,...]   ★奖励回路★(10-04): 分配序号, 起点 = current; 各序号的决策提示(留出序号的除外)当训练提示,
#                 每轮: 模型自己抽 G 份答案(训练器 sample_n, 温 1) → 适配器 <域>_reward 按次日实际结算 → one-hot × (收益 − 组均值) + 同串 token 的 kl 锚(教师 = 本轮起点 ③, β) + 回放 + 保持料
#                 → 一遍训练 → 轮末再抽(下一轮样本 + 本轮读数) → 各轮 wt2 门 + 套件门, 过门里训练提示收益最高者 ACCEPTED。kdrft @序号 [同参] = 续跑/重跑(第 0 轮抽过就不重抽)
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
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
    [ -n "$KD_PICK" ] || DIE "选轮: 没有一轮过 wt2 门, 不进任何对比(读数见 $W/gate_*.txt)"
    LOG "选轮结果: $KD_PICK(过门的轮里留出 KL 最低, $bkl)"
}
stage_kdpick(){ kd_pick "${1:?③根目录}"; LOG "KDPICK_DONE $KD_PICK"; }

# kdrun <料目录> <训练轮数> [层 0-39] [lr 2e-4] [出题轮数 1] [标签] [k=v,...]: ★料目录放好, 一条命令走完★(10-02)
#   kdgen(模型读 chunks/ 每块自出问答 + 部署态答 hold/ 通用题) → kdtrain(照轮数训满, 每轮存 ckpt_eNN) → kdpick(过 wt2 门的轮里留出 KL 最低)
#   → 每轮 kddiag eval(分叉位跟上率, 只打不判) → 逐轮对照表。出题轮数给 0 = 跳过 kdgen, 用 qa/ 现有问答(只改训练配置重训时用)。
#   料目录: chunks/*.txt = 要写进 ③ 的文本; hold/questions.txt = 通用题(教师 = 部署态自己, 把 ③ 钉住; 别放和要写的事实冲突的题,
#   比如写 1+1=3 时放"一加一等于几"等于让两头对拉); 可选 probe/questions.txt = 自定义探针(第 0 步和每轮末原样答一遍),
#   seeds/s0~s2.txt = 出题提示(缺省 = 复盘三种子)。层/lr 缺省 = 10-02 全量全层那趟的配方。
stage_kdrun(){
    local DOCD="${1:?料目录}" EP="${2:?训练轮数}" LAYERS="${3:-0-39}" LR="${4:-2e-4}" ROUNDS="${5:-1}" TAG="${6:-}" EXTRA="${7:-}"
    [ -d "$DOCD/chunks" ] || DIE "没有 $DOCD/chunks(要写进 ③ 的文本放这里)"
    [ -s "$DOCD/hold/questions.txt" ] || DIE "没有 $DOCD/hold/questions.txt(通用题, 不给的话 ③ 没有东西钉着, 10-01 不加约束料局部性崩)"
    local NAME; NAME="$(basename "$DOCD")"; [ -n "$TAG" ] || TAG="L${LAYERS}-lr${LR}-e${EP}"
    # 出题轮数 0 = 不出题, 用料目录现有问答(同一批题只改训练配置时用; 重新出题会换掉题目, 前后两趟就不可比了)
    if [ "$ROUNDS" = 0 ]; then
        ls "$DOCD"/qa/*.train.qa >/dev/null 2>&1 || DIE "出题轮数 0 = 用现有问答, 但 $DOCD/qa 里没有 *.train.qa(先跑一遍 kdgen)"
        LOG "kdrun: 出题轮数 0, 不出题, 用 $DOCD/qa 现有问答($(ls "$DOCD"/qa/*.train.qa | wc -l) 份训练题文件)"
    else stage_kdgen "$DOCD" . "$ROUNDS" 1; fi
    stage_kdtrain "$DOCD" "$LAYERS" "$LR" "$EP" . "$TAG" "$EXTRA"
    local R="$FTD/kd-$NAME-$TAG" n
    stage_kdpick "$R"
    # 逐轮分叉位核对(只打不判, 选轮仍按 kdpick): 留出 KL 是整段平均, 分叉位跟上率才直接说"材料写进去了多少"(10-02 加)
    for n in $(seq 1 "$EP"); do stage_kddiag "$R" "$(printf 'ckpt_e%02d' "$n")" eval; done
    kd_diag_table "$R"
    LOG "KDRUN_DONE $R 选中 $KD_PICK"
}

# 逐轮对照表: 留出 KL(train.log 最后一趟) + 分叉位挂③跟上率 / 教师有把握的分叉位跟上率 / 含数字参考位挂③答对率(各轮 diag.out 汇总行)
kd_diag_table(){
    local R="$1" f n kl a b c
    LOG "逐轮对照(只打不判): 轮 | 留出 KL | 分叉位挂③跟上教师 | 教师有把握(p≥0.5)的分叉位 | 含数字参考位挂③答对"
    for f in "$R"/diag-ckpt_e*-eval/diag.out; do
        [ -s "$f" ] || continue
        n="$(basename "$(dirname "$f")" | sed 's/diag-ckpt_e0*\([0-9]*\)-eval/\1/')"
        kl="$(awk -v n="$n" '$1=="step" && $2=="0"{split("",v)} $1=="epoch" && $2==n{v[n]=$4} END{print v[n]}' "$R/train.log")"
        a="$(grep -a '分叉位(教师榜首≠部署榜首)' "$f" | tail -1 | sed 's/.*挂③榜首 = 教师榜首 \([0-9.]*%\).*/\1/')"
        b="$(grep -a '分叉位且教师榜首' "$f" | tail -1 | sed 's/.*挂③榜首 = 教师榜首 \([0-9.]*%\).*/\1/')"
        c="$(grep -a '含数字的参考位' "$f" | tail -1 | sed 's/.*挂③ \([0-9.]*%\) \/ 部署.*/\1/')"
        LOG "  e$n | $kl | $a | $b | $c"
    done
}

# kdeval <③根目录> <请求 JSON>...: 第八版训完的验收一条龙(10-02)。
#   ① kd_pick 自动选轮(见上);
#   ② 每条真实请求按温 0 跑: 不挂 ③ + 挂选中的 ③, sse 落 <根>/eval/<请求名>.{base,ckpt_eNN}.sse, CFO_JSON 进 nightly.log ——
#      只为把原样输出摆出来看行为, 不参与选轮;
#   ③ 挂选中的 ③ 过解码图门(d1_kv_ring_gate.sh graph): 放大器的 cuBLAS 核进了整步图, 走图 == 直发逐字节且没有"捕获失败/PDL 改边失败"。
stage_kdeval(){
    local R0="${1:?③根目录}"; shift
    [ $# -ge 1 ] || DIE "kdeval 至少要一条请求 JSON"
    local q; for q in "$@"; do [ -s "$q" ] || DIE "没有请求 $q"; done
    kd_pick "$R0"
    local W="$R0/eval" n pick="$KD_PICK"
    for q in "$@"; do
        n="$(basename "$q" .json)"
        stage_reviewrun "$q" - "$W/$n.base.sse" 0
        stage_reviewrun "$q" "$pick" "$W/$n.$(basename "$pick").sse" 0
    done
    # 门脚本自己不查实例锁: 服务被 pkill -9 后要等它真退出再装下一份模型(need_idle 不认 ds4-server)
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    pgrep -x ds4-server >/dev/null && DIE "服务还没退干净, 不装第二份模型"
    need_idle
    LOG "解码图门: 挂 $pick"
    bash "$ROOT/speed-bench/d1_kv_ring_gate.sh" - "$MDL" 64 graph "$ZCH" "--posttrain $pick" 2>&1 | tee -a "$LOGF" | tail -14
    LOG "KDEVAL_DONE $R0 选中 $pick"
}

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

# ---------------- ③ 第八版: 上下文蒸馏(2026-10-01 夜, 用户"发车") ----------------
# 方法(成熟方案的共识, fable5 10-01 夜检索段): 教师 = ①+② 把复盘块放进上下文; 学生 = ①+②+③ 不看块; 料 = 教师围着块自己出的
# 大量多样问答; 目标 = 学生在答案位逼近教师分布(KL), 走反传训练 ③。对原文做 next-token(六趟 doc3 的做法)各家都判了负。
# kdgen <料目录> [块名正则] [每种种子轮数] [并发路数]: 起服务(①+②, 不挂 ③) → 每块 × 三种种子提示 × 轮数 让模型读块出问答
#   → 按"问：/答："拆对 → 按问题哈希五取一留作 eval(措辞没进训练) → qa/<块>.train.qa / .eval.qa → 停服务。
#   三种种子(Cartridges 的五类收成三类, 都要求问题自带完整指代 —— 学生不看材料, "这份复盘"指不到任何东西):
#     s0 事实问答 8 组 / s1 推理与教训问答 6 组 / s2 概括 2 组 + 交易员请教 2 组。采样走模型卡(温 1), 不思考。
KD_SEED0='请仔细阅读上面的复盘材料，然后提出 8 个具体问题并逐一作答。要求：1. 每个问题都必须写明完整指代（如股票代码和名称、具体日期），让没看过这份材料的人也知道问的是哪一件事，不许出现“这份材料”“上文”之类的说法；2. 问题覆盖材料里不同的具体事实（价格、日期、数字、错误类型、结论、原因等），不要重复；3. 答案只依据材料，简洁准确。严格按下面的格式输出，不要输出别的内容：
问：……
答：……'
KD_SEED1='请仔细阅读上面的复盘材料，然后提出 6 个需要理解和推理才能回答的问题并作答，例如：当时的判断错在哪里、为什么会错、应该怎样修正、这次复盘的教训是什么、以后遇到类似情况该怎么做。要求每个问题都写明完整指代（如股票代码和名称、具体日期），不许出现“这份材料”“上文”之类的说法；答案要有依据、讲清理由。严格按下面的格式输出，不要输出别的内容：
问：……
答：……'
KD_SEED2='请根据上面的复盘材料写 4 组问答：第 1、2 组请对方概括这次复盘的要点（问题里写明股票代码和名称或具体日期）；第 3、4 组是一位交易员向你请教在类似行情里怎样避免同样的错误（问题里点明是哪只股票或哪一天的复盘）。不许出现“这份材料”“上文”之类的说法；答案要具体。严格按下面的格式输出，不要输出别的内容：
问：……
答：……'
# 问答拆对(kdgen 末尾调; 也可单独 kdsplit <料目录> 重拆, 不起服务): gen/*.json → qa/<块>.{train,eval}.qa, 留出 = md5(题) % 5 == 0
kd_split(){
    local DOCD="${1:?料目录}"
    python3 - "$DOCD" <<'PY' || DIE "问答拆对失败"
import glob, hashlib, json, os, re, sys
d = sys.argv[1]
by = {}
bad = 0
hq = [l.strip() for l in open(os.path.join(d, "hold", "questions.txt"))] if os.path.exists(os.path.join(d, "hold", "questions.txt")) else []
hq = [q for q in hq if q]
for p in sorted(glob.glob(os.path.join(d, "gen", "hold_q*.json"))):   # 保持料: 题目在 hold/questions.txt 第 N 行, 答案是整段回答
    i = int(re.search(r"hold_q(\d+)\.json$", p).group(1)) - 1
    a = (json.load(open(p))["choices"][0]["message"].get("content") or "").strip()
    if 0 <= i < len(hq) and a: by.setdefault("hold_general", []).append((hq[i], a))
for p in sorted(glob.glob(os.path.join(d, "gen", "*.json"))):
    if os.path.basename(p).startswith("hold_q"): continue
    chunk = re.sub(r"_s\d+r\d+\.json$", "", os.path.basename(p))
    txt = json.load(open(p))["choices"][0]["message"].get("content") or ""
    txt = txt.replace("**", "")
    cur_q, cur_a, mode, pairs = [], [], None, []
    for line in txt.splitlines():
        # 标记行: "问：/答：", 前面可带序号("1. 问："), 后面也可带序号("问1：/答1：", 10-01 有 13 份生成这样写, 原正则整份拆不出)
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
    by.setdefault(chunk, []).extend(p for p in pairs if p[0] and p[1])
ntr = nev = 0
for chunk, pairs in by.items():
    seen, tr, ev = set(), [], []
    for q, a in pairs:
        k = re.sub(r"\s+", "", q)
        if k in seen: continue
        seen.add(k)
        (ev if int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0 else tr).append((q, a))
    for name, lst in (("train", tr), ("eval", ev)):
        with open(os.path.join(d, "qa", "%s.%s.qa" % (chunk, name)), "w") as f:
            for q, a in lst: f.write("#Q\n%s\n#A\n%s\n" % (q, a))
    ntr += len(tr); nev += len(ev)
print("问答拆对: %d 块, 训练 %d 组 / 留出 %d 组; 没拆出问答的生成 %d 份" % (len(by), ntr, nev, bad))
PY
}
stage_kdsplit(){ kd_split "${1:?料目录}"; }
stage_kdgen(){
    # 并发默认 1(10-01 实撞): --batch 3 的并发调度器在"采样 + 长提示"下每对请求一条首 token 就 "V4.1 decode failed"、另一条百来个 token 后同样报错
    #   (服务端日志 finish=error, 24 份生成全废)。并发路的病单独立案, 这里走单路串行, 给 >1 才开 --batch。
    local DOCD="${1:?料目录}" PAT="${2:-.}" ROUNDS="${3:-1}" PAR="${4:-1}"
    [ -d "$DOCD/chunks" ] || DIE "没有块目录 $DOCD/chunks(先在 Mac 上 qtf_requests_mac.sh corpus <料名>)"
    local -a CH=(); local f
    # 空块跳过: hold_general.txt 是保持料的空块(本段第一次跑完才落盘), 续跑时若当普通块去生成, 模型会对着空材料自编问答混进保持料
    # 账本块跳过: ledger_<域>_<序号>.txt 与它的问答都是 kd_ledger 代码产的, 不让模型出题(10-06 实撞: kdtake 之后 kdinc @序号, 账本块已在 chunks/ 里,
    #   白生成 3 份且 kd_split 把代码问答盖掉, 全靠随后的 kd_ledger 再盖回来)
    for f in "$DOCD"/chunks/*.txt; do b="$(basename "$f" .txt)"; [[ "$b" == ledger_* ]] && continue; [ -s "$f" ] && [[ "$b" =~ $PAT ]] && CH+=("$f"); done
    [ "${#CH[@]}" -gt 0 ] || DIE "块名正则 $PAT 一个都没选中"
    mkdir -p "$DOCD/gen" "$DOCD/qa"
    LOG "kdgen: ${#CH[@]} 块 × 3 种子 × $ROUNDS 轮, 并发 $PAR → $DOCD/gen"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    local -a BATCH=(); [ "$PAR" -gt 1 ] && BATCH=(--batch "$PAR")
    bash "$SC/serve_1m_spark.sh" start "$MDL" "$ZCH" ${BATCH[@]+"${BATCH[@]}"} >>"$LOGF" 2>&1 || DIE "服务起不来(看 ~/ds4-server-1m.log)"
    local t0; t0=$(date +%s)
    gen_one(){   # <块文件> <种子号> <轮> —— 已有非空产物就跳过(断点续跑)
        local cf="$1" s="$2" r="$3" b out seed
        b="$(basename "$cf" .txt)"; out="$DOCD/gen/${b}_s${s}r${r}.json"
        [ -s "$out" ] && return 0
        # 料目录自带 seeds/s<号>.txt 就用它: 内置三种子句句是"复盘材料/股票代码/交易员", 拿去问一篇不是复盘的文档会答非所问
        if [ -s "$DOCD/seeds/s$s.txt" ]; then seed="$(cat "$DOCD/seeds/s$s.txt")"
        else case "$s" in 0) seed="$KD_SEED0";; 1) seed="$KD_SEED1";; *) seed="$KD_SEED2";; esac; fi
        jq -n --rawfile c "$cf" --arg q "$seed" '{model:"deepseek-chat", max_tokens:4096, messages:[{role:"user", content:($c + "\n" + $q)}]}' \
            | curl -s -m 1800 "http://127.0.0.1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d @- > "$out.tmp" \
            && jq -e '.choices[0].finish_reason == "stop"' "$out.tmp" >/dev/null 2>&1 && mv "$out.tmp" "$out" \
            || { echo "★生成失败 $b s$s r$r: $(head -c 300 "$out.tmp" 2>/dev/null)★"; rm -f "$out.tmp"; }
    }
    # ★保持料★(10-01 夜, wt2 守门中位 KLD 0.031 → 0.055 之后加): hold/questions.txt 每行一道通用题, 部署态不看任何块直接答 →
    #   训练时这些题的教师 = 部署态自己(同一个提示), KL 把 ③ 在通用问题上钉住不许漂。产物 gen/hold_qNNN.json → qa/hold_general.*.qa
    hold_one(){
        local i="$1" q="$2" out="$DOCD/gen/hold_q$(printf '%03d' "$1").json"
        [ -s "$out" ] && return 0
        jq -n --arg q "$q" '{model:"deepseek-chat", max_tokens:2048, messages:[{role:"user", content:$q}]}' \
            | curl -s -m 1800 "http://127.0.0.1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d @- > "$out.tmp" \
            && jq -e '.choices[0].finish_reason == "stop"' "$out.tmp" >/dev/null 2>&1 && mv "$out.tmp" "$out" \
            || { echo "★保持料生成失败 第 $i 题: $(head -c 300 "$out.tmp" 2>/dev/null)★"; rm -f "$out.tmp"; }
    }
    if [ -s "$DOCD/hold/questions.txt" ]; then
        local hi=0 hq
        while IFS= read -r hq || [ -n "$hq" ]; do
            [ -n "$hq" ] || continue; hi=$((hi+1))
            while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
            hold_one "$hi" "$hq" >>"$LOGF" 2>&1 &
        done < "$DOCD/hold/questions.txt"
        wait
        : > "$DOCD/chunks/hold_general.txt"   # 空块 = 教师不看任何材料(训练器认空块: 教师提示 = 学生提示)
        LOG "kdgen 保持料: $hi 题, 产出 $(ls "$DOCD"/gen/hold_q*.json 2>/dev/null | wc -l)"
    fi
    local s r k=0 tot=$(( ${#CH[@]} * 3 * ROUNDS ))
    for r in $(seq 1 "$ROUNDS"); do for f in "${CH[@]}"; do for s in 0 1 2; do
        k=$((k+1))
        # 续跑: 已有产物的先在这里跳过 —— 放进后台再判, 每份都要白等一轮 sleep 2(10-01 续跑实撞: 122 份已有产物空等 4 分钟)
        [ -s "$DOCD/gen/$(basename "$f" .txt)_s${s}r${r}.json" ] && continue
        while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
        gen_one "$f" "$s" "$r" >>"$LOGF" 2>&1 &
        [ $((k % 6)) = 0 ] && LOG "kdgen 已发 $k/$tot, $(( $(date +%s) - t0 )) s, 产出 $(ls "$DOCD"/gen/*.json 2>/dev/null | wc -l)"
    done; done; done
    wait
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1
    kd_split "$DOCD"
    LOG "KDGEN_DONE $(( $(date +%s) - t0 )) s, $(ls "$DOCD"/gen/*.json | wc -l) 份生成"
}

# kdtrain <料目录> [层 39|a-b] [lr] [轮数] [块名正则] [标签]: 拼清单(qa/<块>.{train,eval}.qa ↔ chunks/<块>.txt) + 配置 → ./ds4 --ptrain
#   (src/core/core_ptrain*.c: 教师 top-K 带缓存 → 第 0 步评估 → 训练 → 每轮留出 KL + 贪心探针) → ③ 落 $FTD/kd-<料名>-<标签>/。
#   看门狗同解算段(MemAvailable < 2500 MB 连续两次杀); 探针走生成路, 关投机(省草稿塔那 1.2 GB, 温 0 下两条路逐字节同)。
# 拼训练清单: qa/<块>.{train,eval}.qa ↔ chunks/<块>.txt, 一行 "<块> <问答> <train|eval>"(kdtrain / kdprof 共用)
# 保持料训练题在清单里列 KD_HOLD_REP 遍(= 这些题每轮练 KD_HOLD_REP 次): 通用题只有 ~50 道, 复盘题上千, 不加权时只占 ~5%,
#   拉回力太弱 —— 小子集那趟(116 题 × 3 轮)wt2 中位 KLD 已 0.031 → 0.055, 全量步数是它的 9 倍。4 遍 ≈ 17%。
#   增量(kdinc)一次只有一两百题, 保持料列 1 遍就占 15~25%, 那边按份额给(见 kd_inc_run)。
KD_HOLD_REP=4
kd_list(){
    local DOCD="$1" PAT="$2" LIST="$3" f b rep
    : > "$LIST"
    for f in "$DOCD"/qa/*.train.qa; do
        b="$(basename "$f" .train.qa)"; [[ "$b" =~ $PAT ]] || continue
        rep=1; [ "$b" = hold_general ] && rep=$KD_HOLD_REP
        for _ in $(seq 1 "$rep"); do echo "$DOCD/chunks/$b.txt $f train" >> "$LIST"; done
        [ -s "$DOCD/qa/$b.eval.qa" ] && echo "$DOCD/chunks/$b.txt $DOCD/qa/$b.eval.qa eval" >> "$LIST"
    done
    [ -s "$LIST" ] || DIE "清单为空(先跑 kdgen; 块名正则 $PAT)"
}

# 同料同清单的教师表跨目录复用: 教师 = 部署态读块, 只跟料有关、跟训练配置无关。同料的别的 kd-<料名>-* 目录里清单相同(按文件名比,
#   不比路径前缀)的 teacher.bin 拷过来, 引擎再按料哈希(全部 id + topk)核对, 对不上自动重算。为什么: 新目录现算教师表要 ~31 min,
#   而且在训练进程里现算会留下一块不还的内存(10-02 full6: 预热峰值余量 3067 MB, 读缓存 3494 MB), 贴着看门狗线。
kd_teacher_cache(){
    local OUT="$1" NAME="$2" sib
    [ -s "$OUT/teacher.bin" ] && return 0
    for sib in "$FTD"/kd-"$NAME"-*; do
        [ "$sib" != "$OUT" ] && [ -s "$sib/teacher.bin" ] && [ -s "$sib/data.list" ] || continue
        diff -q <(sed 's#[^ ]*/##g' "$sib/data.list") <(sed 's#[^ ]*/##g' "$OUT/data.list") >/dev/null || continue
        cp "$sib/teacher.bin" "$OUT/teacher.bin" && LOG "教师表从 $(basename "$sib") 拷来(引擎按料哈希核对, 对不上自动重算)"
        return 0
    done
}

# kdprof <料目录> [步数 5] [nsys|-] [k=v,...]: 训练计时(./ds4 --ptrain prof=1 max_steps=N, 10-02)。配置照 kdrun 缺省(全层 0-39, lr 2e-4, batch 4)
#   + maxlen 880(复盘全量口径), 第 4 参追加/覆盖。只跑 N 步: 不做第 0 步评估/探针、不评估不存盘; 每 5 步打一张整步分段表
#   (前向 / 损失 / 出口反传 / 逐层反传 = 重算 + routed 专家反向 + 注意力半层 + 层内其余 + 层间 / Adam, 段和对每题墙钟)。
#   要现成的教师表(kd_teacher_cache 从同料目录拷), 没有就停车 —— 不让 31 分钟的教师那遍混进计时。
#   第 3 参给 nsys = 套 nsys 采 GPU 时间线, 出逐核名合计表(cuda_gpu_kern_sum; 训练路 prof 每段同步、单流, 核时长不重叠, 直接加和可信);
#   nsys 下墙钟会虚高, 只看比例。第 5 参 = 用哪个引擎二进制(缺省 ./ds4; 改反传前 cp 一份 ds4.base_xxx, 同一题同一配置 A/B,
#   例: 第 4 参 gradcheck=2,gclayers=0/14/20/30/39 两个二进制各跑一遍, 比每层"反传"梯度与有限差分比值)。输出按二进制名分文件, 不互相覆盖。
#   第 6 参 = 额外的引擎命令行参数(原样追加), 例 --v41-prof: 训练前向里每层把逐专家 token 数落 /tmp/v41_route_Lnn_n<行数>.txt
#   (给 gguf-tools/bench/v41_vq_train_bench.cu 当训练形状的真路由; 每层要同步读回 sel, 计时作废)。
stage_kdprof(){
    local DOCD="${1:?料目录}" STEPS="${2:-5}" NS="${3:-}" EXTRA="${4:-}" BIN="${5:-./ds4}" XARGS="${6:-}"
    [ -x "$BIN" ] || DIE "没有引擎二进制 $BIN"
    local NAME OUT LIST RC; NAME="$(basename "$DOCD")"; OUT="$FTD/kd-$NAME-prof"; mkdir -p "$OUT"; LIST="$OUT/data.list"
    kd_list "$DOCD" . "$LIST"
    printf 'data=%s\nout=%s\nlayers=0-39\nrank=64\nlr=2e-4\nepochs=1\nbatch=4\ntopk=64\nmaxlen=880\nprof=1\nmax_steps=%s\n' "$LIST" "$OUT" "$STEPS" > "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    kd_teacher_cache "$OUT" "$NAME"
    [ -s "$OUT/teacher.bin" ] || DIE "kdprof 要现成的教师表(同料同清单的 kd-$NAME-* 目录里没有), 先跑一趟 kdtrain"
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

stage_kdtrain(){
    # 第 7 参: 额外配置项, 逗号分隔 k=v(如 gradcheck=2 只做梯度检查就退出; batch=8), 原样追加进配置(后写的覆盖前面的默认)
    # 第 4 参 = 训几轮(照训满, 每轮存 ckpt_eNN), 挑哪一轮归 kdpick。日志每轮一行"留出 KL 较上轮降 X ± 标准误":
    #   末轮还降 2 个标准误以上 = 下次可以多给几轮(10-02 全量 3 轮, 第 3 轮还在降 0.069)。
    #   ★10-03 起一轮 = 一个 epoch_tok 切片(≈ 1/3 料), 3 轮才覆盖全量一遍 —— 给轮数时按"想过几遍 × 3"给; 轮末那行比的是相邻切片★
    local DOCD="${1:?料目录}" LAYERS="${2:-39}" LR="${3:-3e-4}" EP="${4:-3}" PAT="${5:-.}" TAG="${6:-}" EXTRA="${7:-}"
    local NAME; NAME="$(basename "$DOCD")"; [ -n "$TAG" ] || TAG="L${LAYERS}-lr${LR}-e${EP}"
    local OUT="$FTD/kd-$NAME-$TAG"; mkdir -p "$OUT"
    local LIST="$OUT/data.list"
    kd_list "$DOCD" "$PAT" "$LIST"
    # epoch_tok(10-03 用户定): 一轮 = 打乱序里连续的一段, 行数(含提示)累计到它即收, 下一轮接着取, 取完才重新打乱(一个周期内每题恰好一次)。
    #   review_v2 全量训练料 1287 题 232023 行, 78000 ≈ 三分之一 ⇒ 3 轮过一遍; 按 10-03 的 7.4 ms/行, 一轮训练 ≈ 9.6 分钟 + 轮末评估/探针 ≈ 2.5 分钟。
    #   为什么不是全量一轮: 全层 + 全量一轮 30 分钟低于这台机器的带宽地板(每步 800 token 要流三遍几乎整层专家权重), 10 分钟一轮只能切片。
    printf 'data=%s\nout=%s\nlayers=%s\nrank=64\nlr=%s\nepochs=%s\nbatch=4\ntopk=64\nmaxlen=1024\nprobe_n=6\nprobe_tok=96\nepoch_tok=78000\n' \
        "$LIST" "$OUT" "$LAYERS" "$LR" "$EP" > "$OUT/ptrain.cfg"
    # 料目录里有 probe/questions.txt(一行一题)就当自定义探针: 第 0 步和每轮末部署态 / 挂 ③ 各答一遍, 原样进 probe_eNN.txt
    [ -s "$DOCD/probe/questions.txt" ] && echo "probe_q=$DOCD/probe/questions.txt" >> "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    kd_teacher_cache "$OUT" "$NAME"
    LOG "kdtrain: 层 $LAYERS lr $LR 轮 $EP"
    kd_train_run "$OUT"
    LOG "KDTRAIN_DONE $OUT"
}
# 训练发车(kdtrain / kdinc 共用): 停服务 → 看门狗(train_guard) → ./ds4 --ptrain <输出目录>/ptrain.cfg → 日志尾部上屏, 失败停车
kd_train_run(){
    local OUT="$1"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    LOG "训练: $(wc -l < "$OUT/data.list") 行清单 → $OUT; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
    ( sleep 20; train_guard ) &
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --no-dspark --ptrain "$OUT/ptrain.cfg" > "$OUT/train.out" 2>&1 </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/train.out" | grep -E "ptrain|探针|部署态|挂 ③|教师参考|问:" | tail -80
    [ "$RC" = 0 ] || DIE "训练失败(rc=$RC, 见 $OUT/train.out)"
}

# kddiag <③根目录> <ckpt_eNN|-> <问答.qa|eval> [块.txt]: 逐位诊断(./ds4 --ptrain diag=1, 10-02)。不训练, 一次装载打完所有题:
#   每道题的答案逐位并排 教师(读块) / 挂 ③ / 部署态(③ 置零) 三份前 5 名 + 参考 token 的概率, 末尾汇总"分叉位"
#   (教师榜首 ≠ 部署态榜首 = 材料改了模型选择的位置: 教师选的是不是参考答案、挂 ③ 跟上没有) → <③根目录>/diag-<ckpt>[-块名]/diag.txt。
#   为什么: 选轮用的留出 KL 是整段平均, 量不到"答二还是答三"那一位(10-02 1+1=3: 留出 KL 降 80%, 探针照样答二)。
#   第 3 参给 eval = 用这份 ③ 训练时的全部留出题(清单里 eval 行, 不含保持料), 各题配各自的块; 给 .qa 文件 = 只用这些题 + 一个块
#   (块缺省 = 清单里第一个非保持料块)。.qa 的答案填学生自己的贪心输出 = 打它自己走的那条路上的决策位。层与秩照 ③ 的 ptrain.cfg。
stage_kddiag(){
    local R0="${1:?③根目录}" CK="${2:?ckpt 名(ckpt_eNN; - = 根目录那份)}" QA="${3:?问答 .qa 或 eval}" CH="${4:-}"
    [ -s "$R0/ptrain.cfg" ] && [ -s "$R0/data.list" ] || DIE "$R0 不是 kdtrain 的输出目录(缺 ptrain.cfg / data.list)"
    local PT="$R0"; [ "$CK" = - ] || PT="$R0/$CK"
    [ -s "$PT/amp_L39.bin" ] || DIE "没有 $PT/amp_L39.bin"
    local OUT LAYERS RANK NQ
    # 显式给了块 = 换教师上下文做对比, 输出目录带块名, 不覆盖默认那份; eval = 整份留出集, 目录带 -eval
    if [ "$QA" = eval ]; then OUT="$R0/diag-$(basename "$PT")-eval"
    else OUT="$R0/diag-$(basename "$PT")${CH:+-$(basename "$CH" .txt)}"; fi
    mkdir -p "$OUT"
    if [ "$QA" = eval ]; then
        grep ' eval$' "$R0/data.list" | grep -v hold_general > "$OUT/data.list"
        [ -s "$OUT/data.list" ] || DIE "$R0/data.list 里没有留出行"
        # 同一份留出集的教师表与 ③ 无关(教师 = 部署态读块): 别的 ckpt 的 eval 诊断目录有同清单的 teacher.bin 就拷来,
        #   引擎按料哈希核对, 对不上自动重算 —— 逐轮诊断时每轮省掉教师那一遍(复盘 70 块 381 s)
        local sib; for sib in "$R0"/diag-*-eval; do
            [ "$sib" != "$OUT" ] && [ -s "$sib/teacher.bin" ] && cmp -s "$sib/data.list" "$OUT/data.list" && {
                cp "$sib/teacher.bin" "$OUT/"; LOG "kddiag: 教师表从 $(basename "$sib") 拷来(引擎按料哈希核对)"; break; }
        done
        NQ=$(awk '{print $2}' "$OUT/data.list" | xargs grep -h -c '^#Q' | awk '{s+=$1} END{print s}')
        CH="$(wc -l < "$OUT/data.list") 块"
    else
        [ -s "$QA" ] || DIE "没有问答 $QA"
        [ -n "$CH" ] || CH="$(awk '$1 !~ /hold_general/ {print $1; exit}' "$R0/data.list")"
        [ -s "$CH" ] || DIE "没有块 $CH"
        echo "$CH $QA eval" > "$OUT/data.list"
        NQ=$(grep -c '^#Q' "$QA"); CH="$(basename "$CH")"
    fi
    LAYERS="$(awk -F= '$1=="layers"{v=$2} END{print v}' "$R0/ptrain.cfg")"
    RANK="$(awk -F= '$1=="rank"{v=$2} END{print v}' "$R0/ptrain.cfg")"
    # maxlen 照训练那趟(配置里最后一个生效): 同一批题进出一致, 训练缓冲也不比训练时大(全层训练内存余量只 ~3 GB)
    local MAXLEN; MAXLEN="$(awk -F= '$1=="maxlen"{v=$2} END{print v}' "$R0/ptrain.cfg")"
    printf 'data=%s\nout=%s\ninit=%s\nlayers=%s\nrank=%s\ntopk=64\nmaxlen=%s\ndiag=1\n' "$OUT/data.list" "$OUT" "$PT" "$LAYERS" "$RANK" "${MAXLEN:-1024}" > "$OUT/ptrain.cfg"
    # 训练那趟用了共用的按题教师缓存(kdinc 的 teacher=)就接着用: 同一批留出题的教师全命中, 不用再算一遍
    local TCH; TCH="$(awk -F= '$1=="teacher"{v=$2} END{print v}' "$R0/ptrain.cfg")"; [ -z "$TCH" ] || echo "teacher=$TCH" >> "$OUT/ptrain.cfg"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 1; need_idle
    LOG "kddiag: $NQ 题, 块 $CH, 挂 $PT(层 $LAYERS 秩 $RANK) → $OUT"
    ( sleep 20; train_guard ) &
    local GUARD=$!
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --no-dspark --ptrain "$OUT/ptrain.cfg" > "$OUT/diag.out" 2>&1 </dev/null
    local RC=$?
    kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
    tr '\r' '\n' < "$OUT/diag.out" | grep -E "ptrain" | tail -40
    [ "$RC" = 0 ] && [ -s "$OUT/diag.txt" ] || DIE "诊断失败(rc=$RC, 见 $OUT/diag.out)"
    LOG "KDDIAG_DONE $OUT/diag.txt"
}

# ---------------- ★增量后训练序列★(2026-10-03 用户: "第一天在日期文件夹下面训练, 第二天叠加训练, 而不是重新训" → 当晚 "文件夹换成自增序号, 每一次都是增量, 记住进度序号"
#                  → 10-04 "跟每天没有关系, 只需要执行训练命令, 直接开始训练": 料不再按天给, kdinc 自己从料池取没训过的) ----------------
# 目录 $FTD/incr/:
#   pool/chunks/ docs/ qa/ hold/   料池 = qtf corpus 产物原样同步进来(块 <篇>_cNN.txt, 整篇 docs/<篇>.txt, 已出的问答 qa/<块>.{train,eval}.qa); kdinc 只读不改
#   SEQ                     进度: 最后分配出去的序号(kdinc 每次 +1, 四位零填充; 目录已存在就继续加)
#   hold/questions.txt      通用题, 各次共用(第一次 kdgen 让部署态答一遍 → hold/qa/hold_general.{train,eval}.qa, 以后直接用这一份)
#   teacher.bin             按题教师缓存(训练器 cfg teacher=), 各次共用: 回放的旧题永远命中, 一次只算新题
#   ledger.txt              一次几行: 序号 / 料的量 / 转移读数 / 选中轮 / 留出 KL / current 指向
#   ledger/<序号>.tsv       决策账本逐笔(kd_ledger), ledger/stats_<序号>.txt 累计统计
#   current -> <序号>/out/ckpt_eNN   上线的那份 = 过门的最新一次; 回滚 = 改指向
#   <序号>/source.txt       这次的料从哪来(料目录 / 块名正则 / 时间); chunks/ docs/ 文本; qa/ 问答(kdgen 或从料目录带); out/ 训练输出(data.list ptrain.cfg
#              train.log ckpt_eNN heldout_eNN.tsv …); fwd/ 转移测试; gate/wt2_ckpt_eNN.txt 守门全文; ACCEPTED | REJECTED 一行结论
# 叠加 = 训练器 init=<current>(逐层读 A/B, Adam 动量从零起; init 的 base.fnv 与现挂 ② 对不上就停车 —— 换底座只能从零重蒸馏)。
# 三条规矩写死: ①本次 eval 题永远不进任何一次的 train(留出 + 套件先剔掉在任何 train 文件里出现过的题, 剔了就打出来);
#   ②回放 1:1(前面各次 train 题等距抽 ≈ 本次题数); ③没过门那次的料不丢 —— 以后的回放照抽。
# 一次 = 料池里还没训过的前 K 篇(篇 = 块名去掉 _cNN 的前缀, 一篇的块永远同一次进; 名字序 = stock_<日期>_<代码> 的时间序)。"训过" = 块名前缀在任何
# 序号文件夹的 chunks/ 里出现过 —— 进度就是盘上这些文件夹, 没有第二份账要对。K 默认 4 篇: 10-03 实测 2 篇 121 题(+回放 121)一轮 4 分钟, 4 篇 ≈ 200 题
# 对着用户定的"一轮 10 分钟"。料池空了 kdinc 不分配序号直接退。
INCR="$FTD/incr"
# 下一个序号: SEQ + 1, 目录已占就继续加(手工放过目录也不撞); 写回 SEQ
kd_seq_next(){
    local n; n=$(cat "$INCR/SEQ" 2>/dev/null || echo 0); n=$((10#$n + 1))
    while [ -e "$INCR/$(printf '%04d' "$n")" ]; do n=$((n + 1)); done
    mkdir -p "$INCR"; printf '%04d' "$n" > "$INCR/SEQ"; printf '%04d' "$n"
}
# 上一次 ACCEPTED 的序号(< 给定序号; 没有打空)
kd_prev_accepted(){
    local N="$1" dd p=""
    for dd in "$INCR"/[0-9]*/; do dd="$(basename "$dd")"; [[ "$dd" < "$N" ]] && [ -s "$INCR/$dd/ACCEPTED" ] && p="$dd"; done
    echo "$p"
}

# 块名 → 篇名(去掉 _cNN); 一行一个, 去重排序。保持料块 hold_general 不是料
kd_units(){ xargs -rn1 basename | sed -E 's/(_c[0-9]+)?\.txt$//' | grep -vx hold_general | sort -u; }
# 料池里还没训过的篇(名字序): 料池全部篇 − 任何序号文件夹 chunks/ 里出现过的篇。只打名字, 不打日志(调用方要捕获)
kd_pool_new(){
    local seen; seen="$(ls "$INCR"/[0-9]*/chunks/*.txt 2>/dev/null | kd_units)"
    ls "$INCR"/pool/chunks/*.txt 2>/dev/null | kd_units | grep -vxF -f <(printf '%s\n' "$seen")
}
# kd_ingest <序号> <篇>...: 把料池里这些篇的块(与整篇 docs/、已出的问答)放进序号文件夹, 写 source.txt; 第一次顺带把保持料放进 incr/hold/
kd_ingest(){
    local N="${1:?序号}" POOL="$INCR/pool" D="$INCR/$N" u f b s n=0; shift
    mkdir -p "$D/chunks" "$D/qa" "$INCR/hold"
    { printf '料池 %s\n时间 %s\n' "$POOL" "$(date '+%F %T')"; printf '篇 %s\n' "$@"; } > "$D/source.txt"
    for u in "$@"; do
        for f in "$POOL/chunks/$u.txt" "$POOL/chunks/${u}"_c[0-9]*.txt; do
            [ -s "$f" ] || continue; b="$(basename "$f" .txt)"
            cp "$f" "$D/chunks/"; n=$((n+1))
            for s in train eval; do [ -s "$POOL/qa/$b.$s.qa" ] && cp "$POOL/qa/$b.$s.qa" "$D/qa/"; done
        done
        # 整篇复盘(docs/)也带上: 账本的决策日材料节选从整篇里切(块里每块重复开头的结论段, 切不干净会把结果带进决策提示)
        [ -s "$POOL/docs/$u.txt" ] && { mkdir -p "$D/docs"; cp "$POOL/docs/$u.txt" "$D/docs/"; }
    done
    [ "$n" -gt 0 ] || DIE "篇 $* 在 $POOL/chunks 一个块都没有"
    if [ ! -s "$INCR/hold/questions.txt" ] && [ -s "$POOL/hold/questions.txt" ]; then
        cp "$POOL/hold/questions.txt" "$INCR/hold/"
        if ls "$POOL"/qa/hold_general.train.qa >/dev/null 2>&1; then mkdir -p "$INCR/hold/qa"; cp "$POOL"/qa/hold_general.*.qa "$INCR/hold/qa/"; fi
    fi
    LOG "第 $N 次 取料: $# 篇 $n 块, $(ls "$D"/qa/*.train.qa 2>/dev/null | wc -l) 份训练问答 ← 料池; 篇 = $*; 保持料 $([ -s "$INCR/hold/qa/hold_general.train.qa" ] && echo 已有答案 || echo 待第一次 kdgen)"
}
# kdpool: 只看进度 —— 序号 / 上线的 ③ / 料池里还没训过的篇(下一次 kdinc 会取前 K 篇)。不动模型
stage_kdpool(){
    local -a NEW=(); local u
    while IFS= read -r u; do [ -n "$u" ] && NEW+=("$u"); done < <(kd_pool_new)
    LOG "进度: SEQ=$(cat "$INCR/SEQ" 2>/dev/null || echo 0)(已跑 $(ls -d "$INCR"/[0-9]*/ 2>/dev/null | wc -l) 次, ACCEPTED $(ls "$INCR"/[0-9]*/ACCEPTED 2>/dev/null | wc -l) 次), current → $(readlink -f "$INCR/current" 2>/dev/null || echo 无)"
    LOG "料池 $INCR/pool: 共 $(ls "$INCR"/pool/chunks/*.txt 2>/dev/null | kd_units | wc -l) 篇, 没训过 ${#NEW[@]} 篇: ${NEW[*]:-无}"
}
# kdtake [篇数上限 4]: 只取料不训练 —— 分配下一个序号, 从料池取没训过的前 K 篇, 出账本 + 校准料 + 决策提示(kd_ledger); 之后 kdinc @序号 训它。
#   用处: 先把下一批的决策提示做出来, 当本批训练的 probe_q(kdinc @本批 3 1e-4 probe_q=<下一批>/decide/questions.txt): 训练中每轮末对没见过的决策答一遍,
#   kdscore 逐轮打分 = 一次训练里就看到先验第几轮带过去, 不用等下一次增量。
stage_kdtake(){
    local CAP="${1:-4}" N u; local -a NEW=()
    [[ "$CAP" =~ ^[0-9]+$ ]] || DIE "kdtake 第 1 参是篇数上限(数字)"
    while IFS= read -r u; do [ -n "$u" ] && NEW+=("$u"); done < <(kd_pool_new)
    [ "${#NEW[@]}" -gt 0 ] || { stage_kdpool; DIE "料池里没有没训过的料"; }
    [ "$CAP" -gt 0 ] && [ "${#NEW[@]}" -gt "$CAP" ] && NEW=("${NEW[@]:0:$CAP}")
    N="$(kd_seq_next)"; kd_ingest "$N" "${NEW[@]}"
    kd_ledger "$N"
    LOG "KDTAKE_DONE $N: 料在 $INCR/$N/(chunks qa docs decide), 训它 = kdinc @$N"
}
# kdscore <序号(提示与真值所属)> <probe 文件> [规则 stats 截至的序号 = 序号−1]: 对任意一份训练器 probe 文件(out/probe_eNN.txt)按领域切片打三臂分(只打不写账本)
stage_kdscore(){
    local N="${1:?序号}" P="${2:?probe 文件}" PD="${3:-}" dom nq k0=1 ST
    [ -s "$P" ] || DIE "没有 $P"
    [ -n "$PD" ] || PD="$(printf '%04d' $((10#$N - 1)))"
    kd_domain_load
    for dom in $(kd_domains); do
        [ -s "$INCR/$N/decide/$dom/questions.txt" ] || continue
        nq=$(wc -l < "$INCR/$N/decide/$dom/questions.txt"); ST="$(kd_stats_upto "$dom" "$PD")"
        kd_probe_slice "$P" "$k0" "$nq" > "${P%.txt}_$dom.txt"; k0=$((k0 + nq))
        [ -n "$ST" ] || { LOG "kdscore [$dom]: 第 $PD 次为止这个领域没有 stats, 规则臂算不了"; continue; }
        LOG "kdscore [$dom] $(basename "$(dirname "$P")")/$(basename "$P") (真值 第 $N 次, 规则 stats ≤ $PD):"
        "${dom}_score" "${P%.txt}_$dom.txt" "$INCR/$N/decide/$dom/truth.tsv" "$ST" | tee -a "$LOGF"
    done
}
# kdinc [篇数上限 4] [训练轮数 3] [lr 1e-4] [k=v,...]: ★一次增量后训练★ —— 从料池取没训过的前 K 篇, 分配下一个序号, 在上一次 ACCEPTED 的 ③ 上叠加训练, 过门, 更新 current。
#   kdinc @序号 [轮] [lr] [k=v] = 不取新料, 对已有的序号文件夹(重)跑(失败后续跑 / 手工放好料的文件夹)。进度打在日志: "第 N 次, 起点 第 M 次"。
stage_kdinc(){
    local A="${1:-}" EP="${2:-3}" LR="${3:-1e-4}" EXTRA="${4:-}" N CAP u; local -a NEW=()
    if [ -n "$A" ] && [ "${A#@}" != "$A" ]; then N="${A#@}"; [ -d "$INCR/$N" ] || DIE "没有序号文件夹 $INCR/$N"
    else
        CAP="${A:-4}"; [[ "$CAP" =~ ^[0-9]+$ ]] || DIE "kdinc 第 1 参是篇数上限(数字)或 @序号, 不是 $CAP"
        while IFS= read -r u; do [ -n "$u" ] && NEW+=("$u"); done < <(kd_pool_new)
        [ "${#NEW[@]}" -gt 0 ] || { stage_kdpool; LOG "料池里没有没训过的料, 本次不分配序号; 喂料 = 把 qtf corpus 产物的 chunks/ docs/ [qa/] 同步进 $INCR/pool/"; return 0; }
        [ "$CAP" -gt 0 ] && [ "${#NEW[@]}" -gt "$CAP" ] && NEW=("${NEW[@]:0:$CAP}")
        N="$(kd_seq_next)"; kd_ingest "$N" "${NEW[@]}"
    fi
    LOG "进度: 本次 = 第 $N 次增量(SEQ=$(cat "$INCR/SEQ")), 起点 = $(p=$(kd_prev_accepted "$N"); [ -n "$p" ] && echo "第 $p 次(ACCEPTED)" || echo "①+②(第一次)")"
    kd_inc_run "$N" "$EP" "$LR" "$EXTRA"
}

# ---------------- 决策账本 + 校准料(10-03 夜, 用户 "继续 rsi 设计, 完成代码编写, 跑两天, 再看看有没有变聪明") ----------------
# 为什么: 蒸馏复盘叙事只让 ③ "记住"(10-03 转移测试: 对没见过的下一天, 数字位 52.2 → 53.6%, 只带过去了答题格式)。要"越预测越准", 教师得是
# 结果本身: 每笔决策的次日实际(代码从复盘头部逐笔核定)进账本, 账本算出校准规则(目标/止损该留多少空间 = 历史实现涨跌幅的中位数), 规则 + 逐笔
# 事实 + 规则应用示例写成一块代码产的材料和问答(答案全是代码算的, 不让模型编), 跟复盘块一起蒸进 ③。测"聪明"= 下一天训练前(kdfwd): 用前面的 ③
# 对下一天各股的【决策日材料节选】(不含结果)要目标价/止损位, 与部署态并排, 对账本里的次日实际最高/最低算误差 —— 这才是决策位上的转移。
# ★领域适配器★(10-04, 用户 "我不是说了泛化的吗"): 校准回路里随领域变的三件(解析器 / 规则与料 / 打分)住 $SC/kd_domain/<域>.sh, 每个定义
#   <域>_parse / <域>_material / <域>_score(契约与为什么见 kd_domain/_template.sh; 金融版 finance.sh 是从这里原样拆出去的, 产物逐字节没变)。
#   主回路对每个领域各走一遍, 认不出的篇适配器自己跳过; 哪个领域有笔哪个领域就出材料。加领域 = 加一个文件, 这里不改。
kd_domains(){ local a; for a in "$SC"/kd_domain/[a-z]*.sh; do [ -s "$a" ] && basename "$a" .sh; done; }
kd_domain_load(){ local d; for d in $(kd_domains); do . "$SC/kd_domain/$d.sh"; done; }
# kd_stats_upto <域> <序号>: 该领域截至该序号的最新 stats 文件(那次没笔就没有 stats, 往前找; 一个都没有打空)
kd_stats_upto(){ local f b p=""; for f in "$INCR/ledger/$1"/stats_[0-9]*.txt; do [ -s "$f" ] || continue; b="$(basename "$f" .txt)"; b="${b#stats_}"; [[ "$b" > "$2" ]] || p="$f"; done; echo "$p"; }
# kd_ledger <序号>: 决策账本 + 校准料, 逐领域: docs/ 整篇 → ledger/<域>/<序号>.tsv(逐笔) → 累计(序号 ≤ 本次; 最后一列 = 序号, 本次的笔按它挑) → ledger/<域>/stats_<序号>.txt +
#   chunks/ledger_<域>_<序号>.txt + qa/ledger_<域>_<序号>.{train,eval}.qa + decide/<域>/(<id>.txt 决策提示, questions.txt @文件行, truth.tsv);
#   各领域 questions.txt 按领域名序拼成 decide/questions.txt(kdfwd 的决策探针一次装载全问, 打分再切回去)。各领域都没笔 = 本次没有校准料, 只走知识回路(不停车)。
kd_ledger(){
    local N="${1:?序号}" D="$INCR/${1:?}" LG="$INCR/ledger" dom msg n ALL tot=0
    mkdir -p "$D/decide" "$D/qa" "$D/chunks"; : > "$D/decide/questions.txt"
    [ -d "$D/docs" ] || { LOG "账本 第 $N 次: 没有 $D/docs(整篇), 本次没有校准料"; return 0; }
    kd_domain_load
    for dom in $(kd_domains); do
        mkdir -p "$LG/$dom" "$D/decide/$dom"; : > "$LG/$dom/$N.tsv"
        msg="$("${dom}_parse" "$D/docs" "$N" "$LG/$dom/$N.tsv")" || DIE "账本[$dom] 第 $N 次: 解析停车($msg)"
        n=$(wc -l < "$LG/$dom/$N.tsv")
        [ "$n" -gt 0 ] || { LOG "账本[$dom] 第 $N 次: 本次 0 笔($msg)"; continue; }
        ALL="$D/decide/$dom/all.tsv"; cat $(ls "$LG/$dom"/[0-9]*.tsv | awk -v d="$N" -F/ '{n = $NF; sub(/\.tsv$/, "", n); if (n <= d) print}') > "$ALL"
        msg="$("${dom}_material" "$ALL" "$N" "$LG/$dom/stats_$N.txt" "$D/chunks/ledger_${dom}_$N.txt" "$D/qa/ledger_${dom}_$N.train.qa" "$D/qa/ledger_${dom}_$N.eval.qa" "$D/decide/$dom")" \
            || DIE "账本[$dom] 第 $N 次: 出料停车($msg)"
        LOG "账本[$dom] $msg"
        [ ! -s "$D/decide/$dom/questions.txt" ] || cat "$D/decide/$dom/questions.txt" >> "$D/decide/questions.txt"
        tot=$((tot + n))
    done
    [ "$tot" -gt 0 ] || LOG "账本 第 $N 次: 各领域都没有逐笔记录, 本次没有校准料(只走知识回路)"
}
stage_kdledger(){ kd_ledger "${1:?序号}"; LOG "KDLEDGER_DONE $1: $INCR/$1/chunks/ledger_<域>_$1.txt + qa/ledger_<域>_$1.{train,eval}.qa + decide/<域>/"; }

# kd_replay <当日训练题数 N> <输出目录> <清单> <前面各日期目录...>: 回放抽样。前面各日的 train 题(不含保持料)按文件序排成一串(M 题),
#   第 j 题取当 floor(j·N/M) 比上一题进了位 —— 每天每块按比例出人, 同输入同输出(可复现)。抽中的题按来源块写进 <输出目录>/<日期>_<块>.qa,
#   清单加一行 "<那天的块> <抽样问答> train"(教师缓存按题命中, 回放不重算教师)。
kd_replay(){
    local N="$1" RD="$2" LIST="$3"; shift 3
    local -a SRC=(); local d f
    for d in "$@"; do for f in "$d"/qa/*.train.qa; do [ -s "$f" ] && [ "$(basename "$f")" != hold_general.train.qa ] && SRC+=("$f"); done; done
    [ "${#SRC[@]}" -gt 0 ] || return 0
    rm -rf "$RD"; mkdir -p "$RD"
    local M; M=$(cat "${SRC[@]}" | grep -c '^#Q')
    [ "$M" -gt 0 ] && [ "$N" -gt 0 ] || return 0
    awk -v N="$N" -v M="$M" -v RD="$RD" -v LIST="$LIST" '
        function flush(   out) { if (blk == "") return
            if (int(j * N / M) > int((j - 1) * N / M)) { out = RD "/" tag ".qa"; printf "%s", blk >> out; close(out)
                if (!(out in seen)) { seen[out] = 1; print chunk, out, (b ~ /^ledger_/ ? "train hard" : "train") >> LIST; close(LIST) } }   # 回放的账本料同样是硬目标
            blk = "" }
        FNR == 1 { flush(); n = split(FILENAME, p, "/"); b = p[n]; sub(/\.train\.qa$/, "", b); tag = p[n - 2] "_" b
                   chunk = FILENAME; sub(/\/qa\/[^\/]*\.train\.qa$/, "/chunks/" b ".txt", chunk) }
        /^#Q$/ { flush(); j++; blk = $0 "\n"; next }
        { blk = blk $0 "\n" }
        END { flush() }' "${SRC[@]}"
}

# kd_leakfilter <train 问答文件清单(一行一个路径)> <清单> <输出目录> <eval 源文件...>: 留出题剔重 —— 在【任何一天】的 train 问答里出现过
#   (问题去空白后逐字同)的题不算留出: 哪怕今天没回放到它, ③ 是叠着来的, 前几天练过的题今天也不是"没见过"。过滤后的副本落
#   <输出目录>/<日期>_<块>.eval.qa, 清单加 eval 行; 打一行"留 X 剔 Y"。为什么必须: 重复出题 / 回放都可能把一道留出题带进训练,
#   带进去"学会了没 / 忘了没"两道门都是假的。
kd_leakfilter(){
    local TRL="$1" LIST="$2" OD="$3"; shift 3
    rm -rf "$OD"; mkdir -p "$OD"
    local -a TR=(); local f src b day out kept=0 leak=0 k l
    while read -r f; do [ -s "$f" ] && TR+=("$f"); done < "$TRL"
    awk '/^#Q$/ { q = ""; m = 1; next } /^#A$/ { if (m) print q; m = 0; next } m { l = $0; gsub(/[ \t\r]/, "", l); q = q l }' "${TR[@]}" | sort -u > "$OD/train_keys.txt"
    for src in "$@"; do
        b="$(basename "$src" .eval.qa)"; day="$(basename "$(dirname "$(dirname "$src")")")"; out="$OD/${day}_$b.eval.qa"
        awk -v K="$OD/train_keys.txt" '
            BEGIN { while ((getline l < K) > 0) keys[l] = 1; close(K) }
            function flush() { if (blk == "") return; if (q in keys) leak++; else { printf "%s", blk; kept++ } blk = "" }
            /^#Q$/ { flush(); blk = $0 "\n"; q = ""; m = 1; next }
            /^#A$/ { m = 0 }
            { blk = blk $0 "\n"; if (m) { l = $0; gsub(/[ \t\r]/, "", l); q = q l } }
            END { flush(); printf "%d %d\n", kept + 0, leak + 0 > "/dev/stderr" }' "$src" > "$out" 2> "$out.cnt"
        read -r k l < "$out.cnt"; rm -f "$out.cnt"; kept=$((kept + k)); leak=$((leak + l))
        [ -s "$out" ] && echo "$(dirname "$(dirname "$src")")/chunks/$b.txt $out eval" >> "$LIST"
    done
    LOG "留出剔重: 留 $kept 题, 剔掉 $leak 题(在 train 里出现过)"
}

# kd_paired <heldout_e00.tsv> <heldout_eNN.tsv> <块路径前缀> <in|out>: 逐题配对(in = 块路径以前缀起头的题 = 当日; out = 其余 = 套件)。
#   与训练器 pt_paired_gain 同口径: 降幅 D = Σ(起点_i − 本轮_i)/Σm_i(按答案 token 加权), 题当独立单位, 比值估计线性化的标准误。
#   打 "题数 降幅 标准误 本轮KL 起点KL"(没题全 0)。
kd_paired(){
    awk -F'\t' -v P="$3" -v IN="$4" 'NR == FNR { a[$2] = $4; next }
        ((index($1, P) == 1) == (IN == "in")) && ($2 in a) { q++; d = a[$2] - $4; sd += d; sm += $3; dd[q] = d; mm[q] = $3; cur += $4; prev += a[$2] }
        END { if (!q || !sm) { print 0, 0, 0, 0, 0; exit } g = sd / sm; for (i = 1; i <= q; i++) { u = dd[i] - g * mm[i]; ss += u * u }
              se = q > 1 ? sqrt(ss * q / (q - 1)) / sm : 0; printf "%d %.5f %.5f %.5f %.5f\n", q, g, se, cur / sm, prev / sm }' "$1" "$2"
}

# kd_wt2_se <docgate 全文>: 两臂(② / ②+③)逐位行配对 → ΔΣmin、ΔKLD 的均值 ± 标准误。32 token 一块取块均值再算标准误 —— 相邻 token 高度相关,
#   按 token 算会把误差低估好几倍。只打不判(门仍是 gate_read 的 Σmin ≥ −0.5pp 且 KLD ≤ +3%); 读法: |Δ| 不到 2 个标准误 = 分不清是 ③ 还是噪声。
kd_wt2_se(){
    local a b; read -r a b < <(grep -o "逐位行 → [^ ]*" "$1" | awk '{print $3}' | head -2 | tr '\n' ' ')
    [ -n "${a:-}" ] && [ -s "$a" ] && [ -n "${b:-}" ] && [ -s "$b" ] || { echo "(无逐位行)"; return 0; }
    awk -v B=32 'NR == FNR { k[$1] = $2; s[$1] = $3; next } ($1 in k) { i = int(n / B); dk[i] += $2 - k[$1]; ds[i] += $3 - s[$1]; c[i]++; n++ }
         END { nb = 0; for (i in c) { nb++; mk = dk[i] / c[i]; ms = ds[i] / c[i]; sk += mk; ss += ms; qk += mk * mk; qs += ms * ms }
               if (nb < 2) { print "(块数不足)"; exit }
               mk = sk / nb; ms = ss / nb; vk = qk / nb - mk * mk; vs = qs / nb - ms * ms; if (vk < 0) vk = 0; if (vs < 0) vs = 0
               printf "ΔΣmin %+.2f ± %.2f pp, ΔKLD %+.4f ± %.4f(%d 块 × %d token 配对)", 100 * ms, 100 * sqrt(vs / (nb - 1)), mk, sqrt(vk / (nb - 1)), nb, B }' "$a" "$b"
}

# kdfwd <日期> [③目录]: ★时间向前的转移测试★(10-03 用户 "测一下有没有变聪明") —— 用【该日期之前训出的 ③】(缺省 = 该日期前最近一个 ACCEPTED 的选中轮)
#   对该日期的【全部】问答(train + eval, 这份 ③ 一道都没见过)跑 kddiag: 挂③ 对部署态在分叉位 / 含数字参考位 / 参考 token 概率上的差 = 前几天学的对新一天
#   有没有用。为什么只认这个: 训完当日再量(留出 KL、数字位)量的是"记住了多少"; "越来越聪明"只能用它没见过的下一天来量, 而且要连着几十天看曲线, 一两天说明不了。
#   产物 incr/<序号>/fwd/diag-<标签>-eval/diag.txt, ledger 加行 "转移"/"决策转移"。kdinc 有起点时训练前自动跑。
#   第 3 参 = 标签(缺省 prev; 对比别的 ③ 时给, 如 v1), 产物目录按它分: diag-<标签>-eval / decide-<标签>。
#   ② 决策探针(10-03 夜加): 前面的 ③ 与部署态对今天各股的【决策日材料节选】(decide/<代码>.txt, 不含结果)各给一次目标价/止损位(训练器 epochs=0 +
#   init + probe_q, 一次装载两臂), 对账本里的次日实际最高/最低算 |目标−最高|/最高 等; 第三臂 = 账本校准规则本身(前一天的统计)。
#   ③ 比它只是"记住了规则还是会用": 规则臂赢部署 = 从结果学到的先验有用; 挂③ 贴近规则臂 = ③ 把先验用上了(不提规则也用)。
stage_kdfwd(){
    local N="${1:?序号}" PT="${2:-}" TAG="${3:-prev}" D="$INCR/${1:?}" FWD="$INCR/${1:?}/fwd" dd f b
    if [ -z "$PT" ]; then
        for dd in "$INCR"/[0-9]*/; do dd="$(basename "$dd")"; [[ "$dd" < "$N" ]] && [ -s "$INCR/$dd/ACCEPTED" ] && PT="$(awk '{print $3}' "$INCR/$dd/ACCEPTED")"; done
    fi
    [ -n "$PT" ] && [ -s "$PT/amp_L39.bin" ] || DIE "kdfwd $N: 没有在它之前训出的 ③(给第 2 参, 或先跑前一次)"
    PT="$(readlink -f "$PT")"   # 第 2 参给相对路径时 fwd/<标签> 符号链会按链所在目录解析 → 断链(10-03 实撞: kddiag 报 "没有 …/fwd/e03/amp_L39.bin")
    mkdir -p "$FWD"; : > "$FWD/data.list"
    for f in "$D"/qa/*.qa; do b="$(basename "$f")"; b="${b%%.*}"; [ "$b" = hold_general ] && continue; echo "$D/chunks/$b.txt $f eval" >> "$FWD/data.list"; done
    [ -s "$FWD/data.list" ] || DIE "kdfwd $N: $D/qa 里没有问答"
    printf 'data=%s\nout=%s\nlayers=0-39\nrank=64\nmaxlen=1024\nteacher=%s\n' "$FWD/data.list" "$FWD" "$INCR/teacher.bin" > "$FWD/ptrain.cfg"
    ln -sfn "$PT" "$FWD/$TAG"
    local Q="$D/decide/questions.txt" DEC="$FWD/decide-$TAG"
    if [ -s "$Q" ]; then
        mkdir -p "$DEC"
        { cat "$FWD/data.list"; echo "$D/chunks/hold_general.txt $D/qa/hold_general.train.qa train"; } > "$DEC/data.list"   # 预热探底要一道训练题, 拿保持料
        printf 'data=%s\nout=%s\ninit=%s\nlayers=0-39\nrank=64\ntopk=64\nmaxlen=1024\nepochs=0\nprobe_n=0\nprobe_tok=120\nprobe_q=%s\nteacher=%s\n' \
            "$DEC/data.list" "$DEC" "$PT" "$Q" "$INCR/teacher.bin" > "$DEC/ptrain.cfg"
        kd_train_run "$DEC"
        # 逐领域打分: 探针文件按各领域题数切片(questions.txt 是按领域名序拼的); 规则臂用 ③ 所在那次(PD)为止该领域的 stats —— ③ 学到的只到那次
        local PD dom nq k0=1 ST; PD="$(grep -o '/incr/[0-9]\{4\}/' <<< "$PT/" | head -1 | tr -dc '0-9')"
        # ③ 来自比本次更晚的序号(kdtake 取料后先跑了奖励回路)时, stats ≤ PD 会含本次这几笔的次日实际 = 规则臂偷看答案; 封到本次之前(10-06)
        [[ "$PD" < "$N" ]] || PD="$(printf '%04d' $((10#$N - 1)))"
        kd_domain_load
        for dom in $(kd_domains); do
            [ -s "$D/decide/$dom/questions.txt" ] || continue
            nq=$(wc -l < "$D/decide/$dom/questions.txt"); ST="$(kd_stats_upto "$dom" "$PD")"
            kd_probe_slice "$DEC/probe_e00.txt" "$k0" "$nq" > "$DEC/probe_$dom.txt"; k0=$((k0 + nq))
            [ -n "$ST" ] || { LOG "kdfwd $N [$dom]: 第 $PD 次为止这个领域没有 stats, 规则臂算不了, 不打分"; continue; }
            "${dom}_score" "$DEC/probe_$dom.txt" "$D/decide/$dom/truth.tsv" "$ST" | tee "$DEC/score_$dom.txt" | tee -a "$LOGF"
            echo "$N 决策转移[$dom] ③=$PT $(tail -1 "$DEC/score_$dom.txt")" >> "$INCR/ledger.txt"
        done
    else LOG "kdfwd $N: 没有 decide/questions.txt(kd_ledger 没跑或各领域都没有决策日材料), 跳过决策探针"; fi
    stage_kddiag "$FWD" "$TAG" eval
    local S="$FWD/diag-$TAG-eval/diag.out" a n p nq
    a="$(grep -a '分叉位(教师榜首≠部署榜首)' "$S" | tail -1 | sed 's/.*: \([0-9]*\) 位 .*挂③榜首 = 教师榜首 \([0-9.]*%\).*/\1 位, 挂③跟上教师 \2/')"
    n="$(grep -a '含数字的参考位' "$S" | tail -1 | sed 's/.*: \([0-9]*\) 位 | 榜首 = 参考: 教师 \([0-9.]*%\) \/ 挂③ \([0-9.]*%\) \/ 部署 \([0-9.]*%\).*/\1 位, 教师 \2 挂③ \3 部署 \4/')"
    p="$(grep -a '参考 token 平均概率(榜外按 0)' "$S" | tail -1 | sed 's/.*: 教师 \([0-9.]*\) \/ 挂③ \([0-9.]*\) \/ 部署 \([0-9.]*\).*/教师 \1 挂③ \2 部署 \3/')"
    nq="$(grep -c '^#Q' "$D"/qa/*.qa | awk -F: '$1 !~ /hold_general/ {s += $2} END {print s + 0}')"
    LOG "kdfwd $N ← $PT(本次 $nq 题全没见过): 分叉位 $a | 含数字位 $n | 参考 token 平均概率 $p"
    echo "$N 转移 ③=$PT 本次 $nq 题 分叉位 $a 含数字位 $n 参考概率 $p" >> "$INCR/ledger.txt"
}

# kd_probe_slice <probe 文件> <起始题号> <题数>: 训练器 probe 文件按 "自定义#k 问:" 分题, 取第 k0..k0+n−1 题, 题号重编从 1 起(各领域的决策提示拼在一个
#   questions.txt 里一次装载, 打分按领域切开; 各领域 _score 只看自己那片。打分本身在 kd_domain/<域>.sh 的 <域>_score 里)
kd_probe_slice(){
    LC_ALL=C.UTF-8 gawk -v k0="$2" -v n="$3" '/^自定义#[0-9]+ 问: / { k++; on = (k >= k0 && k < k0 + n); if (on) sub(/^自定义#[0-9]+/, "自定义#" (k - k0 + 1)) } on' "$1"
}

# kd_inc_run <序号> [训练轮数 3] [lr 1e-4] [k=v,...]: 一次增量后训练的主流程(stage_kdinc 分配序号取料后调; 目录与规矩见本段头注释)。
#   当日文本 → (没问答就 kdgen) → 清单 = 当日 train + 回放(kd_replay) + 保持料 1 遍 + 留出(当日 eval + 套件 = 前面各日 eval, 都过 kd_leakfilter)
#   → init=current 训练(epoch_tok=0: 料小, 一轮 = 过一遍; lr 比首训 2e-4 低一半, 少扰动已学的) → 每轮 ckpt 三道读数:
#     wt2(gate_read 口径判, kd_wt2_se 只打) / 套件(③_t 对 ③_{t-1} 逐题配对, 退 ≤ 2 个标准误才过; 第一天套件空 = 过) / 当日学到(只打)
#   → 过门的轮里当日留出 KL 最低者 → ACCEPTED + current 指过去; 没有 → REJECTED, current 不动 → 选中轮(没选中 = 末轮) kddiag eval → ledger。
#   不起服务(训练段都要停服务; 上线是另一件事): bash serve_1m_spark.sh start "$MDL" "$ZCH" --posttrain $FTD/daily/current
kd_inc_run(){
    local N="${1:?序号}" EP="${2:-3}" LR="${3:-1e-4}" EXTRA="${4:-}"
    [[ "$N" =~ ^[0-9]{4}$ ]] || DIE "序号要四位数字"
    local D="$INCR/$N" OUT="$INCR/$N/out" G="$INCR/$N/gate" HQ="$INCR/hold/questions.txt"
    ls "$D"/chunks/*.txt >/dev/null 2>&1 || DIE "没有 $D/chunks/*.txt(本次文本放这里; kdinc 会从料池取)"
    [ -s "$HQ" ] || DIE "没有 $HQ(通用题, 各次共用; kdinc 第一次会从料池 pool/hold/ 带过来)"
    mkdir -p "$OUT" "$G" "$D/qa"
    # 保持料答案只出一次(第一天 kdgen 让部署态答): 每天重答是另一组采样, 教师缓存全不命中, 还白烧几分钟
    if [ -s "$INCR/hold/qa/hold_general.train.qa" ]; then cp "$INCR"/hold/qa/hold_general.*.qa "$D/qa/"; : > "$D/chunks/hold_general.txt"
    else mkdir -p "$D/hold"; cp "$HQ" "$D/hold/questions.txt"; fi
    # 本次问答已有就不重出: 日期文件夹里的料是定格的, 重出会换掉题目, 和前后两天就不可比。
    #   账本问答(ledger_*, kdtake 先出的)不算"已有"(10-06 实撞: kdtake → kdinc @序号 把 kdgen 整个跳过, 复盘块一道题都没有, 清单只剩账本硬目标)
    if ! ls "$D"/qa/*.train.qa 2>/dev/null | grep -v hold_general | grep -qv '/ledger_'; then stage_kdgen "$D" . 1 1; fi
    if [ ! -s "$INCR/hold/qa/hold_general.train.qa" ]; then
        [ -s "$D/qa/hold_general.train.qa" ] || DIE "kdgen 没出保持料问答($D/qa/hold_general.train.qa)"
        mkdir -p "$INCR/hold/qa"; cp "$D"/qa/hold_general.*.qa "$INCR/hold/qa/"
    fi
    # 决策账本 + 校准料(kd_ledger, 逐领域适配器 kd_domain/<域>.sh): 代码产的块 chunks/ledger_<域>_<序号>.txt 与问答 qa/ledger_<域>_<序号>.*.qa 跟复盘块一起进清单; decide/ 给 kdfwd 的决策探针用
    kd_ledger "$N"
    # 前面各日(按日期字符串比, 只认有训练问答的) + 起点
    local -a PREV=(); local dd
    for dd in "$INCR"/[0-9]*/; do dd="$(basename "$dd")"; [[ "$dd" < "$N" ]] && ls "$INCR/$dd"/qa/*.train.qa >/dev/null 2>&1 && PREV+=("$INCR/$dd"); done
    local INIT=""; [ -e "$INCR/current" ] && INIT="$(readlink -f "$INCR/current")"
    [ -z "$INIT" ] || [ -s "$INIT/amp_L39.bin" ] || DIE "current 指向的 $INIT 没有 amp_L39.bin"
    # ★训练前先量"前面学到的对今天有没有用"★(kdfwd): 本次全部问答这份 ③ 一道没见过, 挂③ 对部署态的决策位差就是转移; 训完再量的是记住了多少
    [ -z "$INIT" ] || stage_kdfwd "$N" "$INIT"
    # 清单: 本次 train(保持料 1 遍, 份额 15~25%, 与全量料的 4 遍 ≈ 17% 同一个量级) + 回放 + 留出
    local LIST="$OUT/data.list" b f nq=0 nrep=0
    : > "$LIST"
    for f in "$D"/qa/*.train.qa; do
        # 账本料(ledger_<域>_<序号>)的答案全是代码算的 → 清单第 4 列 hard: 训练器用答案 one-hot 当目标 + 数字位加权, 不过"教师信不信规则"这一道(10-04)
        b="$(basename "$f" .train.qa)"; echo "$D/chunks/$b.txt $f train$([[ "$b" == ledger_* ]] && echo " hard")" >> "$LIST"
        [ "$b" = hold_general ] || nq=$((nq + $(grep -c '^#Q' "$f")))
    done
    [ "$nq" -gt 0 ] || DIE "本次没有训练题($D/qa)"
    if [ "${#PREV[@]}" -gt 0 ]; then kd_replay "$nq" "$OUT/replay" "$LIST" "${PREV[@]}"; nrep=$(cat "$OUT"/replay/*.qa 2>/dev/null | grep -c '^#Q'); fi
    local -a EV=(); local p
    ls "$D"/qa/*.train.qa > "$OUT/train_files.txt"   # 剔重的依据 = 今天 + 前面各日的全部 train 问答(不只是今天抽到的回放)
    for f in "$D"/qa/*.eval.qa; do [ -s "$f" ] && EV+=("$f"); done
    for p in ${PREV[@]+"${PREV[@]}"}; do
        ls "$p"/qa/*.train.qa >> "$OUT/train_files.txt"
        # 套件 = 前面各次的复盘留出题, 不含前面各次的账本留出题: 账本题的答案是"截至那时"的规则数字, 规则按设计随账本变(0001 涨幅中位 +1.9% → 0005 −0.1%),
        # 学了新规则之后旧规则的题 KL 必然上去(10-04 实撞: 0005 e01 套件-账本 17 题 0.403 → 0.512), 那不是忘, 是改口 —— 当门会把"学会新规则"判成退
        for f in "$p"/qa/*.eval.qa; do b="$(basename "$f")"; [ -s "$f" ] && [ "$b" != hold_general.eval.qa ] && [[ "$b" != ledger_* ]] && EV+=("$f"); done
    done
    kd_leakfilter "$OUT/train_files.txt" "$LIST" "$OUT/evalq" "${EV[@]}"
    # 本次留出题数不含保持料的留出题(它们另有"保持料留出 KL"一栏, 门里的"本次 N 题"也不含), 与 kd_paired 的题数对得上
    local nev nsu; nev=$(ls "$OUT"/evalq/"${N}"_*.eval.qa | grep -v hold_general | xargs -r cat | grep -c '^#Q'); nsu=$(ls "$OUT"/evalq/*.eval.qa | grep -v "/${N}_" | xargs -r cat | grep -c '^#Q')
    printf 'data=%s\nout=%s\nlayers=0-39\nrank=64\nlr=%s\nepochs=%s\nbatch=4\ntopk=64\nmaxlen=1024\nprobe_n=6\nprobe_tok=96\nepoch_tok=0\nteacher=%s\n' \
        "$LIST" "$OUT" "$LR" "$EP" "$INCR/teacher.bin" > "$OUT/ptrain.cfg"
    [ -z "$INIT" ] || echo "init=$INIT" >> "$OUT/ptrain.cfg"
    [ -s "$D/probe/questions.txt" ] && echo "probe_q=$D/probe/questions.txt" >> "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    LOG "kdinc 第 $N 次: 本次 $nq 题 + 回放 $nrep 题(前 ${#PREV[@]} 次) + 保持料 $(grep -c '^#Q' "$D/qa/hold_general.train.qa") 题; 留出 本次 $nev 题 + 套件 $nsu 题; 起点 ${INIT:-①+②(第一次)}; 轮 $EP lr $LR → $OUT"
    kd_train_run "$OUT"
    # 每轮三道读数 → 选轮
    local n ck c tq tg tse tkl tkl0 sq sg sse skl skl0 r w se pass best="" bkl="" e0="$OUT/heldout_e00.tsv" en
    [ -s "$e0" ] || DIE "没有 $e0(训练器第 0 步没落逐题留出表)"
    for n in $(seq 1 "$EP"); do
        ck="$(printf 'ckpt_e%02d' "$n")"; c="$OUT/$ck"; en="$OUT/heldout_e$(printf '%02d' "$n").tsv"
        [ -s "$c/amp_L39.bin" ] && [ -s "$en" ] || { LOG "kdinc 第 $N 次 $ck: 没有 ckpt 或逐题留出表, 不参选"; continue; }
        read -r tq tg tse tkl tkl0 <<< "$(kd_paired "$e0" "$en" "$D/chunks/" in)"
        read -r sq sg sse skl skl0 <<< "$(kd_paired "$e0" "$en" "$D/chunks/" out)"
        stage_docgate "$c" 512 "$G/wt2_$ck.txt" >/dev/null
        w=0; r=$(gate_read "$G/wt2_$ck.txt") || w=$?
        se=$(kd_wt2_se "$G/wt2_$ck.txt")
        # ★本域门★(10-07): 金融 j 2048 同一口径(Σmin 退 ≤0.5pp 且 KLD 涨 ≤3%); 只守 wt2 时本域漂了九次没人拦
        stage_docgate "$c" 2048 "$G/finj_$ck.txt" "$FINJ" >/dev/null
        local wj=0 rj; rj=$(gate_read "$G/finj_$ck.txt") || wj=$?
        pass=1; [ "$w" = 0 ] || pass=0; [ "$wj" = 0 ] || pass=0
        [ "$sq" = 0 ] || awk -v g="$sg" -v s="$sse" 'BEGIN{exit !(g >= -2 * s)}' || pass=0   # 套件: 起点 − 本轮 < −2SE = 显著退
        LOG "kdinc 第 $N 次 $ck: 本次留出 KL $tkl0 → $tkl(配对降 $tg ± $tse, $tq 题) | 套件 $([ "$sq" = 0 ] && echo "空" || echo "$skl0 → $skl(配对降 $sg ± $sse, $sq 题)") | wt2 $r | $se | 金融j $rj$([ "$wj" = 0 ] || echo "★本域没过★") | $([ "$pass" = 1 ] && echo 过门 || echo ★没过门★)"
        if [ "$pass" = 1 ] && { [ -z "$bkl" ] || awk -v a="$tkl" -v b="$bkl" 'BEGIN{exit !(a < b)}'; }; then best="$c"; bkl="$tkl"; fi
    done
    if [ -n "$best" ]; then
        ln -sfn "$best" "$INCR/current"; rm -f "$D/REJECTED"
        echo "$(date '+%F %T') $best 本次留出 KL $bkl, 起点 ${INIT:-①+②}" > "$D/ACCEPTED"
        LOG "kdinc 第 $N 次: ★ACCEPTED★ $(basename "$best"), current → $best(上线: bash $SC/serve_1m_spark.sh start \"\$MDL\" \"\$ZCH\" --posttrain $INCR/current)"
    else
        rm -f "$D/ACCEPTED"; echo "$(date '+%F %T') 没有一轮同时过 wt2 门与套件门, current 不动(${INIT:-无}); 读数见 gate/ 与 nightly.log" > "$D/REJECTED"
        LOG "kdinc 第 $N 次: ★REJECTED★ current 不动(${INIT:-无})"
    fi
    echo "$N 本次 $nq 回放 $nrep 留出 $nev 套件 $nsu 轮 $EP lr $LR 选中 ${best:-无} 本次KL ${bkl:--} current $(readlink -f "$INCR/current" 2>/dev/null || echo 无)" >> "$INCR/ledger.txt"
    # 逐位核对(只打不判): 选中轮(没选中 = 末轮)在全部留出题(本次 + 套件)上 —— 教师数字位答对率 = 本次料教师信不信材料, 挂 ③ 跟上率 = 写进去多少
    stage_kddiag "$OUT" "$(basename "${best:-$OUT/$(printf 'ckpt_e%02d' "$EP")}")" eval
    kd_diag_table "$OUT"
    LOG "KDINC_DONE $N ${best:-REJECTED}"
}

# ---------------- 奖励回路 kdrft(10-04, 用户 "奖励写进去") ----------------
# 为什么: 六次增量三组指标都不动的根因在信号 —— 教师 = 自己读材料(上限 = 记住), 硬目标 = 代码算的中位数规则(上限 = 规则, 位移到 1 封顶),
# 损失里从来没有"这个决策后来对不对"。这里把结果放进损失: 模型自己对决策提示抽 G 份答案(训练器 sample_n, 模型卡配方温 1, 种子固定), 代码按次日
# 实际结算每份收益(适配器 <域>_reward), 同题 G 份互为基线(权重 = 收益 − 组均值, 训练器 pt_rft_group 算), 答案 one-hot × 权重当目标(= REINFORCE 的梯度,
# 负的往下压); 同一串 token 再配一行 kl 锚(教师 = 本轮起点的 ③ 自己答, 权重 β; 不锚部署态 —— 那是没学规则的那个): 奖励推、锚拉; 回放 + 保持料护知识;
# 各轮 wt2 门 + 套件门照旧。一轮 = 结算上一轮末抽的样本 → 清单 → 一遍训练 → 轮末再抽(既是下一轮的样本, 也是这轮的读数)。第 0 轮只抽不训(起点 = current)。
# 读数(只打不判, 账本不到百笔时留出收益是噪声): 训练提示平均收益逐轮 / 留出提示平均收益 / 失败份数 / 锚 KL / 贪心决策探针三臂(同 kdscore)。
# 目录 incr/<序号>/rft/: questions.txt(训练提示在前、留出提示在后, 一次装载全答) layout.txt(每片 "领域 片 题数") truth_<域>_{train,held}.tsv r0/ r1/ …
# 序号照常分配(source.txt 写"奖励回路", 没有 chunks/qa ⇒ 后面的 kdinc 不把它当料); 过门的轮里训练提示收益最高者 ACCEPTED → current。

# kd_rft_prompts <序号> <留出序号>: 其余各序号 decide/<域>/ 的提示 = 训练提示, 留出序号的 = 留出提示; questions.txt 训练片在前, layout.txt 记切片
kd_rft_prompts(){
    local N="$1" HELD="$2" R="$INCR/$1/rft" dom M part q n
    mkdir -p "$R"; : > "$R/questions.txt"; : > "$R/layout.txt"
    for part in train held; do for dom in $(kd_domains); do
        : > "$R/truth_${dom}_$part.tsv"; n=0
        for M in $(ls -d "$INCR"/[0-9]*/ | xargs -n1 basename | sort); do
            q="$INCR/$M/decide/$dom/questions.txt"
            [ "$M" != "$N" ] && [ -s "$q" ] || continue
            if [ "$part" = held ]; then [ "$M" = "$HELD" ] || continue; else [ "$M" != "$HELD" ] || continue; fi
            cat "$q" >> "$R/questions.txt"; cat "$INCR/$M/decide/$dom/truth.tsv" >> "$R/truth_${dom}_$part.tsv"; n=$((n + $(wc -l < "$q")))
        done
        echo "$dom $part $n" >> "$R/layout.txt"
    done; done
}
# kd_rft_reward <样本文件> <序号> <输出目录>: 按 layout 切片 → <域>_reward 结算 → <输出目录>/reward_<域>_<片>.tsv; 汇总一行进 reward.txt 并打出:
#   "train <均值> <份数> <失败> held <均值> <份数> <失败>"(均值含失败份的最坏值; 失败 = 没解析到 / 止损≥入场 / 目标≤入场)
kd_rft_reward(){
    local S="$1" N="$2" O="$3" R="$INCR/$2/rft" dom part n k0=1 f
    mkdir -p "$O"; rm -f "$O"/reward_*.tsv "$O"/sample_*_*.tsv
    while read -r dom part n; do
        if [ "$n" -gt 0 ]; then
            f="$O/reward_${dom}_$part.tsv"
            awk -F'\t' -v OFS='\t' -v k0="$k0" -v n="$n" '$1 >= k0 && $1 < k0 + n { $1 = $1 - k0 + 1; print }' "$S" > "$O/sample_${dom}_$part.tsv"
            "${dom}_reward" "$O/sample_${dom}_$part.tsv" "$R/truth_${dom}_$part.tsv" > "$f"
        fi
        k0=$((k0 + n))
    done < "$R/layout.txt"
    # 汇总一行 "train <奖励均值> <份数> <失败> <第 5 列均值(金融 = 持仓一天收益, 只记)> held <同四项>"
    { for part in train held; do printf '%s ' "$part"; cat "$O"/reward_*_"$part".tsv 2>/dev/null | awk -F'\t' '{ s += $3; c++; if ($4 ~ /fail/) fl++; x += $5 } END { printf "%.3f %d %d %.3f ", c ? s / c : 0, c + 0, fl + 0, c ? x / c : 0 }'; done; echo; } > "$O/reward.txt"
    cat "$O/reward.txt"
}
# kd_rft_score <probe 文件> <序号> <标签> <规则 stats 截至的序号>: 贪心决策探针三臂打分(同 kdscore), 训练片 / 留出片各一行汇总进日志。
#   规则臂只许用留出序号之前的 stats(第 4 参 = 留出序号 − 1): kdtake 出的 stats_<留出序号> 累计里含留出那几笔的次日实际, 拿它当规则臂 = 规则偷看答案
#   (10-06 改; 第 0008 次留出 0006 时它就是最新 stats, 读数不变)
kd_rft_score(){
    local P="$1" N="$2" TAG="$3" PD="${4:?规则 stats 截至的序号}" R="$INCR/$2/rft" dom part n k0=1 ST
    while read -r dom part n; do
        if [ "$n" -gt 0 ]; then
            ST="$(kd_stats_upto "$dom" "$PD")"
            kd_probe_slice "$P" "$k0" "$n" > "${P%.txt}_${dom}_$part.txt"
            [ -z "$ST" ] || LOG "奖励回路 $TAG 贪心决策探针[$dom/$part] $("${dom}_score" "${P%.txt}_${dom}_$part.txt" "$R/truth_${dom}_$part.tsv" "$ST" | tail -1)"
        fi
        k0=$((k0 + n))
    done < "$R/layout.txt"
}
# kd_rft_list <轮目录> <样本文件> <奖励目录> <序号> <β> <清单>: 训练片里收口的样本 → rft_<域>.qa(#Q 提示 / #A 答案 / #W 收益) + anchor_<域>.qa(同题同答)
#   + 清单两行(块 = 空文件 anchor.txt: 教师提示 = 学生提示, 锚行的教师由训练器 anchor= 定)。打收进来的份数(回放按它定量)
kd_rft_list(){
    local RD="$1" S="$2" RW="$3" N="$4" BETA="$5" LIST="$6" R="$INCR/$4/rft" dom part n k0=1 c tot=0
    : > "$RD/anchor.txt"
    while read -r dom part n; do
        if [ "$part" = train ] && [ "$n" -gt 0 ] && [ -s "$RW/reward_${dom}_train.tsv" ]; then
            c=$(LC_ALL=C.UTF-8 gawk -F'\t' -v Q="$R/questions.txt" -v k0="$k0" -v n="$n" -v RW="$RW/reward_${dom}_train.tsv" -v QF="$RD/rft_$dom.qa" -v AF="$RD/anchor_$dom.qa" '
                function unesc(s,   o, i, ch) { o = ""; for (i = 1; i <= length(s); i++) { ch = substr(s, i, 1); if (ch == "\\" && i < length(s)) { i++; ch = substr(s, i, 1); o = o (ch == "n" ? "\n" : ch == "t" ? "\t" : ch) } else o = o ch } return o }
                BEGIN { while ((getline l < Q) > 0) { nq++; p = l; sub(/^@/, "", p); txt = ""; while ((getline x < p) > 0) txt = txt (txt == "" ? "" : "\n") x; close(p); prompt[nq] = txt } close(Q)
                        while ((getline l < RW) > 0) { split(l, f, "\t"); r[f[1] "," f[2]] = f[3] } close(RW); printf "" > QF; printf "" > AF }
                $1 >= k0 && $1 < k0 + n && $3 == 1 { key = ($1 - k0 + 1) "," ($2 + 0); if (!(key in r)) next; a = unesc($4); q = prompt[$1]
                    printf "#Q\n%s\n#A\n%s\n#W %s\n", q, a, r[key] >> QF; printf "#Q\n%s\n#A\n%s\n", q, a >> AF; c++ }
                END { close(QF); close(AF); print c + 0 }' "$S")
            if [ "${c:-0}" -gt 0 ]; then
                echo "$RD/anchor.txt $RD/rft_$dom.qa train rft" >> "$LIST"; echo "$RD/anchor.txt $RD/anchor_$dom.qa train kl $BETA" >> "$LIST"; tot=$((tot + c))
            fi
        fi
        k0=$((k0 + n))
    done < "$R/layout.txt"
    echo "$tot"
}
# kd_rft_run <序号> <轮数> <每题份数> <留出序号> <lr> <β> [k=v,...]: 主流程(见本段头注释)
kd_rft_run(){
    local N="$1" T="${2:-3}" G="${3:-8}" HELD="${4:-}" LR="${5:-1e-4}" BETA="${6:-1}" EXTRA="${7:-}"
    local R="$INCR/$N/rft" START O LIST PREV3 S RW t nrft dd f b p ntr nhe
    START="$(readlink -f "$INCR/current" 2>/dev/null || true)"; [ -n "$START" ] && [ -s "$START/amp_L39.bin" ] || DIE "奖励回路要有 current(先跑 kdinc)"
    kd_domain_load
    [ -n "$HELD" ] || HELD="$(for dd in "$INCR"/[0-9]*/; do dd="$(basename "$dd")"; [ "$dd" != "$N" ] && [ -s "$INCR/$dd/decide/questions.txt" ] && echo "$dd"; done | tail -1)"
    kd_rft_prompts "$N" "$HELD"
    local RPD; RPD="$(printf '%04d' $((10#${HELD:-1} - 1)))"   # 规则臂 stats 截至留出序号之前(见 kd_rft_score)
    ntr=$(awk '$2 == "train" {s += $3} END {print s + 0}' "$R/layout.txt"); nhe=$(awk '$2 == "held" {s += $3} END {print s + 0}' "$R/layout.txt")
    [ "$ntr" -gt 0 ] || DIE "奖励回路: 没有训练提示(各序号 decide/ 都空?)"
    local HOLDQ="$INCR/hold/qa/hold_general.train.qa"; [ -s "$HOLDQ" ] || DIE "没有 $HOLDQ"
    : > "$R/hold_general.txt"
    LOG "奖励回路 第 $N 次: 起点 $START, 训练提示 $ntr 道(留出 = 第 ${HELD:-无} 次 $nhe 道), 每道抽 $G 份, $T 轮, lr $LR, 锚 β $BETA"
    # probe_base: 自定义探针的部署态答案各轮不变, 第 0 轮算一次落 $R/probe_base.txt, 后面各轮直接读(10-06: 每轮重装再算 20 道单流 ≈ 3 分钟白算)
    local CFG="layers=0-39\nrank=64\ntopk=64\nmaxlen=1024\nbatch=4\nprobe_n=0\nprobe_tok=120\nepoch_tok=0\nprobe_base=$R/probe_base.txt\n"
    # 第 0 轮: 只抽样(epochs=0: 第 0 步的 init 就是产物); 抽过就不重抽(kdrft @序号 续跑)
    O="$R/r0"; mkdir -p "$O"
    if [ ! -s "$O/sample_e00.txt" ]; then
        echo "$R/hold_general.txt $HOLDQ train" > "$O/data.list"   # 预热探底要一道训练题
        printf "data=%s\nout=%s\ninit=%s\nepochs=0\nprobe_q=%s\nsample_n=%s\nteacher=%s\n$CFG" "$O/data.list" "$O" "$START" "$R/questions.txt" "$G" "$INCR/teacher.bin" > "$O/ptrain.cfg"
        [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$O/ptrain.cfg"   # 第 0 轮也吃 k=v(probe_tok 等采样口径三轮必须一致; 10-04 第 0008 次第一发漏了)
        kd_train_run "$O"
        [ -s "$O/sample_e00.txt" ] || DIE "奖励回路 轮 0: 训练器没落 sample_e00.txt"
    fi
    LOG "奖励回路 轮 0(起点 ③ 抽样): 收益 $(kd_rft_reward "$O/sample_e00.txt" "$N" "$O")"
    kd_rft_score "$O/probe_e00.txt" "$N" "轮 0" "$RPD"
    # 套件(前面各次的复盘留出题, 剔重; 不含账本题, 理由见 kd_inc_run)与回放来源 = 全部有训练问答的序号
    local -a PREV=() EV=()
    for dd in "$INCR"/[0-9]*/; do dd="$(basename "$dd")"; [ "$dd" != "$N" ] && ls "$INCR/$dd"/qa/*.train.qa >/dev/null 2>&1 && PREV+=("$INCR/$dd"); done
    for p in ${PREV[@]+"${PREV[@]}"}; do for f in "$p"/qa/*.eval.qa; do b="$(basename "$f")"; [ -s "$f" ] && [ "$b" != hold_general.eval.qa ] && [[ "$b" != ledger_* ]] && EV+=("$f"); done; done
    for t in $(seq 1 "$T"); do
        if [ "$t" = 1 ]; then PREV3="$START"; RW="$R/r0"; S="$R/r0/sample_e00.txt"; else PREV3="$R/r$((t - 1))/ckpt_e01"; RW="$R/r$((t - 1))"; S="$R/r$((t - 1))/sample_e01.txt"; fi
        [ -s "$PREV3/amp_L39.bin" ] && [ -s "$S" ] || DIE "奖励回路 轮 $t: 缺上一轮的 ③($PREV3)或样本($S)"
        O="$R/r$t"; mkdir -p "$O"; LIST="$O/data.list"; : > "$LIST"
        # 留出(套件)先写且各轮同序 ⇒ 题号跨轮一致, 逐题配对可比; 然后保持料、自采样 + 锚、回放
        : > "$O/train_files.txt"; for p in ${PREV[@]+"${PREV[@]}"}; do ls "$p"/qa/*.train.qa >> "$O/train_files.txt"; done
        [ "${#EV[@]}" -eq 0 ] || kd_leakfilter "$O/train_files.txt" "$LIST" "$O/evalq" "${EV[@]}"
        echo "$R/hold_general.txt $HOLDQ train" >> "$LIST"
        nrft=$(kd_rft_list "$O" "$S" "$RW" "$N" "$BETA" "$LIST")
        [ "${nrft:-0}" -gt 0 ] || DIE "奖励回路 轮 $t: 没有可训的样本(全没收口?)"
        [ "${#PREV[@]}" -eq 0 ] || kd_replay "$nrft" "$O/replay" "$LIST" "${PREV[@]}"
        printf "data=%s\nout=%s\ninit=%s\nanchor=%s\nepochs=1\nlr=%s\nprobe_q=%s\nsample_n=%s\nteacher=%s\n$CFG" "$LIST" "$O" "$PREV3" "$PREV3" "$LR" "$R/questions.txt" "$G" "$INCR/teacher.bin" > "$O/ptrain.cfg"
        # 第 0 步的探针每轮都跳(起点 ③ 的探针与采样上一轮末 / 第 0 轮已出过); 第 0 步评估只第 1 轮做(它的 heldout_e00 = 套件门的基线), 第 2 轮起跳
        # (起点 = 上一轮末同一份 ③, 轮末评估就是它的读数; 10-06: 一轮 40 分钟里这两样白算 5 分多)
        printf 'probe0=0\neval0=%s\n' "$([ "$t" = 1 ] && echo 1 || echo 0)" >> "$O/ptrain.cfg"
        [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$O/ptrain.cfg"
        LOG "奖励回路 轮 $t: 自采样 $nrft 份(+ 同数锚行) + 回放 $(cat "$O"/replay/*.qa 2>/dev/null | grep -c '^#Q') 题 + 保持料; 起点 $PREV3"
        kd_train_run "$O"
        [ -s "$O/ckpt_e01/amp_L39.bin" ] && [ -s "$O/sample_e01.txt" ] || DIE "奖励回路 轮 $t: 训练器没落 ckpt_e01 或 sample_e01.txt"
        LOG "奖励回路 轮 $t: 收益 $(kd_rft_reward "$O/sample_e01.txt" "$N" "$O") | $(tr '\r' '\n' < "$O/train.out" | grep -a '锚 KL' | tail -1 | sed 's/.*ptrain\] //')"
        kd_rft_score "$O/probe_e01.txt" "$N" "轮 $t" "$RPD"
    done
    # 门(各轮 ckpt_e01): wt2 + 套件(对第 1 轮第 0 步 = 起点, 逐题配对) + ★训练提示奖励必须高过轮 0(起点 ③ 自己的样本)★ → 过门里奖励最高者(同分取后面的轮)。
    #   没有第三道: 三轮都比起点差照样 ACCEPTED 一个(10-04 第 0007 次第 1 轮实撞: −4.15 → −4.52 时才看见这个缺口)
    local GD="$INCR/$N/gate" c w r se pass best="" brw="" sq sg sse skl skl0 rw e0="$R/r1/heldout_e00.tsv" rw0
    rw0=$(awk '{print $2}' "$R/r0/reward.txt")
    mkdir -p "$GD"
    for t in $(seq 1 "$T"); do
        c="$R/r$t/ckpt_e01"; [ -s "$c/amp_L39.bin" ] || continue
        rw=$(awk '{print $2}' "$R/r$t/reward.txt")
        stage_docgate "$c" 512 "$GD/wt2_r$t.txt" >/dev/null
        w=0; r=$(gate_read "$GD/wt2_r$t.txt") || w=$?
        se=$(kd_wt2_se "$GD/wt2_r$t.txt")
        stage_docgate "$c" 2048 "$GD/finj_r$t.txt" "$FINJ" >/dev/null   # 本域门(10-07, 同 kd_inc_run)
        local wj=0 rj; rj=$(gate_read "$GD/finj_r$t.txt") || wj=$?
        pass=1; [ "$w" = 0 ] || pass=0; [ "$wj" = 0 ] || pass=0
        sq=0
        if [ -s "$e0" ] && [ -s "$R/r$t/heldout_e01.tsv" ]; then
            read -r sq sg sse skl skl0 <<< "$(kd_paired "$e0" "$R/r$t/heldout_e01.tsv" "$R/" out)"
            [ "$sq" = 0 ] || awk -v g="$sg" -v s="$sse" 'BEGIN{exit !(g >= -2 * s)}' || pass=0
        fi
        awk -v a="$rw" -v b="$rw0" 'BEGIN{exit !(a > b)}' || pass=0
        LOG "奖励回路 轮 $t ckpt: 训练提示奖励 $rw(轮 0 $rw0) | 套件 $([ "$sq" = 0 ] && echo 空 || echo "$skl0 → $skl(配对降 $sg ± $sse, $sq 题)") | wt2 $r | $se | 金融j $rj$([ "$wj" = 0 ] || echo "★本域没过★") | $([ "$pass" = 1 ] && echo 过门 || echo ★没过门★)"
        if [ "$pass" = 1 ] && { [ -z "$brw" ] || awk -v a="$rw" -v b="$brw" 'BEGIN{exit !(a >= b)}'; }; then best="$c"; brw="$rw"; fi
    done
    if [ -n "$best" ]; then
        ln -sfn "$best" "$INCR/current"; rm -f "$INCR/$N/REJECTED"
        echo "$(date '+%F %T') $best 奖励回路 训练提示收益 $brw, 起点 $START" > "$INCR/$N/ACCEPTED"
        LOG "奖励回路 第 $N 次: ★ACCEPTED★ $(basename "$(dirname "$best")")/ckpt_e01(训练提示收益 $brw), current → $best"
    else
        rm -f "$INCR/$N/ACCEPTED"; echo "$(date '+%F %T') 没有一轮同时过 wt2 门与套件门, current 不动($START)" > "$INCR/$N/REJECTED"
        LOG "奖励回路 第 $N 次: ★REJECTED★ current 不动($START)"
    fi
    echo "$N 奖励回路 起点 $START 轮 $T G $G 训练提示 $ntr 留出 $nhe(第 ${HELD:-无} 次) 奖励(训/留 失败 附带) $(for t in $(seq 0 "$T"); do [ -s "$R/r$t/reward.txt" ] && awk -v t="$t" '{printf "r%s %s/%s %s/%s %s/%s; ", t, $2, $7, $4, $9, $5, $10}' "$R/r$t/reward.txt"; done)选中 ${best:-无} current $(readlink -f "$INCR/current" 2>/dev/null || echo 无)" >> "$INCR/ledger.txt"
    LOG "KDRFT_DONE $N ${best:-REJECTED}"
}
# ptgate <训练器输出目录> [标签 gate]: ★生成侧改动的逐字节门★(10-06, 探针/采样改合批): 拿该目录的 ptrain.cfg 原样再跑一趟(只换 out=<目录>_<标签>,
#   probe_base 另起一份让部署态也重算), 然后 diff probe_e00.txt / sample_e00.txt —— 温 0 探针与按份种子的采样都必须逐字节同(构造上同一批核, 差一字节 = bug),
#   再打两趟各自的探针/采样用时。只对 epochs=0 的目录有意义(第 0 步就是产物); 不动 current、不写账本。
#   第 3 参 k=v,...(可选)追加进配置: 例 probe_batch=1,sample_batch=1 = 全 1 路(必须逐字节同), sample_batch=8 = 采样 8 路(分布级同, 看用时)。
stage_ptgate(){
    local SRC="${1:?训练器输出目录(含 ptrain.cfg, epochs=0)}" TAG="${2:-gate}" EXTRA="${3:-}" OUT="${1%/}_${2:-gate}" f
    [ -s "$SRC/ptrain.cfg" ] && [ -s "$SRC/probe_e00.txt" ] || DIE "ptgate: $SRC 缺 ptrain.cfg 或 probe_e00.txt"
    grep -q '^epochs=0' "$SRC/ptrain.cfg" || DIE "ptgate: 只对 epochs=0 的目录(第 0 步就是产物)"
    mkdir -p "$OUT"
    grep -v '^out=\|^probe_base=' "$SRC/ptrain.cfg" > "$OUT/ptrain.cfg"
    printf 'out=%s\nprobe_base=%s/probe_base.txt\n' "$OUT" "$OUT" >> "$OUT/ptrain.cfg"
    [ -z "$EXTRA" ] || tr ',' '\n' <<< "$EXTRA" >> "$OUT/ptrain.cfg"
    LOG "ptgate: 重跑 $SRC 的配置 → $OUT${EXTRA:+ (+ $EXTRA)}"
    kd_train_run "$OUT"
    local bad=0
    for f in probe_e00.txt sample_e00.txt; do
        [ -s "$SRC/$f" ] || continue
        if cmp -s "$SRC/$f" "$OUT/$f"; then LOG "ptgate: $f 逐字节同($(wc -c < "$OUT/$f") B)"
        else bad=1; LOG "ptgate: ★$f 不同★ $(diff "$SRC/$f" "$OUT/$f" | grep -c '^[<>]') 行有差; 前 6 行差异:"; diff "$SRC/$f" "$OUT/$f" | head -6 | cut -c1-200 | tee -a "$LOGF"; fi
    done
    LOG "ptgate 用时 旧: $(tr '\r' '\n' < "$SRC/train.out" | grep -a -E 'ptrain (探针|采样) e00\]' | sed -E 's/.*\] //' | tr '\n' ' ')"
    LOG "ptgate 用时 新: $(tr '\r' '\n' < "$OUT/train.out" | grep -a -E 'ptrain (探针|采样) e00\]' | sed -E 's/.*\] //' | tr '\n' ' ')"
    [ "$bad" = 0 ] && LOG "PTGATE_DONE 过门" || DIE "PTGATE 没过门(产物在 $OUT)"
}

# kdrft [轮数 3] [每题份数 8] [留出序号] [lr 1e-4] [锚 β 1] [k=v,...]; kdrft @序号 [同参] = 对已分配的序号续跑/重跑
stage_kdrft(){
    local A="${1:-}" N
    if [ -n "$A" ] && [ "${A#@}" != "$A" ]; then N="${A#@}"; [ -d "$INCR/$N" ] || DIE "没有序号文件夹 $INCR/$N"; shift
    else
        N="$(kd_seq_next)"; mkdir -p "$INCR/$N/rft"
        printf '奖励回路\n时间 %s\n起点 %s\n' "$(date '+%F %T')" "$(readlink -f "$INCR/current" 2>/dev/null || echo 无)" > "$INCR/$N/source.txt"
    fi
    LOG "进度: 本次 = 第 $N 次(奖励回路, SEQ=$(cat "$INCR/SEQ")), 起点 = current"
    kd_rft_run "$N" "${1:-3}" "${2:-8}" "${3:-}" "${4:-1e-4}" "${5:-1}" "${6:-}"
}

case "${1:-all}" in
  dkgen)   stage_dkgen "${2:-}" "${3:-1}" "${4:-1024}" "${5:-1.0}";;
  dktrain) stage_dktrain "${2:-}" "${3:-}" "${4:-}" "${5:-.}";;
  dkgate)  stage_dkgate "${2:-}" "${3:-}" "${4:-1.0}" "${5:-512}";;
  dkrun)   stage_dkrun "${2:-}" "${3:-}" "${4:-4}" "${5:-3}" "${6:-1536}";;
  dkspeed) stage_dkspeed "${2:-}" "${3:-}" "${4:-3}" "${5:-1024}" "${6:-}";;
  dkloop)  stage_dkloop "${2:-}" "${3:-}" "${4:-3}" "${5:-4096}";;
  dkab)    stage_dkab "${2:-}" "${3:-}" "${4:-512}";;
  kdinc)   stage_kdinc "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  kdrft)   stage_kdrft "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" "${8:-}";;
  ptgate)  stage_ptgate "${2:-}" "${3:-}" "${4:-}";;
  kdpool)  stage_kdpool;;
  kdtake)  stage_kdtake "${2:-}";;
  kdscore) stage_kdscore "${2:-}" "${3:-}" "${4:-}";;
  kdfwd)   stage_kdfwd "${2:-}" "${3:-}" "${4:-}";;
  kdledger) stage_kdledger "${2:-}";;
  kdgen)   stage_kdgen "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  kdtrain) stage_kdtrain "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" "${8:-}";;
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
  kdtable) kd_diag_table "${2:-}";;   # 只重打逐轮对照表(kdrun 在 kdpick 停车后补跑 kddiag 时用)
  kdrun)   stage_kdrun "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" "${8:-}";;
  kdeval)  shift; stage_kdeval "$@";;
  kdsplit) stage_kdsplit "${2:-}";;
  kddiag)  stage_kddiag "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
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
