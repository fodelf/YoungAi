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
    local REQ="${1:?请求 JSON}" PT="${2:?③目录或 -}" OUTF="${3:?输出路径}"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3; need_idle
    local extra=(); [ "$PT" = - ] || extra=(--posttrain "$PT")
    bash "$SC/serve_1m_spark.sh" start "" "" ${extra[@]+"${extra[@]}"} >>"$LOGF" 2>&1 || DIE "服务没起来(看 $LOGF)"
    LOG "重跑 $(basename "$REQ") 挂 ③=$PT"
    python3 - "$REQ" "$OUTF" <<'PYEOF' 2>&1 | tee -a "$LOGF"
import json, re, sys, urllib.request
req, outf = sys.argv[1], sys.argv[2]
b = json.load(open(req, encoding="utf-8")); b.pop("_note", None); b["stream"] = True
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
        if n % 500 == 0 and not text and len(think) > 8000 and think.count(think[-2000:]) >= 3:
            loop = True; break
def first(t):
    for m in re.finditer(r"大盘(上涨|下跌)", t):
        if re.match(r'["”」]?\s*(or|或|/|还是)', t[m.end():m.end() + 8]): continue
        if re.search(r'(or|或|/|还是)\s*["“「]?$', t[max(0, m.start() - 8):m.start()]): continue
        return m.group(0), m.start(), t[max(0, m.start() - 80):m.end()].replace("\n", "⏎")
    return None
f = first(think)
print("思考 %d 字 / 正文 %d 字%s" % (len(think), len(text), " / ★复读(尾部 2000 字周期 ≥3 遍), 已停流★" if loop else ""))
print("思考段第一次拍板: %s" % ("%s @%d …%s" % f if f else "无"))
c = re.findall(r"大盘(?:上涨|下跌)", text)
print("正文结论: %s" % (c[0] if c else "(正文没写到结论)"))
print("REVIEW_VERDICT content=%s loop=%d" % (c[0][2:] if c else "none", int(loop)))   # 迭代段按这一行判停
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

case "${1:-all}" in
  sample)  stage_sample "${2:-}" "${3:-4}" "${4:--}" "${5:-}" "${6:-1.0}" "${7:-1.0}" "${8:-}";;
  reward)  stage_reward "${2:-}";;
  solve3)  stage_solve3 "${2:-}" "${3:-}" "${4:-}";;
  demo)    stage_demo "${2:-}" "${3:-4}";;
  reviewiter) shift; stage_reviewiter "$@";;
  review)  stage_review "${2:-}" "${3:-}" "${4:-}" "${5:-}";;
  reviewrun) stage_reviewrun "${2:-}" "${3:-}" "${4:-}";;
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
