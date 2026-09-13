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
VQF="$ROOT/gguf/go-onebit/vqfin"
D2="$ROOT/gguf/go-onebit/vqnight"
MDL="$ROOT/gguf/ds4-fin86q8ve.gguf"       # ① 量化模型(夜间不改)
ZCH="$VQF/champ86amp/zchain.bin"          # ② 反修放大器(夜间不改)
FTD="$D2/finetune"                        # ③ 微调文件版本库
BACKEND="http://192.168.2.203:8001"
SAMPLES_N=8                               # 一晚最多取几条样本(每天产 ~2 条 ≈ 四个交易日滚动窗)
RANK=16                                   # 微调秩: 学的是行为增量, 不是还原, 十几维够; 上限 1024
LAMBDA=0.05                               # 岭正则(样本少, 宁欠拟合不过拟合)
MIN_GAIN=0.02                             # 层门槛: held-out 残差能量下降比, 不到就不注入该层
LOGF="$D2/nightly.log"
LOG(){ echo "[znight $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGF"; }
DIE(){ LOG "★$*★"; exit 1; }
mkdir -p "$D2" "$FTD"
need_idle(){ local b; b=$(for p in ds4 ds4-bench ds4quant_run zlayer vq_merge_v4 finetune_solve; do pgrep -x "$p"; done)
             [ -z "$b" ] || DIE "机器非空(实例锁): $b"; }

# ---------------- ① 样本 → ids ----------------
# ids 两条: 教师 [上下文 ‖ 正文], 学生 [正文]。上下文全拼在前面 = 只需要一个偏移区间
# (row_layout 的 xshift 只支持一段), 语义上也顺: 教师先读完当日复盘, 再看这几份材料。
# ★训练/判决必须隔离★(2026-09-10 用户点破): 此前 samples 与 nll 两段各自拉同一个 API,
# 于是解算用的样本和判决用的样本是同一批 —— 判据里"真泛化"和"记住了这几篇报告"分不开。
# 现在一次拉全量, 按日期切: 早的 N-EVAL_N 条训练, ★最近 EVAL_N 条留作判决★(时序切法
# 而不是随机切: 实际部署就是"今晚学过去的, 明天面对没见过的", 随机切会高估泛化)。
# 判决集样本单独落 eval_samples.json, nll 段只读它, 不再自己拉 API。
EVAL_N=4
stage_samples(){
    local N="${1:-$SAMPLES_N}"
    LOG "① 拉样本(最多 $N 条) ← $BACKEND"
    curl -sf -m 60 "$BACKEND/api/review/finetune-samples?limit=$N&full=true" -o "$D2/samples_all.json" \
        || DIE "拉样本失败(Mac 后端没起? ./start.sh --backend)"
    python3 - "$D2" "$EVAL_N" <<'PY' || DIE "训练/判决切分失败"
import json, os, sys
d, ev = sys.argv[1], int(sys.argv[2])
items = json.load(open(os.path.join(d, "samples_all.json")))["items"]
# 按日期升序: 早的在前。切在末尾 ⇒ 判决集是时间上最靠后的那几条。
items.sort(key=lambda x: (x.get("date") or "", x.get("symbol") or ""))
if len(items) <= ev:
    print("样本只有 %d 条, 不够切出 %d 条判决集" % (len(items), ev)); sys.exit(1)
tr, te = items[:-ev], items[-ev:]
json.dump({"items": tr}, open(os.path.join(d, "samples.json"), "w"), ensure_ascii=False)
json.dump({"items": te}, open(os.path.join(d, "eval_samples.json"), "w"), ensure_ascii=False)
print("切分: 训练 %d 条(%s..%s) | ★判决 %d 条(%s..%s), 解算一行都不看★"
      % (len(tr), tr[0].get("date"), tr[-1].get("date"),
         len(te), te[0].get("date"), te[-1].get("date")))
PY
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

# ---------------- ② 两遍捕获(部署同路) ----------------
# 引擎 --cap-dir 逐层落 raw_ffn_in(x) 与 raw_ffn_out(routed, ★zchain 之后★),
# 正是"量化+zchain"那个部署态的真值 —— 不需要 FP 教师, 不需要建锚。
stage_capture(){
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
        # ★--ctx 必须给足★(2026-09-08 实撞): 会话默认 32768, 喂到第 32779 个 token 直接
        # "cuda decode failed"(报错还不说是上下文满了), 捕获被静默截断成 32779 行 ——
        # 解算器靠行数越界才发现。这里按 ids 行数 ×1.1 给, 至少 8192。
        local ctx=$(( $(wc -l < "$ids") * 11 / 10 )); [ "$ctx" -lt 8192 ] && ctx=8192
        ./ds4 --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 --ctx "$ctx" \
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


# ---------------- SFT 梯度靶: 取料 + 解算(2026-09-10) ----------------
# 与 capture/solve 那条"教师−学生差"的老路并列, 不替换它 —— 老路的读数还要能复现。
# 取料只跑【学生一遍】: SFT 的靶来自损失梯度, 不需要教师。
#   $1 = 上一轮的微调(逗号分隔, 空=从裸底座 θ₀ 起)  $2 = η(默认 100)  $3 = 输出名
stage_sft(){
    local FT="${1:-}" ETA="${2:-100}" TAG="${3:-sft_$(date +%H%M)}"
    [ -s "$D2/ids_s.txt" ] || DIE "先跑 samples(训练集 ids 不在)"
    local rows; rows=$(wc -l < "$D2/ids_s.txt")
    LOG "SFT① 停服 → 训练集取料($rows 行${FT:+, 基于 $FT})"
    bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 5
    need_idle
    local cap="$D2/sft_cap" top="$D2/sft.top"
    rm -rf "$cap"; mkdir -p "$cap"
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" ${FT:+--finetune "$FT"} --mem-budget-mb 110000 \
        --eval-no-bos --eval-ids "$D2/ids_s.txt" --eval-topk 64 "$top" \
        > "$D2/sft_top.log" 2>&1 </dev/null || { tail -5 "$D2/sft_top.log"; DIE "topk 失败"; }
    local ctx=$(( rows * 11 / 10 )); [ "$ctx" -lt 8192 ] && ctx=8192
    ./ds4 --cuda -m "$MDL" --zchain "$ZCH" ${FT:+--finetune "$FT"} --mem-budget-mb 110000 --ctx "$ctx" \
        --score-ids "$D2/ids_s.txt" --score-out /dev/null --cap-dir "$cap" --cap-layers 42-42 \
        > "$D2/sft_cap.log" 2>&1 </dev/null || { tail -5 "$D2/sft_cap.log"; DIE "capture 失败"; }
    # ★查产出件数, 别只看退出码★: 实例锁拒启动时进程也可能"正常"返回, 目录却是空的(09-10 实撞)
    [ "$(ls "$cap" | wc -l)" -ge 1 ] || DIE "capture 目录空 — 机器上是不是还有别的 ds4 在跑?"
    LOG "SFT② 解算(η $ETA rank $RANK λ 1)"
    set -o pipefail
    "$AMP/finetune_solve" --student "$cap" --out "$FTD/$TAG.bin" --mode sft \
        --topk "$top" --gguf "$MDL" --rank "$RANK" --lambda 1 --eta "$ETA" --min-gain -99 2>&1 | tee -a "$LOGF"
    [ "${PIPESTATUS[0]}" = 0 ] || DIE "SFT 解算失败"
    LOG "SFT③ 产物 $FTD/$TAG.bin ($(du -h "$FTD/$TAG.bin" | cut -f1))"
}

# ---------------- ④ 上线 ----------------
stage_deploy(){
    local ft=""
    [ -e "$FTD/current.bin" ] && ft="$(readlink -f "$FTD/current.bin")"
    if [ -n "$ft" ]; then LOG "④ 起服务: 量化模型 + zchain + 微调 $(basename "$ft")"
    else LOG "④ 起服务: 量化模型 + zchain(无微调文件)"; fi
    bash "$SC/serve_1m_spark.sh" start "$MDL" "$ZCH" plain ${ft:+--finetune "$ft"} 2>&1 | tee -a "$LOGF" | tail -12
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
stage_nll(){
    local N="${1:-3}" FT="${2:-}"   # FT: 判决时挂的微调(逗号分隔多个); 空=基线
    local BEN="$ROOT/gguf-tools/bench"
    [ -x "$BEN/anchor_metrics" ] || make -C "$ROOT/gguf-tools" anchor_metrics >>"$LOGF" 2>&1
    # ★只用判决集★: samples 段切出来的 eval_samples.json(解算一行都没看过)。
    # 它不在就硬停 —— 回退去拉 API 会悄悄把训练样本混进判决, 那正是这次要堵的漏。
    [ -s "$D2/eval_samples.json" ] || DIE "缺 $D2/eval_samples.json — 先跑 samples 做训练/判决切分"
    cp "$D2/eval_samples.json" "$D2/nll_samples.json"
    LOG "尺① 判决集(解算未见): $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(len(d[\"items\"]),'条:', ','.join(x.get('symbol','?') for x in d['items']))" "$D2/eval_samples.json")"
    # 一次分词拿到所有段的精确边界: 用特殊 token 当分隔符 —— tokenizer 遇到特殊 token 会
    # 断开 span(core_bpe.c: tokenize_rendered_chat_vocab), 所以每段各自成词, 边界零歧义。
    # 分隔符放在文件最前面 ⇒ ids[0] 就是它自己的 id, 不用猜也不用写死。
    python3 - "$D2" <<'PY' || DIE "拼分词输入失败"
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
PY
    ./ds4 --cuda -m "$MDL" --dump-tokens --prompt-file "$D2/nll_all.txt" 2>/dev/null > "$D2/nll_all.dump" \
        || DIE "分词失败"
    python3 - "$D2" <<'PY' || DIE "ids 组装失败"
import os, re, sys
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
# 两条序列, 同样的材料前缀, 只有续写不同:
#   [材料1][对版1][EOS][材料2][对版2][EOS]...
#   [材料1][错版1][EOS][材料2][错版2][EOS]...
# EOS 留在序列里当文档边界(挡住上一条报告去条件化下一条材料), 它本身也是个该被预测的目标。
# ★判决行只取两版真正不同的那些 token★(09-08 夜第二轮收紧):
# 两版报告 97.3% 逐字相同, 相同 token 的 NLL 差恒为 0 却全进分母 —— 首轮 +0.0178 就是这么被
# 稀释了 35 倍。真正在做决策的是那 2.7%: 目标价数字、止损算法、以及推翻原判的那句理由。
# 用 SequenceMatcher 在【token id 序列】上求最长公共子序列, 取每个版本各自的非公共块。
# 这只是在挑"判哪几行", 不碰任何数值 —— NLL 还是 anchor_metrics 那一份 C 实现算。
# 每块再往前带 1 个 token: 位置 i 预测 ids[i+1], 所以要判"第一个分歧 token"就得从它前一行起判。
import difflib
env = ["N=%d" % N]
diffrows = {"right": [], "wrong": []}
for i in range(N):
    sm = difflib.SequenceMatcher(None, right[i], wrong[i], autojunk=False)
    blk = {"right": [], "wrong": []}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1: blk["right"].append((i1, i2))
        if j2 > j1: blk["wrong"].append((j1, j2))
    diffrows["right"].append(blk["right"])
    diffrows["wrong"].append(blk["wrong"])
for name, rep in (("right", right), ("wrong", wrong)):
    seq, segs_all, segs_diff = [], [], []
    for i in range(N):
        a = len(seq) + len(pro[i])                 # 报告第一个 token 的行号
        seq += pro[i] + rep[i] + [sep]
        segs_all.append((a - 1, a + len(rep[i])))  # ★位置 i 预测 ids[i+1]★ ⇒ 判决行整体左移一格
        for (b, e) in diffrows[name][i]:
            segs_diff.append((a + b - 1, a + e))   # 同样左移一格, 末尾多带一个"改动后第一个 token"
    ndiff = sum(hi - lo for lo, hi in segs_diff)
    open(os.path.join(d, "nll_ids_%s.txt" % name), "w").write("\n".join(map(str, seq)) + "\n")
    env.append("S_%s=%d" % (name.upper(), len(seq)))
    env.append("ROWS_%s=%s" % (name.upper(), ",".join("%d:%d" % t for t in segs_all)))
    env.append("NTGT_%s=%d" % (name.upper(), sum(hi - lo for lo, hi in segs_all)))
    env.append("DROWS_%s=%s" % (name.upper(), ",".join("%d:%d" % t for t in segs_diff)))
    env.append("NDIFF_%s=%d" % (name.upper(), ndiff))
    print("%-5s 序列 %6d 行 | 报告段 %5d token | ★分歧段 %4d token (%d 块, 占 %.1f%%)★"
          % (name, len(seq), sum(hi - lo for lo, hi in segs_all), ndiff, len(segs_diff),
             100.0 * ndiff / max(1, sum(hi - lo for lo, hi in segs_all))))
open(os.path.join(d, "nll_rows.env"), "w").write("\n".join(env) + "\n")
PY
    . "$D2/nll_rows.env"
    LOG "尺② 停服(清理=关进程, 不删文件)"; bash "$SC/serve_1m_spark.sh" stop >>"$LOGF" 2>&1; sleep 3
    need_idle
    : > "$D2/nll_verdict.txt"
    for role in right wrong; do
        local ids="$D2/nll_ids_$role.txt" out="$D2/nll_$role.nll"
        local rows rws ntg
        rows=$(wc -l < "$ids")
        # 不用 ${role^^}+eval 取变量: 那套写法在老 bash 上会静默取空(2026-08 看门狗就栽在这)。
        local drw
        if [ "$role" = right ]; then rws="$ROWS_RIGHT"; ntg="$NTGT_RIGHT"; drw="$DROWS_RIGHT"
        else                        rws="$ROWS_WRONG"; ntg="$NTGT_WRONG"; drw="$DROWS_WRONG"; fi
        [ -n "$rws" ] || DIE "${role} 的判决行为空(nll_rows.env 没生成对?)"
        [ -n "$drw" ] || DIE "${role} 的分歧行为空(两版一模一样?)"
        # ★走 --eval-nll 而不是 --eval-logits★(2026-09-10 换出口, 口径审计差 3.5e-05 已过):
        # 判决只要目标 token 的那 4 字节, 全词表 logits 是 517 KB/位置。旧口径下 3 条样本
        # 就吃 8.1 GB, 全 16 条要 46 GB —— 盘放不下, 而且统一内存机器上写它等于掏 GPU 内存,
        # 09-08 夜就是这么把 spark 写崩的(NVRM Out of memory, wrong 遍读数全丢)。
        # 换出口后 16 条全量也只有几百 KB, 两个条件可以同时留在盘上供复查。
        LOG "尺③ ${role} 遍打分($rows 行, prefill 批路)"
        ./ds4 --cuda -m "$MDL" --zchain "$ZCH" ${FT:+--finetune "$FT"} \
            --mem-budget-mb 110000 --eval-no-bos \
            --eval-ids "$ids" --eval-nll "$out" \
            > "$D2/nll_$role.log" 2>&1 </dev/null || { tail -5 "$D2/nll_$role.log"; DIE "${role} 遍打分失败"; }
        [ -s "$out" ] || DIE "${role} 遍没出 nll"
        LOG "尺④ ${role} 判决(只算报告段 $ntg 个目标 token, $(du -h "$out" | cut -f1) nll)"
        # anchor_metrics 单文件模式跑完会打个"冒烟判决"并按 PPL 区间返回 0/1 —— 这里只取读数,
        # 不拿它的退出码当成败(报告段 PPL 落在区间外不代表打分失败)。
        # 两个行集合各判一次: 全报告段(看整体) + ★分歧段(看真正做决策的那些 token)★
        "$BEN/anchor_metrics" --ref-nll "$out" --ids "$ids" --rows "$rws" \
            2>&1 | sed "s/^/[$role] /" | tee -a "$D2/nll_verdict.txt" | tee -a "$LOGF" || true
        "$BEN/anchor_metrics" --ref-nll "$out" --ids "$ids" --rows "$drw" \
            2>&1 | sed "s/^/[${role}-diff] /" | tee -a "$D2/nll_verdict.txt" | tee -a "$LOGF" || true
        grep -q "^\[$role\] 参考 PPL\[行段" "$D2/nll_verdict.txt" || DIE "${role} 没出行段读数"
    done
    LOG "尺⑤ 收口"
    python3 - "$D2" <<'PY' 2>&1 | tee -a "$LOGF"
import os, re, sys
d = sys.argv[1]
v = {}
for ln in open(os.path.join(d, "nll_verdict.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"\[(right|wrong|right-diff|wrong-diff)\] 参考 PPL\[行段 n=(\d+)\] = ([\d.]+)\s+\(平均 NLL ([\d.]+)\)", ln)
    if m: v[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
if not {"right", "wrong"} <= set(v):
    print("读数不全, 拿不到判决"); sys.exit(1)
for tag, title in (("", "全报告段(含两版逐字相同的部分)"), ("-diff", "★分歧段(两版真正不同的 token)★")):
    if ("right" + tag) not in v or ("wrong" + tag) not in v:
        continue
    nr, pr, lr = v["right" + tag]; nw, pw, lw = v["wrong" + tag]
    d_ = lr - lw
    print("  %s" % title)
    print("    对版 n=%-6d PPL=%-8.4f 平均NLL=%.4f" % (nr, pr, lr))
    print("    错版 n=%-6d PPL=%-8.4f 平均NLL=%.4f" % (nw, pw, lw))
    print("    判据 NLL(对版)−NLL(错版) = %+.4f  (总 log 差 %+.1f nats)" % (d_, lr * nr - lw * nw))
print("  ★%s★" % ("模型现在更愿意写【错版】, 这就是后训练要抹平的差"
                  if v.get("right-diff", v["right"])[2] > v.get("wrong-diff", v["wrong"])[2]
                  else "模型已经更愿意写【对版】, 权重里不缺这个知识"))
PY
    LOG "尺⑥ 恢复服务"; stage_deploy
}

# ★换工具前的机制审计★: prefill 批路(--eval-ids)与逐 token 解码路(--score-ids)在同一批 ids 上
# 必须给出同一个 NLL。判决尺从解码路换到批路是为了速度, 但速度不能换口径 —— 两条路要是不等,
# 后面所有读数都不能和历史比。用上一轮 nll 段留下的解码路 logits 做基准, 不重跑它。
stage_evalaudit(){
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

case "${1:-all}" in
  samples) stage_samples "${2:-}";;
  capture) stage_capture "${2:-}";;
  solve)   stage_solve "${2:-}";;
  deploy)  stage_deploy;;
  probe)   shift; stage_probe "${1:-}";;
  nll)     stage_nll "${2:-}" "${3:-}";;
  sft)     stage_sft "${2:-}" "${3:-}" "${4:-}";;
  nlltext) stage_nlltext;;
  nllaudit) stage_nllaudit "${2:-}";;
  evalaudit) stage_evalaudit;;
  restore) stage_deploy;;
  all)     stage_samples "${2:-}" && stage_capture && stage_solve && stage_deploy && LOG "ZNIGHT_ALL_DONE";;
  *) echo "用法: $0 [samples|capture|solve|deploy|probe <标签>|nll [N] [微调]|sft [上轮微调] [η] [名]|nllaudit [N]|evalaudit|restore|all]"; exit 2;;
esac
