#!/bin/bash
# entity_probe_spark.sh — 实体特征知识召回: 探针尺 + "低秩能不能还原知识"的分钟级判决(2026-11-01, fable5 11-01 方案 ①②)。
#
# 病: 鹈鹕画出来像鸭 = "鹈鹕长什么样"的知识召回位被 2-bit 专家压扁(官方写 长喙/大喉囊 的位置学生 p 只有 FP 的 0.5~0.8, 个别 70 倍);
#   这是一类病(任何具体实体都退成同类模板), 所以尺和料都对着"实体 → 辨识特征"这一类, 鸟整类留出只量不练(禁词表本来就挡鸟)。
# 教师 = 官方 API 关思考作答(带逐位 top-20, api_teacher.py collect --no-think), 学生 = ./ds4 --score-ids(部署同路)在同一串 ids 上打分;
#   尺 = bugmd_ids_tools.py ptfprobe: 教师 ≥ 0.7 的内容词位上 学生÷教师 的几何均值比(留出组 / 训练探针组 分开报)。
#
# 用法(spark 仓库根):
#   entity_probe_spark.sh prep                       实体表 → 题单 → 官方采集 → 训练行 / 教师表 / 探针流(只打 API, 不占模型)
#   with_model_stopped_spark.sh gguf-tools/scripts/entity_probe_spark.sh score <标签> [③目录]   学生臂打分 → gguf/go-onebit/evo/probe/ent_<标签>.tsv
#   with_model_stopped_spark.sh gguf-tools/scripts/entity_probe_spark.sh train <标签>           ③ 一轮(叠在现役 ② 上, ② 不动) → gguf/v41/posttrain/entity-<标签>/ckpt_e01
# 判读(fable5 11-01 方案 ②): 训前 score c5f2 → train → score 带 ③; 留出组比值往 1 靠 = 低秩能还原知识召回, 只有训练探针组动 = 只会背。
# 出错会怎样: 教师表对不上序列的条被丢(ptf 打印"对上 N 条"); 锚行数与流不等 ptfprobe 直接停(★锚只有 … 行★), 那是 covered 顺序与 ids 流不一致。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; PD=gguf-tools/data/posttrain; KEY=$PD/deepseek_api.key
WORK=gguf/go-onebit/evo/entity; TOOLS=speed-bench/bugmd_ids_tools.py; APT=gguf-tools/scripts/api_teacher.py; TTC=gguf-tools/scripts/think_trace_collect.py
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"
M(){ echo "[ent $(date '+%m-%d %H:%M:%S')] $*"; }
CMD="${1:?prep|score|train}"; shift

case "$CMD" in
prep)
    mkdir -p "$WORK"
    python3 "$TOOLS" entq "$PD/entity_list.tsv" "$WORK" || exit 1
    # 训练题再过一遍禁词表(鸟 / 骑行 / 留出动物): 命中的题整条丢, 留出题不筛(留出本来就是被禁的那类)
    BANRE=$(grep -v '^#' "$PD/evo_ban.txt" | grep -v '^[[:space:]]*$' | paste -sd'|' -)
    nb=$(grep -ciE "$BANRE" "$WORK/train_q.jsonl" || true); [ "$nb" = 0 ] || { grep -viE "$BANRE" "$WORK/train_q.jsonl" > "$WORK/train_q.tmp"; mv "$WORK/train_q.tmp" "$WORK/train_q.jsonl"; }
    nb2=$(grep -ciE "$BANRE" "$WORK/ptrain_q.jsonl" || true); [ "$nb2" = 0 ] || { grep -viE "$BANRE" "$WORK/ptrain_q.jsonl" > "$WORK/ptrain_q.tmp"; mv "$WORK/ptrain_q.tmp" "$WORK/ptrain_q.jsonl"; }
    M "禁词筛掉训练题 $nb 条 / 训练探针题 $nb2 条; 训练题 $(wc -l < "$WORK/train_q.jsonl") / 留出 $(wc -l < "$WORK/hold_q.jsonl") / 训练探针 $(wc -l < "$WORK/ptrain_q.jsonl")"
    for g in hold ptrain train; do
        M "官方采集 $g(关思考, 带 top-20, 8 路, 上限 512 位)"
        "$PY" "$APT" collect --key "$KEY" --questions "$WORK/${g}_q.jsonl" --out "$WORK/${g}_raw.jsonl" --hf "$HF" --no-think --max-tokens 512 --jobs 8 --timeout 300 | tail -1
        "$PY" -c "import json,sys; [sys.stdout.write(l) for l in open(sys.argv[1]) if json.loads(l).get('ok')]" "$WORK/${g}_raw.jsonl" > "$WORK/${g}_ok.jsonl"
        # --head-prefix: 题头(<｜User｜>题<｜Assistant｜></think>)进 prefix 只当上下文。不给会怎样(11-01 实撞): 题面也算进损失, 教师在题面上是 one-hot,
        #   训练器第 0 步 1.30 对引擎答案位 0.96, ③ 把力气花在背题面模板上, 训练器报 −57% 而引擎答案位只动 3%
        "$PY" "$TTC" chunks --hf "$HF" --raw "$WORK/${g}_ok.jsonl" --out "$WORK/${g}_rows.jsonl" --max-tok 1024 --max-chunks 1 --head-prefix | tail -1
        "$PY" "$APT" simids --hf "$HF" --rows "$WORK/${g}_rows.jsonl" --out "$WORK/$g.seqs" | tail -1
        "$PY" "$APT" ptf --ids "$WORK/$g.seqs" --raw "$WORK/${g}_ok.jsonl" --hf "$HF" --out "$WORK/$g.ptf" --covered "$WORK/$g.covered" --topk 20 | tail -1
    done
    # 探针流 = 留出覆盖序列 + 训练探针覆盖序列首尾相接(与 ptf 记录同序), --score-ids 要一行一个 id
    cat "$WORK/hold.covered" "$WORK/ptrain.covered" | tr ' ' '\n' | grep -v '^$' > "$WORK/probe.ids"
    M "探针流 $(wc -l < "$WORK/probe.ids") 位(留出 $(wc -l < "$WORK/hold.covered") 条 + 训练探针 $(wc -l < "$WORK/ptrain.covered") 条); 训练行 $(wc -l < "$WORK/train_rows.jsonl") 行, 教师表 $WORK/train.ptf"
    echo "ENT_PREP_DONE";;
score)
    # score <标签> [③目录] [②目录]: ② 不给 = serve_pick 的现役; 给 = 候选侧车目录(段 B 带路由系数的候选等, 11-01 晚), ③ 给空串跳过
    TAG="${1:?标签}"; PT="${2:-}"; [ -z "${3:-}" ] || ARM="${3%/}"; OUT=gguf/go-onebit/evo/probe/ent_$TAG.tsv; mkdir -p "$(dirname "$OUT")"
    STU=$WORK/stu_$TAG.bin; ZARG="--zchain $ARM"; [ -z "$PT" ] || ZARG="$ZARG --posttrain ${PT%/}"
    N=$(wc -l < "$WORK/probe.ids"); M "学生臂 $ZARG 打分 $N 位 → $STU"
    ./ds4 -m "$GG" --cuda --mem-budget-mb 40000 --weight-cache-mb 88000 --score-ids "$WORK/probe.ids" --score-out "$STU" $ZARG > "/tmp/bugmd/ent_score_$TAG.log" 2>&1 || { tail -5 "/tmp/bugmd/ent_score_$TAG.log"; echo "★打分失败★"; exit 1; }
    grep -a "完成\|PPL" "/tmp/bugmd/ent_score_$TAG.log" | tail -2
    "$PY" "$TOOLS" ptfprobe "$STU" "$HF/tokenizer.json" "$OUT" "hold=$WORK/hold.ptf" "ptrain=$WORK/ptrain.ptf" || exit 1
    rm -f "$STU" "$STU".*   # 学生锚 ~0.5 MB/位, 用完即删; 流 ≤2048 位时 --score-ids 还落逐层 x_L/y_L/hce 旁件(core_v41_score.c), 一起删
    echo "ENT_SCORE_DONE $OUT";;
scoreshort)
    # scoreshort <标签> [③目录]: 短流尺(11-01 21:38 的做法固化, 11-02): 留出行 1~5 / 6~10 各成一条 ≤1k 位的流(hold_sA/sB.ids + 对应 ptf), 量"短上下文"的知识召回,
    #   与 score 的 16.8k 长流对着看 —— ent04/ent06 两轮长流 +9~12% 而探针的短知识题反退, 得知道短上下文到底动没动。产物 probe/ent_short_<sA|sB>_<标签>.tsv
    TAG="${1:?标签}"; PT="${2:-}"; ZARG="--zchain $ARM"; [ -z "$PT" ] || ZARG="$ZARG --posttrain ${PT%/}"
    for X in sA sB; do
        STU=$WORK/stu_short_${X}_$TAG.bin; OUT=gguf/go-onebit/evo/probe/ent_short_${X}_$TAG.tsv
        M "短流 $X: 学生臂 $ZARG 打分 $(wc -l < "$WORK/hold_$X.ids") 位"
        ./ds4 -m "$GG" --cuda --mem-budget-mb 40000 --weight-cache-mb 88000 --score-ids "$WORK/hold_$X.ids" --score-out "$STU" $ZARG > "/tmp/bugmd/ent_short_${X}_$TAG.log" 2>&1 || { tail -3 "/tmp/bugmd/ent_short_${X}_$TAG.log"; echo "★短流 $X 打分失败★"; exit 1; }
        "$PY" "$TOOLS" ptfprobe "$STU" "$HF/tokenizer.json" "$OUT" "hold=$WORK/hold_$X.ptf" > /dev/null || exit 1   # 汇总在 tsv 的 # 行里, stdout 没有(11-02 实撞: 管道接 grep 把空输出当失败)
        grep -a "^# hold" "$OUT"
        rm -f "$STU" "$STU".*
    done
    echo "ENT_SCORESHORT_DONE $TAG";;
train)
    # train <标签> [训练行] [教师表] [轮数=1] [预算秒=0]: 缺省短行 train_rows.jsonl / train.ptf; 长前缀行(entlong + ptfprefix, 11-01 晚)传 long_rows.jsonl / long.ptf, 训练器按 maxlen 分段接力;
    #   预算秒 = 训练器 budget_s(11-02 用户定一轮 ≤ 1 小时: 到点收轮评估落盘, 不开下一轮), 0 = 不限
    TAG="${1:?标签}"; DATA="${2:-$WORK/train_rows.jsonl}"; TPTF="${3:-$WORK/train.ptf}"; EP="${4:-1}"; BUDGET="${5:-0}"; OUT=gguf/v41/posttrain/entity-$TAG; mkdir -p "$OUT"
    {   echo "# entity_probe_spark.sh $TAG: ③ 一轮(./ds4 --ptrain), 料 = 实体特征描述 × 官方 API top-20, 叠在 ② $(basename "$ARM") 之上, ② 原封不动"
        echo "data=$DATA"; echo "out=$OUT"; echo "teacher=fp:$TPTF"; echo "lr=2e-5"
        printf '%s\n' layers=0-39 rank=64 epochs=$EP batch=4 topk=20 maxlen=1024 expand=0 ctxlen=65536 doc_tail=0 budget_s=$BUDGET
    } > "$WORK/train_$TAG.cfg"
    M "③ 训: $WORK/train_$TAG.cfg(料 $(wc -l < "$DATA") 行, 教师 $TPTF) → $OUT"
    ./ds4 --cuda -m "$GG" --zchain "$ARM" --mem-budget-mb 110000 --no-dspark --ptrain "$WORK/train_$TAG.cfg" > "/tmp/bugmd/ent_train_$TAG.log" 2>&1; rc=$?
    grep -aE "ptrain|★|教师表|epoch|留出|hold" "/tmp/bugmd/ent_train_$TAG.log" | tail -8
    ls -d "$OUT"/ckpt_e* 2>/dev/null | tail -1
    echo "ENT_TRAIN_DONE rc=$rc";;
vstep)
    # vstep <标签> [层=15] [轮=1] [预算秒=600] [每层热门专家=4] [每步改几份=0.01]: V 步(core_ptrain_vstep.c, 11-02): 不训低秩, 端到端重选这几层专家权重的离散码字,
    #   料 = 实体混合前缀行(mix_rows/mix.ptf, 短中长各 1/3), 第 1 轮带机制审计(vq_check=1: 最负 64 组的一阶预测 ΔL 对实测, 反了就停)。
    #   产物 gguf/v41/posttrain/vstep-<标签>/ckpt_vNN/vqpatch_Lnn.bin + base.fnv, 挂法同 ③(--posttrain 目录)。eval_n=16 只评 16 段留出(探针要分钟级)。
    #   为什么 layers 从 V 步首层起: 反传只要穿到那一层; 低秩 A 留 0 不步进, 只占显存不改输出。
    #   v01(11-02 10:2x) 实撞: 8 题一批取 1% = 17.7 万组, 本批降留出 +17%(背批) ⇒ 缺省改成批 16 题两半一致门 + 每步 0.1%
    #   第 8 参 = 每趟梯度几层一组(两份半批梯度缓冲 = 2 × 组层数 × 热门 × 142 MB; 训练态可用 ~8 GB ⇒ 5 层 × 4 专家 = 5.7 GB 是上限)
    #   第 9 参 = 料: mix(短中长前缀混合, 缺省) | short(315 短行, 全是靶位: v04 实撞 16 题长前缀行只有 1,900 个答案位出梯度, 九成算力花在只当上下文的前缀上; 短行 130 位全出梯度, 同代价靶位多 5 倍)
    #   第 10 参 = batch(一份的题数; 份数 S = 批 ÷ batch, 门 = S 份全同意; 缺省 4)
    TAG="${1:?标签}"; VL="${2:-15}"; EP="${3:-1}"; BUDGET="${4:-600}"; HOT="${5:-4}"; FRAC="${6:-0.001}"; BS="${7:-16}"; GRP="${8:-1}"; ROWS="${9:-mix}"; BATCH="${10:-4}"; OUT=gguf/v41/posttrain/vstep-$TAG; mkdir -p "$OUT"
    LO="${VL%%-*}"; DATA=$WORK/mix_rows.jsonl; TPTF=$WORK/mix.ptf; [ "$ROWS" = short ] && { DATA=$WORK/train_rows.jsonl; TPTF=$WORK/train.ptf; }
    {   echo "# entity_probe_spark.sh vstep $TAG: V 步(./ds4 --ptrain vq_layers=$VL), 料 = 实体行($ROWS) × 官方 API top-20, ② $(basename "$ARM") 挂着不动, 产物 = 索引补丁"
        echo "data=$DATA"; echo "out=$OUT"; echo "teacher=fp:$TPTF"; echo "lr=2e-5"
        printf '%s\n' "layers=$LO-39" rank=64 "epochs=$EP" "batch=$BATCH" topk=20 maxlen=1024 expand=0 ctxlen=65536 doc_tail=0 eval_n=16 "budget_s=$BUDGET"
        printf '%s\n' "vq_layers=$VL" "vq_hot=$HOT" vq_knn=16 "vq_group=$GRP" "vq_bs=$BS" vq_check=1 "vq_frac=$FRAC"
    } > "$WORK/vstep_$TAG.cfg"
    M "V 步: $WORK/vstep_$TAG.cfg(层 $VL, $EP 轮, 预算 $BUDGET s) → $OUT"
    ./ds4 --cuda -m "$GG" --zchain "$ARM" --mem-budget-mb 110000 --no-dspark --ptrain "$WORK/vstep_$TAG.cfg" > "/tmp/bugmd/vstep_$TAG.log" 2>&1; rc=$?
    grep -aE "V步|V 步|★|审计|补丁|第 0 步|留出" "/tmp/bugmd/vstep_$TAG.log" | tail -14 | cut -c1-220
    echo "ENT_VSTEP_DONE rc=$rc";;
*) echo "用法: $0 prep | score <标签> [③目录] | scoreshort <标签> [③目录] | train <标签> … | vstep <标签> [层] [轮] [预算秒]"; exit 2;;
esac
