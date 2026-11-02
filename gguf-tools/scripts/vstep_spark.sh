#!/bin/bash
# vstep_spark.sh — V 步(端到端重选专家权重的离散码字, core_ptrain_vstep.c, 2026-11-02)的整条链: 料 → 一小时轮 → 四把尺。
#
# 目标(用户 11-02 定): 编程指标不掉或提升 + 细节(知识)还原。两个目标都写进选码门: 每轮的批 = 一份编程流(fit.ids 整条 15,360 位, 教师 = 原件 FP 全 logits 锚)
#   + 若干份实体知识短行(官方 API top-20); 一组码字要全部份都说"该换"才换; 轮末编程判决份(judge.ids, 泛化留出)与知识留出任一不降就整轮回退。
# 用法(spark 仓库根, 模型停着跑: with_model_stopped_spark.sh 包):
#   vstep_spark.sh prep [域…] [agent=目录]                  料: ids 行(各域 fit 流) + 实体短行(按实体切: 训练进 rows.jsonl, 留出实体进 know_hold.jsonl) [+ agent 头行按任务切], 教师表 = 各锚转表(anchor2ptf K=20) + 各短表 cat 起来
#   vstep_spark.sh run <标签> [层=10-19] [轮=12] [预算=2400] [热门=4] [份额=0.0005] [短题数=24] [组=5] [batch=8] [eval_n=64] [留出=ent|agent|row] [尾权τ=0] [前缀篇/轮=4] [起点③目录=空]
#     第 14 参(11-02 深夜) = 起点补丁目录(cfg vq_init): 续训(同层接着选)或跨层叠(起点动 L10-19, 这趟选 L20-29), 落盘的 ckpt 自包含, 直接挂
#   vstep_spark.sh pick <ckpt 根目录> [须≤第0步=0]         选轮: 训练日志里判决份最低的那个 ckpt(第 2 参 1 = 还得不比第 0 步高, 否则空)
#   vstep_spark.sh judge <标签> <ckpt 目录>                 四把尺: 实体长流 / 短流 / 编程 judge 15360 + wt2 512 / 规划位探针
# 判读: 训练日志 /tmp/bugmd/vstep_<标签>.log 里 "★V 步轮 N 末" 的本批 / 留出 / 判决份三列, "回退" 行 = 信赖域起作用; 产物 gguf/v41/posttrain/vstep-<标签>/ckpt_vNN(vqpatch_Lnn.bin + base.fnv)。
# 出错会怎样: 锚行数与 ids 不等 anchor2ptf 直接停; 训练器配不上教师表停车(★有题没对上★); 补丁回读不同停车。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; TOOLS=speed-bench/bugmd_ids_tools.py
WORK=gguf/go-onebit/evo/vstep; ENT=gguf/go-onebit/evo/entity; JD=gguf/v41judge
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"
M(){ echo "[vstep $(date '+%m-%d %H:%M:%S')] $*"; }
CMD="${1:?prep|run|judge}"; shift
case "$CMD" in
prep)
    # prep [流域…=code graphics restore]: 训练流 = 各域 gguf/go-onebit/<域>/fit.ids(各 15,360 位, 教师 = v41judge/teacher_<域>_fit_n15360 锚), 轮流当每轮的第 0 份;
    #   泛化留出 = code/judge.ids(编程判决份, 训练从没见过)。v06 实撞: 只有一条编程流, 每轮同一条, 第 3 轮起全同意的组从 37 万掉到 11 万(容易拿的拿完了) ⇒ 多条轮换。
    #   每条表先过 ptfhit(教师 top-1 命中下一 token ≥ 50%, FP 锚约 70%): 锚与 ids 配错了训练器照样能配上表、教的却是别处的分布, 只有这一关能抓。
    # prep [域…] [agent=目录](11-02 晚): agent=<evo/v002 这种目录> 把官方 API 对齐的 agent 头行(rows.jsonl: 10k 前缀 + 官方思考头 2k 靶, teacher.ptf)并进料,
    #   按任务(首个 <｜User｜> 后的任务句)每 5 个任务留 1 个(第 3 个起)整任务进 agent_hold.jsonl(训练器 eval_kl=, run 第 11 参 agent), 其余进训练 —— 留出任务整条没见过, 量的是决策位的跨任务泛化
    mkdir -p "$WORK"; : > "$WORK/rows.jsonl"; : > "$WORK/streams.ptf"; AGENT=""; DOMS=""
    for a in "$@"; do case "$a" in agent=*) AGENT="${a#agent=}";; *) DOMS="$DOMS $a";; esac; done
    DOMS="${DOMS:-code graphics restore}"; ns=0
    for dm in $DOMS; do
        IDS=gguf/go-onebit/$dm/fit.ids; TB=$JD/teacher_${dm}_fit_n15360_$(basename "$HF").bin
        [ -s "$IDS" ] && [ -s "$TB" ] || { echo "★$dm: 流 $IDS 或 FP 锚 $TB 缺★"; exit 1; }
        "$PY" -c "import sys,json; ids=[int(x) for x in open(sys.argv[1]).read().split()]; print(json.dumps({'ids': ids}))" "$IDS" >> "$WORK/rows.jsonl"   # ids 行: 流原样, 不加 BOS(core_ptrain_ids.c)
        "$PY" "$TOOLS" anchor2ptf "$TB" "$IDS" "$WORK/fit_$dm.ptf" 20 || exit 1
        "$PY" "$TOOLS" ptfhit "$WORK/fit_$dm.ptf" || exit 1
        cat "$WORK/fit_$dm.ptf" >> "$WORK/streams.ptf"; ns=$((ns + 1))
    done
    # 知识短行按实体整只切(11-02 v06/v09 实撞: 按行五取一的留出 −5% 一点没迁到没练过的鸟, 降的是背同实体的别种问法): 每类实体按题单序每 5 只留 1 只(第 3 只起),
    #   那只的三种问法全进 know_hold.jsonl(训练器 eval_kl=), 其余进 rows.jsonl 当训练。实体从题单 train_q.jsonl 的 id(类|实体|语言|题号)来, 训练行按题文对回题单, 对不上停。
    "$PY" - "$ENT/train_q.jsonl" "$ENT/train_rows.jsonl" "$WORK/rows.jsonl" "$WORK/know_hold.jsonl" <<'EOF' || exit 1
import sys, json, collections
q = [json.loads(l) for l in open(sys.argv[1])]; rows = [json.loads(l) for l in open(sys.argv[2])]
cls = collections.OrderedDict()
for a in q:
    c, e, _, _ = a["id"].split("|"); cls.setdefault(c, [])
    if e not in cls[c]: cls[c].append(e)
hold = set(e for es in cls.values() for i, e in enumerate(es) if i % 5 == 2)
tr = open(sys.argv[3], "a"); ho = open(sys.argv[4], "w"); nt = nh = 0
for r in rows:
    m = [a for a in q if a["q"] in r["prefix"]]
    if len(m) != 1: sys.exit("★实体行对不上题单(%d 条命中): %s★" % (len(m), r["prefix"][:60]))
    if m[0]["id"].split("|")[1] in hold: ho.write(json.dumps(r, ensure_ascii=False) + "\n"); nh += 1
    else: tr.write(json.dumps(r, ensure_ascii=False) + "\n"); nt += 1
print("知识行: 训练 %d(实体 %d) / 按实体留出 %d(实体 %d: %s)" % (nt, sum(len(v) for v in cls.values()) - len(hold), nh, len(hold), " ".join(sorted(hold))))
EOF
    if [ -n "$AGENT" ]; then
        [ -s "$AGENT/rows.jsonl" ] && [ -s "$AGENT/teacher.ptf" ] || { echo "★agent 料 $AGENT 缺 rows.jsonl / teacher.ptf★"; exit 1; }
        "$PY" - "$AGENT/rows.jsonl" "$WORK/rows.jsonl" "$WORK/agent_hold.jsonl" <<'EOF' || exit 1
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1])]
def task(r):
    p = r["prefix"]; i = p.find("<｜User｜>"); t = p[i + 9:] if i >= 0 else p
    j = t.find("Current runtime"); return (t[:j] if j > 0 else t[:80]).strip()
tasks = []
for r in rows:
    if task(r) not in tasks: tasks.append(task(r))
hold = set(t for i, t in enumerate(tasks) if i % 5 == 2)
tr = open(sys.argv[2], "a"); ho = open(sys.argv[3], "w"); nt = nh = 0
for r in rows:
    if task(r) in hold: ho.write(json.dumps(r, ensure_ascii=False) + "\n"); nh += 1
    else: tr.write(json.dumps(r, ensure_ascii=False) + "\n"); nt += 1
print("agent 头行: 训练 %d 行(任务 %d) / 按任务留出 %d 行(任务 %d: %s)" % (nt, len(tasks) - len(hold), nh, len(hold), " | ".join(sorted(hold))[:300]))
EOF
        cat "$AGENT/teacher.ptf" >> "$WORK/streams.ptf"
    else : > "$WORK/agent_hold.jsonl"; fi
    JUDGE=gguf/go-onebit/code/judge.ids; TJ=$JD/teacher_code_judge_n15360_$(basename "$HF").bin
    "$PY" -c "import sys,json; ids=[int(x) for x in open(sys.argv[1]).read().split()]; print(json.dumps({'ids': ids}))" "$JUDGE" > "$WORK/eval.jsonl"
    "$PY" "$TOOLS" anchor2ptf "$TJ" "$JUDGE" "$WORK/judge.ptf" 20 || exit 1
    "$PY" "$TOOLS" ptfhit "$WORK/judge.ptf" || exit 1
    cat "$WORK/streams.ptf" "$WORK/judge.ptf" "$ENT/train.ptf" > "$WORK/teacher.ptf"
    M "料 $(wc -l < "$WORK/rows.jsonl") 行(流 $ns: $DOMS + 短行 $(wc -l < "$ENT/train_rows.jsonl")), 泛化留出 = judge 流; 教师表 $(du -m "$WORK/teacher.ptf" | cut -f1) MB"
    echo "VSTEP_PREP_DONE";;
run)
    # 第 10 参 = eval_n(留出按篇等间隔抽几段; 知识短行一行一段 ⇒ 就是留出题数上限, v06 16 题的抖动 ±0.005 误触过两次回退 ⇒ 缺省 64 = 短行留出全评)
    # 第 11 参 = 知识留出怎么切: ent(缺省, 11-02 起) = prep 切好的按实体留出 know_hold.jsonl(eval_kl=); row = 老路按行五取一(data_eval=1, v06~v10 的口径, 只作对照)
    # 第 12 参 = 尾权 τ(11-02 晚, cfg vq_tail): 选码趟按行 w = min(16, max(1, KL/τ)), 0 = 关(v06~v11 口径)。天花板诊断: c5f2 177 个尾位里 L10-19 换原件只修 12 个, L20-29 修 39 个 ⇒ 层也换 20-29
    # 第 13 参 = 每轮前缀文档几篇(cfg vq_docs, 11-02 晚): agent 头行一篇只 1~3 段, 单篇一份选不出码字
    TAG="${1:?标签}"; VL="${2:-10-19}"; EP="${3:-12}"; BUDGET="${4:-2400}"; HOT="${5:-4}"; FRAC="${6:-0.0005}"; BS="${7:-24}"; GRP="${8:-5}"; BATCH="${9:-8}"; EVN="${10:-64}"; KH="${11:-ent}"; TAIL="${12:-0}"; NDOC="${13:-4}"; INIT="${14:-}"
    OUT=gguf/v41/posttrain/vstep-$TAG; mkdir -p "$OUT"; LO="${VL%%-*}"
    [ -s "$WORK/teacher.ptf" ] || { echo "★先 prep★"; exit 1; }
    [ -z "$INIT" ] || [ -s "$INIT/base.fnv" ] || { echo "★起点 $INIT 不是 V 步 ckpt(没有 base.fnv)★"; exit 1; }
    case "$KH" in ent) [ -s "$WORK/know_hold.jsonl" ] || { echo "★按实体留出要先 prep(know_hold.jsonl)★"; exit 1; }; HOLDL="eval_kl=$WORK/know_hold.jsonl";;
                 agent) [ -s "$WORK/agent_hold.jsonl" ] || { echo "★agent 留出要先 prep agent=目录★"; exit 1; }; HOLDL="eval_kl=$WORK/agent_hold.jsonl";;   # 留出 = 没见过的 agent 任务的官方思考头
                 row) HOLDL="data_eval=1";; *) echo "留出方式只认 ent|agent|row"; exit 2;; esac
    {   echo "# vstep_spark.sh run $TAG: V 步 vq_layers=$VL, 料 = 编程流(fit, FP 锚, 轮换) + 实体短行(API top-20), 泛化留出 = judge 流, 知识留出 = $KH; ② $(basename "$ARM") 挂着不动, 产物 = 索引补丁"
        echo "data=$WORK/rows.jsonl"; echo "eval=$WORK/eval.jsonl"; echo "out=$OUT"; echo "teacher=fp:$WORK/teacher.ptf"; echo "lr=2e-5"
        printf '%s\n' "layers=$LO-39" rank=64 "epochs=$EP" "batch=$BATCH" topk=20 maxlen=1024 expand=0 ctxlen=65536 doc_tail=0 "eval_n=$EVN" "$HOLDL" "budget_s=$BUDGET"
        printf '%s\n' "vq_layers=$VL" "vq_hot=$HOT" vq_knn=16 "vq_group=$GRP" "vq_bs=$BS" vq_check=1 "vq_frac=$FRAC" "vq_tail=$TAIL" "vq_docs=$NDOC"
        [ -z "$INIT" ] || echo "vq_init=$INIT"
    } > "$WORK/run_$TAG.cfg"
    M "V 步: $WORK/run_$TAG.cfg(层 $VL, ≤$EP 轮, 预算 $BUDGET s, 热门 $HOT, 份额 $FRAC, 短题 $BS/轮 分 $((BS / BATCH)) 份 + 流 1 份, 知识留出 $KH, 尾权 τ=$TAIL, 起点 ${INIT:-裸 ①+②}) → $OUT"
    ./ds4 --cuda -m "$GG" --zchain "$ARM" --mem-budget-mb 110000 --no-dspark --ptrain "$WORK/run_$TAG.cfg" > "/tmp/bugmd/vstep_$TAG.log" 2>&1; rc=$?
    grep -aE "V 步轮|回退|审计|★|第 0 步" "/tmp/bugmd/vstep_$TAG.log" | tail -16 | cut -c1-240
    echo "VSTEP_RUN_DONE rc=$rc";;
judge)
    TAG="${1:?标签}"; CK="${2:?ckpt 目录}"
    gguf-tools/scripts/entity_probe_spark.sh score "$TAG" "$CK" 2>&1 | grep -aE "V步补丁|★"; grep -a "^# hold\|^# ptrain" gguf/go-onebit/evo/probe/ent_$TAG.tsv
    gguf-tools/scripts/entity_probe_spark.sh scoreshort "$TAG" "$CK" 2>&1 | grep -aE "^# hold|★"
    for ids in "gguf/go-onebit/code/judge.ids 15360" "gguf/go-onebit/g7/wt2.ids 512"; do
        set -- $ids; lg=/tmp/bugmd/judge_${TAG}_$(basename "$1" .ids).log
        bash gguf-tools/scripts/v41_judge.sh "$1" "$2" --hf "$HF" "engine:$GG:$ARM::$CK" > "$lg" 2>&1 && grep -aE "Same top|Mean KLD|Σmin" "$lg" | sed "s|^ *|  $(basename "$1" .ids): |" || echo "★判决 $1 失败(看 $lg)★"
        # 尾位数(11-02 晚定的尺): 均值四趟都动、尾巴一位不动, 鹈鹕的决策全在尾巴上 ⇒ 尾位数与均值并排报; c5f2 判决份 = 123 / 177 / 634
        RW=$(grep -a "逐位行 →" "$lg" | tail -1 | sed 's/.*→ //'); [ -s "$RW" ] && "$PY" -c "
import sys; r=[l.split() for l in open(sys.argv[1])]; kl=[float(x[1]) for x in r]; sm=[float(x[2]) for x in r]
print('  %s 尾位: Σmin<0.2 %d / KLD>2 %d / KLD>1 %d (共 %d 位)' % (sys.argv[2], sum(s<0.2 for s in sm), sum(k>2 for k in kl), sum(k>1 for k in kl), len(r)))" "$RW" "$(basename "$1" .ids)"
    done
    bash gguf-tools/scripts/plan_adjective_probe_spark.sh "$TAG" "$CK" 2>&1 | tail -1
    grep -av "^#" gguf/go-onebit/evo/probe/adj_$TAG.tsv | head -6
    echo "VSTEP_JUDGE_DONE";;
pick)
    # 选轮: 每个落盘的 ckpt_vNN 对应训练日志里 "V 步轮 NN 泛化留出(判决份) X"(回退的轮没有 ckpt, 自然不选); 取 X 最低的; 第 2 参 1 = X 还得 ≤ 第 0 步, 否则输出空
    ROOTD="${1:?ckpt 根目录}"; STRICT="${2:-0}"; LG="/tmp/bugmd/vstep_$(basename "$ROOTD" | sed 's/^vstep-//').log"   # 判决份那行只在 stderr 日志里, train.log 是 lf 的精简账
    [ -s "$LG" ] || { echo "★没有训练日志 $LG★" >&2; exit 1; }
    K0=$(grep -a "第 0 步泛化留出(判决份)" "$LG" | tail -1 | sed -E 's/.*判决份\) ([0-9.]+).*/\1/')
    best=""; bv=""
    for ck in "$ROOTD"/ckpt_v??; do
        n=$(basename "$ck" | sed 's/ckpt_v0*//'); x=$(grep -a "V 步轮 $n 泛化留出(判决份)" "$LG" | tail -1 | sed -E 's/.*判决份\) ([0-9.]+).*/\1/')
        [ -n "$x" ] || continue
        if [ "$STRICT" = 1 ] && [ -n "$K0" ] && awk "BEGIN{exit !($x > $K0)}"; then continue; fi
        if [ -z "$bv" ] || awk "BEGIN{exit !($x < $bv)}"; then bv=$x; best=$ck; fi
    done
    echo "$best";;
*) echo "用法: $0 prep | run <标签> … | judge <标签> <ckpt> | pick <ckpt 根目录> [1]"; exit 2;;
esac
