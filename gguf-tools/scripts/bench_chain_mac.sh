#!/bin/bash
# bench_chain_mac.sh — 无人值守把 unsloth 口径的 9 项任务基准跑完(10-31 晚用户: "每指标 1.5 小时预算, 不跑进过训练料的题, Docker 好了把 Aider 接进链,
#   盯着别中途异常中断, 明早更新指标图")。六项在 spark 上跑(bench_suite_spark.sh), Aider 在 Mac 的 Docker 里跑(aider_bench_mac.sh), 所以由 Mac 这边编排。
#
# 用法(Mac 仓库根, 工作台模型在 spark 8001 开着):
#   mkdir -p /tmp/bugmd; caffeinate -i nohup bash gguf-tools/scripts/bench_chain_mac.sh 1101 90 > /tmp/bugmd/bench_chain_1101.log 2>&1 &
#   caffeinate 防 Mac 睡着(睡了 ssh 轮询和 aider 的隧道都断)。日志只在 /tmp, 产物在 gguf/v41/bench/<标签>/(两机同份)。
# 怎么走:
#   ① 六项逐项: ssh spark 发 bench_suite_spark.sh <标签> <剩余预算> 1 <任务>(一项一次调用, 预算各算各的), 每 60 s 看一眼:
#      进程没了且 summary.json 在 = 这项完; 进程没了 summary 不在 = 异常中断(模型子进程死 / 请求失败后 bench_suite 自停 / ssh 断) →
#      核 8001, 不在就 with_model_stopped_spark.sh --load 装回等冒烟过, 再用【剩余预算】续跑(bench_suite.py 按 answers.jsonl 断点续, 已答的题不重发);
#      一项最多 4 次, 剩余预算 < 3 分钟也发一次(只为写出 summary.json)。GPU 时钟 < 1000 MHz 打 ★(10-17 PD 供电卡 611 MHz 只能冷断电, 脚本修不了, 人看到再说)。
#   ② 六项完: rsync 产物回 Mac, ifeval_grade.py(Google 官方判分器只装在 Mac)判 IFEval, 判完的 ifeval/ 同步回 spark。
#   ③ Aider: aider_bench_mac.sh <标签> <预算> 1(trained_exclude.jsonl 里 task=aider 的题自动排除, 隧道 18001 自建)。
#   ④ 打印六项 summary 一行一项 + aider stats。图 docs/img/quant_compare.html 的 QUANTS 本项目行按这些数手工填, 再 docs/img/render_arch.sh quant_compare 出 png。
# 排除: bench_suite.py 缺省读 gguf-tools/bench/data/trained_exclude.jsonl(只有真进了现役侧车的轮, 10-31 = t002 的 9 题, kept c5f2), 候选清单 trained_pending.jsonl 不排。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
TAG="${1:?标签, 如 1101}"; BUDGET="${2:-90}"
TASKS=(aime25 gpqa_diamond lcb_v6 ifeval mmlu_pro hmmt_feb26)
OUT=gguf/v41/bench/$TAG
L(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
rt(){ ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 spark "cd ~/ds4-main && $*"; }
# 模型直连口在不在; 不在就按 serve_pick.txt 装回(工作台主进程不自动重拉子进程, 10-17)并等冒烟过
ensure_model(){
    rt 'curl -s --noproxy "*" -m 10 http://127.0.0.1:8001/v1/models' | grep -q deepseek && return 0
    L "★8001 没在服务, 装回模型★"; rt 'bash gguf-tools/scripts/with_model_stopped_spark.sh --load' | tail -2
    for _ in $(seq 90); do
        rt 'f=$(ls -t gguf/v41/posttrain/ui_logs/serve_*.log | head -1); grep -q "冒烟 过" "$f" && echo UP; grep -q "冒烟 ★失败★\|★没起来\|★余量" "$f" && echo DOWN' | grep -q UP && { L "模型回来了"; return 0; }
        sleep 10
    done
    L "★模型 15 分钟没回来★"; return 1
}
clock(){ rt 'nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader'; }
alive(){ rt "pgrep -f '[b]ench_suite.py --data gguf-tools/bench/data/$1.jsonl' > /dev/null"; }   # [b] 防 pgrep 匹配到这条 ssh 命令自己
rt "mkdir -p $OUT"
L "起跑 tag=$TAG 每项预算 ${BUDGET} min, 六项 ${TASKS[*]} 在 spark, Aider 在 Mac Docker; 排除清单 $(grep -c . gguf-tools/bench/data/trained_exclude.jsonl) 题; GPU $(clock)"
for t in "${TASKS[@]}"; do
    spent=0; n=0
    while :; do
        n=$((n + 1)); [ "$n" -le 4 ] || { L "★$t 四次都没跑完, 放弃这项, 接着跑下一项★"; break; }
        left=$((BUDGET - spent)); [ "$left" -ge 3 ] || left=3
        ensure_model || break
        L "▶ $t 第 $n 次, 预算 $left min(已用 $spent); GPU $(clock)"
        t0=$(date +%s)
        rt "nohup bash gguf-tools/scripts/bench_suite_spark.sh $TAG $left 1 $t > $OUT/chain_${t}_$n.nohup 2>&1 < /dev/null &"
        sleep 30
        while alive "$t"; do
            sleep 60
            c=$(clock | cut -d, -f1 | tr -dc 0-9); [ -n "$c" ] && [ "$c" -lt 1000 ] && L "★GPU 时钟 $c MHz(< 1000): 10-17 那种限频, 这趟的分不作数, 要冷断电★"
        done
        spent=$((spent + ($(date +%s) - t0 + 59) / 60))
        if rt "test -s $OUT/$t/summary.json"; then
            # 汇总行用 json.dumps 不用花括号推导: 11-01 实撞 ssh 两层引号把 {k: d[k] for k in (…)} 拆碎, 打出来一地 "■ … items_total"
            L "■ $t 完($n 次, 共 $spent min): $(rt "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(dict((k, d.get(k)) for k in sys.argv[2:])))' $OUT/$t/summary.json answers items_total pass_at_1 se_pct truncated_pct mean_completion_tokens wall_min")"
            break
        fi
        L "★$t 第 $n 次中断(进程没了, summary 不在; 已用 $spent min): $(rt "tail -3 $OUT/chain_${t}_$n.nohup | cut -c1-300" | tr '\n' ' ')★"
        [ "$spent" -lt "$BUDGET" ] || { L "★$t 预算用完还没 summary, 发一次 3 分钟的只为写 summary★"; spent=$((BUDGET - 3)); }
    done
done
L "六项完, 拉回产物"
rsync -a --exclude 'chain_*.nohup' "spark:ds4-main/$OUT/" "$OUT/" && L "rsync 回 Mac 完 $(du -sh "$OUT" | cut -f1)"
if [ -s "$OUT/ifeval/answers.jsonl" ]; then
    L "IFEval 官方判分(Mac)"; python3 gguf-tools/scripts/ifeval_grade.py --out "$OUT/ifeval" --data gguf-tools/bench/data/ifeval.jsonl 2>&1 | tail -3
    rsync -a "$OUT/ifeval/" "spark:ds4-main/$OUT/ifeval/" && L "ifeval 判完同步回 spark"
fi
L "▶ Aider Polyglot(Mac Docker, 预算 $BUDGET min)"
ensure_model && bash gguf-tools/scripts/aider_bench_mac.sh "$TAG" "$BUDGET" 1 2>&1 | grep -E "^#|pass_rate|percent_cases|test_cases|tests_outdated|error_outputs|num_malformed|syntax_errors|lazy_comments" | tail -20
L "全部跑完. 六项 summary:"
for t in "${TASKS[@]}"; do
    [ -s "$OUT/$t/summary.json" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], {k: d.get(k) for k in ('answers','items_total','pass_at_1','se_pct','truncated_pct','no_answer','mean_completion_tokens','wall_min')})" "$OUT/$t/summary.json" "$t" || echo "$t: 没有 summary"
done
D=$(ls -d gguf/v41/bench/aider/aider/tmp.benchmarks/*--"$TAG" 2>/dev/null | tail -1); [ -n "$D" ] && { echo "aider: $(find "$D" -name .aider.results.json | wc -l | tr -d ' ') 题, $D"; grep -E "pass_rate_1|percent_cases_well_formed" "$D/stats.txt" 2>/dev/null; }
L "CHAIN_DONE"
