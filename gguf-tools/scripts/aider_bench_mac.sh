#!/bin/bash
# aider_bench_mac.sh — 在 Mac 上用 aider 官方 Docker harness 跑 Aider Polyglot(225 题), 模型是 spark 上的 ds4-server(经 ssh 隧道)。
# spark 没有 docker 权限, 所以 harness 跑在 Mac, 只有请求发到 spark。
#
# 用法: aider_bench_mac.sh <标签> [预算分钟=60] [并发=1] [题数=-1 全部] [benchmark.py 额外参数, 如 --verbose] [模型=openai/ds4]
#   冒烟: aider_bench_mac.sh smoke 15 1 1 --verbose(1 道题, 打印发给模型的 kwargs)
#   对照: 模型给 openai/deepseek-flash = 官方 API 上的原版(密钥 gguf-tools/data/posttrain/deepseek_api.key, 段 B 同一把), 其余口径不变;
#        第 7 个参数换温度(如 0.0), 不给 = yml 里的 1.0;
#        指定题用额外参数 --keywords go/exercises/practice/wordy,python/exercises/practice/pov(子串匹配相对路径)
#   预算到点给 benchmark.py 发 INT: 已跑完的题各有 .aider.results.json, 正在跑的那道丢掉; 然后用 --stats 汇总。
#   题序由 benchmark.py 自己 random.shuffle(无种子), 所以预算内跑到的是随机子集, 报 n 与 ±SE。
#   每题只试 1 次(--tries 1): unsloth 报的是 pass@1, 第二次尝试只影响 pass_rate_2, 省下的时间多跑几题。
# 前置(一次性): gguf/v41/bench/aider/aider = Aider-AI/aider 浅克隆, 其 tmp.benchmarks/polyglot-benchmark = Aider-AI/polyglot-benchmark;
#   镜像 docker build -f benchmark/Dockerfile -t aider-benchmark . (在 aider 目录里)。
# 产物: gguf/v41/bench/aider/aider/tmp.benchmarks/<日期>--<标签>/ (每题目录 + .aider.results.json), 汇总打印在最后 + 写 stats.txt。
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TAG="${1:?标签}"; BUDGET="${2:-60}"; THREADS="${3:-1}"; NUM="${4:--1}"; EXTRA="${5:-}"; MODEL="${6:-openai/ds4}"; TEMP="${7:-}"; RECORD="${8:-}"
# 第 8 个参数 = 录请求的 jsonl(api_record_proxy.py): 容器改打本机 18002, 转发器把每次请求体 / 响应体逐字存下(段 B 拿真实 agent 请求当料要逐字节的消息列表)
AIDER="$ROOT/gguf/v41/bench/aider/aider"
PORT=18001   # Mac 本机 8001 被别的容器占着(quant_trading_flow 后端), 隧道走 18001
[ -d "$AIDER/tmp.benchmarks/polyglot-benchmark" ] || { echo "缺 $AIDER/tmp.benchmarks/polyglot-benchmark" >&2; exit 2; }
docker image inspect aider-benchmark > /dev/null 2>&1 || { echo "镜像 aider-benchmark 没建" >&2; exit 2; }
if [ "$MODEL" = openai/ds4 ]; then
    # 隧道: Mac:$PORT → spark 127.0.0.1:8001(工作台的模型子进程直连口)
    curl -s --noproxy '*' -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" \
        || ssh -fN -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:8001" spark
    curl -s --noproxy '*' -m 10 -o /dev/null "http://127.0.0.1:$PORT/v1/models" || { echo "隧道 $PORT 不通" >&2; exit 2; }
    API_BASE="http://host.docker.internal:$PORT/v1"; API_KEY=ds4
    if [ -n "$RECORD" ]; then
        python3 gguf-tools/scripts/api_record_proxy.py --listen 18002 --upstream "http://127.0.0.1:$PORT" --out "$RECORD" & RPID=$!
        sleep 1; kill -0 "$RPID" 2>/dev/null || { echo "转发器没起来" >&2; exit 2; }
        API_BASE="http://host.docker.internal:18002/v1"; trap 'kill $RPID 2>/dev/null' EXIT
    fi
else
    API_BASE="https://api.deepseek.com/v1"; API_KEY=$(cat "$ROOT/gguf-tools/data/posttrain/deepseek_api.key")
fi
LOG="$AIDER/tmp.benchmarks/$TAG.log"
# 进过训练料的题跳过(gguf-tools/bench/data/trained_exclude.jsonl 里 task=aider 的行, bench_fail_to_task.py 写的): 没显式给 --keywords 时, 把其余全部题目列成 --keywords
EXF=$ROOT/gguf-tools/bench/data/trained_exclude.jsonl
if [ -s "$EXF" ] && [[ "$EXTRA" != *--keywords* ]]; then
    KW=$(cd "$AIDER/tmp.benchmarks/polyglot-benchmark" && ls -d */exercises/practice/* | grep -v -x -F -f <(python3 -c "import json,sys; [print(r['id']) for r in map(json.loads, open(sys.argv[1])) if r['task'] == 'aider']" "$EXF") | paste -sd, -)   # macOS 的 paste 不给文件名就报 usage(11-01 实撞: 题目表空, benchmark.py 收到裸 --keywords 秒退), 要显式写 -
    NEX=$(python3 -c "import json,sys; print(sum(1 for r in map(json.loads, open(sys.argv[1])) if r['task'] == 'aider'))" "$EXF")
    EXTRA="$EXTRA --keywords $KW"; echo "# 排除进过训练料的题 $NEX 道($EXF)" | tee -a "$LOG"
fi
# 第 7 个参数给温度(如 0.0)时, 从基础 yml 派生一份 settings_<标签>.yml 换掉 use_temperature; 不给就用 yml 里的 1.0
SETTINGS=aider_model_settings.yml
if [ -n "$TEMP" ]; then
    SETTINGS="settings_$TAG.yml"
    sed "s/use_temperature: 1.0/use_temperature: $TEMP/" "$ROOT/gguf-tools/bench/aider_model_settings.yml" > "$AIDER/tmp.benchmarks/$SETTINGS"
    cp "$AIDER/tmp.benchmarks/$SETTINGS" "$ROOT/gguf-tools/bench/$SETTINGS"   # 挂进容器的是 /settings = gguf-tools/bench
fi
echo "# $(date '+%F %T') 起跑 tag=$TAG 模型=$MODEL 温度=${TEMP:-1.0} 预算=${BUDGET}min 并发=$THREADS 题数=$NUM $EXTRA" | tee -a "$LOG"
# 代理环境全清空: 容器里 litellm 读 http_proxy 会把 host.docker.internal 的请求发到 Mac 的代理去
docker run --rm -i --memory=12g --memory-swap=12g --add-host=host.docker.internal:host-gateway \
    -v "$AIDER:/aider" -v "$AIDER/tmp.benchmarks:/benchmarks" -v "$ROOT/gguf-tools/bench:/settings:ro" \
    -e "OPENAI_API_KEY=$API_KEY" -e "OPENAI_API_BASE=$API_BASE" \
    -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e NO_PROXY=host.docker.internal \
    -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
    aider-benchmark bash -c "
        cd /aider && timeout -s INT -k 120 ${BUDGET}m ./benchmark/benchmark.py '$TAG' --new --model $MODEL --edit-format diff \
            --tries 1 --threads $THREADS --num-tests $NUM --exercises-dir polyglot-benchmark --read-model-settings /settings/$SETTINGS $EXTRA;
        echo '# 预算到/跑完, 汇总:'; ./benchmark/benchmark.py --stats '$TAG' --exercises-dir polyglot-benchmark" 2>&1 | tee -a "$LOG"
[ -n "$TEMP" ] && rm -f "$ROOT/gguf-tools/bench/$SETTINGS"   # 派生件留在 tmp.benchmarks 里存档, 不留在仓库目录
D=$(ls -d "$AIDER"/tmp.benchmarks/*--"$TAG" 2>/dev/null | tail -1)
[ -n "$D" ] && { n=$(find "$D" -name .aider.results.json | wc -l | tr -d ' '); echo "# $(date '+%F %T') 完成 $n 题, 目录 $D" | tee -a "$LOG"; }
