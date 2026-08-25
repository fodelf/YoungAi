#!/usr/bin/env bash
# spark_merge_and_bench.sh — 在 DGX Spark(Linux aarch64 + GB10) 上把 en86 分层量化产物
# 合并成可跑的 GGUF, 然后跑 HumanEval 164 题全套。
#
#   bash gguf-tools/go-onebit/scripts/spark_merge_and_bench.sh skeleton   # 只抽骨架
#   bash .../spark_merge_and_bench.sh merge                               # 只合并
#   bash .../spark_merge_and_bench.sh bench [humaneval|humaneval-x-go]    # 只跑基准
#   bash .../spark_merge_and_bench.sh all                                 # 全流程
#
# 为什么单独一个脚本而不是直接用 r30_campaign.sh: 那个是**战役**驱动(锚/plan/量化/反修/
# 学生/合并/指标 全链), 这里只要它的 merge 一段, 且 Spark 上没有锚也不需要重量化。
# 合并本身仍然调它同一套工具(vq_merge_v4.c / dql_down_offset.py / skel_from_hf.sh),
# 没有另起炉灶。
set -uo pipefail

ROOT="${ROOT:-$HOME/ds4-main}"
SC="$ROOT/gguf-tools/go-onebit/scripts"
. "$SC/_portable.sh"

R30="$ROOT/gguf/go-onebit/r30"
OUTF="${OUTF:-$R30/en86}"                 # 战役目录: layers/ 在这下面
LAYERS="$OUTF/layers"
SKEL="${SKEL:-$R30/r30_skeleton.gguf}"
MDL="${MDL:-$ROOT/gguf/ds4-en86.gguf}"
NL="${NL:-43}"                            # 层数, 与 manifest 行数一致
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
export PATH="$HOME/opt/go/bin:$PATH"      # Go 判定器(基准阶段用)

LOG(){ echo "[spark $(date +%H:%M:%S)] $*" >&2; }

do_skeleton(){
    [ -f "$SKEL" ] && { LOG "骨架已在: $SKEL ($(file_apparent_bytes "$SKEL") B)"; return 0; }
    [ -d "$DS4_HF" ] || { LOG "★HF 原始缺: $DS4_HF — 还没传完?★"; exit 2; }
    # 阈值放开到 Spark 档: 121G 内存 / 3.5T 盘, 16G Mac 的 6G 内存闸会误杀多线程量化
    LOG "抽骨架(0731 → --experts-hole → 紧凑骨架), 线程=${SKEL_TH:-16}"
    SKEL_WDOG_MB="${SKEL_WDOG_MB:-40000}" \
    SKEL_FREE_GB="${SKEL_FREE_GB:-40}" \
    SKEL_WDOG_FREE_GB="${SKEL_WDOG_FREE_GB:-20}" \
        bash "$SC/skel_from_hf.sh" "$SKEL" "${SKEL_TH:-16}" || { LOG "★骨架失败★"; exit 3; }
}

do_merge(){
    [ -f "$LAYERS/manifest.txt" ] || { LOG "★manifest 缺: $LAYERS — 层件还没传完?★"; exit 4; }
    [ -f "$SKEL" ] || { LOG "★骨架缺, 先跑 skeleton 段★"; exit 4; }
    local n_vq; n_vq=$(ls "$LAYERS" | grep -c "^dql_vq_L") || true
    LOG "层件: manifest $(wc -l < "$LAYERS/manifest.txt") 行 / dql_vq $n_vq 个"
    [ "$n_vq" -ge "$NL" ] || { LOG "★dql_vq 只有 $n_vq < $NL — 传输没完★"; exit 4; }
    # 真账(manifest 实算): blob 70.149 + down 11.42 + 骨架 8.202 = 89.77 GiB = 96.39 GB
    local free; free=$(disk_free_gb "$(dirname "$MDL")")
    LOG "输出盘可用 ${free} GiB (合并产物 ~89.8 GiB / 96.4 GB)"
    [ "${free:-0}" -ge 100 ] || { LOG "★盘不足 100G 停★"; exit 7; }

    LOG "生成 down 偏移表"
    python3 "$SC/dql_down_offset.py" "$LAYERS" "$NL" > "$OUTF/down_offsets.txt" \
        || { LOG "★down 偏移表失败★"; exit 5; }

    LOG "起合并(VQ 冠军形态: blob+down+骨架, 非消费式=源全保留)"
    rm -f "$MDL"
    "$(dirname "$0")/../quant/vq_merge_v4" --merge \
        --skeleton "$SKEL" \
        --blob-sizes "$LAYERS/manifest.txt" \
        --down-offsets "$OUTF/down_offsets.txt" \
        --dql-host 127.0.0.1 --dql-dir "$LAYERS" \
        --out "$MDL" 2>&1 | tail -20
    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] || { LOG "★合并失败 rc=$rc★"; exit $rc; }
    LOG "合并 ✓ $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}') — 口径=Finder十进制GB"
}

do_bench(){
    local suite="${1:-humaneval}"
    [ -f "$MDL" ] || { LOG "★模型缺: $MDL — 先合并★"; exit 6; }
    cd "$ROOT" || exit 1
    # server 单实例: 起之前先确认没有别的巨模型进程(项目铁律: 不并发跑大模型)
    pgrep -f "ds4-server" >/dev/null && { LOG "已有 ds4-server 在跑, 复用"; } || {
        LOG "起 ds4-server (--ctx 8192, CUDA)"
        nohup ./ds4-server -m "$MDL" --ctx 8192 > /tmp/ds4_server.log 2>&1 &
        for i in $(seq 1 180); do
            curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
            sleep 5
        done
    }
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
        || { LOG "★server 没起来, 看 /tmp/ds4_server.log★"; tail -20 /tmp/ds4_server.log >&2; exit 6; }

    LOG "跑 $suite 全 164 题"
    cd "$ROOT/gguf-tools/go-onebit" || exit 1
    PUBBENCH_CACHE="$PWD/pubbench_data" "$(dirname "$0")/../calib/pubbench" \
        --suite "$suite" --limit 164 --tag "${TAG:-en86}" --api completions
}

case "${1:-all}" in
    skeleton) do_skeleton ;;
    merge)    do_merge ;;
    bench)    do_bench "${2:-humaneval}" ;;
    all)      do_skeleton && do_merge && do_bench humaneval ;;
    *) echo "用法: $0 [skeleton|merge|bench <suite>|all]" >&2; exit 1 ;;
esac
