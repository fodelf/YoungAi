#!/bin/bash
# r29_bench.sh — R29 双机流水线加载 + 40 题基准 + 报告(2026-08-01 用户令)。
#
# 前置: M1 已产出 gguf/go-onebit/ds4-r29.gguf(量化+反修+合并+路由烘焙完成)。
# 步骤: ①模型摆渡到 M4(双机各持一份, 层切片各读各的) ②svc.sh 起双机 lane
#       ③pubbench_serial 逐题隔离跑 HumanEval-Py 20 + Go 20 ④汇总报告
#
# 逐题隔离的原因(pubbench_serial.sh 原注): 常驻 server 跨请求 live-KV rewind 在分布式
# VQ lane 断路(worker KV prefix hash mismatch → 全量 rebuild → 路由挂死), 根修归 v2.3。
# 每题重启 server = fresh session 全量 prefill, 零 rewind。
#
# 用法: r29_bench.sh [ferry|serve|bench|report|all]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
REL=gguf/go-onebit/ds4-r29.gguf
MDL="$ROOT/$REL"
TAG="${TAG:-r29}"
RPT="$ROOT/gguf-tools/reports/pubbench/REPORT_${TAG}_$(date +%Y-%m-%d).md"
LOG(){ echo "[bench $(date +%H:%M:%S)] $*" >&2; }

stage_ferry(){
    # 侧车先行(几十 MB, 比模型快): 双机各跑各的层, 两台都要 zchain
    mkdir -p "$ROOT/gguf/go-onebit/r29/full"
    scp -q $M1:$M1DIR/gguf/go-onebit/r29/full/zchain.bin "$ROOT/gguf/go-onebit/r29/full/" 2>/dev/null \
        && LOG "zchain 侧车已摆渡" || LOG "★zchain 摆渡失败(反修可能还没产出)★"
    SZ=$(ssh $M1 "stat -f %z $M1DIR/$REL 2>/dev/null" || echo 0)
    [ "$SZ" -gt 0 ] || { LOG "M1 上模型不存在, 停"; exit 2; }
    LOG "M1 模型 $(awk -v s="$SZ" 'BEGIN{printf "%.2f",s/2^30}') GiB"
    if [ -f "$MDL" ] && [ "$(stat -f %z "$MDL")" = "$SZ" ]; then
        LOG "M4 已有同尺寸副本, 跳过摆渡"; return 0
    fi
    FREE=$(df -g "$ROOT" | awk 'NR==2{print $4}')
    NEED=$(awk -v s="$SZ" 'BEGIN{printf "%d",s/2^30+2}')
    [ "$FREE" -ge "$NEED" ] || { LOG "★M4 free ${FREE}G < 需 ${NEED}G 停★"; exit 6; }
    LOG "摆渡 M1 → M4(Thunderbolt 直连)"
    scp "$M1:$M1DIR/$REL" "$MDL" || { LOG "摆渡失败"; exit 3; }
    [ "$(stat -f %z "$MDL")" = "$SZ" ] || { LOG "★尺寸不符, 摆渡损坏★"; exit 4; }
    LOG "摆渡 ✓ 尺寸一致"
}

stage_serve(){
    # RESID 显式空: R28v2 是独立量化世代, v3 主线的 prog 残差侧车与之不同代, 挂上=错配
    # CORR 空: 后训练侧车尚未产出(有了再挂)
    # ★zchain 反修侧车必须挂★: R28v2 分离实验实测 —— 挂 zchain 时前段连贯, 不挂立刻字节级
    # 乱码。侧车是独立文件(没内嵌进 GGUF), 且双机各跑各的层, **两台都要有这个文件**。
    ZC="gguf/go-onebit/r29/full/zchain.bin"
    [ -f "$ROOT/$ZC" ] || { LOG "★zchain 侧车不在 M4($ZC) — 先跑 ferry★"; exit 3; }
    ssh $M1 "[ -f $M1DIR/$ZC ]" || { LOG "★zchain 侧车不在 M1 — 反修产物缺★"; exit 3; }
    LOG "起双机 lane(MODEL=$REL, zchain=$ZC)"
    MODEL="$REL" RESID= CORR= CTX="${CTX:-8192}" \
        EXTRA_ENV="DS4_ZCHAIN=$ZC" "$ROOT/tools/svc.sh" up 2>&1 | tail -5 >&2
    sleep 5
    curl -s -m 10 "http://127.0.0.1:${PORT:-8013}/v1/models" >/dev/null 2>&1 \
        && LOG "server 就绪 ✓" || LOG "★server 未响应(基准仍会逐题重启, 继续)★"
}

stage_bench(){
    for SUITE in humaneval humaneval-x-go; do
        LOG "跑 $SUITE 20 题(逐题隔离)"
        SUITE="$SUITE" TAG="$TAG" N=20 "$ROOT/gguf-tools/scripts/pubbench_serial.sh" \
            > "/tmp/r28v2_bench_${SUITE}.log" 2>&1
        LOG "$SUITE rc=$? → reports/pubbench/pubbench_${SUITE}_${TAG}.jsonl"
    done
}

stage_report(){
    OUT="$ROOT/gguf-tools/reports/pubbench"
    {
        echo "# R28 v2 基准报告 ($(date '+%Y-%m-%d %H:%M'))"
        echo
        echo "## 模型"
        echo '```'
        ls -l "$MDL" 2>/dev/null | awk '{printf "%-50s %.3f GiB\n",$9,$5/2^30}'
        echo '```'
        echo
        echo "## 体积账(量化器 manifest, 事前=事后)"
        echo '```'
        MAN="$ROOT/gguf/go-onebit/r28v2/full/layers/manifest.txt"
        ssh $M1 "cat $M1DIR/gguf/go-onebit/r28v2/full/layers/manifest.txt" 2>/dev/null \
            | awk '{s+=$2; n++} END{printf "%d 层 载荷 %.3f GiB + backbone 8.202 = %.3f GiB\n",n,s/2^30,s/2^30+8.202}'
        echo '```'
        echo
        for SUITE in humaneval humaneval-x-go; do
            J="$OUT/pubbench_${SUITE}_${TAG}.jsonl"
            echo "## $SUITE"
            if [ -f "$J" ]; then
                python3 - "$J" <<'PY'
import json,sys
p=sys.argv[1]; ok=n=0
for line in open(p):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    n+=1
    if d.get("passed") or d.get("pass") or d.get("ok"): ok+=1
print(f"通过 {ok}/{n}" if n else "无有效记录")
PY
            else
                echo "(无结果文件)"
            fi
            echo
        done
    } > "$RPT"
    LOG "报告 → $RPT"
    cat "$RPT" >&2
}

case "${1:-all}" in
    ferry)  stage_ferry ;;
    serve)  stage_serve ;;
    bench)  stage_bench ;;
    report) stage_report ;;
    all)    stage_ferry && stage_serve && stage_bench; stage_report ;;
    *) echo "用法: r29_bench.sh [ferry|serve|bench|report|all]" >&2; exit 1 ;;
esac
