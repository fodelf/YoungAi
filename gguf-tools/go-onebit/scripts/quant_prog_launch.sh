#!/bin/bash
# quant_prog_launch.sh — 编程全域锚满档量化夜跑启动器(2026-07-20 域放大第三刀)。
# 纪律: ①长跑前机制审计(2层 fast 探针绿才放满档) ②S=530 独占锚名(历史 S 集合
# {16,64,128,305,512}, 同名锚静默复用=实验污染; 530 且正好罩住 526tok 代码区)
# ③DS4_SKIP_MERGE=1 判决停点(M1 盘 31G < 45.6G 合一 GGUF; rr 裁判在 dql 消费前)
# ④nohup+setpgrp 自成进程组(07-17 连带杀教训) ⑤quant_layer.sh 自带看门狗+盘感知。
# 用法: ./quant_prog_launch.sh probe   # 只跑 2 层机制探针
#       ./quant_prog_launch.sh launch  # 探针绿之后满档起跑(后台, 日志 M1:/tmp/quant_prog.launch.log)
set -uo pipefail
M1="${M1:-192.168.1.2}"
M1DIR="${M1DIR:-/Users/fodelf/ds4-main}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STEP="${1:-probe}"

sync_inputs() {
  scp -q "$ROOT/quant_layer.sh" "$M1:$M1DIR/quant_layer.sh"
  scp -q "$ROOT/gguf-tools/go-onebit/scripts/rr_verdict.sh" "$M1:$M1DIR/gguf-tools/go-onebit/scripts/rr_verdict.sh"
  # C 源也同步(2026-07-21 判决前置改动在 ds4quant_run.c; M1 侧 quant_layer M1构建段重编)
  scp -q "$ROOT/gguf-tools/go-onebit/quant/ds4quant_run.c" "$M1:$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.c"
  scp -q /tmp/rr_calib_prog_v1.ids /tmp/rr_code.ids /tmp/rr_hard.ids "$M1:/tmp/"
  echo "[launch] 输入已同步(quant_layer.sh + rr_verdict.sh + ds4quant_run.c + 3 份 ids)" >&2
}

case "$STEP" in
probe)
  sync_inputs
  echo "[launch] 2层 fast 机制探针起跑(分钟级)…" >&2
  ssh "$M1" "cd $M1DIR && DS4_FAST_LAYERS=2 DS4_CORPUS=/tmp/rr_calib_prog_v1.ids ./quant_layer.sh fast" \
      > /tmp/quant_prog_probe.log 2>&1 || true   # L2 fast 尾段 backfit 闸拒跑(exit 3)=预期
  # 探针判据: 2 层 dql+opt 在位 + TABREC 行出现
  OK=1
  ssh "$M1" "ls $M1DIR/gguf/go-onebit/layers/dql_L00.bin $M1DIR/gguf/go-onebit/layers/dql_L01.bin" >/dev/null 2>&1 || OK=0
  grep -q "TABREC L=1" /tmp/quant_prog_probe.log || ssh "$M1" "grep -q 'TABREC L=1' /tmp/quant_all.out" 2>/dev/null || OK=0
  if [ "$OK" = 1 ]; then echo "[launch] ★探针绿★ 2层 dql/opt 在位+TABREC 出行 → 可 launch" >&2
  else echo "[launch] ✗探针失败 — 看 /tmp/quant_prog_probe.log 尾部:" >&2; tail -8 /tmp/quant_prog_probe.log >&2; exit 1; fi
  ;;
launch)
  # CORPUS_IDS/NTOK 可覆盖(v2 起参数化; S 必须历史未用过=锚名独占, 见铁律③)
  CIDS="${CORPUS_IDS:-/tmp/rr_calib_prog_v1.ids}"
  NT="${NTOK:-530}"
  sync_inputs
  scp -q "$CIDS" "$M1:/tmp/" 2>/dev/null || true
  echo "[launch] 满档起跑: 语料=$CIDS S=$NT 独占锚 + DS4_SKIP_MERGE=1 判决停点" >&2
  # EXTRA_ENV 透传口(2026-07-22): 如 "DS4_BF_TERM_MAXP=0 DS4_BWD=0" 关终端反修(行为回退假设检验)
  ssh "$M1" "cd $M1DIR && rm -f /tmp/quant_prog.launch.log && ( \
    DS4_CORPUS=/tmp/$(basename "$CIDS") DS4_NTOK=$NT DS4_SKIP_MERGE=1 ${EXTRA_ENV:-} \
    nohup perl -e 'setpgrp(0,0); exec @ARGV or die \$!' ./quant_layer.sh \
    > /tmp/quant_prog.launch.log 2>&1 < /dev/null & ) && echo started"
  echo "[launch] 已后台; 进度: ssh $M1 tail -f /tmp/quant_prog.launch.log" >&2
  ;;
merge)
  # 手动合并 + ★行为门入环(2026-07-22 用户三修正之一: done=merge+行为门, 坏模型不隔天暴露)★
  # 链条(M1 本地 nohup 自洽, 不依赖发起会话): quant_layer merge → MERGE_GGUF 完成 →
  # 单机 server(chunk512 单机铁律) → behavior_gate_at 4针 → 报告落 /tmp/merge_gate.report → server 收。
  # 交接铁律注: 行为门绿之前不删任何前代模型(检查清单, 非本脚本自动删)。
  ssh "$M1" "cd $M1DIR && rm -f /tmp/quant_prog.merge.log /tmp/merge_gate.report && ( nohup perl -e 'setpgrp(0,0); exec @ARGV or die \$!' bash -c '
    ./quant_layer.sh merge > /tmp/quant_prog.merge.log 2>&1
    grep -q \"MERGE_GGUF.*完成\" /tmp/quant_prog.merge.log || { echo merge未完成 > /tmp/merge_gate.report; exit 1; }
    rm -f /tmp/ds4-svc-m1.log /tmp/ds4-kv-m1/*.kv
    DS4_RESIDUAL=gguf/sidecars/code-hot-res-v3p.gguf DS4_MEM_BUDGET_MB=12000 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_EXPERT_PREAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_METAL_NO_MODEL_WARMUP=1 DS4_PRIMER_BATCH_INJECT=1 DS4_PRIMER_FREE_BUDGET=96 DS4_BASE_NATIVE=1 nohup ./ds4-server -m gguf/go-onebit/ds4-code1b.gguf -c 32768 --port 8013 --kv-disk-dir /tmp/ds4-kv-m1 --kv-disk-space-mb 4096 --max-output-tokens 128 --nothink --tool-primer --soul gguf-tools/go-onebit/corpus/soul/soul_server_v3.txt > /tmp/ds4-svc-m1.log 2>&1 &
    SPID=\$!
    until grep -q listening /tmp/ds4-svc-m1.log 2>/dev/null; do sleep 3; done
    bash gguf-tools/go-onebit/scripts/behavior_gate_at.sh > /tmp/bg_at.log 2>&1
    cp /tmp/behavior_gate_at.report /tmp/merge_gate.report
    kill \$SPID 2>/dev/null
  ' > /tmp/merge_chain.log 2>&1 & ) && echo started"
  echo "[launch] merge+行为门链已后台; 终态: ssh $M1 cat /tmp/merge_gate.report" >&2
  ;;
sentinel)
  # GSWEEP 拦截哨(07-16 配方可比裁决同款): v2/v3p/s512 基线全没跑过 GSWEEP, 且回扫
  # 全缓存路径有 12G OOM 前科 → 全局回扫一露头就杀 ds4quant_run。之后 quant_layer.sh
  # 异常分支自动接管: dql 43/43 → rr 双判决 → MODE=merge → M4.4 SKIP_MERGE 停点。
  # 在 M1 本地跑(不依赖发起会话存活); bash 3.2 安全(无嵌套转义)。
  while true; do
    if grep -aq "全局回扫" /tmp/quant_all.log 2>/dev/null \
       || grep -aq "GSWEEP 起点" /tmp/quant_all.out 2>/dev/null; then
      pkill -f ds4quant_run
      echo "$(date) GSWEEP 起点截停(配方可比+OOM 前科)" >> /tmp/gsweep_sentinel.log
      exit 0
    fi
    pgrep -f 'quant_layer.sh' >/dev/null || exit 0   # 主跑已结束 → 哨兵自终
    sleep 10
  done
  ;;
full)
  # ★全链一键(2026-07-22 用户令: 正向操作链式调用)★:
  # probe(2层机制审计) → launch(满档基线, 反修默认关) → 等判决停点 → merge+行为门 →
  # backfit_decide(指标+真实场景自动裁决)。每段失败即停, 全程无人值守可跑。
  "$0" probe || exit 1
  "$0" launch
  echo "[full] 等判决停点(满档 ~4-6h + 冷锚可能 +4h)…" >&2
  until ssh "$M1" "grep -q '判决停点' /tmp/quant_prog.launch.log 2>/dev/null && ! pgrep -f quant_layer.sh >/dev/null" 2>/dev/null; do sleep 300; done
  echo "[full] 停点到 → merge+行为门链" >&2
  "$0" merge
  until ssh "$M1" "[ -s /tmp/merge_gate.report ]" 2>/dev/null; do sleep 120; done
  echo "[full] 行为门报告到 → 反修自动裁决" >&2
  bash "$ROOT/gguf-tools/go-onebit/scripts/backfit_decide.sh" /tmp/quant_prog.launch.log /tmp/merge_gate.report
  echo "[full] ★全链完成★ 后续: prog_sweep.sh(43针全扫) + agent_loop_probe.sh(两轮回路)" >&2
  ;;
*) echo "用法: $0 probe|launch|merge|sentinel|full" >&2; exit 2 ;;
esac
