#!/usr/bin/env bash
# go2b_requant.sh — go2b 等体积重量化总控(2026-07-24, 用户漂移方案)。
# ★全阶段脚本化, 不私跑; 反修在链内(runbook 集成点 B: 反修在 go2b 前向消漂移)★。
# 权威计划: go2b_requant_runbook.md。
#
# ★无-NFS 分储(2026-07-24 用户裁决: 删旧模型腾空间, 不挂NFS, 量化直接在M1)★:
#   M1(HF全量本地) = 量化+SEARCH, 产物落规范 gguf/go-onebit/layers/, 逐层被 M4 拉走(滚动, M1峰值~5G)
#   M4(76G, 旧模型已删) = 全部层文件(冷dql热槽稀疏洞26.3G + go2b侧车17.5G) + 骨干HF子集(~≤15G)
#   反修(BF_ONLY)/rr_verdict 在 M4(回放只读骨干+dql/侧车字节, 不读专家FP — 已代码级核实)
#
# 链: quality(已过) → emit(M1量化, SEARCH含 z/GE/CE, 反修移后) → migrate(自动) → backbone(HF骨干子集→M4)
#     → backfit(M4, BF_ONLY终局收敛=全层反修在go2b合并态) → verdict(rr_verdict smin) → merge → validate
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
LDIR=$ROOT/gguf/go-onebit/layers                    # M4 侧层文件汇聚地(规范位置)
BACKBONE=$ROOT/hf/DeepSeek-V4-Flash-Backbone        # M4 侧骨干子集
HOTTAB_REL=gguf-tools/go-onebit/corpus/prog_active_top64.txt
CORPUS=${CORPUS:-/tmp/rr_calib_prog_v1.ids}
RR_FLOOR=${RR_FLOOR:-0.3610}   # 现役 v3 rr_code smin, go2b 须 > 此
log(){ echo "[go2b-requant] $*" >&2; }

case "${1:-all}" in
quality)
  "$HERE/go2b_layer_quality.sh" all
  RPT="$ROOT/gguf-tools/go-onebit/reports/go2b_layer_quality_$(date +%F).txt"
  [ -s "$RPT" ] || { log "质量表缺"; exit 1; }
  FAILS=$(grep -oE "失败层=\[[^]]*\]" "$RPT" | tail -1)
  echo "$FAILS" | grep -q "\[\]" || { log "★有层 go2b 不优于 stacked — 停★"; exit 3; }
  log "★逐层质量门全过 → 可进 emit★"
  ;;
sync)
  # 源码/脚本/热表 → M1 + 强制重建(M1 二进制必须含 go2b 侧车集成)
  log "同步量化器源码+脚本+热表 → M1 并重建"
  rsync -a "$ROOT/gguf-tools/go-onebit/quant/" "$M1:$M1DIR/gguf-tools/go-onebit/quant/" \
        --include='*.c' --include='*.h' --exclude='*' 2>/dev/null
  scp -q "$ROOT/quant_layer.sh" "$M1:$M1DIR/quant_layer.sh"
  scp -q "$ROOT/$HOTTAB_REL" "$M1:$M1DIR/$HOTTAB_REL"
  scp -q "$ROOT/gguf-tools/go-onebit/scripts/quant_verify.sh" "$M1:$M1DIR/gguf-tools/go-onebit/scripts/" 2>/dev/null || true
  ssh "$M1" "cd $M1DIR/gguf-tools/go-onebit/quant && cc -O3 -Wall -Wextra -Wno-unused-parameter -lm -framework Accelerate -I$M1DIR -o ./ds4quant_run ds4quant_run.c -lpthread" \
      || { log "M1 构建失败"; exit 1; }
  log "M1 构建 ✓"
  ;;
emit)
  # M1 满档量化(SEARCH 含 z/GE/CE 逐层调优; DS4_BACKFIT_INCR=0+GSWEEP=0 → 反修整体移到 M4 BF_ONLY 阶段,
  # 因层文件滚动迁移不驻 M1)。每层 GO2B_GATE(重建 cos 门) 实时可见 = 每一层输出质量。
  M1FREE=$(ssh "$M1" "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'")
  M4FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  log "盘账: M1=${M1FREE}G(需≥8, 滚动峰值~5G) M4=${M4FREE}G(需≥46)"
  [ "$M1FREE" -ge 8 ] || { log "★M1 盘不足★"; exit 6; }
  [ "$M4FREE" -ge 46 ] || { log "★M4 盘不足★"; exit 6; }
  "$0" sync || exit $?
  ssh "$M1" "[ -f $CORPUS ]" || { scp -q "$CORPUS" "$M1:$CORPUS" 2>/dev/null || { log "语料 $CORPUS 缺"; exit 2; }; }
  ssh "$M1" "rm -f $M1DIR/gguf/go-onebit/layers/dql_go2b_L*.bin 2>/dev/null; true"   # 侧车遗留(quant_layer 只清 dql/opt)
  mkdir -p "$LDIR"
  log "起飞: M1 满档量化(nohup, 日志 M1:/tmp/go2b_emit_m1.log)"
  ssh "$M1" "cd $M1DIR && nohup env DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE=$M1DIR/$HOTTAB_REL \
      DS4_CORPUS=$CORPUS DS4_NTOK=${DS4_NTOK:-530} DS4_SKIP_MERGE=1 \
      DS4_BACKFIT_INCR=0 DS4_GSWEEP=0 \
      ./quant_layer.sh > /tmp/go2b_emit_m1.log 2>&1 < /dev/null & echo EMIT_PID=\$!" < /dev/null
  ( nohup "$HERE/go2b_migrate_watch.sh" > /tmp/go2b_migrate.log 2>&1 < /dev/null & )
  log "搬运看守已起(M4, /tmp/go2b_migrate.log)。质量跟踪: ssh $M1 grep -a GO2B_GATE /tmp/go2b_emit_m1.log"
  ;;
backbone)
  # HF 骨干子集(剔 routed 专家) → M4:$BACKBONE (BF_ONLY/verdict 回放用)
  [ -d "$BACKBONE" ] && [ -f "$BACKBONE/model.safetensors.index.json" ] && { log "骨干子集已在 ✓"; exit 0; }
  scp -q "$HERE/go2b_backbone_extract.py" "$M1:/tmp/"
  log "M1 抽取骨干子集(纯字节拷贝) → /tmp/backbone_sub"
  ssh "$M1" "cd $M1DIR && DS4_HF=$M1DIR/hf/DeepSeek-V4-Flash-Base python3 /tmp/go2b_backbone_extract.py /tmp/backbone_sub" || { log "抽取失败"; exit 1; }
  mkdir -p "$BACKBONE"
  log "拉取子集 → $BACKBONE"
  rsync -a --progress "$M1:/tmp/backbone_sub/" "$BACKBONE/" || { log "拉取失败"; exit 1; }
  ssh "$M1" "rm -rf /tmp/backbone_sub"
  log "骨干子集就位: $(du -sh "$BACKBONE" | cut -f1)"
  ;;
backfit)
  # ★全层反修在 go2b 合并态(用户方案核心)★: BF_ONLY 终局收敛 sweep, 热专家经侧车回放。
  # ★铁律修正(2026-07-24 M4 死机事故)★: 反修=全链最吃内存阶段(HQE+锚+逐层fp16缓存爬升),
  # 必须跑在盘余量充足(swap 有地方长)的 M1 — M4 盘 3-4G 时 swap 饿死 → 内核 panic 实证。
  # 层文件先传回 M1 规范 layers/(M4 留双份备份); 收紧 DS4_BF_MEMGB=8 + 外部看门狗 10.5G。
  N=$(ssh "$M1" "ls $M1DIR/gguf/go-onebit/layers/dql_L*.bin 2>/dev/null | wc -l" | tr -d ' ')
  NG=$(ssh "$M1" "ls $M1DIR/gguf/go-onebit/layers/dql_go2b_L*.bin 2>/dev/null | wc -l" | tr -d ' ')
  [ "$N" -ge 43 ] && [ "$NG" -ge 43 ] || { log "M1 层文件不齐 dql=$N go2b=$NG (先把 M4 layers 传回 M1)"; exit 3; }
  M1FREE=$(ssh "$M1" "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'")
  [ "$M1FREE" -ge 12 ] || { log "★M1 盘余 ${M1FREE}G <12G — swap 饿死风险, 拒跑★"; exit 6; }
  ssh "$M1" "[ -f $CORPUS ]" || scp -q "$CORPUS" "$M1:$CORPUS"
  log "M1 反修(BF_ONLY 合并态, MEMGB=8 + 外部看门狗 10.5G, nohup 日志 M1:/tmp/go2b_backfit_m1.log)"
  ssh "$M1" "cd $M1DIR && nohup env DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE=$M1DIR/$HOTTAB_REL \
      DS4_CORPUS=$CORPUS DS4_NTOK=${DS4_NTOK:-530} DS4_SKIP_MERGE=1 DS4_BF_JUSTIFIED=1 DS4_BF_MEMGB=8 \
      ./quant_layer.sh backfit > /tmp/go2b_backfit_m1.log 2>&1 < /dev/null & echo BF_PID=\$!" < /dev/null
  ssh "$M1" "nohup bash -c 'while :; do P=\$(pgrep -nf ds4quant_run || true); [ -n \"\$P\" ] || { sleep 5; continue; }; \
      MB=\$(footprint -p \$P 2>/dev/null | grep -Eo \"Footprint: *[0-9.]+ *[KMG]B\" | head -1 | awk \"{v=\\\$2;u=\\\$3; if(u==\\\"GB\\\")v*=1024; else if(u==\\\"KB\\\")v/=1024; printf \\\"%d\\\",v}\"); \
      [ -n \"\$MB\" ] && [ \"\$MB\" -gt 10752 ] && { echo \"[wdog] \${MB}MB>10.5G kill\" >> /tmp/go2b_backfit_wdog.log; kill -9 \$P; }; sleep 5; done' >/dev/null 2>&1 < /dev/null & echo WDOG_PID=\$!" < /dev/null
  ;;
verdict)
  # rr_verdict teacher-forced smin(分钟级) vs 现役 0.3610 — M1 跑(层文件+HF 同机; 反修同因死机教训)。
  RRANCH=/tmp/ds4quant_anchor_rr_rr_code_s305.bin
  ssh "$M1" "[ -f $RRANCH ]" || { [ -f "$RRANCH" ] && scp -q "$RRANCH" "$M1:$RRANCH" || { log "rr锚两机都缺 — 停"; exit 3; }; }
  ssh "$M1" "cd $M1DIR && DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE=$M1DIR/$HOTTAB_REL \
      bash gguf-tools/go-onebit/scripts/rr_verdict.sh /tmp/rr_code.ids 305" | tee /tmp/go2b_verdict_summary.txt
  log "★判据: smin 显著 > $RR_FLOOR = go2b 兑现漂移收益★"
  ;;
merge)
  log "merge 等体积(稀疏冷base+热go2b)。前置=引擎稀疏base(runbook D)。M4 已有空间(旧模型已删)。"
  [ "${DS4_GO2B_SPARSE_BASE_READY:-0}" = 1 ] || { log "★引擎稀疏base未实现 — 拒, 见 runbook §2D★"; exit 4; }
  DS4_HF="$BACKBONE" "$ROOT/quant_layer.sh" merge
  ;;
validate)
  "$HERE/prog_sweep.sh"
  ;;
all)
  "$0" emit || exit $?
  log "emit 已后台。后续: 等 migrate 完 → $0 backbone && $0 backfit && $0 verdict"
  ;;
*) log "用法: quality|sync|emit|backbone|backfit|verdict|merge|validate|all"; exit 2 ;;
esac
