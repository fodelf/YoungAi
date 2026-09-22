#!/bin/bash
# bugmd_gates_spark.sh — bug.md 的门与对拍, spark 本机跑(2026-09-21)。每个阶段都是分钟级; 日志全落 /tmp/bugmd/。
#
# 前提: ds4-server 已停(GPU 只有一份模型); 三份二进制在仓库根:
#   ds4.base.prefix     修 CED 之前的(E0 版, 23:00), 只给门当基线
#   ds4                 当前树(CED 两处修复, --use_fast_math)
#   ds4.base.nofastmath 同一棵树去掉 --use_fast_math 重编的
# 用法: bugmd_gates_spark.sh prep|ced|fast|parity
#   prep    从 E0 trace 抽 002490 的渲染提示与 ids(bugmd_ids_tools.py extract), 造短尾提示
#   ced     §6 门: G1 --decoder-full 2K 64 步 prefix vs 新 逐字节; G2 002490 提示: CED(老/新) vs decoder-full 的 [emit] 首分歧;
#           G3 短尾提示(token 数 mod 512 < 128): 同 G2
#   fast    §5.2: wt2 512 引擎 vs 文件态学生, 新(fast-math) vs nofastmath 五指标
#   parity  §5.4(a): 002490 序列 --score-ids top-5(全 40 层预填路) vs 解码路真吐 id
#   full <提示文件> [生成上限; 不给=跑到 ctx 边界]   ★端到端 CED vs --decoder-full★(2026-09-22): 同一二进制、同一条真实提示, 温 0 各生成一整趟,
#            比"停没停(EOS)/复读段/尾部". CED 是我们为提速选的近似(预填只跑一半层), 这一档回答"它有没有把产品带进死循环"。
#            单趟 = 预填(CED 65 s / 精确 128 s) + 上限/24 t/s, 两趟约 26 min。
# 出错会怎样: 二进制不在 = 直接退; 看门狗 available < 8 GB 杀 ds4(与 v41_engine_parity_spark.sh 同一条线)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
STAGE="${1:?prep|ced|fast|parity|full|nograph|statecmp|stopgate}"
MODEL="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
AMP="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine"
OUT=/tmp/bugmd; mkdir -p "$OUT"
# ★上下文只有 1M 一个取值, 脚本里不许出现任何上下文数★(用户 2026-09-22 "不要任何写死的上下文"): 引擎从模型元数据
# deepseek4.context_length 读, --ctx 已不存在(传了直接拒)。生成上限同理不编数: 不传 -n = 不设上限(引擎默认, 生成到 EOS 或
# 上下文边界); 要复现某个产品配置才显式传上限。
# 实撞代价: 09-22 全天 CFO 探针带着 16384 上限(再加 ctx 写死 32768 ⇒ 提示 14k 时最多生成 18.6k), 每趟"顶格",
# 被我判成"写完不停 + 整篇复读"; 不设上限一跑, 23,607 位模型自己吐 EOS, 前半截与带上限那趟逐字相同。
TOOLS="$ROOT/speed-bench/bugmd_ids_tools.py"
TRACE=/tmp/ds4-trace-e0.txt
LOG(){ echo "[bugmd $(date '+%H:%M:%S')] $*"; }
watchdog(){ while sleep 10; do av=$(free -g | awk '/^内存|^Mem/{print $7}'); if [ "${av:-99}" -lt 8 ]; then echo "★看门狗: available ${av}G < 8G, 停车★" | tee -a "$OUT/mem.log"; pkill -f "ds4[.a-z]* -m .*gguf/v41[/]"; return 1; fi; done; }
watchdog & WD=$!; trap 'kill $WD 2>/dev/null' EXIT
pgrep -x ds4-server >/dev/null && { LOG "★ds4-server 还在跑, 先停★"; exit 1; }

# $1=二进制 $2=标签 $3..=其余参数; 生成路与 d1_kv_ring_gate.sh 同一套旗(温 0 seed 1, 不投机)。
run(){ local bin=$1 tag=$2; shift 2
  [ -x "./$bin" ] || { LOG "★没有 ./$bin★"; return 1; }
  "./$bin" -m "$MODEL" --zchain "$AMP" --cuda --temp 0 --seed 1 --no-dspark "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?; LOG "$tag: rc=$rc $(grep -a -h 'prefill .* token' "$OUT/$tag.err" | head -1) $(grep -a -h '★' "$OUT/$tag.err" | head -2 | tr '\n' ' ')"; return $rc; }

case "$STAGE" in
prep)
  python3 "$TOOLS" extract "$TRACE" last "$OUT/002490" || exit 2   # 最后一个请求 = 002490(前面只有冒烟)
  # 短尾提示: 提示 13710 token ≈ 22800 字 ⇒ 0.6 token/字; 要 N mod 512 ∈ [1,127] 即 N ∈ [13313,13439], 砍 271~397 token ≈ 450~660 字
  for cut in 550 450 650; do python3 "$TOOLS" cut "$OUT/002490.prompt.txt" "$cut" "$OUT/short_$cut.txt"; done
  for b in ds4.base.prefix ds4 ds4.base.nofastmath; do ls -la "$b" | awk '{print $5, $6, $7, $8, $9}'; done ;;
ced)
  P2K="$OUT/p2k.txt"; head -c 8000 speed-bench/readme_en_x80.txt > "$P2K"
  LOG "== G1: --decoder-full 2K 64 步, prefix vs 新 逐字节"
  run ds4.base.prefix g1_prefix -n 64 --decoder-full --emit-trace --prompt-file "$P2K"
  run ds4             g1_new    -n 64 --decoder-full --emit-trace --prompt-file "$P2K"
  if cmp -s "$OUT/g1_prefix.out" "$OUT/g1_new.out"; then LOG "G1 逐字节同 ✓"; else LOG "★G1 不同★ $(cmp "$OUT/g1_prefix.out" "$OUT/g1_new.out" 2>&1 | head -1)"; fi
  LOG "== G2: 002490 提示(13710 token, mod 512 = 398): CED 老 / CED 新 / decoder-full, 256 步 [emit]"
  run ds4.base.prefix g2_ced_old  -n 256 --emit-trace --prompt-file "$OUT/002490.prompt.txt"
  run ds4             g2_ced_new  -n 256 --emit-trace --prompt-file "$OUT/002490.prompt.txt"
  run ds4             g2_full     -n 256 --emit-trace --decoder-full --prompt-file "$OUT/002490.prompt.txt"
  LOG "老 CED vs decoder-full: $(python3 "$TOOLS" cmp-emit "$OUT/g2_ced_old.err" "$OUT/g2_full.err")"
  LOG "新 CED vs decoder-full: $(python3 "$TOOLS" cmp-emit "$OUT/g2_ced_new.err" "$OUT/g2_full.err")"
  LOG "老 CED vs 新 CED:       $(python3 "$TOOLS" cmp-emit "$OUT/g2_ced_old.err" "$OUT/g2_ced_new.err")"
  LOG "== G3: 短尾提示(找 N mod 512 < 128 的那条)"
  # 实撞 09-21: 尾巴是中文, 砍 550 字只少 183 token(0.33 token/字), 要砍 ~900 字才落进 [13313, 13439]; 砍多少可由 $2.. 覆盖
  CUTS="${*:2}"; [ -n "$CUTS" ] || CUTS="900 1000 800"
  for cut in $CUTS; do
    python3 "$TOOLS" cut "$OUT/002490.prompt.txt" "$cut" "$OUT/short_$cut.txt" >/dev/null || continue
    run ds4 "g3_probe_$cut" -n 1 --decoder-full --prompt-file "$OUT/short_$cut.txt" || continue
    N=$(grep -a -h -o 'prefill [0-9]* token' "$OUT/g3_probe_$cut.err" | head -1 | awk '{print $2}')
    [ -n "$N" ] || continue
    LOG "砍 $cut 字 → $N token, mod 512 = $((N % 512))"
    if [ $((N % 512)) -lt 128 ]; then
      run ds4.base.prefix g3_ced_old -n 256 --emit-trace --prompt-file "$OUT/short_$cut.txt"
      run ds4             g3_ced_new -n 256 --emit-trace --prompt-file "$OUT/short_$cut.txt"
      run ds4             g3_full    -n 256 --emit-trace --decoder-full --prompt-file "$OUT/short_$cut.txt"
      LOG "老 CED vs decoder-full: $(python3 "$TOOLS" cmp-emit "$OUT/g3_ced_old.err" "$OUT/g3_full.err")"
      LOG "新 CED vs decoder-full: $(python3 "$TOOLS" cmp-emit "$OUT/g3_ced_new.err" "$OUT/g3_full.err")"
      LOG "新 CED 日志里的钳位停车/分块: $(grep -a -h '钳位\|留够\|★' "$OUT/g3_ced_new.err" | head -3 | tr '\n' ' ')"
      break
    fi
  done ;;
fast)
  IDS="$ROOT/gguf/v41judge/eng_wt2_n512.bin.ids"; REF="$ROOT/gguf/v41judge/stu_g7_wt2_n512_file_DeepSeek-V4.1-Flash-vq8sh14-q4k.bin"
  [ -s "$IDS" ] && [ -s "$REF" ] || { LOG "★wt2 512 的 ids 或文件态学生 logits 不在★"; exit 1; }
  for b in ds4 ds4.base.nofastmath; do
    LOG "== §5.2 $b: wt2 512 --score-ids(全 40 层, 不挂反修, 与文件态学生同口径)"
    "./$b" -m "$MODEL" --cuda --mem-budget-mb 40000 --weight-cache-mb 88000 --score-ids "$IDS" --score-out "$OUT/fast_$b.bin" > "$OUT/fast_$b.log" 2>&1
    LOG "$b: $(grep -a -h 'PPL' "$OUT/fast_$b.log" | tail -1)"
    ./gguf-tools/bench/anchor_metrics --ref-raw "$REF" --ids "$IDS" --student "$OUT/fast_$b.bin" 2>&1 | tee "$OUT/fast_$b.metrics" | grep -a -i "kld\|same\|ppl\|top" | head -6
  done ;;
parity)
  IDS="$OUT/002490.ids"; [ -s "$IDS" ] || { LOG "★先跑 prep★"; exit 1; }
  N=$(wc -l < "$IDS")
  LOG "== §5.4(a): 002490 序列 $N token, --score-ids top-5(全 40 层预填路, 挂 grrb) vs 解码路真吐 id"
  ./ds4 -m "$MODEL" --zchain "$AMP" --cuda --mem-budget-mb 40000 --weight-cache-mb 88000 \
      --score-ids "$IDS" --score-topk 5 "$OUT/002490_top5.bin" --score-no-logits > "$OUT/parity.log" 2>&1
  LOG "rc=$? $(grep -a -h '完成\|★' "$OUT/parity.log" | tail -2 | tr '\n' ' ')"
  NP=$(head -c 400 "$OUT/002490.prompt.txt" >/dev/null; grep -a -o 'prompt [0-9]*' "$TRACE" | tail -1 | awk '{print $2}')
  python3 "$TOOLS" cmp-topk "$OUT/002490_top5.bin" "$IDS" "$NP" | tee "$OUT/parity.cmp" ;;
full)
  # 不传第 3 个参数 = 不传 -n = 不设上限(引擎默认)。要复现某个产品配置才显式传上限。
  P="${2:?提示文件}"; NGEN="${3:-}"
  [ -s "$P" ] || { LOG "★没有提示 $P★"; exit 1; }
  for mode in ced exact; do
    EX=""; [ "$mode" = exact ] && EX="--decoder-full"
    LOG "== 端到端 $mode: $(basename "$P"), 上限 ${NGEN:-不设}"
    run ds4 "full_$mode" ${NGEN:+-n "$NGEN"} --emit-trace $EX --prompt-file "$P"
    python3 - "$OUT/full_$mode.out" "$OUT/full_$mode.err" <<'PYEOF'
import sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
err = open(sys.argv[2], encoding="utf-8", errors="replace").read()
n = err.count("[emit] ")
c = [ch for ch in txt if not ch.isspace()]
seen, cnt, tot, seg = set(), 0, 0, []
for i in range(max(len(c) - 3, 0)):
    g = "".join(c[i:i + 4]); tot += 1
    if g in seen: cnt += 1
    seen.add(g)
    if tot == 300: seg.append(int(100 * cnt / 300 + 0.5)); cnt = tot = 0
if tot: seg.append(int(100 * cnt / tot + 0.5))
print("  出 %d token / %d 字; 死循环段 %s; JSON 收尾 %s" % (
    n, len(txt), "★有★" if any(s >= 95 for s in seg) else "无",
    "有" if txt.rstrip().endswith(("`", "}", "。", "）")) else "没有(截断)"))
print("  分段:", " ".join(map(str, seg)))
print("  尾 260 字:", txt[-260:].replace("\n", "|"))
PYEOF
  done ;;
nograph)
  # ★长生成上的走图 vs 直发★(2026-09-22): 解码整步 CUDA graph 每 DGRAPH_BUCKET=1024 个位置重捕获一次, 而 d1 门只测过 64~256 步
  # (跨 0~1 个桶)。真实请求要跨 20 多个桶, 桶边界上的任何错(段数上限算小了/垃圾槽被读/位置槽没更新)在门里看不见,
  # 表现就是"长文跑着跑着变样"。这一档: 同提示同上限, 直发一趟, 与 full_ced(走图)逐 token 比第一处不同。
  # 先跑 `full` 拿走图基线(它默认走图), 再跑这一档。
  P="${2:?提示文件}"; NGEN="${3:-}"
  [ -s "$OUT/full_ced.err" ] || { LOG "★没有走图基线 $OUT/full_ced.err, 先跑 full★"; exit 1; }
  LOG "== 直发(--no-graph): $(basename "$P"), 上限 ${NGEN:-不设}"
  run ds4 nograph ${NGEN:+-n "$NGEN"} --emit-trace --no-graph --prompt-file "$P"
  LOG "走图 vs 直发: $(python3 "$TOOLS" cmp-emit "$OUT/full_ced.err" "$OUT/nograph.err")"
  if cmp -s "$OUT/full_ced.out" "$OUT/nograph.out"; then LOG "输出逐字节同 ✓(桶边界无病)"
  else LOG "★输出不同★ $(cmp "$OUT/full_ced.out" "$OUT/nograph.out" 2>&1 | head -1)"; fi
  LOG "走图日志: $(grep -a -h "位置桶\|走图解了" "$OUT/full_ced.err" | tr '\n' ' ' | cut -c1-300)" ;;
statecmp)
  # ★解码路写的状态 vs 重新预填的状态★(2026-09-22, 用户判"复读是引擎 bug"): 同一段上下文, 两条路各生成一次。
  #   原趟: 提示走预填, 后面 K 个 token 是解码路自己写进 KV 的; 现在从第 K 位继续, 看它接下来吐什么(trace 里已有真值)。
  #   本趟: 把"提示 + 那 K 个 token"整段当提示重新预填(--gen-ids, 按 id 原样喂), 再续写同样长度。
  # 判: 两段逐 token 相同 ⇒ 解码路写的状态与预填一致, 复读不是状态污染; 早早分叉 ⇒ 解码路把状态写坏了, 就是引擎 bug。
  # 为什么必须按 id 喂: 真实请求的序列只有 id 是准的, 文本重新分词拼不回去(09-20 实撞 79 vs 75)。
  IDS="${2:?ids 文件(cut_ids.py 造, 同目录要有 .truth)}"; N="${3:-300}"
  [ -s "$IDS" ] && [ -s "$IDS.truth" ] || { LOG "★缺 $IDS 或 $IDS.truth★"; exit 2; }
  tag=$(basename "$IDS" .ids)
  LOG "statecmp $tag: 重新预填 $(wc -w < "$IDS") token, 续写 $N"
  ./ds4 --cuda -m "$MODEL" --zchain "$AMP" --gen-ids "$IDS" -n "$N" --emit-trace \
        > "$OUT/${tag}_refill.out" 2> "$OUT/${tag}_refill.err"
  python3 - "$OUT/${tag}_refill.err" "$IDS.truth" <<'PYEOF'
import re, sys
emit = [int(m.group(1)) for m in re.finditer(r"^\[emit\] \d+ (\d+)", open(sys.argv[1], errors="replace").read(), re.M)]
truth = [int(x) for x in open(sys.argv[2]).read().split()]
n = min(len(emit), len(truth))
same = 0
first = -1
for i in range(n):
    if emit[i] == truth[i]: same += 1
    elif first < 0: first = i
print("  重新预填续写 %d token, 与原趟(解码路状态)真值比: 前 %d 位一致率 %.1f%%; 第一处不同 = 第 %s 位" %
      (len(emit), n, 100.0 * same / max(n, 1), first if first >= 0 else "无(全同)"))
print("  原趟前 24 个: %s" % " ".join(map(str, truth[:24])))
print("  本趟前 24 个: %s" % " ".join(map(str, emit[:24])))
PYEOF
  ;;
stopgate)
  # ★"该停没停"门(2026-09-22, 用户判"还有重复就还有 bug")★
  # 一条真实请求整趟跑完, 一次回答三件事:
  #   ① 停没停 —— 末 id == EOS(1) 是模型自己收的口; 顶到上限就是没停。
  #   ② 产品路(解码, n=1 融合核)与判决尺那条路(预填 GEMM)逐位 top-1 一致率。修 VQ 位平面错位之前
  #      这条真实序列 16384 位里分歧 141 且只有 7% 是近平局 —— 那就是两条路吃着不同权重的指纹。
  #      修后应只剩极少数真平局。★这是"尺和产品是不是同一份权重"的常设复检门★。
  #   ③ 整段逐位扫 EOS 的排名: EOS 当过 top-1 却没停 = 引擎压着模型(引擎 bug);
  #      EOS 从头到尾进不了 top-10 = 模型自己不想停(提示词/贪心, 改引擎没用)。
  # 为什么按 id 喂(--gen-ids): 真实请求只有 id 是准的, 文本重新分词拼不回去(09-20 实撞 79 vs 75)。
  IDS="${2:?提示 ids 文件(bugmd_ids_tools.py extract 产的 *.ids, 或只含提示的那份)}"; NGEN="${3:-}"   # 不传 = 不设上限, 同 full 档
  [ -s "$IDS" ] || { LOG "★没有 $IDS★"; exit 1; }
  tag=$(basename "$IDS" .ids)_stop
  NP=$(wc -w < "$IDS")
  LOG "== stopgate $tag: 提示 $NP token, 上限 ${NGEN:-不设}"
  ./ds4 --cuda -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 --no-dspark \
        --gen-ids "$IDS" ${NGEN:+-n "$NGEN"} --emit-trace > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  LOG "生成 rc=$? $(grep -a -h 'decode .* token' "$OUT/$tag.err" | tail -1)"
  tr -s ' ' < "$IDS" | tr ' ' '\n' | grep -v '^$' > "$OUT/$tag.full.ids"
  grep -a '^\[emit\] ' "$OUT/$tag.err" | awk '{print $3}' >> "$OUT/$tag.full.ids"
  NF=$(wc -l < "$OUT/$tag.full.ids")
  LOG "真吐 $((NF - NP)) token, 末 id = $(tail -1 "$OUT/$tag.full.ids")(EOS = 1); 标题序列:"
  grep -aE '^#{1,4} ' "$OUT/$tag.out" | sed 's/^/    /' | cut -c1-90
  LOG "== 预填路整段重算 top-10(全 40 层, 挂反修), $NF token"
  # 打分不传 --mem-budget-mb/--weight-cache-mb: 与 09-22 12:16 那趟同一套旗(30487 token / 315 s), 换旗就不能跟那次比
  ./ds4 -m "$MODEL" --zchain "$AMP" --cuda \
      --score-ids "$OUT/$tag.full.ids" --score-topk 10 "$OUT/$tag.top10.bin" --score-no-logits > "$OUT/$tag.score.log" 2>&1
  LOG "打分 rc=$? $(grep -a -h '完成\|PPL\|★' "$OUT/$tag.score.log" | tail -2 | tr '\n' ' ')"
  python3 "$TOOLS" cmp-topk  "$OUT/$tag.top10.bin" "$OUT/$tag.full.ids" "$NP" | tee "$OUT/$tag.parity"
  python3 "$TOOLS" eos-scan  "$OUT/$tag.top10.bin" "$OUT/$tag.full.ids" "$NP" | tee "$OUT/$tag.eos" ;;
*) echo "用法: $0 prep|ced|fast|parity|full <提示文件> [上限]|nograph <提示文件> [上限]|statecmp <ids> [N]|stopgate <提示ids> [上限]"; exit 2;;
esac
