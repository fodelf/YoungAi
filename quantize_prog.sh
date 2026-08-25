#!/bin/bash
# 从全精度 HF(双机46片) 裁出 ~18.4G 编程专属模型(每片≤8.6G → 双机16G全驻留)→ 本项目 gguf/
# q2 仅作 --template 只读结构骨架(分词器/架构/形状, 零字节进输出); 输出 100% 来自 HF 全精度。
#
# 质量分配(实测驱动, 2026-06-22 修正——覆盖率 > 单专家精度):
#   hash层0-2  = ffn_gate_tid2eid 是均匀负载均衡哈希(实测每层256/256全命中, 每专家~3030引用,
#                min2703/max3397)。每token映射6个专家, 编程几千唯一token散满全部256 → top-64只覆盖
#                26.6%引用, 裁到64会误路由99.96%的token。【必须全留256】, 档 iq2_xxs/q2_K(=q2零损失)。
#   路由层3-42 = 编程特化, 留 top-K=18(~96%路由覆盖, 远超K=12的90%)。档从高精度(q2k/q4k)降到 q2 精度
#                (iq2_xxs/q2_K)换更多专家: 实测 K=12@q2k/q4k 退化成重复循环, 证明【覆盖比单专家精度重要】。
#   → 净效果: ≈ q2 的编程质量(非 >q2; >q2 需更高精度, 与 18.4G 全驻留物理冲突), 但 18.4G 全驻留 + 9t/s。
#
# 尺寸: hash层(全256)≈5.4G + backbone等固定≈7.6G + 路由层K=18@q2精度≈5.1G → 总≈18.1G(≤18.4双机全驻留)。
#       步骤[4]干跑显示投影尺寸, 不对就改 KEEP_TOP_K 重跑(K↑覆盖↑但尺寸↑; 18.4G 上限约 K=19)。
#
# 用法: bash quantize_prog.sh   (会要 2 次 sudo: M1 NFS导出 + M4 挂载)
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
M1=192.168.1.2
HF_DIR="$ROOT/hf/DeepSeek-V4-Flash-Base"
Q2=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MNT=/tmp/m1_ds4
MERGE=/tmp/hf_full
QUANT="$ROOT/gguf-tools/deepseek4-quantize"
PY=python3
KEEP_TOP_K="${KEEP_TOP_K:-18}"                           # 路由层每层留 top-K 编程特化专家(~96%覆盖, ~18.1G)
HASH_W13="${HASH_W13:-iq2_xxs}"                          # hash层0-2 w1/w3 (=q2档, 全256必须留)
HASH_W2="${HASH_W2:-q2_K}"                               # hash层0-2 w2   (=q2档, 全256必须留)
ROUTED_W13="${ROUTED_W13:-iq2_xxs}"                      # 路由层 w1/w3 (=q2精度, 覆盖优先; 想试高精度少专家改 q2_K)
ROUTED_W2="${ROUTED_W2:-q2_K}"                           # 路由层 w2    (=q2精度; 想试高精度改 q4_K, 但 K 要降)
NORMS="$ROOT/gguf-tools/data/expert-masks/base_router_norms.json" # ★base 模型实测路由排名(非chat; 收集器扫编程语料得)
MASK="$ROOT/gguf-tools/data/expert-masks/mask-specialty-k${KEEP_TOP_K}.bin"
IMAT="$ROOT/gguf-tools/data/expert-masks/reactgo_router.dat"      # chat-q2 算的 imatrix
# USE_IMATRIX: 0=不带imatrix → 量化器从 base 权重自算重要性(对 base 正确; IQ2_XXS fallback sum(col^2))。
#              1=用 chat-q2 的 reactgo_router.dat —— 实测错配 base 权重把 2-bit 量崩(数字汤/重复), 默认关。
USE_IMATRIX="${USE_IMATRIX:-0}"
OUT="$ROOT/gguf/reactgo-prog.gguf"
M4_SHARD="$ROOT/gguf/reactgo-prog-m4.gguf"                               # 旧 M4 分片(已弃用: M4 跑全模型, 分片会 segfault); 清理删掉
M1_WORKER_REMOTE="/Users/fodelf/ds4-main/gguf/reactgo-prog-worker.gguf"  # M1 worker 分片(远程)
CLEAN_OLD="${CLEAN_OLD:-1}"                                              # 1=重量化前删旧产物腾盘(本机2个+M1的1个)
AUTO_SPLIT="${AUTO_SPLIT:-1}"                                            # 1=量化完自动跑 split_prog.sh 切+传 M1
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"                                        # 1=干跑后不暂停自动继续(无人值守自动跑)
SKIP_NFS="${SKIP_NFS:-0}"                                                # 1=跳过 NFS 挂载 sudo 步骤(NFS 已就绪时)
DRY_ONLY="${DRY_ONLY:-0}"                                                # 1=只跑到干跑投影尺寸就退出(验证流水线)
mkdir -p "$ROOT/gguf"

echo "==== [0/7] 重建量化器(确保 --hash-w* 可用)+ 取现成 k${KEEP_TOP_K} 特化 mask ===="
make -C "$ROOT/gguf-tools" deepseek4-quantize 2>&1 | grep -iE "error" && { echo '✗ 量化器编译失败'; exit 1; }
# mask 生成器 make_expert_mask.py 已删(见 git 历史) — 直接消费 data/expert-masks/ 现成 mask。
[ -f "$MASK" ] || { echo "✗ 现成 mask 缺: $MASK (生成器已删; 可用: $(ls "$ROOT/gguf-tools/data/expert-masks/"mask-specialty-k*.bin 2>/dev/null | xargs -n1 basename | tr '\n' ' '))"; exit 1; }
echo "  ✓ mask(现成): $MASK (hash层0-2全留256, 路由层3-42留top-${KEEP_TOP_K})"

mkdir -p "$MNT"
NFS_READY=0; { [ "$SKIP_NFS" = 1 ] || [ -f "$MNT/gguf/$Q2" ]; } && NFS_READY=1
if [ "$NFS_READY" = 1 ]; then
  echo "==== [1-2/7] NFS 已就绪($MNT 可读 q2 模板), 跳过 sudo 挂载 ===="
  [ -f "$MNT/gguf/$Q2" ] || { echo "✗ SKIP_NFS=1 但 $MNT/gguf/$Q2 不可读 — 先手动挂 NFS 再跑"; exit 1; }
  echo "  ✓ 复用现有挂载, 无需 sudo"
else
echo "==== [1/7] M1 NFS 导出+启动服务 (请输入 M1 的 sudo 密码) ===="
ssh -t $M1 'L="/Users/fodelf/ds4-main -ro -mapall=fodelf -network 192.168.1.0 -mask 255.255.255.0"
grep -q "ds4-main" /etc/exports 2>/dev/null || echo "$L" | sudo tee -a /etc/exports >/dev/null
sudo nfsd enable 2>/dev/null
sudo nfsd start  2>/dev/null
sudo nfsd restart 2>/dev/null
sleep 2
sudo nfsd checkexports 2>&1 | head -3
sudo showmount -e localhost 2>/dev/null | grep -q ds4-main && echo "M1_EXPORT_OK" || echo "M1_EXPORT_WARN(继续试挂载)"'
echo "  (M1 步骤完成, 继续挂载)"

echo "==== [2/7] M4 挂载 M1 (请输入本机 sudo 密码) ===="
mkdir -p "$MNT"
mount | grep -q "$MNT" || sudo mount -t nfs -o ro,resvport,nolocks $M1:/Users/fodelf/ds4-main "$MNT" || { echo '✗ 挂载失败'; exit 1; }
[ -f "$MNT/gguf/$Q2" ] || { echo "✗ 挂载后看不到 M1 的 q2 骨架"; exit 1; }
echo "✓ M4 已能读 M1 (q2骨架 + 24片)"
fi

echo "==== [3/7] 合并 46 片 (重叠优先本地, M1 走NFS) ===="
rm -rf "$MERGE" && mkdir -p "$MERGE"
ln -sf "$MNT/hf/DeepSeek-V4-Flash-Base"/*.safetensors "$MERGE"/ 2>/dev/null
ln -sf "$HF_DIR"/*.safetensors "$MERGE"/ 2>/dev/null
ln -sf "$HF_DIR"/*.json "$MERGE"/ 2>/dev/null
N=$(ls "$MERGE"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
echo "  合并 $N 片 (应 46)"
[ "$N" -ge 46 ] || { echo "✗ 缺片(<46)"; exit 1; }

echo "==== [4/7] 干跑校验 + 投影尺寸 (不读张量, 秒级) ===="
"$QUANT" --dry-run --hf "$MERGE" --template "$MNT/gguf/$Q2" \
  --experts-hot-mask "$MASK" \
  --hash-layers 3 --hash-w1 "$HASH_W13" --hash-w2 "$HASH_W2" --hash-w3 "$HASH_W13" \
  --routed-w1 "$ROUTED_W13" --routed-w2 "$ROUTED_W2" --routed-w3 "$ROUTED_W13" \
  --overwrite --out "$OUT" 2>&1 | tee /tmp/prog_dryrun.log | tail -16 || { echo "✗ 干跑失败,先别正式跑"; exit 1; }
echo "---- 投影输出尺寸(应≤18.4G总;每片≤8.6G双机才全驻留)----"
grep -iE "total|output|GiB|GB|bytes|write|emit" /tmp/prog_dryrun.log | tail -4
echo ""
if [ "$DRY_ONLY" = 1 ]; then echo ">>> DRY_ONLY=1: 干跑投影完成, 退出(未清理/未量化)。"; exit 0; fi
if [ "$AUTO_CONFIRM" = 1 ]; then
  echo ">>> AUTO_CONFIRM=1: 不暂停, 直接继续正式量化(K=${KEEP_TOP_K})。"
else
  echo ">>> K=${KEEP_TOP_K} 投影尺寸如上。回车=继续正式量化; Ctrl+C=取消(改 KEEP_TOP_K 重跑)。"
  read -r _
fi

if [ "$CLEAN_OLD" = 1 ]; then
  echo "==== [清理] 删旧量化产物腾盘(仅以下精确文件, 不用通配/不碰目录) ===="
  for f in "$OUT" "$M4_SHARD"; do
    if [ -f "$f" ]; then sz=$(ls -la "$f" | awk '{printf "%.2f",$5/1073741824}'); rm -f "$f" && echo "  ✓ 删本机 $f (${sz}G)"; else echo "  - 本机无 $f(跳过)"; fi
  done
  # M1 远程 worker 分片(铁律: 用户已明确要求删此远程文件; 仅删这一个精确路径)
  ssh "$M1" "if [ -f '$M1_WORKER_REMOTE' ]; then sz=\$(ls -la '$M1_WORKER_REMOTE'|awk '{printf \"%.2f\",\$5/1073741824}'); rm -f '$M1_WORKER_REMOTE' && echo \"  ✓ 删M1 $M1_WORKER_REMOTE (\${sz}G)\"; else echo '  - M1无 $M1_WORKER_REMOTE(跳过)'; fi" 2>/dev/null || echo "  ⚠ M1 不可达, 跳过远程清理(切分上传时会覆盖)"
fi

IMAT_ARG=""; [ "$USE_IMATRIX" = 1 ] && IMAT_ARG="--imatrix $IMAT"
echo "==== [5/7] 正式量化 (路由${ROUTED_W13}/${ROUTED_W2}@K=${KEEP_TOP_K} + hash层全256 ${HASH_W13}/${HASH_W2} + Q8 backbone; imatrix=$([ "$USE_IMATRIX" = 1 ] && echo "chat-q2(危险)" || echo "base自算fallback"); 全来自HF; RSS看门狗10g) ===="
( while true; do p=$(pgrep -f deepseek4-quantize | head -1); [ -z "$p" ] && exit 0
  r=$(ps -o rss= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$r" ] && [ "$r" -gt 10485760 ] && { echo "⚠ 看门狗杀(RSS>10g)"; kill -9 "$p"; exit 1; }; sleep 3; done ) & WD=$!
"$QUANT" --hf "$MERGE" --template "$MNT/gguf/$Q2" \
  --experts-hot-mask "$MASK" \
  --hash-layers 3 --hash-w1 "$HASH_W13" --hash-w2 "$HASH_W2" --hash-w3 "$HASH_W13" \
  --routed-w1 "$ROUTED_W13" --routed-w2 "$ROUTED_W2" --routed-w3 "$ROUTED_W13" $IMAT_ARG --overwrite --out "$OUT"
RC=$?; kill "$WD" 2>/dev/null
[ "$RC" -eq 0 ] || { echo "✗ 量化失败 rc=$RC"; exit "$RC"; }

echo "==== [6/7] 完成 + 每层尺寸(供切分) ===="
ls -la "$OUT" | awk '{printf "  ✓ 编程模型: %.2f GiB  -> %s\n",$5/1073741824,$9}'
# layer_sizes.py 已删(切分功能已下线, 见 git 历史)

# split_prog.sh 已删(纯驱动已删的 split_gguf_layers.py/balanced_split.py, 见 git 历史);
# 自动切分功能待 C 承接, 现在只提示。
echo "==== [7/7] 自动切分已下线(split_prog.sh 已删, 生成器见 git 历史) ===="
echo "  HF 已不需要可删腾盘; 卸载NFS: sudo umount $MNT"
