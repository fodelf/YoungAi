#!/bin/bash
# move_hf_base_shards.sh — 把 M1 上 hf-base/DeepSeek-V4-Flash-Base 的指定 shard
# "移动"到本机项目目录: 逐 shard 传输 -> 双端 sha256 比对 -> 校验通过才删源。
#
# 安全设计 (原始 HF 权重, 删错不可逆):
#   * 串行逐 shard: 每次只有一个文件处于 "两端都在" 的中间态, 峰值额外占用 = 1 个 shard
#   * 先传后校验后删: sha256 不一致 / 传输失败 -> 立刻停止, 源文件保持不动
#   * 本机磁盘底线 RESERVE_GB: 开新 shard 前算账, 不够就停 (本机同时还在下 0731 模型)
#   * 幂等: 目标已存在且 sha 与源一致 -> 跳过传输, 直接进入删源步骤
#   * DELETE_SRC=0 可退化为纯复制
#
# 用户于 2026-08-02 明确确认: 校验通过后删除 M1 源文件 (真移动)。
set -uo pipefail

REMOTE=${REMOTE:-fodelf@192.168.1.2}
SRC=${SRC:-/Users/fodelf/ds4-main/hf-base/DeepSeek-V4-Flash-Base}
DST=${DST:-/Users/fodelf/git/ds4-main/hf-base/DeepSeek-V4-Flash-Base}
SHARDS=${SHARDS:-"1 2 3 4 5 6 7"}
TOTAL_SHARDS=${TOTAL_SHARDS:-46}
RESERVE_GB=${RESERVE_GB:-20}
DELETE_SRC=${DELETE_SRC:-1}
SSH_OPTS=${SSH_OPTS:-"-o ConnectTimeout=15 -o ServerAliveInterval=30"}

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

avail_gb() { df -k "$1" | tail -1 | awk '{printf "%d", $4/1048576}'; }

mkdir -p "$DST" || die "无法创建 $DST"

log "源  : $REMOTE:$SRC"
log "目标: $DST"
log "shard: $SHARDS   删源: $DELETE_SRC   本机底线: ${RESERVE_GB}G"

ssh $SSH_OPTS "$REMOTE" "test -d '$SRC'" || die "M1 上找不到 $SRC"

# ---- 小文件 (config / tokenizer / index): 只复制, 永不删源 ----
log "=== 阶段 0: 复制小文件 (config/tokenizer/index, 不删源) ==="
for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json LICENSE .gitattributes; do
  if ssh $SSH_OPTS "$REMOTE" "test -f '$SRC/$f'"; then
    rsync -a -e "ssh $SSH_OPTS" "$REMOTE:$SRC/$f" "$DST/$f" \
      && log "  ok   $f" || log "  WARN 复制失败: $f"
  fi
done

ok_n=0; skip_n=0; moved_bytes=0
for i in $SHARDS; do
  name=$(printf 'model-%05d-of-%05d.safetensors' "$i" "$TOTAL_SHARDS")
  log "=== shard $i/$(echo $SHARDS | wc -w | tr -d ' ') : $name ==="

  rsize=$(ssh $SSH_OPTS "$REMOTE" "stat -f %z '$SRC/$name' 2>/dev/null" || true)
  [ -n "$rsize" ] || { log "  跳过: 源不存在"; skip_n=$((skip_n+1)); continue; }
  rgb=$((rsize/1073741824))
  log "  源大小: ${rsize} B (~${rgb} GiB)"

  # --- 磁盘算账: 需要 shard 大小 + 保留水位 ---
  free_gb=$(avail_gb "$DST")
  need_gb=$((rgb + RESERVE_GB))
  log "  本机可用: ${free_gb}G, 需要: ${need_gb}G (shard ${rgb}G + 保留 ${RESERVE_GB}G)"
  lsize_pre=$(stat -f %z "$DST/$name" 2>/dev/null || echo 0)
  if [ "$free_gb" -lt "$need_gb" ] && [ "$lsize_pre" != "$rsize" ]; then
    die "磁盘不足, 停在 shard $i (已完成 $ok_n 个)"
  fi

  # --- 传输 (已完整则跳过) ---
  if [ "$lsize_pre" = "$rsize" ]; then
    log "  本地已有同尺寸文件, 跳过传输, 直接校验"
  else
    t0=$(date +%s)
    rsync -a --partial --inplace -e "ssh $SSH_OPTS" \
      "$REMOTE:$SRC/$name" "$DST/$name" || die "rsync 失败: $name"
    t1=$(date +%s); dt=$((t1-t0)); [ "$dt" -gt 0 ] || dt=1
    log "  传输完成: ${dt}s, $((rsize/1048576/dt)) MB/s"
  fi

  lsize=$(stat -f %z "$DST/$name" 2>/dev/null || echo 0)
  [ "$lsize" = "$rsize" ] || die "大小不符: 本地 $lsize vs 源 $rsize ($name)"

  # --- 双端 sha256 (删源前的唯一凭据) ---
  log "  校验 sha256 (双端并行)..."
  rsha_f=$(mktemp); lsha_f=$(mktemp)
  ssh $SSH_OPTS "$REMOTE" "shasum -a 256 '$SRC/$name'" > "$rsha_f" 2>/dev/null &
  rpid=$!
  shasum -a 256 "$DST/$name" > "$lsha_f" 2>/dev/null &
  lpid=$!
  wait $rpid; rrc=$?
  wait $lpid; lrc=$?
  rsha=$(awk '{print $1}' "$rsha_f"); lsha=$(awk '{print $1}' "$lsha_f")
  rm -f "$rsha_f" "$lsha_f"
  [ "$rrc" = 0 ] && [ -n "$rsha" ] || die "源端 sha256 失败: $name"
  [ "$lrc" = 0 ] && [ -n "$lsha" ] || die "本地 sha256 失败: $name"
  if [ "$rsha" != "$lsha" ]; then
    die "sha256 不一致! 源=$rsha 本地=$lsha ($name) — 源文件未删, 请重传"
  fi
  log "  sha256 一致: ${lsha:0:16}..."

  # --- 校验通过, 删源 ---
  if [ "$DELETE_SRC" = "1" ]; then
    ssh $SSH_OPTS "$REMOTE" "rm -f '$SRC/$name'" \
      && log "  已删除 M1 源: $name" || log "  WARN 删源失败 (本地副本完好): $name"
  else
    log "  DELETE_SRC=0, 保留 M1 源"
  fi

  ok_n=$((ok_n+1)); moved_bytes=$((moved_bytes+rsize))
  log "  进度: $ok_n 个完成, 累计 $((moved_bytes/1073741824)) GiB; 本机剩余 $(avail_gb "$DST")G; M1 剩余 $(ssh $SSH_OPTS "$REMOTE" "df -k / | tail -1 | awk '{printf \"%d\", \\$4/1048576}'" 2>/dev/null)G"
done

log "=== 完成: $ok_n 个 shard 已移动 ($((moved_bytes/1073741824)) GiB), 跳过 $skip_n ==="
log "目标目录: $DST"
ls -la "$DST" >&2
