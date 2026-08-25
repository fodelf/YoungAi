#!/usr/bin/env bash
# 在 **M1 (fodelf.local)** 上执行: 把 M1 的模型数据直传 DGX Spark。
# 数据不经 M4 中转 —— M1 -> spark 直连一跳。
#
#   scp tools/migrate_m1_to_spark.sh fodelf.local:~/
#   ssh fodelf.local 'nohup bash ~/migrate_m1_to_spark.sh > /tmp/m1_to_spark.log 2>&1 &'
#
# 传输量: hf 155G + hf-base 6.2G + gguf 227G = 约 388G
# 目标盘: spark / 有 3.5T 可用, 够。
#
# 合并语义: 不加 --delete, 只做并集; 同名文件以 M1 版本为准(M1 是模型产物真身,
# spark 上那 687M 是先前从 M4 同步过去的子集)。
set -u

SRC="${SRC:-$HOME/ds4-main}"
DST_HOST="${DST_HOST:-spark}"
DST_DIR="${DST_DIR:-ds4-main}"
# 先小后大: 6.2G 的 hf-base 当通路冒烟, 过了再上 155G / 227G
DIRS="${DIRS:-hf-base hf gguf}"
# PATHS: 精确到子目录的路径列表(相对 $SRC), 给了就顶掉 DIRS。用于只搬合并真正需要的
# 那部分 —— en86 下 layers(71G) 要, layers_m10snap(72G, 快照)和 ckpt(合并阶段自己
# rm -rf) 都不要, 一来一去省 147G 传输。
PATHS="${PATHS:-}"
[ -n "$PATHS" ] && DIRS="$PATHS"

for d in $DIRS; do
  [ -e "$SRC/$d" ] || { echo "[skip] $SRC/$d 不存在"; continue; }   # -e: 目录和文件都算
  echo "=== [$(date '+%H:%M:%S')] $d 开始 ($(du -sh "$SRC/$d" 2>/dev/null | cut -f1)) ==="
  # openrsync(macOS 自带)不认 --info=progress2, 用老式 --progress
  # --partial: 388G 中途断了能续, 不用从头再来
  if [ -d "$SRC/$d" ]; then
    ssh "$DST_HOST" "mkdir -p ~/$DST_DIR/$d"
    rsync -a --partial --progress --exclude=.DS_Store \
      "$SRC/$d/" "$DST_HOST:~/$DST_DIR/$d/"
  else
    # 单文件(PATHS 常给, 如 template_head.gguf): 目录形态的 $d/ 尾斜杠对文件无意义,
    # 目标要写成"父目录", 否则 rsync 建出同名目录。**曾静默 skip 掉整个条目**。
    ssh "$DST_HOST" "mkdir -p ~/$DST_DIR/$(dirname "$d")"
    rsync -a --partial --progress "$SRC/$d" "$DST_HOST:~/$DST_DIR/$(dirname "$d")/"
  fi
  rc=$?
  echo "=== [$(date '+%H:%M:%S')] $d 结束 rc=$rc ==="
  [ $rc -ne 0 ] && echo "!!! $d 失败, 继续下一个 (重跑本脚本可断点续传)"
done

echo "=== ALL DONE $(date '+%H:%M:%S') ==="
ssh "$DST_HOST" "du -sh ~/$DST_DIR/hf ~/$DST_DIR/hf-base ~/$DST_DIR/gguf 2>/dev/null"
