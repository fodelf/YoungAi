#!/usr/bin/env bash
# 把本仓库 **全量** 同步到 DGX Spark (aarch64 Linux + GB10)。
# 不看 .gitignore: 模型权重 / 捕获数据 / .git 历史 / 一切工作区文件 全部传。
#
# 用法:
#   tools/sync_spark.sh              # 增量同步 (不删对端多余文件)
#   tools/sync_spark.sh --delete     # 镜像同步 (删除对端多余文件)
#   tools/sync_spark.sh --dry-run    # 只看会传什么 + 体积
#   SPARK_HOST=spark SPARK_DIR=ds4-main tools/sync_spark.sh
#
# 注意: macOS 编出来的 *.o / 宿主二进制也会被传过去(用户要求全量)。
#       对端编译前务必先 `make clean`, 否则 make 会拿 Mach-O 的 .o 去链接。
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"          # ~/.ssh/config 里 NVIDIA Sync 建的 Host
SPARK_DIR="${SPARK_DIR:-ds4-main}"         # 相对 spark 上的 $HOME
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_EXTRA=()
for a in "$@"; do
  case "$a" in
    --delete)  RSYNC_EXTRA+=(--delete) ;;
    --dry-run) RSYNC_EXTRA+=(--dry-run --stats) ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# 唯一排除: macOS 的 Finder 垃圾文件, 传过去纯噪声
EXCLUDES=( --exclude=.DS_Store )

echo "==> $SRC  ->  $SPARK_HOST:~/$SPARK_DIR   (全量, 含 .git 与模型权重)"
ssh "$SPARK_HOST" "mkdir -p ~/$SPARK_DIR"
# macOS 自带 openrsync 不认 --info=progress2, 只能用老式 --progress。
# -a 保留权限/软链/时间戳; 权重已是压缩二进制, 不开 -z 省 CPU。
# macOS 自带 bash 3.2: set -u 下空数组展开会报 unbound variable, 必须用 ${A[@]+"${A[@]}"}
rsync -a --progress --partial \
  "${EXCLUDES[@]}" ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
  "$SRC/" "$SPARK_HOST:~/$SPARK_DIR/"

echo
echo "==> done. Spark 侧下一步:"
echo "    ssh $SPARK_HOST 'cd ~/$SPARK_DIR && make clean && make cuda-spark -j\$(nproc)'"
