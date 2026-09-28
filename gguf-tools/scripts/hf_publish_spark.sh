#!/bin/bash
# hf_publish_spark.sh — 把现役部署对发布到 HuggingFace 公开模型仓库(spark 本机跑, 2026-09-25)。
#
# 发什么(用户令"readme、量化模型、侧车、后训练、spark 的可执行文件, 不要上传其他代码"):
#   README.md / README.zh-CN.md          对外介绍(Mac 仓库里的定稿, 先 scp 到 spark 的 ~/ds4-main/)
#   LICENSE / LICENSE-DeepSeek           ★不是代码, 是必须的★: 分发二进制要带 ds4/ggml 的 MIT 声明,
#                                        分发量化权重要带 DeepSeek 的 MIT 声明 —— MIT 唯一的条件就是保留它
#   DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.partNN-of-40 + SHA256SUMS   ① 113.6 GB 切 40 份(见 split_parts 的 why)
#   install.sh                           一键下载/合并/启动(源在 gguf-tools/scripts/hf_install.sh)
#   DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-...-engine/   ② 领域侧车, 每个 ~40 MB(金融 vqfin41 / 编程 code / 法律 law / 医疗 med / 科研 sci);
#                                        清单只认 install.sh 的 sidecar_of 表(用户按 --domain 选), 这里不另抄一份
#   posttrain-experimental-20260924/     ③ 09-24 大盘研判那一条请求上解的候选(只放 gr_L39.bin + base.fnv;
#                                        predict.txt 带请求的行号与 token id, 不发)
#   bin/ds4, bin/ds4-server              spark(aarch64, CUDA 13)上 make cuda-spark 的产物
#
# 用法: hf_publish_spark.sh split|stage|small|big|verify <用户名/仓库名> [--keep-bin]
#   split   备份原件(真复制+哈希核对+只读) → 切 40 份 → SHA256SUMS → 拼回核对与原件逐字节同
#   stage   在 gguf/hf_upload/<仓库名>/ 摆好发布目录。大文件用硬链接(同一文件系统, 不多占一个字节);
#           放在 gguf/ 下是因为 sync_spark.sh 的 --delete 排除了 gguf/, 同步不会把它删掉
#   small   上传分块以外的全部(几十 MB), 仓库不存在就建成公开仓库 —— 先让页面立起来
#           --keep-bin: 不传 bin/, 线上二进制保持原样(09-28 用户令: 加侧车的那次发布只传侧车 + README + install.sh;
#           spark 上的二进制随开发重编, 不带这个开关就会顺手把线上的换掉)
#   big     逐份上传 40 个分块, 走 LFS 通道; 远端已有且大小对的份跳过, 所以断了重跑同一条就是续传
#   verify  列出仓库里的文件和大小, 与本地逐个比对, 对不上退 1
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT" || exit 1   # 脚本在 gguf-tools/scripts/, 仓库根在上两级
HF="$HOME/v41env/bin/hf"          # 非交互 ssh 的 PATH 里没有 hf, 只认 venv 里这一份
CMD="${1:?用法: $0 split|stage|small|big|verify <用户名/仓库名>}"
REPO="${2:?缺仓库名, 例: wenzhouwu/YoungAi-DeepSeek-V4.1-Flash}"
V41="$ROOT/gguf/v41"
GGUF_NAME="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
# 发布哪些侧车 = install.sh 能选哪些领域: 两边各写一份迟早对不上(用户选得到却没发布 = 启动时报缺目录), 所以从它的表里读
AMP_NAMES=$(sed -n 's/^ *[a-z]*) *echo "\(DeepSeek-[^"]*-engine\)";;$/\1/p' "$ROOT/gguf-tools/scripts/hf_install.sh")
PT_SRC="$V41/night/review_iter_20260924_0729/r1_pt/cand_g1_t6_r10_l1_k1e+09_n0"
PT_NAME="posttrain-experimental-20260924"
STAGE="$ROOT/gguf/hf_upload/${REPO#*/}"
LOG(){ echo "[hfpub $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ LOG "★$*★"; exit 1; }

stage() {
    for f in README.md README.zh-CN.md LICENSE ds4 ds4-server; do [ -s "$ROOT/$f" ] || die "缺 $ROOT/$f"; done
    # ★README 必须是新版★: spark 这份工作树不走 git, 没 scp 过来的话这里是上游 V4 的旧 README
    grep -q 'YoungAi — DeepSeek V4.1 Flash' "$ROOT/README.md" || die "spark 上的 README.md 不是新版, 先从 Mac scp 过来"
    # 下载命令里的仓库名要登录拿到用户名后才填得上; 占位符还在 = 读者照抄就 404
    ! grep -q '__HF_REPO__' "$ROOT/README.md" "$ROOT/README.zh-CN.md" || die "README 里还有 __HF_REPO__ 占位符, 先换成 $REPO"
    grep -q "$REPO" "$ROOT/README.md" || die "README.md 里的下载命令不是 $REPO"
    [ -s "$PARTS_DIR/SHA256SUMS" ] && [ -s "$PARTS_DIR/$(part_name "$NPART")" ] || die "缺分块, 先跑 split"
    [ -s "$ROOT/gguf-tools/scripts/hf_install.sh" ] || die "缺 gguf-tools/scripts/hf_install.sh"
    # ★发布件里不许出现本机用户目录★(09-27 实撞: 第一版 install.sh 写死 /home/fodelf/..., 让每个下载用户 sudo 在自己
    # 机器上造一个"fodelf"目录去对 GGUF 里烤死的分片路径)。现在分片目录走 --engram-dir, 发布件里不该再有任何绝对 home 路径。
    [ -n "$AMP_NAMES" ] || die "没从 install.sh 的 sidecar_of 表里读到任何侧车目录名"
    local a mans=()
    for a in $AMP_NAMES; do
        ls "$V41/$a"/gr_L*.bin >/dev/null 2>&1 && [ -s "$V41/$a/manifest.txt" ] || die "侧车 $a 缺 gr_L*.bin 或 manifest.txt"
        mans+=("$V41/$a/manifest.txt")
    done
    ! grep -nE '/home/|/Users/' "$ROOT/gguf-tools/scripts/hf_install.sh" "$ROOT/README.md" "$ROOT/README.zh-CN.md" "${mans[@]}" \
        || die "上面几行写死了本机用户目录, 下载用户的机器上没有, 改掉再发"
    # install.sh 启动服务时传 --engram-dir; 二进制不认它 = 用户那边服务起到一半退出
    local b h
    for b in ds4 ds4-server; do
        h=$("$ROOT/$b" --help 2>&1)
        case "$h" in *--engram-dir*) ;; *) die "$ROOT/$b 不认 --engram-dir, 先 make cuda-spark";; esac
    done
    # install.sh 续传时按 parts_len(k) = k·PART_BASE + min(k, PART_EXTRA) 把合并文件截回整份边界; 常量写错 =
    # 用户那边拼出一个错位的模型。这里拿 40 份的真实大小逐份核一遍(09-25 实撞: 第一版按"每份一样大"写, 25 份起全错)
    local ib ie k want
    ib=$(sed -n 's/^PART_BASE=//p' "$ROOT/gguf-tools/scripts/hf_install.sh"); ie=$(sed -n 's/^PART_EXTRA=//p' "$ROOT/gguf-tools/scripts/hf_install.sh")
    [ -n "$ib" ] && [ -n "$ie" ] || die "install.sh 缺 PART_BASE/PART_EXTRA"
    for k in $(seq 1 "$NPART"); do
        want=$(( ib + (k <= ie ? 1 : 0) ))
        [ "$(stat -c %s "$PARTS_DIR/$(part_name "$k")")" = "$want" ] || die "第 $k 份大小与 install.sh 的 PART_BASE/PART_EXTRA 推算($want)不符"
    done
    [ -s "$PT_SRC/gr_L39.bin" ] && [ -s "$PT_SRC/base.fnv" ] || die "后训练候选缺 gr_L39.bin/base.fnv: $PT_SRC"
    mkdir -p "$STAGE/bin" "$STAGE/$PT_NAME"
    cp -f "$ROOT/README.md" "$ROOT/README.zh-CN.md" "$ROOT/LICENSE" "$STAGE/"
    cp -f "$ROOT/hf/DeepSeek-V4.1-Flash/LICENSE" "$STAGE/LICENSE-DeepSeek"
    cp -f "$ROOT/gguf-tools/scripts/hf_install.sh" "$STAGE/install.sh"
    cp -f "$ROOT/ds4" "$ROOT/ds4-server" "$STAGE/bin/"
    # 早先整文件版本的发布目录里有一个 GGUF 硬链接; 摘掉的只是这个链接(原件与备份不受影响), 否则 verify 会拿它去比远端
    if [ -e "$STAGE/$GGUF_NAME" ]; then
        [ "$(stat -c %i "$STAGE/$GGUF_NAME")" = "$(stat -c %i "$V41/$GGUF_NAME")" ] || die "$STAGE/$GGUF_NAME 不是原件的硬链接, 不动它"
        rm -f "$STAGE/$GGUF_NAME"
    fi
    ln -f "$PARTS_DIR/SHA256SUMS" "$PARTS_DIR/$GGUF_NAME".part*-of-$NPART "$STAGE/"
    for a in $AMP_NAMES; do mkdir -p "$STAGE/$a" && ln -f "$V41/$a"/* "$STAGE/$a/"; done
    ln -f "$PT_SRC/gr_L39.bin" "$PT_SRC/base.fnv" "$STAGE/$PT_NAME/"
    LOG "发布目录 $STAGE:"
    (cd "$STAGE" && find . -type f -not -path './.cache/*' | sort | xargs ls -l | awk '{printf "  %14d  %s\n", $5, $9}')
}

# ★卡死检测★(2026-09-25 实撞): LFS 上传在线路极差时会挂住 —— 连接不断、不报错、不超时, 进度条停在 45% 一动不动,
# 外层重试循环永远等不到它返回。判据: 上传进度输出(tqdm, 每走一点字节就刷新一次)连续 STALL_S 秒没长 ⇒ 杀掉重传这一份。
# 慢不算卡: 几十 kB/s 时进度条照样在刷新, 只有一个字节都不走才会触发。
STALL_S=300
PLOG=/tmp/hf_publish_part.log
run_with_stall_guard() {   # run_with_stall_guard <秒> <输出文件> 命令...  返回命令的退出码; 被判卡死返回 124
    local lim="$1" log="$2"; shift 2
    : >"$log"
    "$@" >"$log" 2>&1 &
    local pid=$! last=-1 cur lastt; lastt=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        cur=$(stat -c %s "$log")
        if [ "$cur" != "$last" ]; then last=$cur; lastt=$(date +%s)
        elif [ $(( $(date +%s) - lastt )) -ge "$lim" ]; then
            LOG "★进度输出 ${lim}s 没有变化, 判卡死, 杀掉(pid $pid)★"
            kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; return 124
        fi
    done
    wait "$pid"
}

remote_sizes() {   # 远端清单(只读元数据, 不参与数值): 每行 "路径 字节数"
    "$HOME/v41env/bin/python" -c "
import sys
from huggingface_hub import HfApi
for f in HfApi().list_repo_tree(sys.argv[1], recursive=True):
    if getattr(f, 'size', None) is not None: print(f.path, f.size)" "$REPO"
}

# 连不上 HF 时 whoami 也失败, 以前一律报"没登录", 让人去重登 —— 09-28 实撞真因是 Mihomo 当前节点对 HF 握手就断(SSL EOF),
# 令牌好好的。连接错误单独报, 并给出绕路: Mac 上 ssh -f -N -R 17897:127.0.0.1:7897 spark, 再带 HTTPS_PROXY=http://127.0.0.1:17897 跑本脚本。
need_login() {
    local out; out=$("$HF" auth whoami 2>&1) && return 0
    case "$out" in
        *ConnectError*|*SSL*|*timed\ out*|*Connection*) die "spark 连不上 HF(网络, 不是登录): 换 Mihomo 节点, 或走 Mac 隧道(见 need_login 上方注释)";;
        *) die "spark 上 HF 没登录: ssh spark 后执行 $HF auth login(要 Write 令牌)";;
    esac
}

NPART=40
PARTS_DIR="$V41/${GGUF_NAME%.gguf}-parts$NPART"
part_name() { printf '%s.part%02d-of-%d' "$GGUF_NAME" "$1" "$NPART"; }

# 备份 + 切 40 份(2026-09-25 用户令)。为什么切: 这条出网线路上 Xet 通道被掐、只剩 LFS 能传, 而 LFS 不能跨进程续传 ——
# 113.6 GB 一个文件断一次就从零来; 切成 40 份后每份单独提交, 断了只重传当前这一份。下载侧由 install.sh 边下边合并。
split_parts() {
    local src="$V41/$GGUF_NAME" bk="$V41/backup-20260925" h hb hc
    [ -s "$src" ] || die "缺原件 $src"
    LOG "① 原件 sha256($(stat -c %s "$src") B)"
    h=$(sha256sum "$src" | cut -d' ' -f1); LOG "   $h"
    mkdir -p "$bk"
    if [ "$(stat -c %s "$bk/$GGUF_NAME" 2>/dev/null)" != "$(stat -c %s "$src")" ]; then
        LOG "② 备份(真复制, 不是硬链接) → $bk/"
        cp "$src" "$bk/$GGUF_NAME.copying" && mv "$bk/$GGUF_NAME.copying" "$bk/$GGUF_NAME" || die "备份失败"
    fi
    hb=$(sha256sum "$bk/$GGUF_NAME" | cut -d' ' -f1)
    [ "$hb" = "$h" ] || die "备份哈希不一致: $hb"
    chmod a-w "$bk/$GGUF_NAME"; LOG "   备份哈希一致, 已设只读"
    LOG "③ 切 $NPART 份 → $PARTS_DIR/"
    mkdir -p "$PARTS_DIR"
    split -n "$NPART" -d -a 2 --numeric-suffixes=1 --additional-suffix="-of-$NPART" "$src" "$PARTS_DIR/$GGUF_NAME.part" || die "切分失败"
    (cd "$PARTS_DIR" && { echo "$h  $GGUF_NAME"; sha256sum "$GGUF_NAME".part*-of-$NPART; } > SHA256SUMS) || die "写 SHA256SUMS 失败"
    LOG "④ 按顺序拼回去再算 sha256, 必须与原件相同"
    hc=$(cat "$PARTS_DIR/$GGUF_NAME".part*-of-$NPART | sha256sum | cut -d' ' -f1)
    [ "$hc" = "$h" ] || die "拼回去的哈希 $hc ≠ 原件 $h"
    LOG "SPLIT_OK $(ls "$PARTS_DIR"/*.part*-of-$NPART | wc -l) 份, 每份 $(stat -c %s "$PARTS_DIR/$(part_name 1)") B, 拼回哈希 = 原件"
}

case "$CMD" in
  selftest-stall)   # 假上传: 打一行就不再动, 阈值 20 s ⇒ 必须在 20~40 s 内被杀并返回 124; 另一个一直在输出的不许被误杀
    t0=$(date +%s); run_with_stall_guard 20 /tmp/hf_stall_selftest.log bash -c 'echo 开始; sleep 600'; rc=$?; dt=$(( $(date +%s) - t0 ))
    [ "$rc" = 124 ] && [ "$dt" -ge 20 ] && [ "$dt" -le 45 ] || die "卡死检测自测失败: rc=$rc 用时 ${dt}s"
    LOG "卡住的假上传: ${dt}s 被杀, rc=124 ✓"
    t0=$(date +%s); run_with_stall_guard 20 /tmp/hf_stall_selftest.log bash -c 'for i in $(seq 1 12); do echo $i; sleep 5; done'; rc=$?; dt=$(( $(date +%s) - t0 ))
    [ "$rc" = 0 ] || die "一直有输出的慢任务被误杀: rc=$rc"
    LOG "一直在走的慢任务: 跑满 ${dt}s 正常结束, 没被误杀 ✓"
    LOG "SELFTEST_STALL_OK";;
  split) split_parts;;
  stage) stage;;
  small)
    need_login; [ -d "$STAGE" ] || die "先跑 stage"
    KEEP=(); BINMSG=", aarch64 CUDA binaries"; [ "${3:-}" = "--keep-bin" ] && { KEEP=(--exclude "bin/*"); BINMSG=""; LOG "--keep-bin: bin/ 不上传, 线上二进制不动"; }
    LOG "上传小文件(不含 GGUF 分块) → $REPO (公开)"
    # 与 big 同一条线路、同一个病(见 big 段的 ★走 LFS★): 侧车 + 二进制几十 MB, 走 Xet 正好撞上 10~20 s 掐断; 卡死同样不超时
    if ! run_with_stall_guard "$STALL_S" "$PLOG" env HF_HUB_DISABLE_XET=1 "$HF" upload "$REPO" "$STAGE" . --repo-type model --no-private \
        --exclude "*.gguf" --exclude "*.part*-of-*" --exclude ".cache/*" "${KEEP[@]}" \
        --commit-message "README, install.sh, licenses, SHA256SUMS, domain sidecars (finance, code, law, medicine, science), experimental post-training file$BINMSG"; then
        tail -c 600 "$PLOG" | tr '\r' '\n' | grep -v '^\s*$' | tail -5; die "小文件上传失败(完整输出 $PLOG)"
    fi
    LOG "SMALL_DONE https://huggingface.co/$REPO";;
  big)
    need_login; [ -s "$STAGE/$(part_name "$NPART")" ] || die "先跑 stage"
    # ★走 LFS, 不走 Xet★(2026-09-25 定位): 这条出网线路(机场的所有节点都一样)会在 10~20 s 内掐断发往
    # cas-server.xethub.hf.co 的上传 —— 整文件 35 趟全卡在 1%, hf_xet 日志满是 "peer closed connection without
    # sending TLS close_notify"。同节点单连接传 Cloudflare 139 s 不断; 关掉 Xet 走 LFS(数据直传 S3), 256 MB 150 s 零掐断。
    # HF_HUB_DISABLE_XET 只给 hf 这个第三方工具, 引擎不读任何环境变量。
    # ★逐份续传★: 每份一个提交; 先取远端清单, 大小对上的份跳过 ⇒ 断了重跑同一条命令就是续传。
    # 一份失败等 30 s 重传这一份。只在"连续 5 次都 2 分钟内就失败"时停(那是令牌/仓库/文件这类固定问题);
    # 网络差导致的慢失败、卡死不算 —— 09-25 线路整体掉到几十 kB/s 时, 卡死会连着发生, 为此停掉整趟上传不对。
    remote=$(remote_sizes) || die "取远端清单失败"
    for k in $(seq 1 "$NPART"); do
        p=$(part_name "$k"); lsz=$(stat -c %s "$STAGE/$p")
        if [ "$(printf '%s\n' "$remote" | awk -v x="$p" '$1==x {print $2}')" = "$lsz" ]; then LOG "[$k/$NPART] 远端已有, 跳过"; continue; fi
        tries=0; fast=0
        while :; do
            tries=$((tries+1)); t0=$(date +%s)
            LOG "[$k/$NPART] $p ($lsz B) 第 $tries 次"
            if run_with_stall_guard "$STALL_S" "$PLOG" env HF_HUB_DISABLE_XET=1 "$HF" upload "$REPO" "$STAGE/$p" "$p" \
                --repo-type model --commit-message "Base model part $k/$NPART"; then
                dt=$(( $(date +%s) - t0 )); LOG "[$k/$NPART] 完成 ${dt}s, $(( lsz / 1048576 / (dt > 0 ? dt : 1) )) MB/s"; break
            fi
            dt=$(( $(date +%s) - t0 ))
            if [ "$dt" -lt 120 ]; then fast=$((fast+1)); else fast=0; fi
            LOG "★[$k/$NPART] 第 $tries 次失败(跑了 ${dt}s; 连续快速失败 $fast 次)★ 最后输出: $(tail -c 300 "$PLOG" | tr '\r' '\n' | grep -v '^\s*$' | tail -1)"
            [ "$fast" -ge 5 ] && die "[$k/$NPART] 连续 5 次 2 分钟内失败, 不是网络慢, 是固定问题(令牌/仓库/文件), 停下查"
            sleep 30
        done
    done
    LOG "BIG_DONE $NPART 份";;
  verify)
    need_login; [ -d "$STAGE" ] || die "先跑 stage"
    remote=$(remote_sizes) || die "取远端清单失败"
    bad=0
    while read -r rel; do
        rel="${rel#./}"; lsz=$(stat -c %s "$STAGE/$rel")
        rsz=$(printf '%s\n' "$remote" | awk -v p="$rel" '$1==p {print $2}')
        if [ "$lsz" = "$rsz" ]; then echo "  ok   $rel $lsz"; else echo "  ★差 $rel 本地 $lsz 远端 ${rsz:-缺}★"; bad=1; fi
    done < <(cd "$STAGE" && find . -type f -not -path './.cache/*' | sort)
    [ $bad = 0 ] && LOG "VERIFY_OK 本地 $(cd "$STAGE" && find . -type f -not -path './.cache/*' | wc -l) 个文件与远端逐个同大小" || die "VERIFY_FAIL";;
  *) die "不认识的子命令 $CMD";;
esac
