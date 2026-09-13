#!/bin/sh
set -e

# Downloader for the official DeepSeek V4 Flash BASE model (HF safetensors):
# deepseek-ai/DeepSeek-V4-Flash-Base, 46 shards + config + tokenizer, ~274.5 GiB.
# That layout is exactly what gguf-tools/deepseek4-quantize --hf <dir> wants.
#
# The full model does not fit on either of our two 16 GB Macs alone (M4 ~149 GiB
# free, M1 ~135 GiB free), but it fits across BOTH (284 GiB combined). So this
# supports a SPLIT mode:
#
#   * config/tokenizer/index + as many shards as fit go to the LOCAL dir (M4).
#   * remaining shards go to the remote (M1) disk.
#
# Remote shards are fetched in one of two ways (--remote-mode):
#
#   * direct (DEFAULT): the remote downloads its OWN shards from HF, in PARALLEL
#     with the local host, using the remote's own proxy (--remote-proxy). Both
#     machines pull from the network at once, so aggregate bandwidth doubles and
#     wall-clock roughly halves. Requires working outbound on the remote.
#   * stream: the legacy path. Remote shards STREAM through this host straight
#     onto the remote disk: `curl (via local proxy) | ssh REMOTE 'cat > file'`.
#     Only this host touches the network; use it when the remote's outbound is
#     unreliable. Slower (single downloader for the whole model).
#
# Downloads are resumable: re-run the same command. Local shards resume mid-file
# via aria2c's control file; remote shards resume mid-file via an HTTP Range
# request appended to the partial remote file, with a self-heal reset if a CDN
# ever ignores the Range (detected by an over-size result). Completed files on
# either side are skipped by a byte-count check.

# Any HF safetensors repo with the same layout works (--repo / HF_REPO); the
# default is the official BASE model. When the repo is overridden and no --dir
# is given, the output dir follows the repo's basename (./hf/<repo-basename>).
REPO=${HF_REPO:-deepseek-ai/DeepSeek-V4-Flash-Base}
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

# --- 双源下载 (--ms-repo, 2026-09-10) ---------------------------------------
# 起因: spark(Linux, 无梯子)从国内直连 huggingface.co 直接超时, 借 Mac 的 clash 走
# ssh 反向隧道只有 0.76 MB/s, hf-mirror 直连 4 并发 10.7 MB/s, 而 modelscope.cn
# 直连 4 并发实测 64.5 MB/s —— 差 6 倍。但 ModelScope 的 V4.1 仓库只同步了 1-44 片,
# 缺 45-48(MTP×2 + layers.1/14.engram×2, 合计 194 GiB), 那 4 片只能回落 HF 侧。
# 所以按"这个文件 ModelScope 有没有"逐个选源: 有就走快路, 没有就走 --endpoint。
# 不分源会怎样: 全压 hf-mirror, 281 GiB 的大头白白慢 6 倍; 全压 ModelScope 则
# 45-48 片 404, 下出来的模型缺 engram 和 MTP, 加载时报张量缺失而不是下载失败。
# --ms-repo 留空 = 关闭双源, 脚本行为与加这段之前完全一致。
MS_REPO=${MS_REPO:-}
MS_ENDPOINT=https://modelscope.cn
MS_LIST=""   # ModelScope 侧实际存在的文件名, 换行分隔; 启动时拉一次

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_HF_DIR:-}
OUT_DIR_SET=0; [ -n "$OUT_DIR" ] && OUT_DIR_SET=1

# Reached through a local proxy. Override with --proxy / HF_PROXY; disable with
# --no-proxy / HF_PROXY=off. Applies to BOTH the API call and file transfers.
PROXY=${HF_PROXY:-http://127.0.0.1:7897}
TOKEN=${HF_TOKEN:-}

# Remote overflow target (M1 worker over the bridge). Empty => local-only.
REMOTE_SSH=${REMOTE_SSH:-}
REMOTE_DIR=${REMOTE_DIR:-}

# How the remote gets its shards: "direct" (remote downloads itself, in parallel
# with us) or "stream" (legacy: pipe through this host). See header comment.
REMOTE_MODE=${REMOTE_MODE:-direct}
# Proxy the REMOTE uses for its own direct downloads. Sentinel __INHERIT__ =>
# default to the same string as the local PROXY (both Macs run clash on 7897).
REMOTE_PROXY=${REMOTE_PROXY:-__INHERIT__}

# Relay-assist: once THIS host finishes its own shards it would otherwise sit
# idle while the remote still has ~2x as much to fetch. With --relay, the idle
# local host also pulls the remote's outstanding shards (descending, while the
# remote works ascending), pushes each over the fast bridge, and deletes the
# local copy — so both proxies drain the remote's set. Needs scratch space:
# RELAY_RESERVE_GIB is kept free locally to stage one shard at a time.
RELAY=${RELAY:-0}
RELAY_RESERVE_GIB=${RELAY_RESERVE_GIB:-14}

# The mirror image of --relay: once the REMOTE finishes its own share it helps
# fetch the LOCAL host's outstanding shards, staging them on its own disk and
# handing them over the bridge. Needed whenever the remote gets the smaller
# slice (see --local-share), which is the normal case when its disk is smaller.
REVERSE_RELAY=${REVERSE_RELAY:-0}

# Free-space headroom to keep on each disk (GiB). The model is a tight fit, so
# these are small by necessity; raise them and free space if you want margin.
LOCAL_RESERVE_GIB=${LOCAL_RESERVE_GIB:-4}
REMOTE_RESERVE_GIB=${REMOTE_RESERVE_GIB:-4}

CONFIG_ONLY=0
PLAN_ONLY=0
FORCE=0
# aria2's multi-range parallelism (-x/-s) breaks against mirrors that redirect
# to a pre-signed CDN URL whose policy pins a single ByteRange: every extra
# segment comes back 403. --no-aria2 forces the single-connection curl path,
# which those mirrors serve fine (and fast).
USE_ARIA2=${USE_ARIA2:-1}
# Concurrent files per host. The mirror throttles per connection, not per host:
# measured 2026-08-02 on M4, one lane 0.89 MB/s vs three lanes 1.45 MB/s. More
# lanes also eat more of the household uplink, so this stays tunable.
LANES=${DL_LANES:-3}
# Fraction of the model to aim at the LOCAL disk (0 = off => local-first fill).
# With both hosts downloading at similar rates, a local-first split finishes one
# host early and leaves the other alone with the tail; a share matched to the
# measured rates has them finish together.
LOCAL_SHARE=${LOCAL_SHARE:-0}

TAB=$(printf '\t')
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
# Non-interactive ssh doesn't load the remote's profile, so brew dirs aren't on
# PATH; prepend them so aria2c (Intel /usr/local or arm /opt/homebrew) resolves.
# Single-quoted: $PATH stays literal and is expanded by the REMOTE shell.
# Also unset all_proxy: clash exports it as socks5://127.0.0.1:7897, but aria2c
# has no SOCKS support and aborts ("unrecognized proxy format") on that env var
# before honoring our explicit --all-proxy=http://. http(s)_proxy stay (valid).
REMOTE_PATH='export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"; unset all_proxy ALL_PROXY;'

usage() {
    cat <<EOF
DeepSeek V4 Flash BASE model downloader (HF safetensors, ~274.5 GiB)

Usage:
  ./download_base_model.sh [options]

Split across both Macs (recommended here; neither disk fits the model alone):
  ./download_base_model.sh \\
    --remote fodelf@192.168.1.2:/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base

Options:
  --remote USER@HOST:DIR  Overflow remaining shards onto this host via ssh.
                          By default the remote downloads them ITSELF in parallel
                          with us (see --remote-mode).
  --remote-mode MODE      direct (default): remote downloads its own shards in
                          parallel using --remote-proxy. stream: pipe through
                          this host (legacy; use if remote outbound is flaky).
  --remote-proxy URL      Proxy the remote uses for direct downloads
                          (default: same as local --proxy).
  --no-remote-proxy       Remote downloads without a proxy (direct to HF).
  --reverse-relay         After the REMOTE finishes its own shards, it helps
                          fetch the local host's remaining ones and hands them
                          over the bridge. Use when the remote has the smaller
                          slice so it does not sit idle.
  --relay                 After finishing local shards, also fetch the remote's
                          outstanding shards here and push them over the bridge
                          (both proxies drain the remote set). Keeps
                          --relay-reserve GiB free locally as staging scratch.
  --relay-reserve N       GiB to keep free locally for relay staging (default 14).
  --repo REPO             HF repo to download (default: $REPO).
  --endpoint URL          HF-compatible host for the API and for any file the
                          ModelScope lane does not have (default: $ENDPOINT).
                          Use https://hf-mirror.com from a host with no VPN.
  --ms-repo REPO          Also pull from ModelScope, per file: anything that repo
                          has comes from modelscope.cn (measured 64.5 MB/s on
                          spark vs 10.7 on hf-mirror), the rest from --endpoint.
                          Sizes are still checked against the --endpoint list.
                          Omit it to keep the old single-source behaviour.
  --local-share F         Aim F (0..1) of the model at the local disk instead of
                          filling local first. Set it to the local host's share
                          of the two download rates so both finish together.
  --lanes N               Files downloaded concurrently per host (default $LANES).
                          The mirror throttles per connection, so >1 helps; it
                          also uses more of the household bandwidth.
  --no-aria2              Force the single-connection curl path. Needed for
                          mirrors whose pre-signed CDN URLs pin one ByteRange
                          (aria2's parallel segments come back 403).
  --dir DIR               Local output dir (default: ./hf/<repo-basename>
                          or \$DS4_HF_DIR).
  --local-reserve N       GiB to keep free on the local disk  (default 4).
  --remote-reserve N      GiB to keep free on the remote disk (default 4).
  --plan                  Compute and print the split, then exit (no downloads).
  --config-only           Only config/tokenizer/index/license (a few MB).
  --proxy URL             Proxy for API + downloads (default: $PROXY).
  --no-proxy              Disable the proxy.
  --token TOKEN           HF token (repo is public; optional).
  --force                 Proceed even if the space preflight says it won't fit.
  -h, --help              This help.

Environment:
  HF_REPO, DS4_HF_DIR, HF_PROXY (off to disable), HF_ENDPOINT, HF_TOKEN,
  REMOTE_SSH, REMOTE_DIR, REMOTE_MODE (direct|stream), REMOTE_PROXY,
  LOCAL_RESERVE_GIB, REMOTE_RESERVE_GIB

Long unattended run: prefix with caffeinate so the Mac stays awake, e.g.
  caffeinate -i ./download_base_model.sh --remote USER@HOST:DIR
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --remote)
            shift; [ $# -gt 0 ] || { echo "Missing value after --remote" >&2; exit 1; }
            REMOTE_SSH=${1%%:*}
            REMOTE_DIR=${1#*:}
            [ "$REMOTE_SSH" != "$1" ] || { echo "--remote must be USER@HOST:DIR" >&2; exit 1; }
            ;;
        --dir)
            shift; [ $# -gt 0 ] || { echo "Missing value after --dir" >&2; exit 1; }
            OUT_DIR=$1; OUT_DIR_SET=1 ;;
        --repo)
            shift; [ $# -gt 0 ] || { echo "Missing value after --repo" >&2; exit 1; }
            REPO=$1 ;;
        --endpoint)
            shift; [ $# -gt 0 ] || { echo "Missing value after --endpoint" >&2; exit 1; }
            ENDPOINT=${1%/} ;;
        --ms-repo)
            shift; [ $# -gt 0 ] || { echo "Missing value after --ms-repo" >&2; exit 1; }
            MS_REPO=$1 ;;
        --remote-mode)
            shift; [ $# -gt 0 ] || { echo "Missing value after --remote-mode" >&2; exit 1; }
            REMOTE_MODE=$1
            case "$REMOTE_MODE" in direct|stream) ;; *) echo "--remote-mode must be direct|stream" >&2; exit 1 ;; esac ;;
        --remote-proxy) shift; [ $# -gt 0 ] || { echo "Missing value after --remote-proxy" >&2; exit 1; }; REMOTE_PROXY=$1 ;;
        --no-remote-proxy) REMOTE_PROXY="" ;;
        --relay) RELAY=1 ;;
        --reverse-relay) REVERSE_RELAY=1 ;;
        --relay-reserve) shift; [ $# -gt 0 ] || { echo "Missing value after --relay-reserve" >&2; exit 1; }; RELAY_RESERVE_GIB=$1 ;;
        --local-reserve)  shift; LOCAL_RESERVE_GIB=$1 ;;
        --remote-reserve) shift; REMOTE_RESERVE_GIB=$1 ;;
        --no-aria2) USE_ARIA2=0 ;;
        --lanes) shift; [ $# -gt 0 ] || { echo "Missing value after --lanes" >&2; exit 1; }; LANES=$1 ;;
        --local-share) shift; [ $# -gt 0 ] || { echo "Missing value after --local-share" >&2; exit 1; }; LOCAL_SHARE=$1 ;;
        --plan) PLAN_ONLY=1 ;;
        --config-only) CONFIG_ONLY=1 ;;
        --proxy) shift; [ $# -gt 0 ] || { echo "Missing value after --proxy" >&2; exit 1; }; PROXY=$1 ;;
        --no-proxy) PROXY="" ;;
        --token) shift; [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }; TOKEN=$1 ;;
        --force) FORCE=1 ;;
        -h|--help|help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 1 ;;
    esac
    shift
done

# Default output dir follows the repo basename, so --repo alone lands in a
# distinct ./hf/<name> instead of overwriting another model's dir.
case "$LANES" in ''|*[!0-9]*|0) echo "--lanes must be a positive integer" >&2; exit 1 ;; esac

[ "$OUT_DIR_SET" = 1 ] || OUT_DIR="$ROOT/hf/${REPO##*/}"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$ROOT/$OUT_DIR" ;; esac

case "$PROXY" in off|none|no) PROXY="" ;; esac
[ "$REMOTE_PROXY" = "__INHERIT__" ] && REMOTE_PROXY=$PROXY
case "$REMOTE_PROXY" in off|none|no) REMOTE_PROXY="" ;; esac
# "No remote proxy" has to mean it: the remote's login env exports http_proxy /
# https_proxy (clash), which curl and aria2 both honor silently. Without this,
# --no-remote-proxy still tunnels every byte through the proxy.
if [ -z "$REMOTE_PROXY" ]; then
    REMOTE_PATH="$REMOTE_PATH unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY;"
fi
if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
command -v python3 >/dev/null 2>&1 || { echo "python3 is required to parse the HF file list" >&2; exit 1; }

mkdir -p "$OUT_DIR"

human() { awk -v b="$1" 'BEGIN { printf "%.1f GiB", b/1073741824 }'; }
gib()   { awk -v g="$1" 'BEGIN { printf "%.0f", g*1073741824 }'; }

curl_get() { # curl_get URL -> stdout (API helper)
    set -- -fsSL "$@"
    [ -n "$PROXY" ] && set -- -x "$PROXY" "$@"
    [ -n "$TOKEN" ] && set -- -H "Authorization: Bearer $TOKEN" "$@"
    curl "$@"
}

# --- file list (size<TAB>name) ----------------------------------------------

echo "Fetching file list for $REPO ..."
LIST=$(curl_get "$ENDPOINT/api/models/$REPO?blobs=true" | python3 -c '
import sys, json
d = json.load(sys.stdin)
sibs = d.get("siblings", [])
if not sibs: sys.exit("no siblings in API response")
for s in sibs:
    n = s.get("rfilename")
    if not n: continue
    sz = s.get("size")
    print("%d\t%s" % (sz if isinstance(sz, int) else -1, n))
') || { echo "Failed to fetch file list (proxy down? endpoint blocked?)" >&2; exit 1; }
[ -n "$LIST" ] || { echo "Empty file list from API" >&2; exit 1; }

if [ "$CONFIG_ONLY" = 1 ]; then
    LIST=$(printf '%s\n' "$LIST" | grep -v '\.safetensors$' || true)
fi

total_bytes=$(printf '%s\n' "$LIST" | awk -F"$TAB" '{ if ($1>0) t+=$1 } END { printf "%.0f", t+0 }')

# ModelScope 侧的文件清单。只用来判断"这个文件能不能走快路", 大小/总量仍以上面的
# HF 清单为准 —— 两边同名文件已逐片核对过字节一致(1-44 片, 首片 970533624)。
# 拉不到就静默退回单源: 少一条快路不该让整个下载失败。
if [ -n "$MS_REPO" ]; then
    echo "Fetching ModelScope file list for $MS_REPO ..."
    MS_LIST=$(curl -fsSL --noproxy '*' --max-time 60 \
        "$MS_ENDPOINT/api/v1/models/$MS_REPO/repo/files?Revision=master&Recursive=true" \
        | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for f in d.get("Data", {}).get("Files", []) or []:
    n = f.get("Name")
    if n: print(n)
' 2>/dev/null) || MS_LIST=""
    if [ -n "$MS_LIST" ]; then
        echo "ModelScope has $(printf '%s\n' "$MS_LIST" | grep -c '\.safetensors$') safetensors shards (fast lane)"
    else
        echo "ModelScope list unavailable -- falling back to $ENDPOINT for everything" >&2
    fi
fi

# 逐文件选源: ModelScope 有就走它(快 6 倍), 否则回落 --endpoint。
shard_url() { # name -> url on stdout
    _su_n=$1
    if [ -n "$MS_LIST" ] && printf '%s\n' "$MS_LIST" | grep -qxF "$_su_n"; then
        printf '%s/api/v1/models/%s/repo?Revision=master&FilePath=%s' \
            "$MS_ENDPOINT" "$MS_REPO" "$_su_n"
    else
        printf '%s/%s/resolve/main/%s' "$ENDPOINT" "$REPO" "$_su_n"
    fi
}

# --- free space on each disk -------------------------------------------------

local_free=$(df -Pk "$OUT_DIR" | awk 'NR==2 { printf "%.0f", $4*1024 }')

remote_free=0
if [ -n "$REMOTE_SSH" ]; then
    [ -n "$REMOTE_DIR" ] || { echo "--remote needs a DIR (USER@HOST:DIR)" >&2; exit 1; }
    echo "Probing remote $REMOTE_SSH ..."
    $SSH "$REMOTE_SSH" "mkdir -p '$REMOTE_DIR'" || { echo "Cannot ssh/mkdir on remote $REMOTE_SSH" >&2; exit 1; }
    remote_free=$($SSH "$REMOTE_SSH" "df -Pk '$REMOTE_DIR'" | awk 'NR==2 { printf "%.0f", $4*1024 }')
fi

local_reserve=$(gib "$LOCAL_RESERVE_GIB")
remote_reserve=$(gib "$REMOTE_RESERVE_GIB")
# --relay keeps extra local space free as staging scratch (shifts the local
# tail shards to the remote, which the relay then helps fetch anyway).
relay_reserve=0
[ "$RELAY" = 1 ] && [ -n "$REMOTE_SSH" ] && relay_reserve=$(gib "$RELAY_RESERVE_GIB")
local_cap=$(awk -v f="$local_free" -v r="$local_reserve" -v rr="$relay_reserve" 'BEGIN { printf "%.0f", f-r-rr }')
remote_cap=$(awk -v f="$remote_free" -v r="$remote_reserve" 'BEGIN { printf "%.0f", f-r }')

# --- already-present sizes (so resume doesn't charge downloaded bytes twice) --
# A shard already on a disk is part of the model and already accounted for in
# that disk's `free`; only the REMAINING bytes (size - present) cost new space.

# In-flight bytes live in "<name>.part" (curl) or "<name>.rr" (reverse relay),
# not under the final name. Counting only the final name would charge a resumed
# shard its FULL size against the disk budget on every re-plan, inventing a
# shortfall that isn't there.
LHAVE=$(mktemp); RMAP=$(mktemp)
printf '%s\n' "$LIST" | while IFS="$TAB" read -r size name; do
    [ -n "$name" ] || continue
    h=0
    if   [ -f "$OUT_DIR/$name" ];      then h=$(wc -c < "$OUT_DIR/$name" | tr -d ' ')
    elif [ -f "$OUT_DIR/$name.part" ]; then h=$(wc -c < "$OUT_DIR/$name.part" | tr -d ' ')
    elif [ -f "$OUT_DIR/$name.rr" ];   then h=$(wc -c < "$OUT_DIR/$name.rr" | tr -d ' ')
    fi
    printf '%s\t%s\n' "$name" "${h:-0}"
done > "$LHAVE"
: > "$RMAP"
if [ -n "$REMOTE_SSH" ]; then
    $SSH "$REMOTE_SSH" "cd '$REMOTE_DIR' 2>/dev/null && find . -type f -exec stat -f '%z %N' {} + 2>/dev/null" \
        | awk '{ sz=$1+0; nm=substr($0, index($0,$2)); sub(/^\.\//,"",nm)
                 sub(/\.part$/,"",nm); sub(/\.rr$/,"",nm)
                 if (nm!="" && sz > m[nm]) m[nm]=sz }
               END { for (k in m) print k "\t" m[k] }' > "$RMAP" || true
fi

# --- placement: keep existing shards where they are; greedily fill remaining ---
# new bytes local-first, overflow remote. Capacity is charged in REMAINING bytes.

PLAN=$(mktemp); STATS=$(mktemp)
printf '%s\n' "$LIST" | awk -F"$TAB" -v lc="$local_cap" -v rc="$remote_cap" \
    -v hasrem="$([ -n "$REMOTE_SSH" ] && echo 1 || echo 0)" -v lhf="$LHAVE" -v rmf="$RMAP" -v statsf="$STATS" \
    -v share="$LOCAL_SHARE" -v totalb="$total_bytes" '
BEGIN {
    while ((getline ln < lhf) > 0) { p=index(ln,"\t"); if (p) lh[substr(ln,1,p-1)]=substr(ln,p+1) }
    while ((getline ln < rmf) > 0) { p=index(ln,"\t"); if (p) rh[substr(ln,1,p-1)]=substr(ln,p+1) }
    target_l = (share > 0) ? share*totalb : -1
}
{
    size=$1; name=$2
    if (name=="") next
    is_shard = (name ~ /\.safetensors$/)
    lhave = (name in lh)? lh[name]+0 : 0
    rhave = (name in rh)? rh[name]+0 : 0
    lcost = size-lhave; if (lcost<0) lcost=0
    rcost = size-rhave; if (rcost<0) rcost=0
    dest="X"
    if (!is_shard)                              { dest="L" }   # config/tokenizer -> local
    else if (lhave>0)                           { dest="L" }   # continue a local partial
    else if (rhave>0 && hasrem && ru+rcost<=rc) { dest="R" }   # continue a remote partial
    # Share mode: both hosts pull at their own rate, so filling one disk first
    # leaves the other idle while the loaded one drags on alone. Cap local at
    # its target share of the model and let the rest go remote; capacity still
    # overrides (either side falls back to the other when it cannot fit).
    else if (target_l >= 0 && hasrem) {
        if      (ltot+size <= target_l && lu+lcost <= lc) { dest="L" }
        else if (ru+rcost <= rc)                          { dest="R" }
        else if (lu+lcost <= lc)                          { dest="L" }
    }
    else if (lu+lcost <= lc)                    { dest="L" }   # keep local / fill local
    else if (hasrem && ru+rcost <= rc)          { dest="R" }   # overflow to remote
    if (dest=="L")      { lu+=lcost; ltot+=size; lnew+=lcost; nl++ }
    else if (dest=="R") { ru+=rcost; rtot+=size; rnew+=rcost; nr++ }
    else                { xtot+=size; xnew+=(lcost<rcost?lcost:rcost) }
    print dest "\t" size "\t" name
}
END { printf "%.0f %.0f %.0f %.0f %.0f %.0f %d %d\n", ltot,rtot,xtot,lnew,rnew,xnew,nl,nr > statsf }' > "$PLAN"

read local_use remote_use unfit_gross local_new remote_new unfit n_local n_remote < "$STATS"
rm -f "$LHAVE" "$RMAP" "$STATS"

echo
echo "Repo:        $REPO"
echo "Proxy:       ${PROXY:-<none>}   Endpoint: $ENDPOINT"
echo "Total model: $(human "$total_bytes")"
echo
echo "LOCAL  $OUT_DIR"
echo "   free $(human "$local_free")  reserve ${LOCAL_RESERVE_GIB} GiB  ->  $n_local files / $(human "$local_use") on disk  ($(human "$local_new") new -> free after ~$(human $((local_free-local_new))))"
if [ -n "$REMOTE_SSH" ]; then
echo "REMOTE $REMOTE_SSH:$REMOTE_DIR"
echo "   free $(human "$remote_free")  reserve ${REMOTE_RESERVE_GIB} GiB  ->  $n_remote files / $(human "$remote_use") on disk  ($(human "$remote_new") new -> free after ~$(human $((remote_free-remote_new))))"
fi
echo

if [ "$unfit" -gt 0 ] 2>/dev/null && [ "$FORCE" != 1 ]; then
    echo "ERROR: $(human "$unfit") of shards (still to download) do not fit in the reserved budgets." >&2
    if [ -z "$REMOTE_SSH" ]; then
        echo "  Add a second disk with --remote USER@HOST:DIR, or --config-only." >&2
    else
        echo "  Lower --local-reserve / --remote-reserve, free space, or add a disk." >&2
    fi
    rm -f "$PLAN"; exit 1
fi

if [ "$PLAN_ONLY" = 1 ]; then
    echo "Plan only (no downloads). Files:"
    awk -F"$TAB" '{ printf "  %s  %s\n", $1, $3 }' "$PLAN"
    rm -f "$PLAN"; exit 0
fi

# --- downloaders -------------------------------------------------------------

file_size_local() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }

# API 报的 size 只对 LFS 文件(权重分片)可信。普通小文件拿到的是 git blob 大小,
# 跟镜像实际吐出的字节常对不上 —— 实测 hf-mirror 上 README.md api=1622 实际=12944,
# encoding/encoding.py api=35338 实际=35310。拿它当完成判据, 这些文件会永远"没下完",
# 每轮重下一次且永不转正。1 MiB 以上按 LFS 严格逐字节卡, 以下只要求非空。
size_trusted() { [ "$1" -ge 1048576 ] 2>/dev/null; }
file_size_remote() { # always emits an integer (0 if missing/unreachable)
    # </dev/null: ssh must NOT read the caller's stdin (it would swallow the
    # PLAN file when called inside a `while read ... < PLAN` loop).
    n=$($SSH "$REMOTE_SSH" "wc -c < '$1' 2>/dev/null" </dev/null 2>/dev/null | tr -d ' ')
    [ -n "$n" ] && echo "$n" || echo 0
}

have_aria2=0
[ "$USE_ARIA2" = 1 ] && command -v aria2c >/dev/null 2>&1 && have_aria2=1

download_local() { # size name
    size=$1; name=$2; out="$OUT_DIR/$name"; url=$(shard_url "$name")
    # 仓库里带子目录的文件(V4.1 有 encoding/ assets/ inference/ evaluation/)必须先
    # 建父目录: aria2c 会自己建, curl 不会 —— 它报 "(23) Failure writing output to
    # destination" 然后被上面的重试循环反复重试到耗尽, 看着像网络问题, 其实是本地
    # 目录不存在。老的 BASE 仓库全是平铺文件, 所以这条路一直没被走到过。
    case "$name" in */*) mkdir -p "$(dirname "$out")" ;; esac
    have=$(file_size_local "$out")
    if size_trusted "$size"; then
        [ "$have" = "$size" ] && { echo "ok  L  $name ($(human "$size"))"; return; }
    else
        [ "$have" -gt 0 ] && { echo "ok  L  $name ($(human "$have"))"; return; }
    fi
    echo "get L  $name ($(human "$size"))"
    if [ "$have_aria2" = 1 ]; then
        # Flaky proxy/CDN + 1h-expiring HF presigned URLs: retry forever (each
        # retry re-resolves a fresh signed URL, so 403-on-expiry self-heals),
        # drop stalled connections, but still give up on a genuine 404.
        set -- -c --auto-file-renaming=false --allow-overwrite=true --file-allocation=none \
            -x 8 -s 8 -k 1M --console-log-level=warn --summary-interval=0 \
            --max-tries=0 --retry-wait=10 --timeout=60 --connect-timeout=30 \
            --max-file-not-found=5 \
            -d "$OUT_DIR" -o "$name"
        # No --proxy => actively disable it; a proxy in the environment would
        # otherwise be picked up and silently used.
        if [ -n "$PROXY" ]; then set -- --all-proxy="$PROXY" "$@"; else set -- --no-proxy='*' "$@"; fi
        [ -n "$TOKEN" ] && set -- --header="Authorization: Bearer $TOKEN" "$@"
        aria2c "$@" "$url" || true
    else
        # -s: this runs for hours into a log file; curl's progress meter would
        # bury the get/ok lines under megabytes of redraw noise. Progress is
        # observed from the file sizes instead (tools/dl_0731_progress.sh).
        # 僵死熔断 (2026-09-10): 这条 curl 以前只有 -fsSL -C -, 没有超时也没有重试。
        # hf-mirror 的连接会挂住既不返回也不报错 —— 实测 encoding/ 里几 KB 的小文件
        # 卡了十分钟 0 字节, 而 aria2c 分支一直有 --timeout/--max-tries, 只有 curl
        # 分支漏了。不加会怎样: 那条 lane 静默停摆, 日志既不报错也不推进, 看着像
        # "还在下载", 实际 0 进度(历史上同样的裸 curl 挂过 6 天)。
        # --speed-limit/--speed-time: 60 秒内平均低于 50 KB/s 就掐断; -C - 让下一次
        # 重试从已落盘的字节接着走, 所以掐断不浪费已下内容。
        set -- -fsSL -C - --connect-timeout 30 --retry 5 --retry-delay 5 \
            --speed-limit 51200 --speed-time 60
        if [ -n "$PROXY" ]; then set -- -x "$PROXY" "$@"; else set -- --noproxy '*' "$@"; fi
        [ -n "$TOKEN" ] && set -- -H "Authorization: Bearer $TOKEN" "$@"
        _try=0
        while [ "$_try" -lt 30 ]; do
            curl "$@" -o "$out.part" "$url" && break
            _rc=$?
            # curl 33 = 服务端不认 Range, 续传没法做。ModelScope 的下载端点对超大文件
            # 就是这样: model-00047/48(各 101.5 GB 的 engram 片)首段请求返回 200 而不是
            # 206(Range 头被整个忽略), 中段请求直接 404 —— 而前 46 片是支持的。
            # 后果: .part 一旦非空, 之后每一次 -C - 都以 33 失败, 那条 lane 一个字节也
            # 推不进去, 日志刷满 "Cannot resume" 却看不出是源的问题(实测卡在 0.7/0.5 GiB)。
            # 只能丢掉已下字节从头拉。代价是断一次就得重来, 所以别在这种源上做长尾重试:
            # 拉不动就让它落到下一轮, 由 hf-mirror 那条(支持 Range)接手。
            if [ "$_rc" = 33 ]; then rm -f "$out.part"; fi
            _try=$((_try + 1))
            sleep 5
        done
        # 只有拿到 API 声明的完整字节才转正, 否则留 .part 等下一轮续传 —— 半截文件
        # 改名成正式名会让下一次运行的 size 检查判定"已完成", 静默交付一个残缺分片。
        _got=$(file_size_local "$out.part")
        if [ "$_got" -gt 0 ] && { ! size_trusted "$size" || [ "$_got" = "$size" ]; }; then
            mv "$out.part" "$out"
        elif [ "$_got" -gt 0 ]; then
            echo "ERR L  $name: $(human "$_got") / $(human "$size") — 留作 .part, 重跑续传" >&2
        fi
    fi
}

download_remote() { # size name  (stream through this host, byte-level resumable)
    size=$1; name=$2; rpath="$REMOTE_DIR/$name"; url="$ENDPOINT/$REPO/resolve/main/$name"
    have=$(file_size_remote "$rpath")
    if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" = "$size" ]; then echo "ok  R  $name ($(human "$size"))"; return; fi
    # Over-size => a prior attempt was corrupted (e.g. CDN ignored Range); reset.
    if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" -gt "$size" ] 2>/dev/null; then
        $SSH "$REMOTE_SSH" ": > '$rpath'"; have=0
    fi
    set -- -fsSL
    [ -n "$PROXY" ] && set -- -x "$PROXY" "$@"
    [ -n "$TOKEN" ] && set -- -H "Authorization: Bearer $TOKEN" "$@"
    if [ "$have" -gt 0 ] 2>/dev/null; then
        echo "get R  $name  resume @ $(human "$have") / $(human "$size") -> $REMOTE_SSH"
        curl "$@" -r "${have}-" "$url" | $SSH "$REMOTE_SSH" "cat >> '$rpath'" || true
        # Self-heal: if the server ignored Range it appended a full body; the file
        # overshoots its size, so reset it for a clean re-download next run.
        got=$(file_size_remote "$rpath")
        if [ "$size" -ge 0 ] 2>/dev/null && [ "$got" -gt "$size" ] 2>/dev/null; then
            echo "warn R  $name range not honored; resetting for clean re-download" >&2
            $SSH "$REMOTE_SSH" ": > '$rpath'"
        fi
    else
        echo "get R  $name ($(human "$size")) -> $REMOTE_SSH"
        curl "$@" "$url" | $SSH "$REMOTE_SSH" "cat > '$rpath'" || true
    fi
}

# --- remote DIRECT downloader (runs ON the remote, in parallel with us) -------

remote_has_aria2=0
remote_check_aria2() {
    [ -n "$REMOTE_SSH" ] || return 0
    [ "$USE_ARIA2" = 1 ] || return 0
    if $SSH "$REMOTE_SSH" "$REMOTE_PATH command -v aria2c >/dev/null 2>&1"; then remote_has_aria2=1; fi
}

# shell-quote a value for safe embedding in the remote command string
shq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

download_remote_direct() { # size name  (remote fetches it itself via its proxy)
    size=$1; name=$2; rpath="$REMOTE_DIR/$name"; url="$ENDPOINT/$REPO/resolve/main/$name"
    have=$(file_size_remote "$rpath")
    if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" = "$size" ]; then echo "ok  R  $name ($(human "$size"))"; return; fi
    echo "get R  $name ($(human "$size")) @ $REMOTE_SSH direct"
    if [ "$remote_has_aria2" = 1 ]; then
        # retry forever through proxy/CDN drops (re-resolves a fresh signed URL
        # each try, so HF's 1h URL-expiry 403 self-heals); give up only on 404.
        rcmd="$REMOTE_PATH aria2c -c --auto-file-renaming=false --allow-overwrite=true --file-allocation=none -x 8 -s 8 -k 1M --console-log-level=warn --summary-interval=0 --max-tries=0 --retry-wait=10 --timeout=60 --connect-timeout=30 --max-file-not-found=5 -d $(shq "$REMOTE_DIR") -o $(shq "$name")"
        [ -n "$REMOTE_PROXY" ] && rcmd="$rcmd --all-proxy=$(shq "$REMOTE_PROXY")"
        [ -n "$TOKEN" ] && rcmd="$rcmd --header=$(shq "Authorization: Bearer $TOKEN")"
        rcmd="$rcmd $(shq "$url")"
    else
        # curl resumes against the .part target (-C -); finalize with mv on success.
        rcmd="curl -fsSL -C - --retry 5 --retry-delay 5"
        [ -n "$REMOTE_PROXY" ] && rcmd="$rcmd -x $(shq "$REMOTE_PROXY")"
        [ -n "$TOKEN" ] && rcmd="$rcmd -H $(shq "Authorization: Bearer $TOKEN")"
        rcmd="$rcmd -o $(shq "$rpath.part") $(shq "$url") && mv $(shq "$rpath.part") $(shq "$rpath")"
    fi
    # </dev/null: keep ssh off the caller's stdin (the PLAN file in the loop).
    $SSH "$REMOTE_SSH" "$rcmd" </dev/null || true
}

# Both passes loop until every assigned shard verifies complete, so one shard
# giving up (proxy died, URL expired) never strands the rest of that machine's
# line — the whole point of running both hosts in parallel.
# Read the PLAN on FD 3, not stdin: the ssh calls below would otherwise consume
# the plan file as their stdin and end the loop after one shard.
# Lanes pull from a shared list instead of owning a fixed slice of it.
#
# The fixed-slice version (lane i takes every LANES-th file) collapses at the
# tail: a lane that finishes its slice EXITS, and since the pass waits for all
# lanes before starting the next round, concurrency decays to however many lanes
# still hold an unfinished file. Measured mid-run: M1 down to 2 live connections
# of 16, 0.81 MB/s instead of ~3.8.
#
# Claiming is an atomic mkdir. A claim is RELEASED only when the file did not
# complete, so a finished file is never re-claimed inside the same round (which
# would spin); the round's retry loop wipes all claims and starts over.
claim_key() { printf '%s' "$1" | tr '/' '_'; }
claim()   { mkdir "$CLAIMD/$(claim_key "$1")" 2>/dev/null; }
unclaim() { rmdir "$CLAIMD/$(claim_key "$1")" 2>/dev/null || true; }

remote_lane() {
    while :; do
        _got=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = R ] || continue
            claim "$name" || continue
            if [ "$REMOTE_MODE" = direct ]; then download_remote_direct "$size" "$name"
            else download_remote "$size" "$name"; fi
            have=$(file_size_remote "$REMOTE_DIR/$name")
            [ "$have" = "$size" ] || unclaim "$name"
            _got=1
            break
        done 3< "$PLAN"
        [ "$_got" = 0 ] && break
    done
}
local_lane() {
    while :; do
        _got=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = L ] || continue
            # Handed to the remote's reverse relay — it owns this .part now.
            [ -d "$TAKEOVER_D/$(claim_key "$name")" ] && continue
            claim "$name" || continue
            download_local "$size" "$name"
            have=$(file_size_local "$OUT_DIR/$name")
            # 判据必须和 download_local 一致。用不可信的小文件 size 比对会判成"没下完"
            # 而 unclaim, 于是下一条 lane 立刻重新 claim 同一个文件, download_local 打
            # 一行 ok 就返回, 再 unclaim —— 所有 lane 死死卡在这 36 个小文件上空转
            # (实测 21154 行 ok), 只剩一条 lane 真在下权重分片, 8 并发退化成 1。
            if size_trusted "$size"; then
                [ "$have" = "$size" ] || unclaim "$name"
            else
                [ "$have" -gt 0 ] || unclaim "$name"
            fi
            _got=1
            break
        done 3< "$PLAN"
        [ "$_got" = 0 ] && break
    done
}

run_remote_pass() {
    while :; do
        CLAIMD=$(mktemp -d)
        # Wait on THESE lanes by pid: a bare `wait` would also block on the
        # other host's pass, which runs as a sibling background job.
        _l=0; _pids=""
        while [ "$_l" -lt "$LANES" ]; do remote_lane & _pids="$_pids $!"; _l=$((_l+1)); done
        wait $_pids
        rm -rf "$CLAIMD"
        pending=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = R ] || continue
            have=$(file_size_remote "$REMOTE_DIR/$name")
            # 与 download_local 同一判据: 小文件的 API size 不可信, 拿它比对会让
            # 这一轮永远 pending, 外层 10s 一轮无限重试, 每轮把已完成的全部重扫重打
            # (实测刷了 55552 行 ok)。只有 LFS 分片才逐字节卡。
            if size_trusted "$size"; then
                [ "$have" != "$size" ] && pending=1
            else
                [ "$have" -gt 0 ] || pending=1
            fi
        done 3< "$PLAN"
        [ "$pending" = 0 ] && break
        echo "remote: shards still incomplete, retry round in 10s ..."
        sleep 10
    done
}
run_local_pass() {
    while :; do
        CLAIMD=$(mktemp -d)
        _l=0; _pids=""
        while [ "$_l" -lt "$LANES" ]; do local_lane & _pids="$_pids $!"; _l=$((_l+1)); done
        wait $_pids
        rm -rf "$CLAIMD"
        pending=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = L ] || continue
            have=$(file_size_local "$OUT_DIR/$name")
            # 与 download_local 同一判据: 小文件的 API size 不可信, 拿它比对会让
            # 这一轮永远 pending, 外层 10s 一轮无限重试, 每轮把已完成的全部重扫重打
            # (实测刷了 55552 行 ok)。只有 LFS 分片才逐字节卡。
            if size_trusted "$size"; then
                [ "$have" != "$size" ] && pending=1
            else
                [ "$have" -gt 0 ] || pending=1
            fi
        done 3< "$PLAN"
        [ "$pending" = 0 ] && break
        echo "local: shards still incomplete, retry round in 10s ..."
        sleep 10
    done
}

# --- reverse relay: idle REMOTE host helps fetch the local's shards ----------
# Whichever host is given the smaller slice finishes first and would then sit
# idle while the other drags the tail alone. This is the remote->local
# direction: the finished remote downloads a shard the local host still owes,
# into a scratch dir on its own disk, and we pull it over the bridge (measured
# 1.0 GB/s, so the transfer is noise next to the ~3 MB/s internet leg).
#
# The two hosts walk the local list from opposite ends (local lanes ascending,
# this descending) so they meet in the middle instead of racing for the same
# file. Overlap is still possible; it costs duplicate bytes, never corruption
# (each side writes its own staging file and the size check gates the rename).

local_actively_downloading() { # name -> true if the local .part was touched recently
    _f="$OUT_DIR/$1.part"
    [ -f "$_f" ] || return 1
    _n=$(date +%s); _m=$(stat -f %m "$_f" 2>/dev/null || echo 0)
    [ $((_n - _m)) -lt 90 ]
}

# Streamed, not staged: the remote's curl writes to stdout and ssh carries the
# bytes straight into the local .part. Nothing lands on the remote's disk (it
# has ~7 GiB left once its own share is done, too little to stage 3 GiB
# shards), and with no disk cost this runs LANES-wide like a normal pass.
#
# It appends to the SAME .part the local lane uses, so an in-flight shard keeps
# the gigabytes already fetched instead of restarting. That makes handoff
# mandatory: the local lane must release the file first, or both hosts write
# the same descriptor and corrupt it. TAKEOVER_D is the shared interlock —
# reverse relay marks a name there and kills the local curl; local_lane skips
# any name marked in it for the rest of the run.
local_curl_pid() { pgrep -f "curl .*$1\$" 2>/dev/null | head -1; }

reverse_relay_fetch() { # size name
    size=$1; name=$2; out="$OUT_DIR/$name"; part="$out.part"
    url="$ENDPOINT/$REPO/resolve/main/$name"
    mkdir -p "$TAKEOVER_D/$(claim_key "$name")" 2>/dev/null    # local_lane stands down
    p=$(local_curl_pid "$name")
    if [ -n "$p" ]; then kill "$p" 2>/dev/null || true; sleep 2; fi
    have=$(file_size_local "$part")
    echo "rrelay take $name ($(human "$size")) resume @ $(human "${have:-0}") via $REMOTE_SSH"
    if [ "${have:-0}" -gt 0 ] 2>/dev/null; then
        $SSH "$REMOTE_SSH" "$REMOTE_PATH curl -fsSL -r ${have}- '$url'" </dev/null >> "$part" || true
    else
        $SSH "$REMOTE_SSH" "$REMOTE_PATH curl -fsSL '$url'" </dev/null > "$part" || true
    fi
    got=$(file_size_local "$part")
    if [ "$got" = "$size" ]; then
        mv -f "$part" "$out"
        echo "rrelay ok   $name delivered"
    elif [ "$size" -ge 0 ] 2>/dev/null && [ "$got" -gt "$size" ] 2>/dev/null; then
        # Range ignored by the CDN: the body was appended whole. Start clean.
        echo "rrelay warn $name range not honored; resetting" >&2
        rm -f "$part"
        rmdir "$TAKEOVER_D/$(claim_key "$name")" 2>/dev/null || true
    else
        # Did not finish: hand the file back so the local lanes resume it.
        rmdir "$TAKEOVER_D/$(claim_key "$name")" 2>/dev/null || true
    fi
}

# How many shards the remote still owes on its OWN share. Reverse relay only
# uses the lanes left over from that, so it ramps up as the remote drains its
# list instead of competing with it from the start.
remote_pending_count() {
    $SSH "$REMOTE_SSH" "ls '$REMOTE_DIR'/*.part 2>/dev/null | wc -l" </dev/null 2>/dev/null | tr -d ' '
}
local_all_done() {
    while IFS="$TAB" read -r size name <&3; do
        [ -n "$name" ] || continue
        [ "$(file_size_local "$OUT_DIR/$name")" = "$size" ] || return 1
    done 3< "$llist"
    return 0
}
local_pending_count() {
    _n=0
    while IFS="$TAB" read -r size name <&3; do
        [ -n "$name" ] || continue
        [ "$(file_size_local "$OUT_DIR/$name")" = "$size" ] || _n=$((_n + 1))
    done 3< "$llist"
    echo "$_n"
}

rrelay_lane() {
    while :; do
        _got=0
        while IFS="$TAB" read -r size name <&3; do
            [ -n "$name" ] || continue
            have=$(file_size_local "$OUT_DIR/$name")
            [ "$have" = "$size" ] && continue
            claim "$name" || continue
            reverse_relay_fetch "$size" "$name"
            have=$(file_size_local "$OUT_DIR/$name")
            [ "$have" = "$size" ] || unclaim "$name"
            _got=1
            break
        done 3< "$llist"
        [ "$_got" = 0 ] && break
    done
}

reverse_relay_pass() {
    [ -n "$REMOTE_SSH" ] || return 0
    llist=$(mktemp)
    # Descending: local lanes walk the list ascending, so the two sweeps meet in
    # the middle rather than fighting over the same shard.
    awk -F"$TAB" '$1=="L" && $3 ~ /\.safetensors$/ { print $2 "\t" $3 }' "$PLAN" | tail -r > "$llist"
    echo "reverse-relay: armed; will use whatever lanes the remote isn't using"
    while :; do
        local_all_done && break
        rleft=$(remote_pending_count)
        rr=$(( LANES - ${rleft:-0} ))                     # lanes the remote has spare
        # ...but never take more than half of what's left: one shard can only be
        # fetched by one host, so grabbing them all idles the LOCAL machine
        # (observed: M4 dropped to 0 connections while M1 ran 13). Split the
        # remaining files between the two instead.
        lleft=$(local_pending_count)
        half=$(( (lleft + 1) / 2 ))
        [ "$rr" -gt "$half" ] && rr=$half
        if [ "$rr" -lt 1 ]; then sleep 30; continue; fi   # remote still busy with its own
        CLAIMD=$(mktemp -d)
        _l=0; _pids=""
        while [ "$_l" -lt "$rr" ]; do rrelay_lane & _pids="$_pids $!"; _l=$((_l+1)); done
        wait $_pids
        rm -rf "$CLAIMD"
        rpending=0
        while IFS="$TAB" read -r size name <&3; do
            [ -n "$name" ] || continue
            have=$(file_size_local "$OUT_DIR/$name")
            if [ "$have" != "$size" ]; then rpending=1; fi
        done 3< "$llist"
        [ "$rpending" = 0 ] && break
        sleep 10
    done
    rm -f "$llist"
}

# --- relay-assist: idle local host helps fetch the remote's shards -----------
# Runs AFTER the local pass, concurrently with the remote's own (ascending)
# pass. Walks the remote shard set DESCENDING so the two meet in the middle;
# each shard is staged here, pushed over the bridge, then deleted locally.

remote_actively_downloading() { # name -> true if remote has a fresh .aria2 for it
    r=$($SSH "$REMOTE_SSH" "f='$REMOTE_DIR/$1.aria2'; [ -f \"\$f\" ] || exit 1; n=\$(date +%s); m=\$(stat -f %m \"\$f\" 2>/dev/null || echo 0); [ \$((n-m)) -lt 90 ] && echo active" </dev/null 2>/dev/null)
    [ "$r" = active ]
}

relay_fetch_local() { # size name -> stage into $scratch
    _name=$2; _out="$scratch/$_name"; _url="$ENDPOINT/$REPO/resolve/main/$_name"
    if [ "$have_aria2" = 1 ]; then
        set -- -c --auto-file-renaming=false --allow-overwrite=true --file-allocation=none \
            -x 8 -s 8 -k 1M --console-log-level=warn --summary-interval=0 \
            --max-tries=0 --retry-wait=10 --timeout=60 --connect-timeout=30 --max-file-not-found=5 \
            -d "$scratch" -o "$_name"
        [ -n "$PROXY" ] && set -- --all-proxy="$PROXY" "$@"
        [ -n "$TOKEN" ] && set -- --header="Authorization: Bearer $TOKEN" "$@"
        aria2c "$@" "$_url" </dev/null || true
    else
        set -- -fSL -C -
        [ -n "$PROXY" ] && set -- -x "$PROXY" "$@"
        [ -n "$TOKEN" ] && set -- -H "Authorization: Bearer $TOKEN" "$@"
        curl "$@" -o "$_out.part" "$_url" </dev/null && mv "$_out.part" "$_out" || true
    fi
}

relay_push() { # name -> copy $scratch/name to remote (temp then atomic mv)
    _name=$1; _src="$scratch/$_name"; _tmp="$REMOTE_DIR/$_name.relaytmp"; _dst="$REMOTE_DIR/$_name"
    if command -v rsync >/dev/null 2>&1; then
        rsync -e "$SSH" --partial --inplace --no-whole-file "$_src" "$REMOTE_SSH:$_tmp" </dev/null || return 1
    else
        scp -o BatchMode=yes -o ConnectTimeout=10 "$_src" "$REMOTE_SSH:$_tmp" </dev/null || return 1
    fi
    $SSH "$REMOTE_SSH" "mv -f '$_tmp' '$_dst'" </dev/null || return 1
}

run_relay_pass() {
    [ -n "$REMOTE_SSH" ] || return 0
    scratch="${OUT_DIR}.relay"; mkdir -p "$scratch"   # sibling dir, same disk
    rlist=$(mktemp)
    awk -F"$TAB" '$1=="R"{print $2"\t"$3}' "$PLAN" | tail -r > "$rlist"   # descending
    echo "relay: local pass done; helping remote with its outstanding shards ..."
    while :; do
        pending=0
        while IFS="$TAB" read -r size name <&3; do
            [ -n "$name" ] || continue
            rhave=$(file_size_remote "$REMOTE_DIR/$name")
            if [ "$size" -ge 0 ] 2>/dev/null && [ "$rhave" = "$size" ]; then
                rm -f "$scratch/$name" "$scratch/$name.aria2" 2>/dev/null; continue
            fi
            if remote_actively_downloading "$name"; then pending=1; continue; fi  # remote owns it
            need=$(awk -v s="$size" 'BEGIN{printf "%.0f", s+1073741824}')
            lf=$(df -Pk "$scratch" | awk 'NR==2{printf "%.0f", $4*1024}')
            if [ "$lf" -lt "$need" ] 2>/dev/null; then echo "relay: low local scratch, waiting ..."; pending=1; sleep 15; continue; fi
            echo "relay get  $name ($(human "$size")) -> staging here"
            relay_fetch_local "$size" "$name"
            lh=$(file_size_local "$scratch/$name")
            if [ "$size" -ge 0 ] 2>/dev/null && [ "$lh" != "$size" ]; then echo "relay: $name incomplete locally, will retry"; pending=1; continue; fi
            # re-check right before delivering: remote may have taken/finished it
            rhave=$(file_size_remote "$REMOTE_DIR/$name")
            if [ "$rhave" = "$size" ]; then rm -f "$scratch/$name" "$scratch/$name.aria2" 2>/dev/null; continue; fi
            if remote_actively_downloading "$name"; then pending=1; continue; fi
            echo "relay push $name -> $REMOTE_SSH:$REMOTE_DIR"
            if relay_push "$name"; then
                rhave=$(file_size_remote "$REMOTE_DIR/$name")
                if [ "$rhave" = "$size" ]; then rm -f "$scratch/$name" "$scratch/$name.aria2" 2>/dev/null; echo "relay ok   $name delivered"
                else pending=1; fi
            else echo "relay: push of $name failed, will retry" >&2; pending=1; fi
        done 3< "$rlist"
        [ "$pending" = 0 ] && break
        sleep 10
    done
    rm -f "$rlist"; rmdir "$scratch" 2>/dev/null || true
}

# Shared interlock: names the remote's reverse relay has taken over, so the
# local lanes stop writing those .part files. Defined for every mode so the
# lanes' lookup never sees an empty path.
TAKEOVER_D=$(mktemp -d)

echo "Downloading (re-run to resume)..."
[ -n "$REMOTE_SSH" ] && echo "Remote mode: $REMOTE_MODE${REMOTE_PROXY:+  remote-proxy $REMOTE_PROXY}"
[ "$RELAY" = 1 ] && [ -n "$REMOTE_SSH" ] && echo "Relay: ON  (local host helps fetch remote shards; ${RELAY_RESERVE_GIB} GiB scratch reserved)"
echo
if [ -n "$REMOTE_SSH" ] && [ "$REMOTE_MODE" = direct ] && [ "$n_remote" -gt 0 ] 2>/dev/null; then
    # Both machines pull from HF at once: remote in the background, local here.
    remote_check_aria2
    run_remote_pass &
    REMOTE_PID=$!
    # Runs alongside, not after: it sizes itself to the remote's spare lanes, so
    # it stays out of the way early and ramps up as the remote drains its share.
    RRELAY_PID=""
    if [ "$REVERSE_RELAY" = 1 ]; then reverse_relay_pass & RRELAY_PID=$!; fi
    run_local_pass
    [ "$RELAY" = 1 ] && run_relay_pass   # idle local host now drains remote's set too
    wait "$REMOTE_PID" 2>/dev/null || true
    [ -n "$RRELAY_PID" ] && { wait "$RRELAY_PID" 2>/dev/null || true; }
    rm -rf "$TAKEOVER_D"
else
    run_local_pass
    [ -n "$REMOTE_SSH" ] && run_remote_pass
fi

# --- verify ------------------------------------------------------------------

echo
incomplete=""
# dest "X" = the placement pass found no disk with room for it. Only --force
# gets this far; the shard was never attempted, so report it separately from a
# genuinely failed transfer — otherwise a partial model reads as "complete".
unplaced=""
while IFS="$TAB" read -r dest size name <&3; do
    [ -n "$name" ] || continue
    [ "$size" -ge 0 ] 2>/dev/null || continue
    case "$dest" in
        L) got=$(file_size_local "$OUT_DIR/$name") ;;
        R) got=$(file_size_remote "$REMOTE_DIR/$name") ;;
        *) unplaced="$unplaced $name"; continue ;;
    esac
    [ "$got" = "$size" ] || incomplete="$incomplete $dest:$name"
done 3< "$PLAN"
rm -f "$PLAN"

if [ -n "$unplaced" ]; then
    echo "NOT DOWNLOADED — no disk had room ($(human "$unfit_gross")); free space and re-run:" >&2
    for f in $unplaced; do echo "  $f" >&2; done
    echo >&2
fi

if [ -n "$incomplete" ]; then
    echo "Incomplete (re-run to resume):" >&2
    for f in $incomplete; do echo "  $f" >&2; done
    exit 1
fi

if [ -n "$unplaced" ]; then
    echo "Placed files complete, but the model is PARTIAL (see above)."
else
    echo "All files complete."
fi
echo "  local : $OUT_DIR"
[ -n "$REMOTE_SSH" ] && echo "  remote: $REMOTE_SSH:$REMOTE_DIR"
echo
echo "For quantization, the --hf dir must contain ALL shards together; mount the"
echo "remote portion over the bridge (e.g. SMB) into $OUT_DIR before running"
echo "  gguf-tools/deepseek4-quantize --hf \"$OUT_DIR\" --template ... --out ..."
echo
echo "Done."
# Partial model => nonzero exit, so a caller/cron never mistakes it for a
# finished download.
if [ -n "$unplaced" ]; then exit 1; fi
