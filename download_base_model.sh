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

REPO="deepseek-ai/DeepSeek-V4-Flash-Base"
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_HF_DIR:-"$ROOT/hf/DeepSeek-V4-Flash-Base"}
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$ROOT/$OUT_DIR" ;; esac

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

# Free-space headroom to keep on each disk (GiB). The model is a tight fit, so
# these are small by necessity; raise them and free space if you want margin.
LOCAL_RESERVE_GIB=${LOCAL_RESERVE_GIB:-4}
REMOTE_RESERVE_GIB=${REMOTE_RESERVE_GIB:-4}

CONFIG_ONLY=0
PLAN_ONLY=0
FORCE=0

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
  --relay                 After finishing local shards, also fetch the remote's
                          outstanding shards here and push them over the bridge
                          (both proxies drain the remote set). Keeps
                          --relay-reserve GiB free locally as staging scratch.
  --relay-reserve N       GiB to keep free locally for relay staging (default 14).
  --dir DIR               Local output dir (default: ./hf/DeepSeek-V4-Flash-Base
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
  DS4_HF_DIR, HF_PROXY (off to disable), HF_ENDPOINT, HF_TOKEN,
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
            OUT_DIR=$1; case "$OUT_DIR" in /*) ;; *) OUT_DIR="$ROOT/$OUT_DIR" ;; esac ;;
        --remote-mode)
            shift; [ $# -gt 0 ] || { echo "Missing value after --remote-mode" >&2; exit 1; }
            REMOTE_MODE=$1
            case "$REMOTE_MODE" in direct|stream) ;; *) echo "--remote-mode must be direct|stream" >&2; exit 1 ;; esac ;;
        --remote-proxy) shift; [ $# -gt 0 ] || { echo "Missing value after --remote-proxy" >&2; exit 1; }; REMOTE_PROXY=$1 ;;
        --no-remote-proxy) REMOTE_PROXY="" ;;
        --relay) RELAY=1 ;;
        --relay-reserve) shift; [ $# -gt 0 ] || { echo "Missing value after --relay-reserve" >&2; exit 1; }; RELAY_RESERVE_GIB=$1 ;;
        --local-reserve)  shift; LOCAL_RESERVE_GIB=$1 ;;
        --remote-reserve) shift; REMOTE_RESERVE_GIB=$1 ;;
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

case "$PROXY" in off|none|no) PROXY="" ;; esac
[ "$REMOTE_PROXY" = "__INHERIT__" ] && REMOTE_PROXY=$PROXY
case "$REMOTE_PROXY" in off|none|no) REMOTE_PROXY="" ;; esac
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

LHAVE=$(mktemp); RMAP=$(mktemp)
printf '%s\n' "$LIST" | while IFS="$TAB" read -r size name; do
    [ -n "$name" ] || continue
    h=0; [ -f "$OUT_DIR/$name" ] && h=$(wc -c < "$OUT_DIR/$name" | tr -d ' ')
    printf '%s\t%s\n' "$name" "${h:-0}"
done > "$LHAVE"
: > "$RMAP"
if [ -n "$REMOTE_SSH" ]; then
    $SSH "$REMOTE_SSH" "cd '$REMOTE_DIR' 2>/dev/null && find . -type f -exec stat -f '%z %N' {} + 2>/dev/null" \
        | awk '{ sz=$1; nm=substr($0, index($0,$2)); sub(/^\.\//,"",nm); if (nm!="") print nm "\t" sz }' > "$RMAP" || true
fi

# --- placement: keep existing shards where they are; greedily fill remaining ---
# new bytes local-first, overflow remote. Capacity is charged in REMAINING bytes.

PLAN=$(mktemp); STATS=$(mktemp)
printf '%s\n' "$LIST" | awk -F"$TAB" -v lc="$local_cap" -v rc="$remote_cap" \
    -v hasrem="$([ -n "$REMOTE_SSH" ] && echo 1 || echo 0)" -v lhf="$LHAVE" -v rmf="$RMAP" -v statsf="$STATS" '
BEGIN {
    while ((getline ln < lhf) > 0) { p=index(ln,"\t"); if (p) lh[substr(ln,1,p-1)]=substr(ln,p+1) }
    while ((getline ln < rmf) > 0) { p=index(ln,"\t"); if (p) rh[substr(ln,1,p-1)]=substr(ln,p+1) }
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
    else if (rhave>0 && hasrem && ru+rcost<=rc) { dest="R" }   # continue a remote partial
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
file_size_remote() { # always emits an integer (0 if missing/unreachable)
    # </dev/null: ssh must NOT read the caller's stdin (it would swallow the
    # PLAN file when called inside a `while read ... < PLAN` loop).
    n=$($SSH "$REMOTE_SSH" "wc -c < '$1' 2>/dev/null" </dev/null 2>/dev/null | tr -d ' ')
    [ -n "$n" ] && echo "$n" || echo 0
}

have_aria2=0; command -v aria2c >/dev/null 2>&1 && have_aria2=1

download_local() { # size name
    size=$1; name=$2; out="$OUT_DIR/$name"; url="$ENDPOINT/$REPO/resolve/main/$name"
    have=$(file_size_local "$out")
    if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" = "$size" ]; then echo "ok  L  $name ($(human "$size"))"; return; fi
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
        [ -n "$PROXY" ] && set -- --all-proxy="$PROXY" "$@"
        [ -n "$TOKEN" ] && set -- --header="Authorization: Bearer $TOKEN" "$@"
        aria2c "$@" "$url" || true
    else
        set -- -fL --progress-meter -C -
        [ -n "$PROXY" ] && set -- -x "$PROXY" "$@"
        [ -n "$TOKEN" ] && set -- -H "Authorization: Bearer $TOKEN" "$@"
        curl "$@" -o "$out.part" "$url" && mv "$out.part" "$out" || true
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
    set -- -fSL
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
        rcmd="curl -fSL -C - --retry 5 --retry-delay 5"
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
run_remote_pass() {
    while :; do
        pending=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = R ] || continue
            if [ "$REMOTE_MODE" = direct ]; then download_remote_direct "$size" "$name"
            else download_remote "$size" "$name"; fi
            have=$(file_size_remote "$REMOTE_DIR/$name")
            if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" != "$size" ]; then pending=1; fi
        done 3< "$PLAN"
        [ "$pending" = 0 ] && break
        echo "remote: shards still incomplete, retry round in 10s ..."
        sleep 10
    done
}
run_local_pass() {
    while :; do
        pending=0
        while IFS="$TAB" read -r dest size name <&3; do
            [ -n "$name" ] || continue
            [ "$dest" = L ] || continue
            download_local "$size" "$name"
            have=$(file_size_local "$OUT_DIR/$name")
            if [ "$size" -ge 0 ] 2>/dev/null && [ "$have" != "$size" ]; then pending=1; fi
        done 3< "$PLAN"
        [ "$pending" = 0 ] && break
        echo "local: shards still incomplete, retry round in 10s ..."
        sleep 10
    done
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

echo "Downloading (re-run to resume)..."
[ -n "$REMOTE_SSH" ] && echo "Remote mode: $REMOTE_MODE${REMOTE_PROXY:+  remote-proxy $REMOTE_PROXY}"
[ "$RELAY" = 1 ] && [ -n "$REMOTE_SSH" ] && echo "Relay: ON  (local host helps fetch remote shards; ${RELAY_RESERVE_GIB} GiB scratch reserved)"
echo
if [ -n "$REMOTE_SSH" ] && [ "$REMOTE_MODE" = direct ] && [ "$n_remote" -gt 0 ] 2>/dev/null; then
    # Both machines pull from HF at once: remote in the background, local here.
    remote_check_aria2
    run_remote_pass &
    REMOTE_PID=$!
    run_local_pass
    [ "$RELAY" = 1 ] && run_relay_pass   # idle local host now drains remote's set too
    wait "$REMOTE_PID" 2>/dev/null || true
else
    run_local_pass
    [ -n "$REMOTE_SSH" ] && run_remote_pass
fi

# --- verify ------------------------------------------------------------------

echo
incomplete=""
while IFS="$TAB" read -r dest size name <&3; do
    [ -n "$name" ] || continue
    [ "$size" -ge 0 ] 2>/dev/null || continue
    case "$dest" in
        L) got=$(file_size_local "$OUT_DIR/$name") ;;
        R) got=$(file_size_remote "$REMOTE_DIR/$name") ;;
        *) continue ;;
    esac
    [ "$got" = "$size" ] || incomplete="$incomplete $dest:$name"
done 3< "$PLAN"
rm -f "$PLAN"

if [ -n "$incomplete" ]; then
    echo "Incomplete (re-run to resume):" >&2
    for f in $incomplete; do echo "  $f" >&2; done
    exit 1
fi

echo "All files complete."
echo "  local : $OUT_DIR"
[ -n "$REMOTE_SSH" ] && echo "  remote: $REMOTE_SSH:$REMOTE_DIR"
echo
echo "For quantization, the --hf dir must contain ALL shards together; mount the"
echo "remote portion over the bridge (e.g. SMB) into $OUT_DIR before running"
echo "  gguf-tools/deepseek4-quantize --hf \"$OUT_DIR\" --template ... --out ..."
echo
echo "Done."
