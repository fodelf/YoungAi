#!/bin/sh
# Pooled 3-host download of one HF/ModelScope model repo with spark as the
# COLLECTOR: every file ends up in spark's ~/ds4-main/hf/<repo-name>, nothing
# stays on the Macs. Repo via REPO= (default DSpark, 155.4 GiB / 48 shards).
#
# Topology is NOT fixed — measure before assuming who the muscle is:
#   2026-08-20 (DSpark run): spark->hf-mirror ~0.1 MB/s per lane, Macs ~1 MB/s
#     per lane via hf-mirror, Mac->spark LAN push ~14 MB/s. Macs = the muscle.
#   2026-09-01 (Vision-Exp run): spark reaches NEITHER huggingface.co NOR
#     hf-mirror (both time out) but pulls 27.7 MB/s from ModelScope, while the
#     Macs manage 0.19 MB/s (hf-mirror) / 0.28 MB/s (ModelScope). Spark = the
#     muscle, ~100x the Macs; a Mac lane is worth roughly one small shard.
# That reversal is why fetch_list falls back to the ModelScope copy of the same
# repo and why DL_BASE exists — see fetch_list.
#
# There is NO fixed split. All hosts claim shards from ONE pool that lives on
# spark's filesystem (atomic mkdir under $RDEST/.claims), so a fast host
# naturally does more, a dead host contributes nothing, and a host that comes
# back mid-run (the M1 today) just starts claiming. Mac flow per shard:
# curl -C - into local staging -> size gate -> rsync to spark tmp -> atomic mv
# -> remote size check -> delete local copy. Spark flow: curl straight into
# the final dir. Small (non-safetensors) files are spark's job only.
#
#   tools/fetch_dspark_pool.sh mac      # downloader lane set on a Mac
#   tools/fetch_dspark_pool.sh spark    # downloader lane set ON spark (run there)
#   tools/fetch_dspark_pool.sh super    # M4 supervisor: (re)attach M1 when it returns
#   tools/fetch_dspark_pool.sh status   # one-shot progress/quality report
#   tools/fetch_dspark_pool.sh verify   # sha256 every landed file on spark vs HF lfs oids
#
# Knobs (env): POOL_LANES (mac 3 / spark 2), RESERVE_GIB (local headroom, 8),
# REPO, HF_ENDPOINT, SPARK (ssh host), M1_SSH.
set -u

REPO=${REPO:-deepseek-ai/DeepSeek-V4-Flash-DSpark}
NAME=${REPO##*/}
ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
SPARK=${SPARK:-spark}                 # ssh alias; resolvable from both Macs
M1_SSH=${M1_SSH:-fodelf@192.168.1.2}
RDEST="ds4-main/hf/$NAME"             # relative to $HOME on spark
# Download-URL prefix. Default = HF layout on $ENDPOINT; override per host,
# e.g. DL_BASE="https://modelscope.cn/models/$REPO/resolve/master" (measured
# 2026-08-20: 15.5 MB/s per connection from spark vs ~0.1 on hf-mirror; byte
# range spot-check MS==HF, final sha256 verify is the hard gate anyway).
DL_BASE=${DL_BASE:-}
MS_BASE=${MS_BASE:-https://modelscope.cn}
MS_REV=${MS_REV:-master}
CLAIMS="$RDEST/.claims"
ROLE=${1:-mac}
RESERVE_GIB=${RESERVE_GIB:-8}
HOSTID=$(hostname -s | tr -c 'A-Za-z0-9' '-')
TAB=$(printf '\t')
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

case "$ROLE" in
    spark) LANES=${POOL_LANES:-2} ;;
    *)     LANES=${POOL_LANES:-3} ;;
esac

# Local staging dir on Macs (repo hf/, same disk budget as everything else).
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STAGE="$ROOT/hf/.dspark-stage"

# SP "cmd": run cmd in spark's $HOME — locally when we ARE spark, over ssh
# otherwise. All paths in commands are relative to spark's home.
if [ "$ROLE" = spark ]; then
    SP() { ( cd "$HOME" && sh -c "$1" ); }
else
    SP() { $SSH "$SPARK" "$1" </dev/null; }
fi

human() { awk -v b="$1" 'BEGIN { printf "%.2f GiB", b/1073741824 }'; }
lsize() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }
rsize() { n=$(SP "wc -c < '$RDEST/$1' 2>/dev/null" 2>/dev/null | tr -d ' '); [ -n "$n" ] && echo "$n" || echo 0; }
ckey()  { printf '%s' "$1" | tr '/' '_'; }

# --- file list: "size<TAB>name", cached per run ------------------------------
LIST=/tmp/dspark_pool_list.$$
fetch_list() {
    # HF API first. Measured 2026-09-01: spark reaches neither huggingface.co
    # nor hf-mirror (both time out), so fall back to the SAME repo on
    # ModelScope -- file count, byte sizes and Sha256 all match HF, and
    # ModelScope's Sha256 *is* the HF lfs oid (both are the file sha256), so
    # `verify` keeps working from either source. When the list came from
    # ModelScope the downloads must come from there too, hence DL_BASE.
    curl -fsSL --max-time 60 "$ENDPOINT/api/models/$REPO?blobs=true" 2>/dev/null | python3 -c '
import sys, json
d = json.load(sys.stdin)
for s in d.get("siblings", []):
    n = s.get("rfilename")
    if not n: continue
    sz = s.get("size"); lfs = s.get("lfs") or {}
    # "-" placeholder: an empty middle field would be collapsed by tab-IFS.
    print("%d\t%s\t%s" % (sz if isinstance(sz, int) else -1, (lfs.get("oid") or "-"), n))
' > "$LIST".full 2>/dev/null
    if [ ! -s "$LIST".full ]; then
        curl -fsSL --max-time 60 "$MS_BASE/api/v1/models/$REPO/repo/files?Revision=$MS_REV&Recursive=true" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for f in d.get("Data", {}).get("Files", []):
    if f.get("Type") != "blob": continue
    # Path, not Name: Name is only the basename, so inference/config.json
    # would collide with the top-level config.json and clobber it.
    p = f.get("Path")
    if not p: continue
    sz = f.get("Size")
    print("%d\t%s\t%s" % (sz if isinstance(sz, int) else -1, (f.get("Sha256") or "-"), p))
' > "$LIST".full || { echo "FATAL: cannot fetch file list from $ENDPOINT or $MS_BASE" >&2; exit 1; }
        [ -s "$LIST".full ] || { echo "FATAL: empty file list from both sources" >&2; exit 1; }
        [ -n "$DL_BASE" ] || DL_BASE="$MS_BASE/models/$REPO/resolve/$MS_REV"
        echo "list: HF unreachable, using ModelScope ($DL_BASE)"
    fi
    awk -F"$TAB" '{ print $1 "\t" $3 }' "$LIST".full > "$LIST"
    [ -s "$LIST" ] || { echo "FATAL: empty file list" >&2; exit 1; }
}

# --- claim pool on spark ------------------------------------------------------
# A claim is a directory (mkdir is atomic on one filesystem). The owner
# refreshes its mtime every 4 min; a claim untouched for >20 min whose target
# is not complete is considered abandoned and gets broken.
claim() { # name -> 0 if we own it
    k=$(ckey "$1")
    if SP "mkdir '$CLAIMS/$k' 2>/dev/null && echo '$HOSTID' > '$CLAIMS/$k/owner'"; then return 0; fi
    stale=$(SP "find '$CLAIMS/$k' -maxdepth 0 -mmin +20 2>/dev/null" 2>/dev/null)
    if [ -n "$stale" ]; then
        SP "rm -rf '$CLAIMS/$k'" 2>/dev/null
        SP "mkdir '$CLAIMS/$k' 2>/dev/null && echo '$HOSTID' > '$CLAIMS/$k/owner'" && return 0
    fi
    return 1
}
unclaim()   { SP "rm -rf '$CLAIMS/$(ckey "$1")'" 2>/dev/null || true; }
heartbeat() { # name -> pid of refresher on stdout
    # >/dev/null on the child: it inherits the command-substitution pipe
    # otherwise, holding its write end open so $(heartbeat ...) never returns.
    ( while :; do sleep 240; SP "touch '$CLAIMS/$(ckey "$1")'" 2>/dev/null || true; done ) >/dev/null 2>&1 &
    echo $!
}

# --- downloaders --------------------------------------------------------------
curl_dl() { # url out.part  (resumable, retry inside)
    curl -fSL -C - --retry 8 --retry-delay 5 --speed-time 60 --speed-limit 10240 \
        -sS -o "$2" "$1"
}

mac_shard() { # size name  -> 0 when the shard is verified on spark
    size=$1; name=$2
    url="${DL_BASE:-$ENDPOINT/$REPO/resolve/main}/$name"
    part="$STAGE/$name.part"
    # Space gate: never start a fetch that could squeeze the system disk.
    need=$(awk -v s="$size" -v r="$RESERVE_GIB" 'BEGIN{printf "%.0f", s + r*1073741824}')
    while :; do
        free=$(df -Pk "$STAGE" | awk 'NR==2{printf "%.0f", $4*1024}')
        [ "$free" -ge "$need" ] && break
        echo "wait  $HOSTID low disk ($(human "$free") free, need $(human "$need")); 60s"
        sleep 60
    done
    echo "get   $HOSTID $name ($(human "$size"))"
    curl_dl "$url" "$part" || true
    have=$(lsize "$part")
    if [ "$have" != "$size" ]; then
        if [ "$have" -gt "$size" ] 2>/dev/null; then rm -f "$part"; fi  # Range ignored: restart clean
        echo "retry $HOSTID $name incomplete ($(human "$have"))"; return 1
    fi
    # Push: unique tmp per host, atomic mv, then byte-verify before local rm.
    tmp="$RDEST/$name.push.$HOSTID"
    echo "push  $HOSTID $name -> $SPARK"
    rsync -e "$SSH" --partial --inplace "$part" "$SPARK:$tmp" </dev/null || { echo "retry $HOSTID push failed $name"; return 1; }
    SP "mv -f '$tmp' '$RDEST/$name'" || return 1
    got=$(rsize "$name")
    [ "$got" = "$size" ] || { echo "retry $HOSTID remote size mismatch $name ($got != $size)"; return 1; }
    rm -f "$part"
    echo "done  $HOSTID $name landed on spark"
    return 0
}

spark_shard() { # size name (runs on spark; straight into final dir)
    size=$1; name=$2
    url="${DL_BASE:-$ENDPOINT/$REPO/resolve/main}/$name"
    out="$HOME/$RDEST/$name"
    mkdir -p "$(dirname "$out")"
    echo "get   spark $name ($(human "$size"))"
    curl_dl "$url" "$out.part" || true
    have=$(lsize "$out.part")
    if [ "$have" != "$size" ]; then
        if [ "$have" -gt "$size" ] 2>/dev/null; then rm -f "$out.part"; fi
        echo "retry spark $name incomplete ($(human "$have"))"; return 1
    fi
    mv -f "$out.part" "$out"
    echo "done  spark $name"
    return 0
}

# --- lane: claim next incomplete shard from the pool, do it, repeat -----------
lane() {
    while :; do
        _got=0
        while IFS="$TAB" read -r size name <&3; do
            [ -n "$name" ] || continue
            case "$name" in *.safetensors) ;; *) continue ;; esac
            [ "$(rsize "$name")" = "$size" ] && continue
            claim "$name" || continue
            hb=$(heartbeat "$name")
            if [ "$ROLE" = spark ]; then spark_shard "$size" "$name"; else mac_shard "$size" "$name"; fi
            ok=$?
            kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null
            # Done files are skipped by the size check before claim, so a
            # completed claim can be dropped; a failed one must be.
            unclaim "$name"
            _got=1
            break
        done 3< "$WLIST"
        [ "$_got" = 0 ] && break
    done
}

all_done() {
    while IFS="$TAB" read -r size name <&3; do
        [ -n "$name" ] || continue
        case "$name" in *.safetensors) ;; *) continue ;; esac
        [ "$(rsize "$name")" = "$size" ] || return 1
    done 3< "$LIST"
    return 0
}

run_worker() {
    fetch_list
    SP "mkdir -p '$RDEST' '$CLAIMS'"
    WLIST=$LIST
    if [ "$ROLE" = spark ]; then
        # spark walks the list descending (Macs ascend): the slow lanes chew the
        # tail instead of racing the fast lanes for the head.
        WLIST=$LIST.rev
        awk '{ a[NR]=$0 } END { for (i=NR; i>=1; i--) print a[i] }' "$LIST" > "$WLIST"
    else
        mkdir -p "$STAGE"
    fi
    while :; do
        if [ "$ROLE" = spark ]; then
            # Small files are spark-only and cheap; retried every round until done.
            while IFS="$TAB" read -r size name <&3; do
                [ -n "$name" ] || continue
                case "$name" in *.safetensors) continue ;; esac
                [ "$(rsize "$name")" = "$size" ] && continue
                spark_shard "$size" "$name" || true
            done 3< "$LIST"
        fi
        _l=0; _pids=""
        while [ "$_l" -lt "$LANES" ]; do lane & _pids="$_pids $!"; _l=$((_l+1)); done
        wait $_pids
        all_done && break
        echo "round: shards still incomplete, next round in 20s"
        sleep 20
    done
    echo "COMPLETE: all shards verified on spark ($ROLE worker exiting)"
    rm -f "$LIST" "$LIST".full "$LIST".rev 2>/dev/null
}

# --- supervisor: pull the M1 back in whenever it becomes reachable ------------
run_super() {
    while :; do
        # Keep the LOCAL mac worker alive too (curl storms / accidental kills).
        # [.] so the pgrep (and any ssh command line carrying the pattern)
        # never matches itself.
        if ! pgrep -f "fetch_dspark_pool[.]sh mac" >/dev/null 2>&1; then
            echo "super: local mac worker not running, launching"
            nohup caffeinate -i sh "$ROOT/tools/fetch_dspark_pool.sh" mac >> /tmp/dspark_pool_m4.log 2>&1 &
        fi
        if $SSH -o ConnectTimeout=6 "$M1_SSH" true 2>/dev/null; then
            if ! $SSH "$M1_SSH" "pgrep -f 'fetch_dspark_pool[.]sh mac' >/dev/null" 2>/dev/null; then
                echo "super: M1 reachable, launching worker"
                scp -o BatchMode=yes "$ROOT/tools/fetch_dspark_pool.sh" "$M1_SSH:ds4-main/tools/" 2>/dev/null \
                    || scp -o BatchMode=yes "$0" "$M1_SSH:fetch_dspark_pool.sh"
                $SSH "$M1_SSH" 'f=ds4-main/tools/fetch_dspark_pool.sh; [ -f "$f" ] || f=fetch_dspark_pool.sh; nohup caffeinate -i sh "$f" mac > /tmp/dspark_pool_m1.log 2>&1 & echo "super: M1 worker pid $!"'
            fi
        fi
        # Stop when the whole set is on spark.
        fetch_list 2>/dev/null || { sleep 300; continue; }
        all_done && { echo "super: COMPLETE, exiting"; break; }
        sleep 300
    done
}

# --- status / verify -----------------------------------------------------------
run_status() {
    fetch_list
    tot=0; got=0; ntot=0; ndone=0; missing=""
    while IFS="$TAB" read -r size name <&3; do
        [ -n "$name" ] || continue
        case "$name" in *.safetensors) ;; *) continue ;; esac
        ntot=$((ntot+1)); tot=$(awk -v a="$tot" -v b="$size" 'BEGIN{printf "%.0f", a+b}')
        h=$(rsize "$name")
        if [ "$h" = "$size" ]; then ndone=$((ndone+1)); got=$(awk -v a="$got" -v b="$size" 'BEGIN{printf "%.0f", a+b}')
        else got=$(awk -v a="$got" -v b="$h" 'BEGIN{printf "%.0f", a+b}'); fi
    done 3< "$LIST"
    parts=$(SP "cd '$RDEST' 2>/dev/null && ls *.part *.push.* 2>/dev/null | wc -l" | tr -d ' ')
    claims=$(SP "ls '$CLAIMS' 2>/dev/null | wc -l" | tr -d ' ')
    echo "shards $ndone/$ntot complete, $(human "$got") / $(human "$tot") on spark; in-flight files: ${parts:-0}, live claims: ${claims:-0}"
    rm -f "$LIST" "$LIST".full
}

run_verify() {
    fetch_list
    fail=0
    while IFS="$TAB" read -r size oid name <&3; do
        [ -n "$name" ] || continue
        [ "$oid" != "-" ] || continue    # only lfs files carry a sha256
        got=$(SP "sha256sum '$RDEST/$name' 2>/dev/null | awk '{print \$1}'")
        if [ "$got" = "$oid" ]; then echo "sha ok   $name"
        else echo "sha FAIL $name (got ${got:-missing})"; fail=1; fi
    done 3< "$LIST".full
    rm -f "$LIST" "$LIST".full
    [ "$fail" = 0 ] && echo "VERIFY: all sha256 match" || { echo "VERIFY: FAILURES above" >&2; exit 1; }
}

case "$ROLE" in
    mac|spark) run_worker ;;
    super)     run_super ;;
    status)    run_status ;;
    verify)    run_verify ;;
    *) echo "usage: $0 mac|spark|super|status|verify" >&2; exit 1 ;;
esac
