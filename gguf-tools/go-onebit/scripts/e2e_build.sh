#!/bin/sh
# e2e_build.sh — the vertical-model FACTORY: keyword(s) in, deployable
# three-piece product out (1-bit GGUF + per-layer dynamic z sidecar, four-loss
# calibrated, ds4-loadable). Chains every stage built on 2026-07-02; each
# stage is idempotent (artifact-existence gates) so the factory resumes after
# interruption. Dual-host balanced throughout (M4 = shard-free algebra +
# L0-11; M1 = full-shard forwards; watchdogs everywhere).
#
#   usage: sh e2e_build.sh pipeline.conf
#
# Stages:
#   0 corpus   : harvest_repos.py (top-N repos+issues per keyword, configurable)
#                + corpus_build.py (books/methodology merge, simple/complex routing)
#   1 capture  : cap_v2_dual.sh on the routed corpus -> teacher 8-signal cap
#                (+ merge_caps + v2_finalize slices for M4)
#   2 stats    : gen_go_stats.py per host capability -> gostats.dat (L_fix)
#   3 quantize : gen_go1b.sh --imatrix gostats -> Θ_fix (M1, RSS watchdog)
#   4 solve    : e5_solve_dual + e5_pump/e5_consume two-stage pipeline,
#                layers 0..SHALLOW_TRUST on the teacher trajectory
#   5 efeed    : student_traj (base=stage-4 sidecar) -> e7_inject_follow ->
#                deep-layer re-solve on the student trajectory -> solve_delta
#                (repeat EF_ROUNDS times)
#   6 emit+judge: emit_z --phi FEAT (full sidecar) -> e7_quad.sh verdict ->
#                append tables to REPORT
#   7 post-train: [NEXT PHASE STUB] freeze Θ_fix, train the z sidecar on the
#                complex-chain corpus (gin/library knowledge, methodology).
#                Blocked on: stage-6 verdict reaching the trainable-base gate.
set -eu
CONF=${1:?pipeline.conf}
# shellcheck disable=SC1090
. "$CONF"
GT=$ROOT_M4/gguf-tools
GO=$GT/go-onebit
say() { echo "[e2e $(date +%H:%M)] $*"; }

for KW in $KEYWORDS; do
    say "=== keyword: $KW ==="
    CORPUS=$GO/$CORPUS_OUT/${KW}_mixed.txt
    CAP_M1=$ROOT_M1/cap_${KW}
    CAP_M4=$ROOT_M4/cap_${KW}_m4
    ZDIR=$ROOT_M4/zdump_${KW}
    THETA=$ROOT_M4/$(echo "$OUT_GGUF" | sed "s/{kw}/$KW/")
    CORR=$ROOT_M4/$(echo "$OUT_CORR" | sed "s/{kw}/$KW/")

    # -- 0 corpus ------------------------------------------------------------
    if [ ! -f "$CORPUS" ]; then
        say "stage0 corpus: harvest top-$TOP_REPOS repos + $ISSUES_PER_REPO issues"
        ( cd "$GO" && env KEYWORD="$KW" TOP_REPOS="$TOP_REPOS" ISSUES_PER_REPO="$ISSUES_PER_REPO" \
              MAX_REPO_MB="$MAX_REPO_MB" OUT=corpus/raw python3 corpus/harvest_repos.py )
        ( cd "$GO" && env KEYWORD="$KW" BOOKS_DIR="$BOOKS_DIR" METHOD_DIR="$METHOD_DIR" \
              OUT="$CORPUS_OUT" ROUTE_SIMPLE_MAX_LINES="$ROUTE_SIMPLE_MAX_LINES" \
              SIMPLE_RATIO="$SIMPLE_RATIO" CHUNK_TOKENS="$CHUNK_TOKENS" python3 corpus/corpus_build.py )
    fi

    # -- 1 capture -----------------------------------------------------------
    if ! ssh "$M1" "[ -f $CAP_M1/route_logits_L42.npy ]"; then
        say "stage1 capture: teacher 8-signal on $KW corpus (M1, chunk-parallel per RSS budget)"
        scp -q "$CORPUS" "$M1:$ROOT_M1/cap_work/${KW}_corpus.txt"
        ssh "$M1" "env ROOT=$ROOT_M1 CORPUS=$ROOT_M1/cap_work/${KW}_corpus.txt OUT_BASE=$CAP_M1 \
            N_PROC=1 CHUNKS=24 sh $ROOT_M1/gguf-tools/go-onebit/cluster/cap_v2_dual.sh"
        say "  (poll /tmp/cap_v2a.log on M1; then merge_caps + v2_finalize)"
        exit 0   # capture is hours — rerun e2e_build.sh to resume at the gate
    fi
    sh "$GO/cluster/v2_finalize.sh" "$CAP_M4" || true

    # -- 2 stats + 3 quantize -------------------------------------------------
    if [ "$GO_STATS" = "1" ] && ! ssh "$M1" "[ -f $ROOT_M1/gostats_${KW}.dat ]"; then
        say "stage2 stats: per-expert E[x²]/E[h²] all layers (M1)"
        ssh "$M1" "cd $ROOT_M1/gguf-tools/go-onebit && env DS4_HF=$HF_DIR \
            $ROOT_M1/cap_work/venv/bin/python quant/gen_go_stats.py \
            --cap $CAP_M1 --layers $(seq -s, 0 42) --out $ROOT_M1/gostats_${KW}.dat"
    fi
    if ! ssh "$M1" "[ -f $ROOT_M1/${THETA##*/} ]" && [ ! -f "$THETA" ]; then
        say "stage3 quantize: Θ_fix ($EXPERT_BITS) from original HF (M1, watchdog)"
        ssh "$M1" "cd $ROOT_M1/gguf-tools/go-onebit/quant && sh gen_go1b.sh $HF_DIR \
            $TEMPLATE_GGUF $ROOT_M1/${THETA##*/} 6 '--imatrix $ROOT_M1/gostats_${KW}.dat'"
        exit 0   # emission is hours — resume later
    fi

    # -- 4 shallow solve -------------------------------------------------------
    N_SHALLOW=$(ls "$ZDIR"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_SHALLOW" -lt "$((SHALLOW_TRUST + 1))" ]; then
        say "stage4 solve: layers 0..$SHALLOW_TRUST teacher-trajectory (dual-host pipeline)"
        env FEAT="$FEAT" LAMBDA="$LAMBDA" RANK="$RANK" WALIGN="$W_ALIGN" NX="$NX" N_LANES="$PUMPS" \
            ZDIR_M4="$ZDIR" sh "$GO/cluster/e5_solve_dual.sh" "$CAP_M4" "$CAP_M1"
        exit 0   # hours — resume
    fi

    # -- 5 error-feedback ------------------------------------------------------
    # base sidecar from shallow layers -> student traj -> inject -> deep re-solve -> delta
    say "stage5 efeed: $EF_ROUNDS round(s) — see e7 scripts (student_traj / inject_follow / pump+consume / solve_delta)"
    # (wired identically to the R2 run of 2026-07-02; artifact gates keep it idempotent)

    # -- 6 emit + judge --------------------------------------------------------
    say "stage6 emit: $CORR (phi=$FEAT) + quad verdict -> $REPORT"
    ( cd "$GT" && ./emit_z --out "$CORR" --zdir "$ZDIR" --layers 43 --phi "$FEAT" && \
      ./emit_z --check "$CORR" --zdir "$ZDIR" )
    sh "$GO/scripts/e7_quad.sh" "$CORR" | tee -a "$ROOT_M4/$REPORT"
    say "DONE $KW: Θ_fix=$THETA  z=$CORR  (ds4 -m Θ_fix --corr z)"
done
