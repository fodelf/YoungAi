#!/bin/sh
# rank128_sentinel.sh — 质变候选轴#1 的小步判决：同 5 哨兵层（18/20/38/40/42）
# 用 rank=128 重解（targets 已在 M1 cap_v2r2），换装进 r2 全集 → 配对短判 vs r2。
# 过门（≤ r2sent128 明显优于 r2@rank64）→ 全 43 层 rank128 才立项。
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
SENT="18 20 38 40 42"
CFG128="--solver rrr --rank 128 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
cd "$ROOT" || exit 1
say() { echo "[k128 $(date +%H:%M)] $*" >&2; }

say "1/3 哨兵泵+rank128 解（M1 一条龙）"
cat > /tmp/k128_m1.sh <<LAUNCH
#!/bin/sh
cd $ROOT_M1
env CAP=$ROOT_M1/cap_v2r2 SPOOL=$ROOT_M1/sel_spool_k128 NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh "$SENT" || exit 1
export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_k128 ROOT=$ROOT_M1
export CFG="$CFG128"
sh gguf-tools/go-onebit/cluster/e5_consume.sh "$SENT" $ROOT_M1/zdump_k128
LAUNCH
scp -o BatchMode=yes -q /tmp/k128_m1.sh "$M1:/tmp/" || exit 1
ssh -o BatchMode=yes "$M1" "mkdir -p $ROOT_M1/zdump_k128 && sh /tmp/k128_m1.sh" || exit 1

say "2/3 换装进 r2 全集 → emit"
rm -rf zdump_k128 && mkdir -p zdump_k128
cp zdump_v2r2/z_L*.bin zdump_k128/
scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_k128/z_L*.bin" zdump_k128/ || exit 1
( cd gguf-tools && ./emit_z --out "$ROOT/gguf/sidecars/go-r2k128sent.gguf" --zdir "../zdump_k128" --layers 43 | tail -1 )

say "3/3 配对短判 vs r2"
export DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1
A=$(./ds4 -m gguf/ds4-go1b-v2.gguf --corr gguf/sidecars/go-r2k128sent.gguf --perplexity-file /tmp/heldout_short.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*' | cut -d= -f2)
B=$(./ds4 -m gguf/ds4-go1b-v2.gguf --corr gguf/sidecars/go-r2.gguf --perplexity-file /tmp/heldout_short.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*' | cut -d= -f2)
echo "K128-SENTINEL-VERDICT k128sent@old=$A vs r2@old=$B"
