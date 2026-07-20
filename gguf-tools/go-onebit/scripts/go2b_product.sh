#!/bin/sh
# go2b_product.sh — go2b-NF 混合模型 完整产品线 (顶层驱动, 双机)。
# 阶段:
#   cap    补采 16 最差层的 ffn_in (EF 建 H 需要; L23-36 曾删) —— M1
#   ef     双机认领制 EF 重编 16 层 go2b → gguf/v3-artifacts/go2b_ef_L{L}.bin
#   mono   组装 EF 单块 (v3 go1b层 + EF go2b层) 流式管道 → M1:gguf/ds4-mono-ef.gguf
#   verify M1 offload 载入 + 300tok 全判 → 还原度
# 背景: 直接NF单块=59.3GB/80.5%(已验); EF 版预计 ~85%, 体积不变。
# 最差层集合 = L2,23-34,36,38,40 (逐层真尺 base<45% + go2b-NF+EF 提升最大)。
set -u
M1=192.168.1.2; M1ROOT=/Users/fodelf/ds4-main; ROOT=/Users/fodelf/git/ds4-main
SSH="ssh -o BatchMode=yes $M1"
STAGE=${1:?cap|ef|mono|verify|all}
LAYERS="2 23 24 25 26 27 28 29 30 31 32 33 34 36 38 40"

do_cap() {   # M1: 补采 L23-36 ffn_in (其余层已有 cap)
  echo "[cap] M1 补采 L23-36 ffn_in $(date +%H:%M:%S)"
  $SSH "cd $M1ROOT && mkdir -p cap_ef && DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 \
        DS4_CAP_DIR=$M1ROOT/cap_ef DS4_CAP_LAYERS=23-36 \
        ./ds4 -m gguf/ds4-mono-mixed.gguf --ctx 16384 --perplexity-file gguf-tools/go-onebit/gocorpus_12k.txt --metal > /tmp/cap_ef.log 2>&1"
  $SSH "cd $M1ROOT && python3 gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py cap_ef cap_ef 23-36"
  echo "[cap] done"
}

do_ef() {    # 双机: M4 本地跑一路, M1 远程跑一路 (认领制自动均衡)
  echo "[ef] 双机 EF 重编 16 层 $(date +%H:%M:%S)"
  $SSH "cd $M1ROOT && nohup sh gguf-tools/go-onebit/scripts/ef_dual.sh m1 > /tmp/ef_dual_m1.log 2>&1 &"
  sh "$ROOT/gguf-tools/go-onebit/scripts/ef_dual.sh" m4
  # 等 M1 那路也收尾
  while [ "$($SSH "ls $M1ROOT/gguf/v3-artifacts/go2b_ef_L*.bin 2>/dev/null | wc -l | tr -d ' '")" -lt 16 ]; do sleep 30; done
  echo "[ef] 16 层 EF bin 齐"
}

do_mono() {  # 组装 EF 单块: v3(M4) + EF bins(M1) → 管道 → M1:gguf/ds4-mono-ef.gguf
  echo "[mono] 组装 EF 单块 $(date +%H:%M:%S)"
  # EF bins 在 M1; v3 在 M4. 在 M1 组装(EF bins 本地, v3 从 M4 流). 需 build_monolithic_ef 读本地 v3 →
  # 简化: 先把 EF bins 收到 M4, 在 M4 用本地 v3 组装, 管道回 M1。
  for L in $LAYERS; do scp -o BatchMode=yes -q "$M1:$M1ROOT/gguf/v3-artifacts/go2b_ef_L$L.bin" "$ROOT/gguf/v3-artifacts/" 2>/dev/null; done
  $SSH "rm -f $M1ROOT/gguf/ds4-mono-ef.gguf"
  python3 -u "$ROOT/gguf-tools/go-onebit/quant/build_monolithic_ef.py" "$ROOT/gguf/ds4-go1b-v3.gguf" "$ROOT/gguf/v3-artifacts" 2>/tmp/mono_ef.log \
      | $SSH "cat > $M1ROOT/gguf/ds4-mono-ef.gguf"
  grep -a MONO-EF-DONE /tmp/mono_ef.log; $SSH "ls -la $M1ROOT/gguf/ds4-mono-ef.gguf"
}

do_verify() {
  echo "[verify] M1 载入 + 300tok 全判 $(date +%H:%M:%S)"
  $SSH "cd $M1ROOT && DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 \
        ./ds4 -m gguf/ds4-mono-ef.gguf --ctx 4096 --perplexity-file gguf-tools/go-onebit/go_heldout_300.txt --metal 2>&1 | grep -aE 'offload|avg_nll|fail|OOM'"
  echo "还原度 = (5.6185 - NLL) / (5.6185 - 0.5522) x 100  [教师锚 0.5522]"
}

case "$STAGE" in
  cap) do_cap ;;
  ef) do_ef ;;
  mono) do_mono ;;
  verify) do_verify ;;
  all) do_cap; do_ef; do_mono; do_verify; echo "GO2B-EF-PRODUCT-DONE $(date +%H:%M:%S)" ;;
esac
