#!/usr/bin/env bash
# go2b_layer_quality.sh — 全43层 go2b 每层输出质量门(2026-07-24, 用户: 重量化每层输出质量)。
# 链: ①全43层激活捕获(cap_algo43, 双机) → ②逐层 go2b(激活最优) vs stacked 单层重建 cos
#     → ③全层质量表(go2b输出relL2 / stacked / 改善%)。
# 这是重量化前的"每层输出质量"判据: 每层 go2b应显著优于stacked(现役)。全脚本, 不私改。
# 用法: go2b_layer_quality.sh capture   # 全43层激活(需 svc down, 双机)
#       go2b_layer_quality.sh quality   # 逐层cos门(M1跑, HF在M1)
#       go2b_layer_quality.sh all
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}
CAP=${CAP:-cap_algo43}
M1CAP=/tmp/$CAP   # capture_alllayers harvest 到 M1:/tmp/CAP
MODEL=gguf/go-onebit/ds4-code1b.gguf
CALIB=$ROOT/gguf-tools/go-onebit/corpus/algo_calib.txt
log(){ echo "[layer-q] $*" >&2; }

case "${1:-all}" in
capture)
  pgrep -f '^\./ds4-server' >/dev/null && { log "server在跑 — 先 tools/svc.sh down"; exit 1; }
  log "全43层激活捕获(双机, algo calib) → $CAPDIR"
  PROMPT_FILE="$CALIB" MODEL="$MODEL" CAP="$CAP" "$HERE/capture_alllayers.sh" || exit 1
  n=$(ssh "$M1" "ls $M1CAP/raw_ffn_in_L* 2>/dev/null | wc -l" | tr -d " ")
  log "捕获收割: $n/43 层 ffn_in"
  [ "$n" -ge 40 ] || { log "捕获层数不足($n) — 停"; exit 1; }
  ;;
quality)
  # M1 跑(HF在M1)。逐层: 热专家 go2b(激活最优) vs stacked, 单层输出重建 cos。
  scp -q "$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt" "$M1:/tmp/prog_active.txt" 2>/dev/null
  ssh "$M1" "cd /Users/fodelf/ds4-main && CAPDIR=$M1CAP python3 - <<'PYEOF'
import os,sys
sys.path.insert(0,'/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant')
sys.path.insert(0,'/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd')
os.environ.setdefault('DS4_HF','/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base')
os.environ['DS4_GO2B_ACT_SCALE']='1'
import numpy as np, ds4reader as R
from go2b_encode import encode_go2b, decode_go2b
def Q1(W):
    m=np.abs(W).mean(1,keepdims=True); return np.sign(W)*m
def orel(Wt,Wq,X): Ot=X@Wt.T; Oq=X@Wq.T; return np.linalg.norm(Ot-Oq)/np.linalg.norm(Ot)
cap=os.environ['CAPDIR']
active={}
for l in open('/tmp/prog_active.txt'):
    if l.startswith('L'): L=int(l.split(':')[0][1:]); active[L]=[int(x) for x in l.split(':')[1].split()]
print('L | stacked | go2b(act) | 改善% | 门(go2b<stacked)')
ts=tj=0.0; nl=0; fails=[]
for L in range(43):
    xp=f'{cap}/raw_ffn_in_L{L}'
    if not os.path.isfile(xp): print(f'L{L} | 无激活'); continue
    Xh=np.fromfile(xp,dtype='<f2').reshape(-1,4096).astype(np.float32)[:256]
    exps=active.get(L,list(range(3)))[:3]
    ss=js=0.0;n=0
    for e in exps:
        try: W=R.read_weight(f'layers.{L}.ffn.experts.{e}.w1.weight').astype(np.float32)
        except: continue
        b=Q1(W); stacked=b+Q1(W-b)
        blk,_=encode_go2b(W,Xh=Xh,mode='nf'); dec=decode_go2b(blk,W.shape[1])
        ss+=orel(W,stacked,Xh); js+=orel(W,dec,Xh); n+=1
    if n==0: continue
    ss/=n; js/=n; imp=100*(ss-js)/ss; ok='✓' if js<ss else '✗'
    if js>=ss: fails.append(L)
    ts+=ss; tj+=js; nl+=1
    print(f'L{L} | {ss:.4f} | {js:.4f} | {imp:+.1f}% | {ok}')
if nl: print(f'★均值{nl}层: stacked={ts/nl:.4f} go2b={tj/nl:.4f} 改善={100*(ts-tj)/ts:+.1f}% 失败层={fails}')
PYEOF" | tee "$ROOT/gguf-tools/go-onebit/reports/go2b_layer_quality_$(date +%F).txt"
  ;;
all)
  "$0" capture && "$0" quality
  ;;
*) log "用法: capture|quality|all"; exit 2 ;;
esac
