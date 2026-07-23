#!/bin/bash
# pillar_probe_srv.sh — 12针四支柱面板 · 经常驻 ds4-server 的 /v1/completions 裸续写口径。
# 场景: 双机部署栈(coordinator+worker)已常驻时的质量重测——零新进程、零模型加载、实例锁友好。
# 与 pillar_probe.sh(单机冷加载)判决口径一致: BOS 裸续写 temp0, 续写对题=过, 汤/复读=不过。
# 用法: [PORT=8013] [NPRED=28] [PROBES=first|all] [CORPUS=corpus/xxx.txt] ./pillar_probe_srv.sh
# 输出: /tmp/<语料名>_srv.report(原始输出逐条, 判读留给人)
set -uo pipefail
cd "$(dirname "$0")/.."     # → go-onebit/
PORT="${PORT:-8013}"
NPRED="${NPRED:-28}"
SEL="${PROBES:-all}"
CORPUS="${CORPUS:-corpus/pillar_probes.txt}"   # 扩域口(2026-07-20): 换语料文件复用同判据
[ -f "$CORPUS" ] || { echo "[probe-srv] 语料 $CORPUS 缺失 — 拒跑" >&2; exit 2; }
REPORT=/tmp/$(basename "$CORPUS" .txt)_srv.report
[ "$CORPUS" = corpus/pillar_probes.txt ] && REPORT=/tmp/pillar_probe_srv.report  # 旧默认路径不变
: > "$REPORT"

mapfile_blocks() {
    awk 'BEGIN{RS="---\n"; n=0}
         { blk=""; nl = split($0, L, "\n");
           for (i = 1; i <= nl; i++) if (L[i] !~ /^#/) blk = blk (blk==""?"":"\n") L[i];
           gsub(/\n+$/, "", blk); sub(/^\n+/, "", blk);
           if (blk != "") { n++; printf "%s\x1e", blk } }' "$CORPUS"
}
IFS=$'\x1e' read -r -a BLOCKS -d '' < <(mapfile_blocks; printf '\0') || true
PICK=(0 3 6 9); [ "$SEL" = all ] && PICK=($(seq 0 $((${#BLOCKS[@]}-1))))

echo "[probe-srv] ${#PICK[@]} 条 (port=$PORT npred=$NPRED)" >&2
i=0
for k in "${PICK[@]}"; do
    i=$((i+1))
    frag="${BLOCKS[$k]}"
    echo "[probe-srv $i/${#PICK[@]}] 块$((k+1)): ${frag%%$'\n'*}" >&2
    {   echo "════ PROBE $((k+1)) ════"
        echo "── 片段:"; printf '%s\n' "$frag"
        echo "── 续写(原始):"
    } >> "$REPORT"
    FRAG="$frag" NPRED="$NPRED" PORT="$PORT" python3 - <<'PEOF' >> "$REPORT" 2>>/tmp/pillar_probe_srv.err \
        || echo "[探针失败, 见 /tmp/pillar_probe_srv.err 尾部]" >> "$REPORT"
import json,os,urllib.request
frag=os.environ["FRAG"]; n=int(os.environ["NPRED"]); port=os.environ["PORT"]
body={"model":"ds4","prompt":"<｜begin▁of▁sentence｜>"+frag,
      "max_tokens":n,"temperature":0,"raw":True}  # raw=裸续写(无chat帧), base模型口径
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕全局代理(07-14 502教训)
with opener.open(req,timeout=600) as r:
    d=json.loads(r.read())
c=d["choices"][0]
print(c.get("text",""))
print(f"[finish={c.get('finish_reason')}]")
PEOF
    echo >> "$REPORT"
done
echo "[probe-srv] 完成 → $REPORT" >&2
