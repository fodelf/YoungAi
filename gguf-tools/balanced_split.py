#!/usr/bin/env python3
"""Compute the balanced dual-host layer split for a DS4 GGUF.

Usage: balanced_split.py MODEL.gguf M4_LIMIT_GiB M1_LIMIT_GiB
Prints "L NLAYER" to stdout (M4 = layers 0:L + embed <= M4_LIMIT;
M1 = layers L+1:NLAYER + output <= M1_LIMIT), diagnostics to stderr.
Prints "-1 -1" if no split fits both limits.
"""
import sys, struct, collections

MAGIC = 0x46554747
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
SCALAR = {T_U8:1,T_I8:1,T_U16:2,T_I16:2,T_U32:4,T_I32:4,T_F32:4,T_BOOL:1,T_U64:8,T_I64:8,T_F64:8}
GIB = 1073741824

def ru32(fp): return struct.unpack("<I", fp.read(4))[0]
def ru64(fp): return struct.unpack("<Q", fp.read(8))[0]
def skip(fp, t):
    if t == T_STR: fp.read(ru64(fp))
    elif t == T_ARR:
        et = ru32(fp); n = ru64(fp)
        for _ in range(n): skip(fp, et)
    else: fp.read(SCALAR[t])

def main():
    model = sys.argv[1]; m4lim = float(sys.argv[2]); m1lim = float(sys.argv[3])
    fp = open(model, "rb")
    assert ru32(fp) == MAGIC, "not a GGUF"
    ru32(fp); nt = ru64(fp); nkv = ru64(fp)
    align = 32
    for _ in range(nkv):
        k = fp.read(ru64(fp)); vt = ru32(fp)
        if k == b"general.alignment" and vt == T_U32: align = ru32(fp)
        else: skip(fp, vt)
    infos = []
    for _ in range(nt):
        nm = fp.read(ru64(fp)).decode("utf-8", "replace")
        nd = ru32(fp); [ru64(fp) for _ in range(nd)]; ru32(fp); off = ru64(fp)
        infos.append([nm, off])
    info_end = fp.tell(); data_start = (info_end + align - 1)//align*align
    fp.seek(0, 2); fsz = fp.tell()
    order = sorted(range(len(infos)), key=lambda i: infos[i][1])
    size = {}
    for k, i in enumerate(order):
        nxt = data_start + infos[order[k+1]][1] if k+1 < len(order) else fsz
        size[i] = nxt - (data_start + infos[i][1])

    if any(v < 0 for v in size.values()):
        sys.stderr.write("  ✗ GGUF 不完整/损坏: 张量 offset 超过文件大小 — 模型可能还在写(量化未完成)?\n")
        print("-1 -1")
        return

    per_layer = collections.defaultdict(float)
    embed = output = other = 0.0
    nlayer = 0
    for i, (nm, _) in enumerate(infos):
        s = size[i] / GIB
        if nm.startswith("blk."):
            L = int(nm.split(".")[1]); per_layer[L] += s; nlayer = max(nlayer, L)
        elif nm == "token_embd.weight":
            embed += s
        elif nm.startswith("output"):       # output.weight / output_norm / output_hc_*
            output += s
        else:
            other += s                       # rope etc. -> both shards keep these

    best = None
    for L in range(0, nlayer):
        m4 = embed + other + sum(per_layer.get(i, 0) for i in range(0, L+1))
        m1 = output + other + sum(per_layer.get(i, 0) for i in range(L+1, nlayer+1))
        if m4 <= m4lim and m1 <= m1lim:
            bal = abs((m4lim - m4) - (m1lim - m1))   # most balanced headroom
            if best is None or bal < best[0]:
                best = (bal, L, m4, m1)
    if best is None:
        sys.stderr.write("  ✗ no split fits both limits (model too big -> lower KEEP_TOP_K)\n")
        print("-1 -1")
        return
    _, L, m4, m1 = best
    sys.stderr.write(f"  ✓ L={L}: M4(0:{L})={m4:.2f}G  M1({L+1}:{nlayer})={m1:.2f}G  "
                     f"(limits {m4lim}/{m1lim}; embed={embed:.2f} output={output:.2f} other={other:.2f})\n")
    print(f"{L} {nlayer}")

if __name__ == "__main__":
    main()
