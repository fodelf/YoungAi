#!/usr/bin/env python3
"""vq_merge_v4.py — v4 战役产物合并(2026-07-28 v4.1): 换代式重打包+路由偏置烘焙。

新一代合一文件 = 骨架(非专家张量, 取自现役 base, gate.bias 烘入 α·Δb 路由偏置)
              + 新 down(M1 dql_L*.bin 的 D 段, ssh 流式)
              + 新 blob(M1 dql_vq_L*.bin 真载荷修剪, ssh 流式, type42)。
v4.1 变更: ①blob 不再要求本地目录 — 直接从 M1 ssh dd 流式(真尺寸由 vq_blob_truesize.py
预扫文件 --blob-sizes 提供, 修剪预留空洞) ②--route-bias/--route-alpha: RBIA 侧车
(ds4quant_run DS4_ROUTE_BIAS_FIT 产物)按 mincnt 门后 α 缩放烘进 blk.L.exp_probs_b.bias
(F32 256, 只影响专家选择不影响混合权重 — 与引擎语义同构; A/B 判决 α=2.5 code agree 84.2)。
down 与 blob 必须同代(冷 w2 的 hc 顺序补偿是对新 q1/q3 算的)。

两阶段:
  --extract-skeleton: 从 base 抽非专家张量(~8.6G; 不含 down/blob/opt) → skeleton.gguf
  --merge: skeleton + ssh(M1 blobs 修剪流式) + ssh(M1 dql D 段) [+gate.bias 烘焙] → 输出
用法:
  vq_merge_v4.py --extract-skeleton --base ds4-vq22.gguf --out skeleton.gguf
  vq_merge_v4.py --merge --skeleton skeleton.gguf --blob-sizes blob_sizes.txt \
      --route-bias route_bias_v4.bin --route-alpha 2.5 \
      --dql-host 192.168.1.2 --dql-dir /Users/fodelf/ds4-main/gguf/go-onebit/layers \
      --out ds4-vq4bf.gguf
"""
import argparse, os, struct, subprocess, sys

ALIGN = 32
N_LAYER, NEXP = 43, 256
SZ_G = 2048 * 16 * 34            # dql G/U 段每专家字节
SZ_D = 4096 * 8 * 34             # dql D 段每专家字节
DQL_HDR = 35104                  # dql_L*.bin 头(实测 dql_L00)
DOWN_LAYER_BYTES = NEXP * SZ_D   # 285,212,672
RB_MINCNT = 8                    # 与 ds4quant_run RB_MINCNT 同步


def rd(f, n):
    b = f.read(n); assert len(b) == n; return b


def parse_header(f):
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(f, 24))
    assert magic == 0x46554747 and ver == 3
    kv_start = f.tell()
    def rstr():
        n, = struct.unpack("<Q", rd(f, 8)); return rd(f, n).decode()
    def skipv(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rstr()
        elif t == 9:
            et, = struct.unpack("<I", rd(f, 4)); n, = struct.unpack("<Q", rd(f, 8))
            for _ in range(n): skipv(et)
        else: rd(f, sz[t])
    for _ in range(n_kv):
        rstr(); t, = struct.unpack("<I", rd(f, 4)); skipv(t)
    kv_end = f.tell(); f.seek(kv_start); kv_raw = rd(f, kv_end - kv_start)
    tens = []
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(f, 4))
        ne = list(struct.unpack("<%dQ" % nd, rd(f, 8 * nd)))
        ty, = struct.unpack("<I", rd(f, 4)); off, = struct.unpack("<Q", rd(f, 8))
        tens.append({"name": nm, "ne": ne, "type": ty, "off": off})
    data0 = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    return n_kv, kv_raw, tens, data0


def sizes_by_offset(tens, fsz, data0):
    order = sorted(tens, key=lambda t: t["off"])
    for i, t in enumerate(order):
        end = order[i + 1]["off"] if i + 1 < len(order) else fsz - data0
        t["bytes"] = end - t["off"]
    return tens


def ser_info(t):
    b = struct.pack("<Q", len(t["name"])) + t["name"].encode()
    b += struct.pack("<I", len(t["ne"])) + struct.pack("<%dQ" % len(t["ne"]), *t["ne"])
    b += struct.pack("<IQ", t["type"], t["off"])
    return b


def is_dropped(nm, skeleton_mode):
    if nm.endswith("ffn_exps_vq.blob"): return True
    if ".opt_" in nm: return True
    if skeleton_mode and nm.endswith("ffn_down_exps.weight"): return True
    return False


def write_gguf(out_p, kv_raw, n_kv, entries, writer):
    """entries: [{name,ne,type,bytes,src}] 顺序即数据序; writer(out,entry) 负责写数据."""
    off = 0
    for e in entries:
        e["off"] = off
        off += (e["bytes"] + ALIGN - 1) // ALIGN * ALIGN
    hdr = struct.pack("<IIQQ", 0x46554747, 3, len(entries), n_kv) + kv_raw
    info = b"".join(ser_info(e) for e in entries)
    data0 = (len(hdr) + len(info) + ALIGN - 1) // ALIGN * ALIGN
    with open(out_p, "wb") as out:
        out.write(hdr); out.write(info)
        out.write(b"\x00" * (data0 - len(hdr) - len(info)))
        for e in entries:
            out.seek(data0 + e["off"])
            writer(out, e)
    return data0


def extract_skeleton(a):
    fsz = os.path.getsize(a.base)
    with open(a.base, "rb") as f:
        n_kv, kv_raw, tens, data0 = parse_header(f)
    sizes_by_offset(tens, fsz, data0)
    keep = [t for t in tens if not is_dropped(t["name"], skeleton_mode=True)]
    ents = [{"name": t["name"], "ne": t["ne"], "type": t["type"],
             "bytes": t["bytes"], "src_off": t["off"]} for t in keep]
    src = open(a.base, "rb")
    def w(out, e):
        src.seek(data0 + e["src_off"])
        left = e["bytes"]
        while left:
            b = src.read(min(1 << 26, left)); out.write(b); left -= len(b)
    write_gguf(a.out, kv_raw, n_kv, ents, w)
    print(f"[skeleton] {a.out} {os.path.getsize(a.out)/2**30:.2f} GiB ({len(ents)} 张量)")


def load_route_bias(path, alpha):
    """RBIA → {L: 256×f32 Δb·α}(mincnt 门后; 全零层省略)"""
    with open(path, "rb") as f:
        hd = struct.unpack("<4I", rd(f, 16))
        assert hd[0] == 0x41494252 and hd[1] == N_LAYER and hd[2] == NEXP, "RBIA 头不对"
        acc = struct.unpack("<%df" % (N_LAYER * NEXP), rd(f, N_LAYER * NEXP * 4))
        cnt = struct.unpack("<%dI" % (N_LAYER * NEXP), rd(f, N_LAYER * NEXP * 4))
    out, armed = {}, 0
    for L in range(N_LAYER):
        row = [alpha * acc[L*NEXP+e] if cnt[L*NEXP+e] >= RB_MINCNT else 0.0
               for e in range(NEXP)]
        if any(v != 0.0 for v in row):
            out[L] = row; armed += sum(1 for v in row if v != 0.0)
    print(f"[route-bias] α={alpha} mincnt={RB_MINCNT} 武装槽={armed} 层={len(out)}")
    return out


def ssh_stream(host, path, nbytes, out, skip=0):
    """ssh 流式拷贝 nbytes 到 out(逐块, 校验总长)。tail/head 而非 dd:
    macOS BSD dd 不支持 GNU iflag=skip_bytes,count_bytes(实测 0B, 2026-07-28)。"""
    cmd = ["ssh", host,
           f"tail -c +{skip+1} {path} | head -c {nbytes}"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    got = 0
    while True:
        b = p.stdout.read(1 << 24)
        if not b: break
        out.write(b); got += len(b)
    p.wait()
    assert got == nbytes, f"{path}: 流式 {got}B ≠ {nbytes}B"


def merge(a):
    fsz = os.path.getsize(a.skeleton)
    with open(a.skeleton, "rb") as f:
        n_kv, kv_raw, tens, data0 = parse_header(f)
    sizes_by_offset(tens, fsz, data0)
    rb = load_route_bias(a.route_bias, a.route_alpha) if a.route_bias else {}
    bsz = {}
    for line in open(a.blob_sizes):
        L, s = line.split(); bsz[int(L)] = int(s)
    assert len(bsz) == N_LAYER, f"blob 尺寸表 {len(bsz)}/43"
    # D 段权威偏移(dql_down_offset.py 记录链解析): 固定 DQL_HDR=35104 是错误假设 —
    # 1bit 记录载荷@128, 实测全层 570,425,472; 仍按每层文件读取防未来漂移。
    doff = {}
    for line in open(a.down_offsets):
        L, s = line.split(); doff[int(L)] = int(s)
    assert len(doff) == N_LAYER, f"down 偏移表 {len(doff)}/43"
    ents = [{"name": t["name"], "ne": t["ne"], "type": t["type"],
             "bytes": t["bytes"], "kind": "skel", "src_off": t["off"]} for t in tens]
    for L in range(N_LAYER):
        ents.append({"name": f"blk.{L}.ffn_down_exps.weight", "ne": [2048, 4096, 256],
                     "type": 40, "bytes": DOWN_LAYER_BYTES, "kind": "down", "L": L})
        ents.append({"name": f"blk.{L}.ffn_exps_vq.blob", "ne": [bsz[L]],
                     "type": 42, "bytes": bsz[L], "kind": "blob", "L": L})
    src = open(a.skeleton, "rb")
    def w(out, e):
        if e["kind"] == "skel":
            src.seek(data0 + e["src_off"])
            nm = e["name"]
            if nm.startswith("blk.") and nm.endswith(".exp_probs_b.bias"):
                L = int(nm.split(".")[1])
                raw = src.read(e["bytes"])
                if L in rb:
                    vals = list(struct.unpack("<%df" % NEXP, raw[:NEXP*4]))
                    vals = [v + d for v, d in zip(vals, rb[L])]
                    raw = struct.pack("<%df" % NEXP, *vals) + raw[NEXP*4:]
                    print(f"[merge] gate.bias L{L} 烘焙 ✓", flush=True)
                out.write(raw); return
            left = e["bytes"]
            while left:
                b = src.read(min(1 << 26, left)); out.write(b); left -= len(b)
        elif e["kind"] == "down":
            ssh_stream(a.dql_host, f"{a.dql_dir}/dql_L{e['L']:02d}.bin",
                       DOWN_LAYER_BYTES, out, skip=doff[e["L"]])
            print(f"[merge] down L{e['L']} ✓", flush=True)
        else:
            ssh_stream(a.dql_host, f"{a.dql_dir}/dql_vq_L{e['L']:02d}.bin",
                       e["bytes"], out)
            print(f"[merge] blob L{e['L']} {e['bytes']/2**20:.0f}MiB ✓", flush=True)
    write_gguf(a.out, kv_raw, n_kv, ents, w)
    print(f"[merge] 完成 {a.out} {os.path.getsize(a.out)/2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-skeleton", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--base"); ap.add_argument("--skeleton")
    ap.add_argument("--blob-sizes"); ap.add_argument("--down-offsets")
    ap.add_argument("--route-bias"); ap.add_argument("--route-alpha", type=float, default=2.5)
    ap.add_argument("--dql-host", default="192.168.1.2")
    ap.add_argument("--dql-dir", default="/Users/fodelf/ds4-main/gguf/go-onebit/layers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.extract_skeleton: extract_skeleton(a)
    elif a.merge:
        assert a.blob_sizes and a.down_offsets, "--merge 需 --blob-sizes + --down-offsets"
        merge(a)
    else: sys.exit("需 --extract-skeleton 或 --merge")


if __name__ == "__main__":
    main()
