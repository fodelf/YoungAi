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
    # ★gate/up 路由专家也必须丢(2026-08-02 修)★
    # 原来没这两条是因为骨架源一直是【冠军 VQ 合一 GGUF】—— 那种文件里 gate/up exps 早已
    # 不存在(专家字节全在 blk.L.ffn_exps_vq.blob 里)。而自产骨架的源是 deepseek4-quantize
    # --experts-hole 产出的留洞文件, 它的张量表沿用 published 模版, gate/up exps **在表里**
    # (只是数据是洞)。不丢的话 extract_skeleton 会把这些洞当骨架张量写出来:
    # 实测文件从 8.2 GiB 撑到 17.7+ GiB, 两次把磁盘写穿(OSError:28)。
    if skeleton_mode and (nm.endswith("ffn_gate_exps.weight") or
                          nm.endswith("ffn_up_exps.weight")): return True
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


def extract_blobs(a):
    """★反修v4逆向(2026-08-10): 从合一 GGUF 反抽 blk.L.ffn_exps_vq.blob → dql_vq_LXX.bin
    (--consume 吃掉层件后的链式反修回收路; 字节=blob 原样, vq_slot/zlayer 直读)。"""
    import os
    with open(a.base, "rb") as f:
        n_kv, kv_raw, tens, data0 = parse_header(f)
        fsz = os.fstat(f.fileno()).st_size
        sizes_by_offset(tens, fsz, data0)   # 原地写 t["bytes"]
        n = 0
        for t in tens:
            nm = t["name"]
            if not nm.endswith("ffn_exps_vq.blob"): continue
            L = int(nm.split(".")[1])
            outp = os.path.join(a.out_dir, "dql_vq_L%02d.bin" % L)
            f.seek(data0 + t["off"])
            remain = t["bytes"]
            with open(outp, "wb") as o:
                while remain > 0:
                    chunk = f.read(min(remain, 1 << 24))
                    if not chunk: break
                    o.write(chunk); remain -= len(chunk)
            n += 1
            print("L%02d blob %.2fGB -> %s" % (L, t["bytes"] / 1e9, outp), flush=True)
        print("extract-blobs done:", n)

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
    if host in ("", "-", "local", "127.0.0.1", "localhost"):
        # 本地直读(2026-07-31 R28: 合并在产物同机跑, 免 ssh 往返/免自连密钥)
        with open(path, "rb") as f:
            f.seek(skip); left = nbytes
            while left:
                b = f.read(min(1 << 24, left))
                if not b: break
                out.write(b); left -= len(b)
        assert left == 0, f"{path}: 本地读 {nbytes-left}B ≠ {nbytes}B"
        return
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



# ★opt 内嵌(2026-08-06 用户令"修通op")★: 反修 op(dql_ops 侧车终值+dql 正位 zl.RRR)
# 编成引擎 zchain_from_model 期望的四类张量内嵌进合并 GGUF(blk.L.opt_chain/ge/v8/zlm
# + kv ds4.zchain.present=true)。模型自包含, 运行时零外挂零开关。
REC_HDR = 116
def _iter_recs(raw):
    if len(raw) < 12 or raw[:4] not in (b"DQL2", b"DQO2"): return
    nrec, = struct.unpack_from("<I", raw, 8)
    off = 12
    for _ in range(nrec):
        if off + REC_HDR > len(raw): break
        nm = raw[off:off+16].split(b"\0")[0].decode("ascii", "replace")
        psz, = struct.unpack_from("<Q", raw, off+88)
        vd, = struct.unpack_from("<i", raw, off+112)
        pay = raw[off+REC_HDR:off+REC_HDR+psz]
        off += REC_HDR + psz
        yield nm, vd, pay

def build_opt_tensors(dql_dir, L):
    """→ [entries(kind=opt, data=bytes)] 引擎 zchain_from_model 格式。"""
    import numpy as np
    chain, v8blocks, ge, zlm = [], [], None, None
    glhc = None   # (gh,gc): 折进 per-expert ge 表(引擎 GE 通道零改动)
    p_ops = f"{dql_dir}/dql_ops_L{L:02d}.bin"
    if os.path.exists(p_ops):
        for nm, vd, pay in _iter_recs(open(p_ops, "rb").read()):
            if vd != 1: continue
            row = [0.0]*16
            if "GLdyn2" in nm and len(pay) >= 16:
                row[0] = 2.0; row[2:6] = struct.unpack("<4f", pay[:16]); chain.append(row)
            elif "GLdyn8" in nm and len(pay) >= 36:
                row[0] = 3.0; row[6:15] = struct.unpack("<9f", pay[:36])
                if len(pay) >= 36 + 8*4096*2:
                    row[15] = float(len(v8blocks)); v8blocks.append(pay[36:36+8*4096*2])
                else:
                    continue                      # V8-less dyn8 = 引擎侧 no-op, 不嵌
                chain.append(row)
            elif "bf.GE" in nm and len(pay) >= 512:
                ge = np.frombuffer(pay[:512], dtype=np.float16).astype(np.float32).tobytes()
            elif "zl.RRR" in nm and len(pay) >= 16:
                zk, tr = struct.unpack_from("<If", pay, 0)
                nh = zk + 2*zk*4096
                if 0 < zk <= 16 and len(pay) >= 16 + nh*2:
                    row[0] = 6.0; row[1] = tr; row[2] = float(zk); chain.append(row)
                    zlm = pay[16:16+nh*2]
            elif "GLhc" in nm and len(pay) >= 8:
                glhc = struct.unpack("<2f", pay[:8])
            elif ".GL" in nm and len(pay) >= 4:
                row[0] = 1.0; row[1], = struct.unpack("<f", pay[:4]); chain.append(row)
            elif ("TREF" in nm or "xlayer" in nm) and len(pay) >= 4:
                row[0] = 4.0; row[1], = struct.unpack("<f", pay[:4]); chain.append(row)
    if zlm is None:   # zl.RRR 正位在 dql 主文件(2026-08-04 正位直写)
        p_dql = f"{dql_dir}/dql_L{L:02d}.bin"
        if os.path.exists(p_dql):
            with open(p_dql, "rb") as f:
                head = f.read(12)
                if head[:4] == b"DQL2":
                    nrec, = struct.unpack_from("<I", head, 8)
                    off = 12
                    for _ in range(nrec):
                        hdr = f.read(REC_HDR)
                        if len(hdr) < REC_HDR: break
                        nm = hdr[:16].split(b"\0")[0].decode("ascii", "replace")
                        psz, = struct.unpack_from("<Q", hdr, 88)
                        if "zl.RRR" in nm and psz >= 16:
                            pay = f.read(psz)
                            zk, tr = struct.unpack_from("<If", pay, 0)
                            nh = zk + 2*zk*4096
                            if 0 < zk <= 16 and len(pay) >= 16 + nh*2:
                                row = [0.0]*16; row[0]=6.0; row[1]=tr; row[2]=float(zk)
                                chain.append(row); zlm = pay[16:16+nh*2]
                            break
                        f.seek(psz, 1)
    if glhc is not None:
        gh, gc = glhc
        import numpy as np
        gev = np.frombuffer(ge, dtype=np.float32).copy() if ge else np.ones(256, dtype=np.float32)
        vq_p = f"{dql_dir}/dql_vq_L{L:02d}.bin"
        hot = set()
        if os.path.exists(vq_p):
            with open(vq_p, "rb") as vf:
                vf.seek(16)
                vt = struct.unpack("<768Q", vf.read(768*8))
                hot = {e for e in range(256) if vt[e*3+2] != 0}
        for e in range(256):
            gev[e] *= gh if e in hot else gc
        ge = gev.tobytes()
    ents = []
    if chain:
        data = struct.pack("<%df" % (len(chain)*16), *[x for r in chain for x in r])
        ents.append({"name": f"blk.{L}.opt_chain.weight", "ne": [len(chain)*16],
                     "type": 0, "bytes": len(data), "kind": "opt", "data": data})
    if ge:
        ents.append({"name": f"blk.{L}.opt_ge.weight", "ne": [256],
                     "type": 0, "bytes": len(ge), "kind": "opt", "data": ge})
    if v8blocks:
        data = b"".join(v8blocks)
        ents.append({"name": f"blk.{L}.opt_v8.weight", "ne": [len(v8blocks)*8*4096],
                     "type": 1, "bytes": len(data), "kind": "opt", "data": data})
    if zlm is not None:
        ents.append({"name": f"blk.{L}.opt_zlm.weight", "ne": [len(zlm)//2],
                     "type": 1, "bytes": len(zlm), "kind": "opt", "data": zlm})
    return ents

def kv_append_bool(kv_raw, n_kv, key, val):
    b = struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I?", 7, val)
    return kv_raw + b, n_kv + 1

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
    if not a.no_down:
        for line in open(a.down_offsets):
            L, s = line.split(); doff[int(L)] = int(s)
        assert len(doff) == N_LAYER, f"down 偏移表 {len(doff)}/43"
    ents = [{"name": t["name"], "ne": t["ne"], "type": t["type"],
             "bytes": t["bytes"], "kind": "skel", "src_off": t["off"]} for t in tens]
    # opt 内嵌(本地 dql_dir 读; --consume 前于 ents 构造期完成全部读取)
    n_opt = 0
    if a.dql_host in ("", "-", "local", "127.0.0.1", "localhost"):
        for L in range(N_LAYER):
            oe = build_opt_tensors(a.dql_dir, L)
            ents.extend(oe); n_opt += len(oe)
    if n_opt:
        kv_raw, n_kv = kv_append_bool(kv_raw, n_kv, "ds4.zchain.present", True)
        print(f"[merge] opt 内嵌 {n_opt} 张量 + ds4.zchain.present=true", flush=True)
    else:
        print("[merge] ★opt 张量零内嵌(侧车缺?)— 反修效果不在此模型, 需人工确认★", flush=True)
    # ★--gud(2026-08-03 code2b 42G 复刻)★: G/U/D 三段 signref 全搬(1bit 记录载荷布局
    # [G 256×szG][U 256×szG][D 256×szD], dql_down_offset 的 D 偏移回推 G/U)。blob(热侧车)
    # 不进 GGUF — 运行时外挂, 42.468G 账目与 code2b 一字对齐。
    if a.gud:
        SZ_G = 2048 * (4096 // 256) * 34
        for L in range(N_LAYER):
            g_off = doff[L] - 2 * NEXP * SZ_G
            ents.append({"name": f"blk.{L}.ffn_gate_exps.weight", "ne": [4096, 2048, 256],
                         "type": 40, "bytes": NEXP * SZ_G, "kind": "gud", "L": L, "skip": g_off})
            ents.append({"name": f"blk.{L}.ffn_up_exps.weight", "ne": [4096, 2048, 256],
                         "type": 40, "bytes": NEXP * SZ_G, "kind": "gud", "L": L, "skip": g_off + NEXP * SZ_G})
            ents.append({"name": f"blk.{L}.ffn_down_exps.weight", "ne": [2048, 4096, 256],
                         "type": 40, "bytes": DOWN_LAYER_BYTES, "kind": "gud", "L": L,
                         "skip": doff[L], "last": True})
    blr = None
    if a.blob_layers:
        _lo, _hi = a.blob_layers.split(":"); blr = (int(_lo), int(_hi))
        print(f"[merge] 切片模式: 只带 L{blr[0]}..L{blr[1]} 的 blob({blr[1]-blr[0]+1} 层)", flush=True)
    for L in range(N_LAYER if not a.gud else 0):
        # --no-down(R28): 冷 w2 已编在 blob 的 which=2 槽里, base down 是纯死重
        # (43 × 0.2656 = 11.42 GiB)。省掉它才是 28 GiB 目标能成立的原因; 引擎侧
        # 由 routed_down_shadow() 合成影子张量顶上(ds4.c)。冠军 vq4bf 那种冷 w2 从
        # base go1b 读的配方不能带这个开关, 否则冷专家权重直接没了。
        if blr and not (blr[0] <= L <= blr[1]):
            # 存根 blob(6160B 零头): 装载器认层过"required tensor"检查, 槽位全 0 →
            # 前向若误触即报"缺 vq 槽"硬错; 分布式非本机层永不前向 ⇒ 纯占位。
            ents.append({"name": f"blk.{L}.ffn_exps_vq.blob", "ne": [6160],
                         "type": 42, "bytes": 6160, "kind": "stub", "L": L})
            continue
        if not a.no_down:
            ents.append({"name": f"blk.{L}.ffn_down_exps.weight", "ne": [2048, 4096, 256],
                         "type": 40, "bytes": DOWN_LAYER_BYTES, "kind": "down", "L": L})
        ents.append({"name": f"blk.{L}.ffn_exps_vq.blob", "ne": [bsz[L]],
                     "type": 42, "bytes": bsz[L], "kind": "blob", "L": L})
    src = open(a.skeleton, "rb")
    def w(out, e):
        if e["kind"] == "opt":
            out.write(e["data"]); return
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
        elif e["kind"] == "gud":
            dql_p = f"{a.dql_dir}/dql_L{e['L']:02d}.bin"
            ssh_stream(a.dql_host, dql_p, e["bytes"], out, skip=e["skip"])
            if a.consume and e.get("last"):
                out.flush(); os.remove(dql_p)
            print(f"[merge] {e['name'].split('.')[2]} L{e['L']} ✓"
                  + (" (dql已消费)" if a.consume and e.get("last") else ""), flush=True)
        elif e["kind"] == "stub":
            out.write(struct.pack("<I", 0x4C565144) + b"\0" * (e["bytes"] - 4))   # DQVL 魔数 + 全零槽表
        elif e["kind"] == "down":
            dql_p = f"{a.dql_dir}/dql_L{e['L']:02d}.bin"
            ssh_stream(a.dql_host, dql_p, DOWN_LAYER_BYTES, out, skip=doff[e["L"]])
            # ★--consume 也吃 dql(2026-08-03 冠军复刻盘账)★: D 段是 dql 里唯一被合并读的段,
            # 读完该层 dql 即死重(反修/回放此时已完成)。54G 输出必须靠这 ~13G 边合并边释放。
            if a.consume:
                out.flush(); os.remove(dql_p)
            print(f"[merge] down L{e['L']} ✓" + (" (dql已消费)" if a.consume else ""), flush=True)
        else:
            blob_p = f"{a.dql_dir}/dql_vq_L{e['L']:02d}.bin"
            ssh_stream(a.dql_host, blob_p, e["bytes"], out)
            # ★--consume(R30)★: 该层 blob 已完整写进输出 → 立删源文件。M1 free 50G 放不下
            # 层27.8+骨架8.2+输出36 三者同存; 消费式把峰值净增压到 ~9G(输出36−层27.8)。
            # 代价: 合并中途失败已删的层要重量化 —— 只在盘账过不去时用, 由调用方显式开。
            if a.consume:
                if a.dql_host not in ("", "-", "local", "127.0.0.1", "localhost"):
                    sys.exit("--consume 只允许本地 dql-host(不做远程删)")
                out.flush(); os.remove(blob_p)
            print(f"[merge] blob L{e['L']} {e['bytes']/2**20:.0f}MiB ✓"
                  + (" (源已消费)" if a.consume else ""), flush=True)
    write_gguf(a.out, kv_raw, n_kv, ents, w)
    print(f"[merge] 完成 {a.out} {os.path.getsize(a.out)/2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-skeleton", action="store_true")
    ap.add_argument("--extract-blobs", action="store_true", help="从合一GGUF反抽vq blob→dql_vq_L*.bin")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--base"); ap.add_argument("--skeleton")
    ap.add_argument("--blob-sizes"); ap.add_argument("--down-offsets")
    ap.add_argument("--no-down", action="store_true",
                    help="不写 base ffn_down_exps(冷 w2 已在 blob which=2 槽; 省 11.42 GiB)")
    ap.add_argument("--route-bias"); ap.add_argument("--route-alpha", type=float, default=2.5)
    ap.add_argument("--consume", action="store_true",
                    help="每层 blob 写完即删源(盘不够三者同存时用; 仅限本地 dql-host)")
    ap.add_argument("--gud", action="store_true",
                    help="code2b 模式: G/U/D 三段 signref 全搬(需 --down-offsets), blob 不进 GGUF")
    ap.add_argument("--blob-layers", default="",
                    help="只带 A:B(含)层的 vq blob — 双机协调器切片, 其余层 blob 缺席(worker 用全量文件)")
    ap.add_argument("--dql-host", default="192.168.1.2")
    ap.add_argument("--dql-dir", default="/Users/fodelf/ds4-main/gguf/go-onebit/layers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.extract_blobs:
        extract_blobs(a); return
    if a.extract_skeleton: extract_skeleton(a)
    elif a.merge:
        assert a.blob_sizes, "--merge 需 --blob-sizes"
        assert a.down_offsets or a.no_down, "--merge 需 --down-offsets(或 --no-down)"
        merge(a)
    else: sys.exit("需 --extract-skeleton 或 --merge")


if __name__ == "__main__":
    main()
