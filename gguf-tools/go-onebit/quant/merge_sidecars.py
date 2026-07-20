#!/usr/bin/env python3
"""merge_sidecars.py — 把逐层 go2b 侧车 GGUF 合并成单个 overlay 侧车 (ds4 --residual 一次加载全部最差层)。
每个输入侧车含 blk.{L}.ffn_{gate,up,down}_exps_res.weight(go2b) + ffn_res_lut(F32) + ds4.residual KVs。
合并=拼接所有层的张量 + 汇总 KV。用法: merge_sidecars.py OUT.gguf IN1.gguf IN2.gguf ...
"""
import sys, struct, os

def parse(f):
    def rd(n):
        b = f.read(n); assert len(b) == n; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24))
    assert magic == 0x46554747
    def rstr(): n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    kvs = []
    def rdval(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: return ("str", rstr())
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            return ("arr", et, [rdval(et) for _ in range(n)])
        return ("raw", rd(sz[t]))
    for _ in range(n_kv):
        k = rstr(); t, = struct.unpack("<I", rd(4)); kvs.append((k, t, rdval(t)))
    tens = []
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens.append([nm, list(ne), ty, off])
    data0 = (f.tell() + 31)//32*32
    return kvs, tens, data0

def tsize(ne, ty):
    if ty == 41: return ne[1]*(ne[0]//256)*68*ne[2]
    n = 1
    for x in ne: n *= x
    return n*4  # F32

out_p = sys.argv[1]; ins = sys.argv[2:]
all_t = []; all_kv = {("ds4.residual.present", 7): b"\x01"}
srcs = []
for ip in ins:
    f = open(ip, "rb"); kvs, tens, d0 = parse(f)
    for k, t, v in kvs:
        if k.startswith("ds4.residual.layer."): all_kv[(k, t)] = v[1]
    for nm, ne, ty, off in tens:
        all_t.append((nm, ne, ty, ip, d0 + off))
    srcs.append(f)

out = open(out_p, "wb")
def w(b): out.write(b)
def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
w(struct.pack("<IIQQ", 0x46554747, 3, len(all_t), len(all_kv)))
for (k, t), v in all_kv.items():
    wstr(k); w(struct.pack("<I", t)); w(v if isinstance(v, bytes) else struct.pack("<B", 1))
align = 32; roff = 0; sizes = []
for nm, ne, ty, ip, absoff in all_t:
    sz = tsize(ne, ty); sizes.append(sz)
    wstr(nm); w(struct.pack("<I", len(ne)))
    for d in ne: w(struct.pack("<Q", d))
    w(struct.pack("<IQ", ty, roff)); roff += (sz+align-1)//align*align
hdr = out.tell(); data0 = (hdr+align-1)//align*align; w(b"\0"*(data0-hdr))
DELETE_SRC = os.environ.get("MERGE_DELETE_SRC") == "1"   # 边合并边删源(空间紧)
last_tensor_of = {}                                       # ip -> 该源最后一个张量的全局下标
for i, (nm, ne, ty, ip, absoff) in enumerate(all_t): last_tensor_of[ip] = i
for i, ((nm, ne, ty, ip, absoff), sz) in enumerate(zip(all_t, sizes)):
    f = open(ip, "rb"); f.seek(absoff)
    remain = sz
    while remain > 0:
        chunk = f.read(min(1<<24, remain)); out.write(chunk); remain -= len(chunk)
    f.close()
    pad = (align - (out.tell()-data0) % align) % align; w(b"\0"*pad)
    if DELETE_SRC and last_tensor_of[ip] == i:
        os.remove(ip); print(f"  merged+rm {os.path.basename(ip)}", flush=True)
out.close()
print(f"MERGED {out_p} {os.path.getsize(out_p)>>20}MiB tensors={len(all_t)} layers={len([1 for k in all_kv if k[0].startswith('ds4.residual.layer')])}", flush=True)
