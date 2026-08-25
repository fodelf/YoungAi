#!/usr/bin/env python3
"""路由分歧针: 教师锚 ridx vs 引擎 ffn_moe_topk dump 的逐层 top-8 重合率。

用法:
  抽教师端 (在锚所在机):  probe_router_agree.py extract <anchor.bin> <S_take> <out.i32>
      -> 输出 [NL][S_take][NACT] i32 (锚头自描述, 任意 S_take<=S)
  比对端 (dump 所在机):    probe_router_agree.py compare <teacher.i32> <dump_dir> <prefix> <S_take> <chunk>
      -> 逐层表: top-8 集合重合率 / slot0(top-1 专家)一致率
"""
import sys, os, struct, glob
import numpy as np

def read_anchor_header(path):
    with open(path, "rb") as f:
        hd = struct.unpack("<8I", f.read(32))
        idh = struct.unpack("<Q", f.read(8))[0]
    assert hd[0] == 0x32415144, "not a DQA2 anchor"
    # {magic, S, HCM, DIM, NL, VOCAB, NACT, 0}
    return dict(S=hd[1], HCM=hd[2], DIM=hd[3], NL=hd[4], VOCAB=hd[5], NACT=hd[6], idh=idh)

def extract(anchor, s_take, out):
    h = read_anchor_header(anchor)
    S, NL, DIM, NACT = h["S"], h["NL"], h["DIM"], h["NACT"]
    assert s_take <= S
    ridx_off = 40 + NL * S * DIM * 4
    buf = np.empty((NL, s_take, NACT), dtype=np.int32)
    with open(anchor, "rb") as f:
        for L in range(NL):
            f.seek(ridx_off + (L * S) * NACT * 4)
            buf[L] = np.fromfile(f, dtype=np.int32, count=s_take * NACT).reshape(s_take, NACT)
    buf.tofile(out)
    print(f"extract: NL={NL} S_take={s_take} NACT={NACT} -> {out} ({buf.nbytes} B)")

def compare(teacher_path, dump_dir, prefix, s_take, chunk):
    # NL/NACT 由文件体积反推 (NACT 试 6/8, 取能整除且 NL 合理的那个)
    t = np.fromfile(teacher_path, dtype=np.int32)
    NACT = next(n for n in (6, 8) if t.size % (s_take * n) == 0 and 40 <= t.size // (s_take * n) <= 48)
    NL = t.size // (s_take * NACT)
    t = t.reshape(NL, s_take, NACT)
    print(f"teacher: NL={NL} S={s_take} NACT={NACT}")
    print(f"{'L':>3} {'topK重合':>8} {'top1同':>7}  分歧位置数(重合<K/K)")
    tot_ov, tot_t1, rows = 0.0, 0.0, 0
    per_layer = []
    for L in range(NL):
        eng = np.empty((s_take, NACT), dtype=np.int32)
        got = 0
        for p0 in range(0, s_take, chunk):
            fp = os.path.join(dump_dir, f"{prefix}_ffn_moe_topk-{L}_pos{p0}.i32")
            if not os.path.exists(fp):
                continue
            a = np.fromfile(fp, dtype=np.int32)
            n = min(a.size // NACT, s_take - p0)
            eng[p0:p0+n] = a[:n*NACT].reshape(n, NACT)
            got += n
        if got < s_take:
            print(f"{L:>3}  [缺 dump: {got}/{s_take}]")
            continue
        ov = np.array([len(set(eng[s]) & set(t[L, s])) for s in range(s_take)])
        t1 = (eng[:, 0] == t[L, :, 0])
        per_layer.append((L, ov.mean(), t1.mean(), int((ov < NACT).sum())))
        tot_ov += ov.mean(); tot_t1 += t1.mean(); rows += 1
        print(f"{L:>3} {ov.mean():>7.2f}/{NACT} {t1.mean()*100:>6.1f}% {int((ov<NACT).sum()):>6}/{s_take}")
    if rows:
        print(f"ALL {tot_ov/rows:>7.2f}/{NACT} {tot_t1/rows*100:>6.1f}%   (层均)")

if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
