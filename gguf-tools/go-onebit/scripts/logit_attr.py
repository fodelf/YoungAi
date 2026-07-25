#!/usr/bin/env python3
"""logit_attr.py — 瑕疵归因: 在指定文本位置对比 FP(锚) vs 量化 逐token top-k。
判据: FP 首选=正确写法而量化翻错 → 量化致伤; FP 也选瑕疵写法 → 模型本身。
用法: logit_attr.py anchor.bin lq_dump.bin ids.txt tokenizer.json "搜索串1" ["搜索串2" ...]
anchor 布局: hd[8]u32 + idh u64 + fin[NL*S*DIM]f32 + ridx[NL*S*NACT]i32 + rw 同 + H[NL*S*HCM*DIM]f32 + logits[S*V]f32
lq_dump: [S,V]i32 头 + f32 数据 (DS4_DUMP_LOGITS)
"""
import sys, struct
import numpy as np
from tokenizers import Tokenizer

anchor_p, lq_p, ids_p, tok_p = sys.argv[1:5]
needles = sys.argv[5:]

ids = [int(x) for x in open(ids_p) if x.strip()]
tok = Tokenizer.from_file(tok_p)

f = open(anchor_p, "rb")
hd = struct.unpack("<8I", f.read(32)); f.read(8)
S, HCM, DIM, NL, V, NACT = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
assert S <= len(ids) + 1, f"锚 S={S} vs ids={len(ids)}"
off = 40 + (NL*S*DIM)*4 + (NL*S*NACT)*4*2 + (NL*S*HCM*DIM)*4
f.seek(off)
fp_logits = np.frombuffer(f.read(S*V*4), dtype="<f4").reshape(S, V)

q_logits = None
if lq_p != "-":   # "-" = 仅FP模式(量化侧top1=temp0生成文本本身, 构造成立)
    g = open(lq_p, "rb")
    S2, V2 = struct.unpack("<ii", g.read(8))
    assert (S2, V2) == (S, V), f"量化dump {S2}x{V2} ≠ 锚 {S}x{V}"
    q_logits = np.frombuffer(g.read(S*V*4), dtype="<f4").reshape(S, V)

# 全文重建 + needle 定位到 token 位置
pieces = [tok.decode([i]) for i in ids[:S]]
text = ""; starts = []
for p in pieces:
    starts.append(len(text)); text += p

def topk(l, k=5):
    idx = np.argsort(l)[-k:][::-1]
    return [(tok.decode([int(i)]).replace("\n", "\\n"), float(l[i])) for i in idx]

for nd in needles:
    at = text.find(nd)
    if at < 0:
        print(f"◆ '{nd}' 未在文中找到"); continue
    ti = max(i for i, s in enumerate(starts) if s <= at)   # needle 首 token
    pred = ti - 1                                          # 预测它的位置
    print(f"\n◆ 瑕疵点 '{nd}' @tok{ti} (上文…{text[max(0,at-40):at]!r})")
    print(f"  实际下一token(=量化贪心首选): {pieces[ti]!r}")
    print(f"  FP  top5: {topk(fp_logits[pred])}")
    fp1 = topk(fp_logits[pred], 1)[0][0]
    if q_logits is not None:
        print(f"  量化 top5: {topk(q_logits[pred])}")
    same = (fp1.strip() == pieces[ti].strip())
    print(f"  → FP首选={fp1!r} vs 实际={pieces[ti]!r} {'一致(模型本身)' if same else '★翻转(量化致伤)★'}")
