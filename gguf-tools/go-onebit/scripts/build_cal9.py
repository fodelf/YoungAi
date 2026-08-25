#!/usr/bin/env python3
# build_cal9.py — 全域全wiki校准语料 cal9(2026-08-15 用户令"全语料都要+每层≤5分钟+从wiki找别自造")。
# 五域全部官方维基来源: en=wikitext-2 train(官方数据集) | prog/fin=英文维基主题条目 |
# zh/it=中文/意大利语维基条目(官方API原文)。每域 fit+ev 连续(拼接律), ev=异条目零重叠。
# S 预算 ≈3000(dchunk 全GPTQ ≈5分/层)。产物: g7/wt2train_cal9.ids + MD_FR/MD_EV 打印。
import re
from tokenizers import Tokenizer

ROOT = "/Users/fodelf/ds4-main"
C = f"{ROOT}/gguf-tools/go-onebit/corpus"
W = f"{C}/wiki9"
tok = Tokenizer.from_file(f"{ROOT}/hf/DeepSeek-V4-Flash-0731/tokenizer.json")
T = lambda s: tok.encode(s, add_special_tokens=False).ids

en_arts = [a for a in re.split(r'\n(?= = [^=\n][^\n]*[^=\n] = \n)', open(f"{C}/wikitext2_cal7_fit.txt").read()) if a.strip()]
en_ev_arts = [a for a in re.split(r'\n(?= = [^=\n][^\n]*[^=\n] = \n)', open(f"{C}/wikitext2_cal7_ev.txt").read()) if a.strip()]
ev_extra = open(f"{W}/en_ev_extra.txt").read()
def art(fname_or_txt, title):
    t = open(fname_or_txt).read() if fname_or_txt.endswith(".txt") else fname_or_txt
    m = re.search(rf'= {re.escape(title)} =\n', t)
    s = m.end() if m else 0
    return t[s:s+20000]

blocks = [  # (名, fit文本, ev文本)
    ("en",   "".join(en_arts[:15]),                    "".join(en_ev_arts[:6])),
    ("prog", open(f"{W}/en_prog.txt").read()[:2100],   art(f"{W}/en_ev_extra.txt","Compiler")[:420]),
    ("fin",  open(f"{W}/en_fin.txt").read()[:1350],    art(f"{W}/en_ev_extra.txt","Interest rate")[:380]),
    ("zh",   open(f"{W}/zh.txt").read()[:620],         open(f"{W}/zh_ev.txt").read()[:220]),
    ("it",   open(f"{W}/it.txt").read()[:860],         open(f"{W}/it_ev.txt").read()[:250]),
]
ids = []; fr = []; ev = []
for nm, ft, et in blocks:
    fi = T(ft); ei = T(et)
    fr.append((len(ids), len(ids)+len(fi))); ids += fi
    ev.append((len(ids), len(ids)+len(ei))); ids += ei
    print(f"{nm}: fit {len(fi)} tok [{fr[-1][0]}:{fr[-1][1]}]  ev {len(ei)} tok [{ev[-1][0]}:{ev[-1][1]}]")
open(f"{ROOT}/gguf/go-onebit/g7/wt2train_cal9.ids","w").write("\n".join(map(str,ids))+"\n")
print(f"S={len(ids)}")
print("MD_FR=" + ",".join(f"{a}:{b}" for a,b in fr))
print("MD_EV=" + ",".join(f"{a}:{b}" for a,b in ev))
