#!/usr/bin/env python3
# build_cal8.py — 全域校准语料 cal8(2026-08-15 用户令"全语料都要+每层≤5分钟")。
# 五域: EN(wikitext train, 判决域前置) prog fin zh it; 每域 fit+ev 连续(拼接律);
# S 预算 ≈2850 行(dchunk 全GPTQ ≈4.8分/层)。EV 多段(zlayer EV multi-seg 已支持)。
# 产物: g7/wt2train_cal8.ids + 打印 MD_FR/MD_EV 范围(战役发车直接抄用)。
import sys, os, re
from tokenizers import Tokenizer

ROOT = "/Users/fodelf/ds4-main"
C = f"{ROOT}/gguf-tools/go-onebit/corpus"
tok = Tokenizer.from_file(f"{ROOT}/hf/DeepSeek-V4-Flash-0731/tokenizer.json")
T = lambda s: tok.encode(s, add_special_tokens=False).ids

# EN: 从 wikitext2_cal7_fit(64篇已建)取前 18 篇做 fit, cal7_ev 前 6 篇做 ev(门禁已证零重叠)
en_fit_all = open(f"{C}/wikitext2_cal7_fit.txt").read()
en_arts = [a for a in re.split(r'\n(?= = [^=\n][^\n]*[^=\n] = \n)', en_fit_all) if a.strip()]
en_fit = "".join(en_arts[:18])
en_ev_all = open(f"{C}/wikitext2_cal7_ev.txt").read()
en_evarts = [a for a in re.split(r'\n(?= = [^=\n][^\n]*[^=\n] = \n)', en_ev_all) if a.strip()]
en_ev = "".join(en_evarts[:6])

prog = open(f"{C}/calib_prog_v5.txt").read()
fin  = open(f"{C}/calib_fin_v2.txt").read()
zh   = open(f"{C}/books/chai2010_advanced-go-programming-book__ch1-basic_ch1-03-array-string-and-slice.md").read()
it   = open(f"{ROOT}/speed-bench/promessi_sposi.txt").read()

blocks = [  # (名, fit文本, ev文本) — ev 取各源不相交的后段
    ("en",   en_fit,        en_ev),
    ("prog", prog[:2100],   prog[2400:2800]),
    ("fin",  fin[:1500],    fin[1800:2100]),
    ("zh",   zh[500:1300],  zh[1500:1700]),
    ("it",   it[1000:2000], it[2300:2500]),
]
ids = []; fr = []; ev = []
for nm, ft, et in blocks:
    fi = T(ft); ei = T(et)
    fr.append((len(ids), len(ids)+len(fi))); ids += fi
    ev.append((len(ids), len(ids)+len(ei))); ids += ei
    print(f"{nm}: fit {len(fi)} tok [{fr[-1][0]}:{fr[-1][1]}]  ev {len(ei)} tok [{ev[-1][0]}:{ev[-1][1]}]")
open(f"{ROOT}/gguf/go-onebit/g7/wt2train_cal8.ids","w").write("\n".join(map(str,ids))+"\n")
print(f"S={len(ids)}")
print("MD_FR=" + ",".join(f"{a}:{b}" for a,b in fr))
print("MD_EV=" + ",".join(f"{a}:{b}" for a,b in ev))
print("NFIT(报告用, 首段fit界)=", fr[0][1])
