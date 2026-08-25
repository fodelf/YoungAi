#!/usr/bin/env python3
# corpus_gate.py — 战役语料门禁(2026-08-15 用户令"别再出现单一乔丹传记这种bug")。
# 用法: corpus_gate.py <fit.txt> <ev.txt> [judge_ids] [tokenizer.json]
# 五条硬检(任一不过=exit 1, 战役禁止发车):
#   G1 fit 文章数 >= 40                (单一来源bug的根)
#   G2 单篇最大份额 <= 5%              (乔丹条款: 任何一篇不得主导)
#   G3 EV 文章数 >= 8 且与 fit 零重叠  (组件门判卷必须多题材且独立)
#   G4 fit 总字符 in [12000, 40000]    (过短=覆盖不足, 过长=时间失控)
#   G5 (可选, 给 judge_ids+tokenizer 时) 判决唯一token覆盖率 >= 22% (实测有效带下限)
import sys, re

def articles(txt):
    parts = re.split(r'\n(?= = [^=\n][^\n]*[^=\n] = \n)', txt)
    # 首块可能无标题头(承接上一篇), 一并计为一篇
    return [p for p in parts if p.strip()]

fit_t = open(sys.argv[1]).read()
ev_t  = open(sys.argv[2]).read()
fa, ea = articles(fit_t), articles(ev_t)
def title(a):
    m = re.match(r' = ([^=\n]+) = ', a)
    return m.group(1).strip() if m else a[:40].strip()
ft, et = set(map(title, fa)), set(map(title, ea))
fail = []
if len(fa) < 40: fail.append(f"G1 fit文章数 {len(fa)} < 40")
shares = [len(a)/len(fit_t) for a in fa]
if max(shares) > 0.05: fail.append(f"G2 单篇最大份额 {max(shares)*100:.1f}% > 5%")
if len(ea) < 8: fail.append(f"G3 EV文章数 {len(ea)} < 8")
if ft & et: fail.append(f"G3 EV与fit重叠: {sorted(ft&et)[:3]}")
if not (12000 <= len(fit_t) <= 40000): fail.append(f"G4 fit字符 {len(fit_t)} 出界[12000,40000]")
cov = None
if len(sys.argv) > 4:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(sys.argv[4])
    fs = set(tok.encode(fit_t, add_special_tokens=False).ids)
    jud = [int(x) for x in open(sys.argv[3])]
    js = set(jud)
    cov = len(js & fs) / len(js)
    if cov < 0.22: fail.append(f"G5 判决唯一token覆盖 {cov*100:.1f}% < 22%")
print(f"[corpus_gate] fit {len(fa)}篇/{len(fit_t)}字 单篇峰值{max(shares)*100:.1f}% | EV {len(ea)}篇独立"
      + (f" | 覆盖{cov*100:.1f}%" if cov is not None else ""))
if fail:
    print("[corpus_gate] ★不过门★: " + "; ".join(fail)); sys.exit(1)
print("[corpus_gate] PASS")
