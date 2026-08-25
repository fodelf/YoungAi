#!/usr/bin/env python3
# build_cal10.py — 开源标准校准语料(2026-08-19 用户令"跟开源一样, 不自定义; 数据可少")。
# 默认=wikitext-2-raw train 头部原样(llama.cpp imatrix 同款); CHUNKS>1 = 全文等距取窗
# (cal11 用法: Bartowski calibration_datav3.txt 小语料高覆盖, 2048 预算下 8窗×256 等距
#  铺满全文拿全域覆盖, 原文原序零自造)。fit/ev = 前80%/后20%。
# 用法: build_cal10.py [N=2048] [SRC=corpus/wiki.train.raw] [OUT=g7/wt2train_cal10.ids] [CHUNKS=1]
import os, sys
from tokenizers import Tokenizer

ROOT = os.environ.get("DS4_ROOT", os.path.expanduser("~/ds4-main"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SRC = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/gguf-tools/go-onebit/corpus/wiki.train.raw"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/gguf/go-onebit/g7/wt2train_cal10.ids"
CH = int(sys.argv[4]) if len(sys.argv) > 4 else 1
# PICK=even|odd (2026-08-19 切半设计, 08-21 补回): 等距窗按奇偶分成两份, 内容零重叠、
# 分布同源 —— 一半喂量化, 另一半喂反修, 保证"反修语料是量化未见数据"。
# 取 2*CH 个窗(每窗 N/CH token), 按奇偶各取 CH 个 ⇒ 两份各 N token。
PICK = os.environ.get("PICK", "").lower()
tok = Tokenizer.from_file(f"{ROOT}/hf/DeepSeek-V4-Flash-0731/tokenizer.json")
raw = open(SRC, encoding="utf-8").read()
if CH <= 1:
    ids = tok.encode(raw[:N*8], add_special_tokens=False).ids[:N]   # 8字符/词粗上界足量, 头部原样
else:
    allids = tok.encode(raw, add_special_tokens=False).ids
    w = N // CH
    assert len(allids) >= N, f"全文仅 {len(allids)} token < {N}"
    if PICK in ("even", "odd"):
        NW = CH * 2                              # 双倍窗数, 奇偶各占一半
        step = (len(allids) - w) // (NW - 1)
        want = 0 if PICK == "even" else 1
        ids = []
        for c in range(NW):
            if c % 2 != want: continue
            st = c * step
            ids += allids[st:st+w]
        ids = ids[:N]
    else:
        step = (len(allids) - w) // (CH - 1)   # 等距窗起点, 首窗=文件头, 末窗贴文件尾
        ids = []
        for c in range(CH):
            st = c * step
            ids += allids[st:st+w]
        ids = ids[:N]
assert len(ids) == N, f"文本不足 {N} token(拿到 {len(ids)})"
open(OUT, "w").write("\n".join(map(str, ids))+"\n")
cut = int(N*0.8)
print(f"S={N} src={os.path.basename(SRC)} chunks={CH} pick={PICK or 'all'} fit=0:{cut} ev={cut}:{N}")
