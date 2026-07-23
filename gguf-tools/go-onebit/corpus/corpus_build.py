#!/usr/bin/env python3
# corpus_build.py — corpus stage 2: merge harvested code/issues + user-supplied
# books (BOOKS_DIR, legal texts only) + methodology docs (METHOD_DIR) into the
# routed calibration/post-training corpus:
#   simple chain : plain continuation chunks (short snippets, <= ROUTE_SIMPLE_MAX_LINES)
#   complex chain: issue -> analysis -> patch style threaded chunks
# Mix ratio + chunk size configurable. Output: CORPUS_OUT/{kw}_simple.txt,
# {kw}_complex.txt, {kw}_mixed.txt (the capture/训练入口用 mixed).
#   env: KEYWORD, RAW=corpus/raw, BOOKS_DIR, METHOD_DIR, OUT=corpus/build,
#        ROUTE_SIMPLE_MAX_LINES=40, SIMPLE_RATIO=0.6, CHUNK_TOKENS=512
import os, pathlib, re

KW = os.environ.get("KEYWORD", "go")
RAW = pathlib.Path(os.environ.get("RAW", "corpus/raw")) / KW
BOOKS = pathlib.Path(os.environ.get("BOOKS_DIR", "corpus/books"))
METHOD = pathlib.Path(os.environ.get("METHOD_DIR", "corpus/method"))
OUT = pathlib.Path(os.environ.get("OUT", "corpus/build")); OUT.mkdir(parents=True, exist_ok=True)
SIMPLE_MAX = int(os.environ.get("ROUTE_SIMPLE_MAX_LINES", "40"))
RATIO = float(os.environ.get("SIMPLE_RATIO", "0.6"))
CHARS = int(os.environ.get("CHUNK_TOKENS", "512")) * 4      # ~4 chars/token

simple, cplx = [], []

# ---- code: function-level split, route by length ---------------------------
# 2026-07-20 域放大: 切分边界覆盖多语言定义头(Go func / Py def·class / Rust fn·impl /
# JS·TS function·export / Java·C 无通用头走文件界标); SPLIT_RE env 可覆盖。
SPLIT_RE = os.environ.get(
    "SPLIT_RE",
    r"\n(?=func |def |class |fn |impl |function |export |// ==== )")
for f in sorted(RAW.glob("*.code.txt")):
    txt = f.read_text(errors="ignore")
    for block in re.split(SPLIT_RE, txt):
        if not block.strip(): continue
        (simple if block.count("\n") <= SIMPLE_MAX else cplx).append(block.strip())

# ---- issues: inherently complex-chain material ------------------------------
for f in sorted(RAW.glob("*.issues.txt")):
    for th in f.read_text(errors="ignore").split("=== ISSUE"):
        if th.strip():
            cplx.append("// ISSUE THREAD (analyze, then propose the fix)\n=== ISSUE" + th.strip())

# ---- books + methodology: user-supplied texts, chunked, complex chain -------
for d, tag in ((BOOKS, "BOOK"), (METHOD, "METHOD")):
    if d.exists():
        for f in sorted(d.rglob("*.txt")) + sorted(d.rglob("*.md")):
            t = f.read_text(errors="ignore")
            for i in range(0, len(t), CHARS * 8):
                cplx.append(f"// {tag}: {f.stem}\n" + t[i:i + CHARS * 8])

def pack(items, path):
    with open(path, "w") as f:
        for it in items:
            f.write(it[:CHARS * 8] + "\n\n")
    return path

pack(simple, OUT / f"{KW}_simple.txt")
pack(cplx, OUT / f"{KW}_complex.txt")
# mixed stream honoring SIMPLE_RATIO (interleave so chunk-level heldout stays balanced)
mixed, si, ci = [], 0, 0
while si < len(simple) or ci < len(cplx):
    take_simple = (si + ci) == 0 or (si / max(1, si + ci)) < RATIO
    if take_simple and si < len(simple): mixed.append(simple[si]); si += 1
    elif ci < len(cplx): mixed.append(cplx[ci]); ci += 1
    else: mixed.append(simple[si]); si += 1
pack(mixed, OUT / f"{KW}_mixed.txt")
print(f"[{KW}] simple={len(simple)} complex={len(cplx)} -> {OUT}/{KW}_mixed.txt", flush=True)
