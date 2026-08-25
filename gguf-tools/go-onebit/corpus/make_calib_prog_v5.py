#!/usr/bin/env python3
"""make_calib_prog_v5.py — v5 标准化校准语料(2026-07-28)。

破案背景(fable5): v4 语料覆盖的是"语言"清单, 边际存活需要覆盖"风格"流形 —
Go 校准全仓库风(1297字符, 占42%)而 HumanEval-x-Go 是教程风 → 教程风边际被
1-bit 压平 → 复读吸引子(Go 7/13); Python 只 351 字符却 18/20(base 预训练厚)。
LFU校准片↔LRU行为门满分 = 同语言内剂量-反应自证。

v5 双标准:
  ①覆盖面标准化: 每语言 × {repo 仓库风, tut 教程风, test 测试风, contract 契约风(含错误处理)} 四格, 教程风为自创
    任务(零 HumanEval/判决锚污染), 刻意含逐行解说注释(边际塌陷区)。
  ②体积统一: 每格等 token 预算 B=110(行粒度截断, 填充率断言 ≥80%)。
HELD 区: 每格 fit 截断点之后的后续行(构造上零重叠) + 非代码尾巴。
用法(M1): python3 make_calib_prog_v5.py   (产出 /tmp/rr_calib_prog_v5.{ids,meta})
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = int(os.environ.get("V5_BUDGET", 110))   # 每格 fit token 预算(env 可调档: mini 校准=25)
OUT_TAG = os.environ.get("V5_TAG", "v5")         # 产物名后缀(mini 版不覆盖全量版)
HELD_LINES = int(os.environ.get("V5_HELD_LINES", 5))   # 每格 held 后续行数(mini=1 控锚RAM)
FILL_MIN = float(os.environ.get("V5_FILL_MIN", 0.80))   # mini 档(小预算撞行粒度)可放宽

# 仓库风提取锚(v4 同源, 行数放宽由 token 预算截)
REPO = {
    "go":         ("raw/go/TheAlgorithms__Go.code.txt", "package cache", 80),
    "python":     ("raw/python/*.code.txt", "    def route(self, rule: str", 40),
    "javascript": ("raw/javascript/*.code.txt", "function dispatchRequest(config) {", 40),
    "rust":       ("raw/rust/*.code.txt", "pub fn is_match<P: AsRef<Path>>", 40),
    "typescript": ("raw/typescript/*.code.txt", "export function reactive(target: object) {", 40),
    "c":          ("raw/c/*.code.txt", "static void assoc_expand(", 40),
    "java":       ("raw/java/*.code.txt", "public <T> T fromJson(String json, Class<T> classOfT)", 40),
}
LANGS = list(REPO)

# 测试风提取锚(raw 真实仓库测试码; c 域 raw 无测试→自创 tests/c.txt)
TESTS = {
    "go":         ("raw/go/avelino__awesome-go.code.txt", "func TestAlpha(t *testing.T) {", 40),
    "python":     ("raw/python/pallets__flask.code.txt", "def test_index(app, client, path, template_name):", 40),
    "javascript": ("raw/javascript/TheAlgorithms__JavaScript.code.txt", "describe('AllCombinationsOfSizeK'", 40),
    "rust":       ("raw/rust/BurntSushi__ripgrep.code.txt", "#[test]", 40),
    "typescript": ("raw/typescript/honojs__hono.code.txt", "describe('removePrivateFields'", 40),
    "java":       ("raw/java/google__gson.code.txt", "@Test", 40),
}


def repo_lines(lang):
    import glob
    pat, anchor, span = REPO[lang]
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        lines = open(f, errors="replace").read().splitlines()
        for i, l in enumerate(lines):
            if anchor in l:
                return lines[i:i + span]
    raise SystemExit(f"[FATAL] {lang} 仓库锚未命中: {anchor!r}")


def tutorial_lines(lang):
    p = os.path.join(HERE, "tutorial", f"{lang}.txt")
    return open(p).read().splitlines()


def file_lines(sub, lang):
    return open(os.path.join(HERE, sub, f"{lang}.txt")).read().splitlines()


def anchored(table, lang):
    import glob
    pat, anchor, span = table[lang]
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        lines = open(f, errors="replace").read().splitlines()
        for i, l in enumerate(lines):
            if anchor in l:
                return lines[i:i + span]
    raise SystemExit(f"[FATAL] {lang} 锚未命中: {anchor!r}")


def test_lines(lang):
    if lang == "c":
        return file_lines("tests", "c")
    return anchored(TESTS, lang)


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(sys.argv[1] if len(sys.argv) > 1 else
                              "/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base/tokenizer.json")
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False).ids)

    fit_parts, held_parts, table, pending = [], [], [], []
    for lang in LANGS:
        for style, lines in (("repo", repo_lines(lang)), ("tut", tutorial_lines(lang)),
                             ("test", test_lines(lang)), ("contract", file_lines("contract", lang))):
            cell, used = [], 0
            for k, l in enumerate(lines):
                cand = "\n".join(cell + [l])
                n = ntok(cand)
                if n > BUDGET:
                    break
                cell, used, cut = cell + [l], n, k + 1
            else:
                cut = len(lines)
            assert used >= FILL_MIN * BUDGET, \
                f"[FATAL] {lang}/{style} 填充 {used}/{BUDGET} <80%(原料不足)"
            fit_parts.append(f"// ==== {lang}.{style} ====\n" + "\n".join(cell))
            pending.append((lang, style, used, len(cell), lines[cut:cut + HELD_LINES * 2]))

    # held 尾巴: 全局 fit 行集过滤(真实仓库码同一行跨函数重复是常态)后收 HELD_LINES 行
    fset_pre = set(l for part in fit_parts for l in part.splitlines() if len(l.strip()) > 12)
    for lang, style, used, ncell, rawtail in pending:
        tail = [l for l in rawtail if l not in fset_pre][:HELD_LINES]
        if tail:
            held_parts.append(f"// ==== held_{lang}.{style} ====\n" + "\n".join(tail))
        table.append((lang, style, used, ncell, len(tail)))

    # 非代码尾巴(v4 同源 method 片)
    mth = os.path.join(HERE, "method", "methodology_core.md")
    if os.path.exists(mth):
        seg, on = [], False
        for l in open(mth).read().splitlines():
            if "## Cache penetration" in l: on = True
            if on: seg.append(l)
            if on and "hammer the database" in l: break
        held_parts.append("// ==== method ====\n" + "\n".join(seg))

    fit = "\n".join(fit_parts) + "\n"
    held = "\n".join(held_parts) + "\n"
    out = fit + held

    # 断言①: fit/held 行零重叠
    fset = set(l for l in fit.splitlines() if len(l.strip()) > 12)
    dup = [l for l in held.splitlines() if l in fset]
    assert not dup, f"[FATAL] fit/held 行重叠 {len(dup)}: {dup[0][:60]!r}"
    # 断言②: 判决锚双签名
    for judge in ("hard_eval.txt", "coding_hard.txt"):
        sig = open(os.path.join(HERE, judge), "rb").read()[60:120]
        assert sig not in out.encode(), f"[FATAL] {judge} 判决锚污染"
    # 断言③: 教程任务名不出现在本地基准题库(防基准污染)
    bench = os.path.join(HERE, "..", "..", "..", "benchmarks")
    names = ["SwapAdjacentPairs", "diagonal_window_sum", "interleave(",
             "tidy_spaces", "countTitleVowelWords", "rotate_right", "mergeSorted",
             "NewRing(", "TokenBucket", "class Emitter", "BoundedStack",
             "TtlCache", "intern_init", "class Retrier", "span_clamp"]
    if os.path.isdir(bench):
        import subprocess
        for nm in names:
            r = subprocess.run(["grep", "-rq", "--exclude-dir=node_modules", "--exclude-dir=.git", nm, bench])
            assert r.returncode != 0, f"[FATAL] 教程任务 {nm} 撞基准题库"

    open(os.path.join(HERE, f"calib_prog_{OUT_TAG}.txt"), "w").write(out)
    ids = tok.encode(out, add_special_tokens=False).ids
    nfit = ntok(fit)
    open(f"/tmp/rr_calib_prog_{OUT_TAG}.ids", "w").write("\n".join(map(str, ids)) + "\n")
    open(f"/tmp/rr_calib_prog_{OUT_TAG}.meta", "w").write(f"NTOK={len(ids)}\nNFIT={nfit}\n")

    print(f"{'语言':<12}{'风格':<6}{'tok':>5}{'行':>4}{'held行':>7}", file=sys.stderr)
    for lang, style, used, nl, nh in table:
        print(f"{lang:<12}{style:<6}{used:>5}{nl:>4}{nh:>7}", file=sys.stderr)
    lo = min(t[2] for t in table); hi = max(t[2] for t in table)
    print(f"v5: NTOK={len(ids)} NFIT={nfit} held={len(ids)-nfit} "
          f"格预算={BUDGET} 实际带宽=[{lo},{hi}] (统一度 {100*lo//hi}%)", file=sys.stderr)
    assert nfit < len(ids) - 32, "held 区太小"


if __name__ == "__main__":
    main()
