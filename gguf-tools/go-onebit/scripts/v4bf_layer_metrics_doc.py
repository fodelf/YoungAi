#!/usr/bin/env python3
"""v4bf_layer_metrics_doc.py — 冠军 v4bf 每层指标落地文档(2026-07-29 用户令)。
来源(已归档): gguf/go-onebit/v4bf/v4bf_backfit_quant_all.{out,log} (M1 /tmp 拉回, 多次运行追加体);
自动选段: OUT 取最后一个含全 43 层 SEARCH_BEST+VERDICT 的 section = 冠军锚路由反修相;
LOG 取含冠军 L42 终值(val/held 与 OUT ELEMENTS 逐字匹配)的量化遍 section。
产物: gguf-tools/go-onebit/layer-tables/V4BF.md (表1=量化态/表2=反修后, 43 层×全指标);
既有 ALL.md/L*.md/raw 为该跑自动表, 本文档是其"量化 vs 反修后"合并对照视图, 不改动它们。
"""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "gguf/go-onebit/v4bf/v4bf_backfit_quant_all.out")
LOG = os.path.join(ROOT, "gguf/go-onebit/v4bf/v4bf_backfit_quant_all.log")
DOC = os.path.join(ROOT, "gguf-tools/go-onebit/layer-tables/V4BF.md")
NL = 43

def sections(lines, is_head):
    idx = [i for i, ln in enumerate(lines) if is_head(ln)] + [len(lines)]
    return [(idx[i], lines[idx[i]:idx[i+1]]) for i in range(len(idx)-1)]

out_lines = open(OUT, errors="replace").read().splitlines()
log_lines = open(LOG, errors="replace").read().splitlines()

# ---- OUT 冠军段: 最后一个 43 层 SEARCH_BEST + VERDICT 的段 ----
cand = []
for off, sec in sections(out_lines, lambda l: l.startswith("INHERIT L=0 ")):
    Ls = {int(m.group(1)) for ln in sec if (m := re.match(r"SEARCH_BEST L=(\d+) ", ln))}
    if len(Ls) == NL and any(ln.startswith("VERDICT ") for ln in sec):
        cand.append((off, sec))
assert cand, "OUT 无完整冠军段"
out_off, S = cand[-1]
prev_verdicts = [ln for off, sec in sections(out_lines, lambda l: l.startswith("INHERIT L=0 "))
                 if off < out_off for ln in sec if ln.startswith("VERDICT ")]

grab = lambda pat: {int(m.group(1)): m for ln in S if (m := re.match(pat, ln))}
inh  = grab(r"INHERIT L=(\d+) 上游累积漂移\(输入 vs FP\)=([\d.]+)")
base = grab(r"SEARCH L=(\d+) base\s+val=([\d.]+) held=([\d.]+) fit=([\d.]+) sc=([\d.eE+-]+) w2flip=([\d.]+)%")
sbst = grab(r"SEARCH_BEST L=(\d+) algo=(\S+) lam=(\S+) g=(\S+) k=(\d+) bwd=(\d) "
            r"val [\d.]+→([\d.]+)\(([-+][\d.]+)%\) held [\d.]+→([\d.]+)\(([-+][\d.]+)%\) payload=(\S+)")
grid = grab(r"STAGE_BEST L=(\d+) 四损失\+感知\(网格内落地\) al=(\d) cl=(\d) fx=(\d) sm=(\d) pc=(\d) val=([\d.]+)")
bwd  = grab(r"STAGE_BEST L=(\d+) 向后 (\S+) 最优t=([\d.]+) 阶段后val=([\d.]+)\(入口([\d.]+)\)")
elem = grab(r"ELEMENTS L=(\d+) base=([\d.]+) \| 乘法\+?([-\d.]+)% \| 动态\+?([-\d.]+)% \| 向后\+?([-\d.]+)% "
            r"\| 终 val=([\d.]+) held=([\d.]+)\(([-+][\d.]+)%\)")
lyrf = grab(r"LAYERFILE \S+ L=(\d+) 记录=(\d+) bytes=(\d+) \(([\d.]+ MiB)\)")
for name, d in [("INHERIT", inh), ("base", base), ("SEARCH_BEST", sbst), ("ELEMENTS", elem)]:
    assert len(d) == NL, f"{name} 只有 {len(d)}/43 层"
verdict = [ln for ln in S if ln.startswith("VERDICT ")][-1]
zfile   = [ln for ln in S if ln.startswith("ZFILE ")]
zchain  = [ln for ln in S if ln.startswith("ZCHAIN ")]

# ---- LOG 冠军段: 含冠军 L42 终值逐字对 + 43 条累积行 ----
key42 = f"val={elem[42].group(6)} held={elem[42].group(7)}"
lsec = None
for _, sec in sections(log_lines, lambda l: l.startswith("量化遍 ")):
    cum = [m for ln in sec if (m := re.search(
        r"累积relL2 fit=([\d.]+) held=([\d.]+) \| 路由一致=\s*([\d.]+)% \| 校准µ=([\d.]+)行 "
        r"空=(\d+)/256 \| z\^L k=(\d+)", ln))]
    if len(cum) == NL and any(key42 in ln for ln in sec):
        lsec, cums = sec, cum
assert lsec is not None, "LOG 冠军段未定位"
zl = {}   # 每层 z^L 闸判决(RRR 秩梯子过闸: k/体积)
for ln in lsec:
    m = re.search(r"【L(\d+)】z\^L 算法=(.*?) 进度=(\S+) 体积=(\S+) 还原度=k=(\d+) val=([\d.]+) held=([\d.]+)", ln)
    if m: zl[int(m.group(1))] = m

W = []
W.append("# v4bf 冠军 — 每层量化指标 vs 反修后指标(2026-07-29 落地)\n")
W.append("冠军 = 573b7f5『v4bf 冠军定版: 锚路由反修+路由偏置侧车α2.5+合并v4.1』的锚路由反修相"
         "(campaign_v4.sh backfit: BWD+JUSTIFIED+TERM_MAXP+ANCHOR_ROUTE, GSWEEP=0; "
         "语料 rr_calib_prog_v4 S=1340 n_fit=816 held=524, LCFG=g×43, 热表 hot_v4.txt Σ热=2752)。\n")
W.append("来源=本目录 raw/all.out 同跑归档 `gguf/go-onebit/v4bf/v4bf_backfit_quant_all.{out,log}` "
         f"(多跑追加体, 冠军段=OUT 第 {out_off+1} 行起; 每个数字可回查); "
         "每层机制明细见 L00.md…L42.md(多相追加体: 每文件最后一组=冠军相), 反修总览见 ALL.md。\n")
W.append("口径: 过程内链式 — 第 L 层评估时前缀 0..L-1 已量化+反修锁定。"
         "val/fit/held=相对 FP 锚累积 relL2(val=204 行调优集/fit=612 行拟合集/held=524 行不参拟合); "
         "sc=统一闸标量; w2flip=w2 符号翻转率; base=该层量化态(1bit 合并态: 冷 go1b 符号精修μ10×3 + "
         "热 go2b 合并 2bit, 反修前)。\n")

W.append("\n## 表1 — 每层量化态(反修前 base)\n")
W.append("| L | 上游累积漂移 | val | held | fit | sc | w2flip% | dql记录 |")
W.append("|---|---|---|---|---|---|---|---|")
for L in range(NL):
    b, f = base[L], lyrf.get(L)
    W.append(f"| L{L:02d} | {inh[L].group(2)} | {b.group(2)} | {b.group(3)} | {b.group(4)} "
             f"| {b.group(5)} | {b.group(6)} | {f.group(2) if f else '-'} |")

W.append("\n## 表2 — 每层反修后(锚路由反修定稿)\n")
W.append("| L | 胜者机制(payload) | 四损失落地 al/cl/fx/sm/pc | 向后t | 乘法/动态/向后% | z^L闸 "
         "| 终val(Δ%) | 终held(Δ%) | 累积fit/held | 路由一致% | 空专家/256 |")
W.append("|---|---|---|---|---|---|---|---|---|---|---|")
for L in range(NL):
    s, g, w, e, c = sbst[L], grid.get(L), bwd.get(L), elem[L], cums[L]
    z = zl.get(L)
    mech = f"{s.group(2)}" + (f" λ={s.group(3)}" if s.group(3) not in ("0", "0.0") else "") \
         + (f" g={s.group(4)}" if s.group(4) not in ("0.0000",) else "") + f" ({s.group(11)})"
    vald = (float(e.group(6)) / float(e.group(2)) - 1) * 100
    zcell = f"k={z.group(5)} {z.group(3)}({z.group(4)})" if z else f"k={c.group(6)}"
    W.append(f"| L{L:02d} | {mech} | {'/'.join(g.group(i) for i in range(2,7)) if g else '-'} "
             f"| {w.group(3) if w else '-'} | {e.group(3)}/{e.group(4)}/{e.group(5)} "
             f"| {zcell} "
             f"| {e.group(6)}({vald:+.2f}) | {e.group(7)}({e.group(8)}) "
             f"| {c.group(1)}/{c.group(2)} | {c.group(3)} | {c.group(5)} |")

W.append("\n## 全模型 rr 终判(判决榜口径 — 冠军名义数字在这里)\n")
W.append("rr_verdict 回放判决器, code=rr_code.ids S=305 held=76 / hard=rr_hard.ids S=64 held=15; "
         "原始文件归档 `gguf/go-onebit/v4bf/rr/`。逐轮:\n")
RR = os.path.join(ROOT, "gguf/go-onebit/v4bf/rr")
def rr_line(fn):
    ls = [ln for ln in open(os.path.join(RR, fn), errors="replace") if ln.startswith("VERDICT ")]
    return ls[-1].strip() if ls else "(缺)"
W.append("| 轮次 | code agree | code Σmin | code KL | code ratio | 原始行 |")
W.append("|---|---|---|---|---|---|")
def rr_row(name, fn):
    ln = rr_line(fn)
    m = re.search(r"ratio=([\d.]+) smin=([\d.]+) kl=([\d.]+) agree=([\d.]+)", ln)
    W.append(f"| {name} | **{m.group(4)}** | {m.group(2)} | {m.group(3)} | {m.group(1)} | `{ln}` |")
rr_row("① v4 emit(第一轮量化定稿)", "rrv_code.log")
rr_row("② 锚路由反修裸(修复态)", "rr_verdict.out")
rr_row("③ +路由偏置Δb(残态首扫 α2.5, 峰值)", "rb_ab_a2.5_firstgrid_RECOVERED.txt")
rr_row("★冠军终配置(修复态Δb@α2.5 烘入)", "rb_ab_a2.5.txt")
W.append("")
W.append("- hard S=64: ① Σmin 0.3654(`rrv_hard.log`) → ★冠军 0.3602/KL 2.1467(`rb_ab_summary.txt` "
         "HARD α2.5; α2.0 对照 0.3794/2.0094)。")
W.append("- ③的 84.2 是残态Δb 首扫峰值(原文件被修复态重跑覆盖, 原始行自会话历史恢复+α2.25/2.75 "
         "首扫残留佐证); 终配置用修复态重拟合Δb, α 网格峰=82.9(=top1f 天花板), code 全维优于 α2.0 "
         "故定 α2.5(fable5 2026-07-28 终判 wave)。")
W.append("\n## 战役内校准语料 VERDICT(v4 校准集 S=1340, 非判决榜; 与上表跨语料不可比)\n")
W.append(f"- **冠军锚路由反修相(+z合并)**: `{verdict}`")
for ln in prev_verdicts:
    W.append(f"- 前相对照(被冠军相取代): `{ln}`")
for ln in zfile[-1:] + zchain[-1:]:
    W.append(f"- `{ln}`")
tot = sum(int(lyrf[L].group(3)) for L in lyrf) / 2**30
W.append(f"- dql 层文件: 43×816.03 MiB = {tot:.2f} GiB(合并前中间态; 冠军 carrier=ds4-vq4bf.gguf, "
         "专家体积账见 VERDICT expgib)")
open(DOC, "w").write("\n".join(W) + "\n")
print(f"[v4bf-doc] 写出 {DOC} ({len(W)} 行)", file=sys.stderr)
