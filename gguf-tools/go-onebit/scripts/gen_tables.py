import sys,re,os
out,tbl=sys.argv[1],sys.argv[2]
txt=open(out,encoding='utf-8',errors='replace').read()
VD={1:"✓ 正向(落地)",2:"✓ 正向(未落地)",3:"探索中",4:"✓ 生效"}
def hb(n):
    n=int(n)
    if n>=1<<30: return f"{n/(1<<30):.2f} GiB"
    if n>=1<<20: return f"{n/(1<<20):.2f} MiB"
    if n>=1024:  return f"{n/1024:.1f} KB"
    return f"{n} B"
recs={}
for m in re.finditer(r"^TABREC L=(\d+) (\S+)\s+vol=(\d+) m=\[([-\d.eE ]+)\] 判定=(\d) \| (.+)$",txt,re.M):
    L,nm,vol,ms,vd,algo=int(m.group(1)),m.group(2),int(m.group(3)),m.group(4).split(),int(m.group(5)),m.group(6).strip()
    recs.setdefault(L,[]).append((nm,vol,ms,vd,algo))
raw=[l for l in txt.splitlines() if re.match(r"^(SEARCH|STACK|ROUND|BWDFIN|BACKX|ELEMENTS|VERDICT|ZFILE)",l)]
verd=re.findall(r"^VERDICT .*?smin=([\d.]+) kl=([\d.]+).*?expgib=([\d.]+)",txt,re.M)
zf=re.search(r"^ZFILE (\S+) layers=(\d+) bytes=(\d+)",txt,re.M)
summary=[]
for L in sorted(recs):
    lls=[l for l in raw if re.search(rf"\bL={L}\b",l)]
    els=[l for l in lls if l.startswith("ELEMENTS")]
    rounds=[l for l in lls if l.startswith("ROUND")or l.startswith("ELEMENTS")]
    lines=[f"# L{L:02d} 元素明细(自动生成, 数字=本次运行原始输出)\n"]
    lines.append("| 元素 | 算法 | 体积 | 实测数值 m1-m4 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for nm,vol,ms,vd,algo in recs[L]:
        lines.append(f"| {nm} | {algo} | {hb(vol)} | {' '.join(ms)} | {VD.get(vd,vd)} |")
    lines.append("\n数值口径: z/bwd 行 m1=val m2=held m3=相对base提升% m4=参数(g/λ/t)。")
    # 探索日志(该层被拒/未落地形态)
    lines.append("\n## 探索日志(尝试但未落地的形态)\n```")
    rej=[l for l in lls if ("✗" in l) or ("未中" in l)]
    lines.extend(rej if rej else ["(该层所有尝试形态均落地)"])
    lines.append("```")
    expl=sorted(set(nm for nm,_,_,vd,_ in recs[L] if vd==3))
    if expl:
        lines.append("仍在探索的元素: "+", ".join(expl)+" — 负数=形态不对, 换算法继续。")
    # 逐元素日志
    lines.append("\n## 逐元素日志(原始行)\n")
    def logs_for(nm):
        if nm=="1bit": return [l for l in lls if " base " in l]
        if nm=="bwd.final": return [l for l in lls if l.startswith("BWDFIN")]
        if nm=="bwd.xlayer": return [l for l in lls if l.startswith("BACKX")]
        if nm.startswith("loss")or nm=="percept": return rounds
        key=re.escape(nm.split(".",1)[-1])
        rx=re.compile(rf"^(SEARCH|STACK)\s+L={L}\b.*?\b{key}(?![\w#])")
        return [l for l in lls if rx.search(l)]
    seen=set()
    for nm,_,_,_,_ in recs[L]:
        if nm in seen: continue
        seen.add(nm)
        lg=logs_for(nm)
        lines.append(f"### {nm}")
        lines.append("```")
        lines.extend(lg if lg else ["(结构性元素 — 见 ROUND/ELEMENTS 收敛记录)"])
        lines.append("```")
    open(os.path.join(tbl,f"L{L:02d}.md"),'w',encoding='utf-8').write("\n".join(lines)+"\n")
    landed=[r for r in recs[L] if r[3]==1]
    bestz=next((r for r in landed if r[0].startswith("z.")),None)
    summary.append((L,els[-1] if els else "",bestz[0] if bestz else "-"))
al=[f"# 全模型量化方案总表(43 层, 自动生成)\n"]
if verd:
    sm,kl,eg=verd[-1]
    al.append(f"- **全模型还原度(vs 原始大模型, FP=100%): Σmin {float(sm)*100:.2f}%, KL {kl}**, 专家体积账 {eg} GiB")
if zf: al.append(f"- 合并动态侧车: `{os.path.basename(zf.group(1))}` {hb(zf.group(3))}({zf.group(2)} 层)")
al.append(f"- 每层明细: L00.md … L42.md; 原始输出 raw/all.out\n")
al.append("| 层 | z 胜者 | 收敛行(ELEMENTS) |")
al.append("|---|---|---|")
for L,els,bz in summary:
    al.append(f"| L{L:02d} | {bz} | `{els}` |")
open(os.path.join(tbl,"ALL.md"),'w',encoding='utf-8').write("\n".join(al)+"\n")
print(f"[M4] 每层明细 {len(summary)} 份 + ALL.md")
