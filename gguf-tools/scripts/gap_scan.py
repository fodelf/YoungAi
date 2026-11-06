#!/usr/bin/env python3
# gap_scan.py — 能力缺口探索器(2026-11-05 夜, 用户: "每一轮结束没有科学的指标指出下一轮的方向, 还是随机探索")。
#
# 干什么: 只读每轮已经落盘的判决产物(不碰模型、不出题), 把"差在哪"拆开成能指方向的数, 再按它定下一轮的料配比。
#   ① 思考留出逐段 KL(heldout_e00.tsv 是轮前态；轮后读账本实际判决件对应的遍次，不固定 e01；eval_n 会整题跳过)
#      → 用每轮真实篇号与训练料记录数映射回原题 → 按题族 code/math/data/writing/svgchar/general 聚合: 每族 KL/位、决策位 KL、本轮 Δ、最差的题(长尾)
#   ② 编程判决份逐位 .rows(gguf/v41judge) + 教师 p1 缓存 → Same 缺口按教师自信拆: p1 ≥ 0.5 的错位 = 可修位(该追), p1 < 0.5 = 教师自己拿不准(别追)
#   ③ 本轮料里各族新鲜思考轨迹条数(r0NN/think*/q.jsonl 配 rows.jsonl)= 曝光 ⇒ 族的"响应度" = 曝光过的轮里该族 KL 降了多少
#   ④ 下一轮配比: 份额 ∝ 当前缺口 × 响应度(没数据的族给探索底额), 60 道预算里每族 4~24 道 → gap/next.json(thinkhold 按它出题)
# 用法(spark 仓库根):
#   gap_scan.py segmap --tag vl071           打印这一轮的精确归属，缺篇号则失败，不覆盖旧文件
#   gap_scan.py scan --tag vl071             一轮: 写 gap/vl071.tsv, 追加 gap/ledger.tsv 一行；--epoch 1 可显式指定第一遍
#   gap_scan.py history                      族 × 轮的 KL 与 Δ 表，列出实际遍次；缺篇号的旧轮跳过
#   gap_scan.py rebuild --out /tmp/gap-v2    重算可确认篇号的历史轮；旧账本不覆盖
#   gap_scan.py next --budget 60             按 ledger 里最近的轮定下一轮各族题数 → gap/next.json
# 出错会怎样: 无篇号、原题不明、前后样本对不上就报 ★ 不写；旧账本不再用于出配比，先 rebuild 后替换。
import argparse, glob, json, math, os, re, sys
from vloop_heldout import read_jsonl, fam_of_prefix, read_config, read_heldout, map_segments, validate_pair, paired_gain, selected_epoch

MAP_VERSION = "doc-v2"
TH = "gguf/go-onebit/evo/think_hold"; VL = "gguf/go-onebit/evo/vloop"; GAP = "gguf/go-onebit/evo/gap"
HF = "hf/DeepSeek-V4.1-Flash-1017"
FAMS = ["code", "math", "data", "writing", "svgchar", "general"]

def run_inputs(tag, directory):
    config = read_config(tag, directory)
    training = read_jsonl(config["data"])
    heldout = read_jsonl(config["eval_kl"])
    questions = read_jsonl(os.path.join(os.path.dirname(config["eval_kl"]), "q.jsonl"))
    return training, heldout, questions

def cmd_segmap(a):
    directory = a.dir or f"gguf/v41/posttrain/pt-{a.tag}"
    records = read_heldout(os.path.join(directory, "heldout_e00.tsv"))
    mapping = map_segments(records, *run_inputs(a.tag, directory), FAMS)
    print("# 段 原题行 题号 题族 答案token数 篇号")
    for i, record in enumerate(records):
        print("\t".join(map(str, (i, *mapping[i], record[0], record[3]))))
    return 0

def fam_agg(h, segmap):
    """按族: (KL 和, 位数, 决策位 KL 和, 决策段数); 按题: KL/位"""
    F = {f: [0.0, 0, 0.0, 0] for f in FAMS}; Q = {}
    for s, (n, kl, dk, _doc, _sample) in enumerate(h):
        row, qid, fam = segmap[s]
        F[fam][0] += kl; F[fam][1] += n
        if dk >= 0: F[fam][2] += dk; F[fam][3] += 1
        q = Q.setdefault(qid, [0.0, 0, fam]); q[0] += kl; q[1] += n
    return F, Q

def exposure(tag, training):
    """只数实际合并进训练文件的本轮思考题，目录里备好的下一轮料不算曝光。"""
    ex = {f: 0 for f in FAMS}
    questions = []
    for path in glob.glob(f"{VL}/r{int(tag[2:]):03d}/think*/q.jsonl"):
        questions.extend(read_jsonl(path))
    # 同一题单可能被保留为多个副本，按题号合并，避免重复匹配。
    questions = list({q["id"]: q for q in questions}.values())
    for row in training:
        prefix = row.get("prefix", "")
        if "<think>" not in prefix:
            continue
        _, family = fam_of_prefix(prefix, questions)
        if family in ex:
            ex[family] += 1
    return ex

def judge_split(tag, epoch):
    """编程判决份: 不同位按教师 p1 拆; 返回 (same%, 可修位, p1<0.5 错位, 位数) 或 None"""
    rows = sorted(glob.glob(f"gguf/v41judge/stu_code_judge_n15360_*pt-{tag}_ckpt_e{epoch:02d}v17.rows"))
    p1c = sorted(glob.glob("gguf/v41judge/teacher_code_judge_n15360_*.bin.p1.f32"))
    if not rows or not p1c: return None
    try:
        import numpy as np
        p1 = np.fromfile(p1c[-1], dtype=np.float32)
        same = [];
        for l in open(rows[-1]):
            c = l.split()
            if len(c) < 4: continue
            same.append(int(c[3]))
        same = np.array(same[:len(p1)]); p1 = p1[:len(same)]
        mism = same == 0
        return (100.0 * same.mean(), int((mism & (p1 >= 0.5)).sum()), int((mism & (p1 < 0.5)).sum()), int(len(same)))
    except Exception as e:
        print(f"(judge 拆分失败: {e})"); return None

def scan_one(tag, d=None, epoch=None):
    d = d or f"gguf/v41/posttrain/pt-{tag}"
    e0 = os.path.join(d, "heldout_e00.tsv")
    if not os.path.exists(e0) or not glob.glob(os.path.join(d, "heldout_e0[1-9].tsv")): return None
    epoch = epoch if epoch is not None else selected_epoch(tag, d, os.path.join(VL, "ledger.tsv"))
    if epoch < 1: raise ValueError("轮后遍次必须大于 0")
    e1 = os.path.join(d, f"heldout_e{epoch:02d}.tsv")
    h0, h1 = read_heldout(e0), read_heldout(e1)
    validate_pair(h0, h1)
    training, heldout, questions = run_inputs(tag, d)
    segmap = map_segments(h0, training, heldout, questions, FAMS)
    F0, Q0 = fam_agg(h0, segmap); F1, Q1 = fam_agg(h1, segmap)
    ex = exposure(tag, training)
    res = {"tag": tag, "epoch": epoch, "fam": {}, "exposure": ex, "mapping": MAP_VERSION, "paired": paired_gain(h0, h1)}
    for f in FAMS:
        k0 = F0[f][0] / max(F0[f][1], 1); k1 = F1[f][0] / max(F1[f][1], 1)
        res["fam"][f] = {"kl0": k0, "kl1": k1, "d": (k1 - k0) / k0 if k0 > 0 else 0.0, "dec0": F0[f][2], "dec1": F1[f][2], "n": F1[f][1]}
    tot0 = sum(F0[f][0] for f in FAMS) / sum(F0[f][1] for f in FAMS); tot1 = sum(F1[f][0] for f in FAMS) / sum(F1[f][1] for f in FAMS)
    res["kl0"], res["kl1"] = tot0, tot1
    res["dec0"], res["dec1"] = sum(F0[f][2] for f in FAMS), sum(F1[f][2] for f in FAMS)
    worst = sorted(((q[0] / max(q[1], 1), qid, q[2]) for qid, q in Q1.items()), reverse=True)[:4]
    res["worst"] = [(round(w, 4), qid, fam) for w, qid, fam in worst]
    res["judge"] = judge_split(tag, epoch)
    return res

def fmt_scan(r):
    s = [f"[缺口 {r['tag']} e00→e{r['epoch']:02d}] 思考留出 全位 KL {r['kl0']:.4f} → {r['kl1']:.4f}({100*(r['kl1']-r['kl0'])/r['kl0']:+.1f}%), 决策位 KL 和 {r['dec0']:.3f} → {r['dec1']:.3f}"]
    p = r["paired"]
    z = f"{p['z']:.2f}" if p["z"] is not None else "不可估计"
    se = f"{p['se']:.6f}" if p["se"] is not None else "不可估计"
    s.append(f"  真实题目 {p['documents']} 道 / {p['segments']} 段，题级配对降幅 {p['gain']:.6f} ± {se}，{z} 个标准误")
    s.append("  族        轮前KL   轮后KL    Δ%   决策位和   曝光(本轮新鲜轨迹条数)")
    for f in FAMS:
        x = r["fam"][f]
        s.append(f"  {f:8s} {x['kl0']:.4f}  {x['kl1']:.4f}  {100*x['d']:+5.1f}  {x['dec1']:8.3f}   {r['exposure'][f]}")
    s.append("  长尾(轮后 KL/位最高的题): " + ", ".join(f"{qid}({fam}) {w}" for w, qid, fam in r["worst"]))
    if r["judge"]:
        sm, fix, unc, n = r["judge"]
        s.append(f"  编程判决份 Same {sm:.2f}%: 不同位 {fix + unc} = 可修位(教师 p1≥0.5) {fix} + 教师拿不准位 {unc}(共 {n} 位)")
    return "\n".join(s)

def append_ledger(r):
    os.makedirs(GAP, exist_ok=True)
    p = os.path.join(GAP, "ledger.tsv"); new = not os.path.exists(p)
    if not new:
        with open(p) as stream:
            header = stream.readline().rstrip("\n").split("\t")
        if not {"mapping", "epoch"}.issubset(header):
            raise ValueError("旧账本没有可靠的题目映射或遍次；先 rebuild --out <新目录>，核对后再替换")
    with open(p, "a") as f:
        if new: f.write("tag\tkl0\tkl1\tdec0\tdec1\t" + "\t".join(f"{x}_kl1\t{x}_d\t{x}_ex" for x in FAMS) + "\tsame\tfix\tunc\tmapping\tepoch\n")
        j = r["judge"] or ("-", "-", "-", "-")
        f.write(f"{r['tag']}\t{r['kl0']:.5f}\t{r['kl1']:.5f}\t{r['dec0']:.3f}\t{r['dec1']:.3f}\t" +
                "\t".join(f"{r['fam'][x]['kl1']:.5f}\t{r['fam'][x]['d']:+.4f}\t{r['exposure'][x]}" for x in FAMS) +
                f"\t{j[0] if j[0]=='-' else f'{j[0]:.2f}'}\t{j[1]}\t{j[2]}\t{MAP_VERSION}\t{r['epoch']}\n")
    with open(os.path.join(GAP, f"{r['tag']}.tsv"), "w") as f:
        f.write(fmt_scan(r) + "\n")

def cmd_scan(a):
    r = scan_one(a.tag, a.dir, a.epoch)
    if not r: print(f"★{a.tag} 没有轮前/轮后 heldout TSV({a.dir or 'pt-' + a.tag})★"); return 1
    print(fmt_scan(r))
    if not a.read_only: append_ledger(r)
    return 0

def cmd_history(a):
    tags = sorted(os.path.basename(d)[3:] for d in glob.glob("gguf/v41/posttrain/pt-vl0*") if os.path.exists(os.path.join(d, "heldout_e01.tsv")))
    print("轮/遍次      全位KL后   Δ%  | " + " | ".join(f"{f[:5]:>5s} KL后  Δ%  曝" for f in FAMS) + " | 决策位后  | Same 可修 拿不准")
    for t in tags:
        try:
            r = scan_one(t)
        except (OSError, ValueError, KeyError) as error:
            print(f"跳过 {t}: {error}", file=sys.stderr)
            continue
        if not r: continue
        j = r["judge"]; js = f"{j[0]:.2f} {j[1]:4d} {j[2]:4d}" if j else "-"
        print(f"{t:6s}/e{r['epoch']:02d} {r['kl1']:.4f} {100*(r['kl1']-r['kl0'])/r['kl0']:+5.1f} | " +
              " | ".join(f"{r['fam'][f]['kl1']:.4f} {100*r['fam'][f]['d']:+5.1f} {r['exposure'][f]:2d}" for f in FAMS) +
              f" | {r['dec1']:8.3f} | {js}")
    return 0

def cmd_rebuild(a):
    global GAP
    if os.path.exists(os.path.join(a.out, "ledger.tsv")):
        raise ValueError("输出目录已有账本；请换新目录，避免覆盖原记录")
    GAP = a.out
    count = 0
    for directory in sorted(glob.glob("gguf/v41/posttrain/pt-vl*")):
        tag = os.path.basename(directory)[3:]
        try:
            result = scan_one(tag, directory)
        except (OSError, ValueError, KeyError) as error:
            print(f"跳过 {tag}: {error}", file=sys.stderr)
            continue
        if result:
            append_ledger(result)
            count += 1
            print(fmt_scan(result))
    if not count:
        raise ValueError("没有可确认题目归属的轮，未生成有效账本")
    print(f"重算 {count} 轮 → {GAP}/ledger.tsv；原账本未改")
    return 0

def cmd_next(a):
    """下一轮各族题数: 份额 ∝ 当前缺口(最近一轮轮后 KL/位) × 响应度(曝光过的轮里该族 Δ 的均值, 取降幅; 没曝光过 = 探索先验)"""
    p = os.path.join(GAP, "ledger.tsv")
    if not os.path.exists(p): print("★先 scan 几轮★"); return 1
    with open(p) as stream:
        hdr = stream.readline().rstrip("\n").split("\t")
        L = [l.rstrip("\n").split("\t") for l in stream if l.strip()]
    ix = {h: i for i, h in enumerate(hdr)}
    if not {"mapping", "epoch"}.issubset(ix) or not L or any(len(r) != len(hdr) or r[ix["mapping"]] != MAP_VERSION for r in L):
        raise ValueError("账本包含旧的猜测映射；先 rebuild，不能据此生成配比")
    seen_tag = {}
    for r in L: seen_tag[r[0]] = r          # 同一轮扫两次取后一次
    L = [seen_tag[t] for t in sorted(seen_tag, key=lambda t: int(t[2:]))]   # 按轮号排(补扫历史轮会追加在后面, 不能拿文件末行当"当前")
    last = L[-1]
    gap = {f: float(last[ix[f + "_kl1"]]) for f in FAMS}
    # 响应度 = 曝光过的轮里该族 Δ 的指数均值(近的权重大, α 0.5; 11-06 04:2x 实撞: math 从同一头起两轮各 −2%, 叠加后第三轮只 −0.5% ——
    #   第一口最大, 平均值会让配比在已经吃饱的族上多停两三轮)
    # 响应度按"净降"记: 本族降幅 − 全留出的升幅(11-06 早实撞: 65 轮 writing 喂 22 道自己 −0.5% 却把 code/math/svgchar 各拉回 1%, 全留出 +0.9%,
    #   只记本族降幅会把它当响应族再配 24 道; 一个族只有不拿别族垫背的降才算响应)
    resp = {}
    for f in FAMS:
        v = [-float(r[ix[f + "_d"]]) - max(0.0, (float(r[ix["kl1"]]) - float(r[ix["kl0"]])) / float(r[ix["kl0"]])) for r in L if int(r[ix[f + "_ex"]]) > 0]
        e = None
        for x in v: e = x if e is None else 0.5 * e + 0.5 * x
        resp[f] = e
    seen = [x for x in resp.values() if x is not None]; mean_r = (sum(seen) / len(seen)) if seen else 0.01
    w = {}
    for f in FAMS:
        r = resp[f]
        rr = max(r, 0.0) / max(mean_r, 1e-6) if r is not None else 1.0   # 没曝光过的族 = 1.0(先验 = 平均响应), 曝光过但不降 = 0 → 只留底额
        w[f] = gap[f] * (0.25 + rr)
    tot = sum(w.values()); q = {f: max(a.minq, min(a.maxq, round(a.budget * w[f] / tot))) for f in FAMS}
    # 配到预算
    d = a.budget - sum(q.values())
    order = sorted(FAMS, key=lambda f: -w[f])
    i = 0
    while d != 0 and i < 200:
        f = order[i % len(FAMS)]
        if d > 0 and q[f] < a.maxq: q[f] += 1; d -= 1
        elif d < 0 and q[f] > a.minq: q[f] -= 1; d += 1
        i += 1
    why = "; ".join(f"{f}: 缺口 {gap[f]:.4f} × 响应 {'无数据' if resp[f] is None else f'{100*resp[f]:+.1f}%'} → {q[f]} 道" for f in order)
    os.makedirs(GAP, exist_ok=True)
    with open(os.path.join(GAP, "next.json"), "w") as stream:
        json.dump({"from": last[0], "mapping": MAP_VERSION, "quotas": q, "gap": gap, "resp": resp, "why": why}, stream, ensure_ascii=False, indent=1)
    print(f"[下一轮配比] 按 {last[0]} 的缺口 × 响应度: " + ", ".join(f"{f}={q[f]}" for f in FAMS)); print("  " + why)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gap-dir", default=GAP); sp = ap.add_subparsers(dest="cmd", required=True)
    sg = sp.add_parser("segmap"); sg.add_argument("--tag", required=True); sg.add_argument("--dir")
    s = sp.add_parser("scan"); s.add_argument("--tag", required=True); s.add_argument("--dir", default=None); s.add_argument("--epoch", type=int); s.add_argument("--read-only", action="store_true")
    sp.add_parser("history")
    rb = sp.add_parser("rebuild"); rb.add_argument("--out", required=True)
    n = sp.add_parser("next"); n.add_argument("--budget", type=int, default=60); n.add_argument("--minq", type=int, default=4); n.add_argument("--maxq", type=int, default=24)
    a = ap.parse_args(); GAP = a.gap_dir
    try:
        rc = {"segmap": cmd_segmap, "scan": cmd_scan, "history": cmd_history, "rebuild": cmd_rebuild, "next": cmd_next}[a.cmd](a)
    except (OSError, ValueError, KeyError) as error:
        print(f"★{error}★", file=sys.stderr); rc = 1
    sys.exit(rc)
