#!/usr/bin/env python3
"""按来源和实际 token 数选完整的 FP 锚；不截断序列，避免教师表的键失配。

在仓库根运行：vloop_anchors.py --round 72 --budget 131072 --domain code
只打印选中的目录。--out 可把来源、位数和文件指纹写成清单供复查。
按新到旧选；下一份同域锚放不下就停止，宁可留余量，不为凑满去捡很旧的小锚。
来源不明的旧锚跳过并报出；本轮必需的锚缺失或来源不对则失败，不能悄悄用旧料顶替。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read_anchor(directory):
    seeds = directory.parent / "seeds.jsonl"
    sources = [json.loads(line).get("src", "") for line in seeds.read_text().splitlines() if line.strip()]
    if not sources:
        raise ValueError("seeds.jsonl 没有来源")
    kinds = {"code" if s.startswith("code_") else "text" if s.startswith("text_") else "unknown" for s in sources}
    if len(kinds) != 1 or "unknown" in kinds:
        raise ValueError("种子来源混合或不明")
    payload = (directory / "rows.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not rows or any(set(r) != {"ids"} or len(r["ids"]) < 2 for r in rows):
        raise ValueError("锚必须是非空 ids 序列")
    if any(type(i) is not int or i < 0 for r in rows for i in r["ids"]):
        raise ValueError("锚含无效 token id")
    if (directory / "teacher.ptf").stat().st_size == 0:
        raise ValueError("teacher.ptf 是空的")
    return {"path": str(directory), "domain": kinds.pop(), "tokens": sum(len(r["ids"]) for r in rows),
            "rows_sha256": hashlib.sha256(payload).hexdigest()}


def select_anchors(loop, through, budget, domain="code", require_current=False):
    if through < 1 or budget < 2:
        raise ValueError("轮号和 token 预算必须为正")
    candidates = sorted((p for p in Path(loop).glob("r[0-9][0-9][0-9]/fp")
                         if int(p.parent.name[1:]) <= through), reverse=True)
    current = Path(loop) / f"r{through:03d}" / "fp"
    if require_current and current not in candidates:
        raise ValueError(f"本轮锚缺失: {current}")
    selected, skipped, total = [], [], 0
    for directory in candidates:
        required = require_current and directory == current
        try:
            item = read_anchor(directory)
            if domain != "any" and item["domain"] != domain:
                raise ValueError(f"来源是 {item['domain']}，本轮只收 {domain}")
        except (OSError, ValueError, TypeError, KeyError) as error:
            if required:
                raise ValueError(f"本轮锚不能使用: {directory}: {error}") from error
            skipped.append((str(directory), str(error)))
            continue
        if item["tokens"] + total > budget:
            reason = f"加入后超过 {budget} token 预算，停止回溯旧锚"
            if required:
                raise ValueError(f"本轮锚不能使用: {directory}: {reason}")
            skipped.append((str(directory), reason))
            break
        selected.append(item)
        total += item["tokens"]
    if not selected:
        raise ValueError("没有来源明确且预算放得下的锚")
    return list(reversed(selected)), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", default="gguf/go-onebit/evo/vloop")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--domain", choices=("code", "text", "any"), default="code")
    ap.add_argument("--require-current", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    try:
        chosen, skipped = select_anchors(args.loop, args.round, args.budget, args.domain, args.require_current)
    except (OSError, ValueError) as error:
        print(f"★选锚失败: {error}★", file=sys.stderr)
        return 1
    if args.out:
        out = Path(args.out)
        out.write_text("path\tdomain\ttokens\trows_sha256\n" + "".join(
            "\t".join(str(x[k]) for k in ("path", "domain", "tokens", "rows_sha256")) + "\n" for x in chosen))
    for directory, reason in skipped:
        print(f"[选锚跳过] {directory}: {reason}", file=sys.stderr)
    print(f"[选锚] {len(chosen)} 份，共 {sum(x['tokens'] for x in chosen)}/{args.budget} token，来源 {args.domain}", file=sys.stderr)
    print("\n".join(x["path"] for x in chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
