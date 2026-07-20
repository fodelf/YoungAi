#!/usr/bin/env python3
# harvest_skills.py — 支柱3(工程化): top-star agent-skill 仓库(superpowers 级)
# 的技能定义文档(SKILL.md / skills/**.md)→ corpus/skills/ 平文本分片(books/ 同形,
# 每源文件一个分片, 供 REF_CORPUS/校准蒸馏消费)。
#   env: SKILL_REPOS(逗号列表覆盖默认), SEARCH_N=6(gh 搜索补充数),
#        OUT=skills, WORK=raw/skills
import json, os, pathlib, shutil, subprocess

CURATED = ["obra/superpowers", "anthropics/skills"]
SEARCH_N = int(os.environ.get("SEARCH_N", "6"))
OUT = pathlib.Path(os.environ.get("OUT", "skills")); OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path(os.environ.get("WORK", "raw/skills")); WORK.mkdir(parents=True, exist_ok=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600).stdout

repos = [r for r in os.environ.get("SKILL_REPOS", "").split(",") if r] or list(CURATED)
if not os.environ.get("SKILL_REPOS") and shutil.which("gh"):
    found = json.loads(sh(f'gh search repos "claude code skills" --sort stars --order desc '
                          f'--limit {SEARCH_N} --json fullName') or "[]")
    for it in found:
        if it["fullName"] not in repos:
            repos.append(it["fullName"])
print(f"[skills] {len(repos)} repos: {', '.join(repos)}", flush=True)

n_shards = 0
for name in repos:
    slug = name.replace("/", "__"); dst = WORK / slug
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{name}.git", str(dst)],
                       capture_output=True, timeout=900)
    if not dst.exists():
        print(f"  clone 失败: {name}", flush=True); continue
    kept = 0
    for p in sorted(dst.rglob("*.md")):
        rel = p.relative_to(dst)
        try:                                  # 悬空 symlink 会让 stat 抛错(claude-skills 实例)
            if ".git" in rel.parts or p.stat().st_size > 200_000:
                continue
        except OSError:
            continue
        text = p.read_text(errors="ignore")
        if len(text.strip()) < 200:                 # 空壳/占位文档不进语料
            continue
        (OUT / f"{slug}__{'_'.join(rel.parts)}").write_text(
            f"// SKILL DOC: {name}/{rel}\n" + text)
        kept += 1; n_shards += 1
    print(f"  {name}: {kept} docs", flush=True)
print(f"[skills] 共 {n_shards} 分片 -> {OUT}/", flush=True)
