#!/usr/bin/env python3
# harvest_issue_fixes.py — 支柱1(Go 真项目问题→修复对): 高信号已关闭 issue +
# 关闭它的 commit/PR diff → "问题 → 讨论 → 修复 diff" 线程语料。这是 agent
# 修复行为的最高信号后训练材料(问题描述与真实修复 diff 成对出现)。
# 输出: raw/issuefix/{owner}__{repo}.issuefix.txt, 线程分隔符 "=== ISSUE-FIX"。
#   env: REPOS(逗号列表, 默认 Go top-10 精选), N_ISSUES=25, DIFF_KB=24,
#        BODY_KB=6, OUT=raw/issuefix
# 需要 gh auth(≈3 API call/issue → 10仓×25issue ≈ 800 call, 5000/h 限额内)。
import json, os, pathlib, subprocess

REPOS = [r for r in os.environ.get("REPOS", "").split(",") if r] or [
    "golang/go", "kubernetes/kubernetes", "ollama/ollama", "gin-gonic/gin",
    "gohugoio/hugo", "fatedier/frp", "syncthing/syncthing",
    "caddyserver/caddy", "prometheus/prometheus", "etcd-io/etcd"]
N = int(os.environ.get("N_ISSUES", "25"))
DIFF_CAP = int(os.environ.get("DIFF_KB", "24")) * 1024
BODY_CAP = int(os.environ.get("BODY_KB", "6")) * 1024
OUT = pathlib.Path(os.environ.get("OUT", "raw/issuefix")); OUT.mkdir(parents=True, exist_ok=True)

def api(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None

for repo in REPOS:
    shards = []
    # 选择器教训(2026-07-15 实测): /issues?sort=comments 选中的是巨型提案/梗帖
    # (golang/go top-1 = #9 玩笑帖 1095 评论), 全无修复 commit → 0 对。
    # search + linked:pr 才是"已被代码解决的 issue"(golang/go 1609 条, k8s 1.1万)。
    issues = (api(f"search/issues?q=repo:{repo}+is:issue+is:closed+linked:pr"
                  f"&sort=comments&order=desc&per_page={N}") or {}).get("items", [])
    for it in issues:
        n = it["number"]
        # 定位修复 diff: 优先 closed 事件带的 commit; 否则第一条已合并的
        # cross-referenced PR 的 merge commit。找不到 = 无代码修复, 跳过。
        sha, via = None, ""
        tl = api(f"repos/{repo}/issues/{n}/timeline?per_page=100") or []
        for ev in tl:
            if ev.get("event") == "closed" and ev.get("commit_id"):
                sha, via = ev["commit_id"], "closing commit"; break
        if not sha:
            for ev in tl:
                src = (ev.get("source") or {}).get("issue") or {}
                if ev.get("event") == "cross-referenced" and (src.get("pull_request") or {}).get("merged_at"):
                    prj = api(f"repos/{repo}/pulls/{src['number']}")
                    if prj and prj.get("merge_commit_sha"):
                        sha, via = prj["merge_commit_sha"], f"PR #{src['number']}"; break
        if not sha:
            continue
        commit = api(f"repos/{repo}/commits/{sha}")
        if not commit:
            continue
        diff, used = [], 0
        for f in commit.get("files", []):
            p = f.get("patch")
            if not p:
                continue
            block = f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{p}\n"
            if used + len(block) > DIFF_CAP:
                break
            diff.append(block); used += len(block)
        if not diff:
            continue
        cms = api(f"repos/{repo}/issues/{n}/comments?per_page=3") or []
        csec = "\n".join(f"[comment @{c['user']['login']}]\n{(c.get('body') or '')[:1500]}" for c in cms)
        shards.append(
            f"=== ISSUE-FIX {repo}#{n}: {it['title']}\n{(it.get('body') or '')[:BODY_CAP]}\n"
            f"--- DISCUSSION ---\n{csec}\n--- FIX DIFF ({via} {sha[:10]}) ---\n{''.join(diff)}\n")
        print(f"  {repo}#{n} ✓ ({via}, diff {used}B)", flush=True)
    out = OUT / (repo.replace("/", "__") + ".issuefix.txt")
    out.write_text("\n".join(shards))
    print(f"[{repo}] {len(shards)} issue-fix 对 -> {out}", flush=True)
