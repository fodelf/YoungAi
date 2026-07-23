#!/usr/bin/env python3
# harvest_repos.py — corpus stage 1: for each keyword, pull the top-N starred
# repos (source files) + their highest-signal issues via the gh CLI, emit
# cleaned text shards under CORPUS_OUT/raw/. All counts configurable
# (pipeline.conf via env). Cluster note: harvesting is network-bound — run on
# one host; the forward/solve stages that consume the corpus use the dual-host
# pipeline as usual.
#
#   env: KEYWORD, LANG=go, TOP_REPOS=10, ISSUES_PER_REPO=50, MAX_REPO_MB=80, OUT=corpus/raw
#   requires: gh auth login done once; git.
# 2026-07-20 域放大: LANG env 泛化搜索语言(缺省 go 保持旧行为); 每语言默认扩展名表。
import json, os, subprocess, sys, pathlib, shutil

KW = os.environ.get("KEYWORD", "go")
LANG = os.environ.get("LANG_Q", os.environ.get("LANG_SEARCH", "go"))
TOP = int(os.environ.get("TOP_REPOS", "10"))
NISS = int(os.environ.get("ISSUES_PER_REPO", "50"))
MAXMB = int(os.environ.get("MAX_REPO_MB", "80"))
OUT = pathlib.Path(os.environ.get("OUT", "corpus/raw")) / KW
OUT.mkdir(parents=True, exist_ok=True)

LANG_EXTS = {  # 每语言默认源扩展名(md 一律带上: README/设计文档是散文支柱原料)
    "go": {".go", ".md", ".mod", ".sum"},
    "python": {".py", ".md", ".toml"},
    "javascript": {".js", ".mjs", ".cjs", ".md", ".json"},
    "typescript": {".ts", ".tsx", ".md", ".json"},
    "rust": {".rs", ".md", ".toml"},
    "c": {".c", ".h", ".md"},
    "java": {".java", ".md", ".gradle"},
    "shell": {".sh", ".bash", ".md"},
}
CODE_EXT = (set(os.environ["EXTS"].split(",")) if os.environ.get("EXTS")
            else {".go", ".md", ".mod", ".sum"} if KW in ("go", "gin", "golang")
            else LANG_EXTS.get(LANG))

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600).stdout

# gh CLI when authenticated; otherwise anonymous GitHub REST (60 req/h — this
# stage needs ~1+TOP requests) + anonymous `git clone` for public repos. The
# anon path skips per-issue comment threads to stay inside the rate budget.
HAVE_GH = shutil.which("gh") is not None

# ---- top-N starred repos for the keyword -----------------------------------
# REPOS=owner/name,owner/name 精确仓库列表覆盖搜索(支柱2 定点采集用)。
q = (f"language:{LANG} stars:>10000" if KW in ("go", "golang") or KW == LANG
     else f"language:{LANG} {KW}")
if os.environ.get("REPOS"):
    repos = [{"fullName": r, "stargazersCount": 0, "size": 0}
             for r in os.environ["REPOS"].split(",") if r]
elif HAVE_GH:
    repos = json.loads(sh(
        f"gh search repos {json.dumps(q)} --sort stars --order desc --limit {TOP} "
        f"--json fullName,stargazersCount,size") or "[]")
else:
    import urllib.parse, urllib.request
    u = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) +
         f"&sort=stars&order=desc&per_page={TOP}")
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "ds4-corpus"}), timeout=60) as resp:
        repos = [{"fullName": it["full_name"], "stargazersCount": it["stargazers_count"],
                  "size": it["size"]} for it in json.load(resp).get("items", [])]
print(f"[{KW}] {len(repos)} repos (gh={'yes' if HAVE_GH else 'anon-rest'})", flush=True)

for r in repos:
    name = r["fullName"]; slug = name.replace("/", "__")
    if r.get("size", 0) > MAXMB * 1024:
        print(f"  skip {name} (> {MAXMB}MB)"); continue
    dst = OUT / slug
    if not dst.exists():
        if HAVE_GH:
            subprocess.run(["gh", "repo", "clone", name, str(dst), "--", "--depth", "1"],
                           capture_output=True, timeout=900)
        else:
            subprocess.run(["git", "clone", "--depth", "1",
                            f"https://github.com/{name}.git", str(dst)],
                           capture_output=True, timeout=900)
    # ---- source text shard ----
    with open(OUT / f"{slug}.code.txt", "w") as f:
        for p in sorted(dst.rglob("*")):
            if p.is_file() and (CODE_EXT is None or p.suffix in CODE_EXT) \
               and p.stat().st_size < 200_000 and ".git" not in p.parts:
                try:
                    f.write(f"\n// ==== {p.relative_to(dst)} ====\n" + p.read_text(errors="ignore"))
                except Exception:
                    pass
    # ---- issues shard (top by comments: highest-signal threads) ----
    if HAVE_GH:
        iss = json.loads(sh(
            f"gh issue list -R {name} --state all --limit {NISS} "
            f"--json number,title,body,comments") or "[]")
    else:
        import urllib.request
        u = (f"https://api.github.com/repos/{name}/issues?state=all&sort=comments"
             f"&direction=desc&per_page={min(NISS, 100)}")
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "ds4-corpus"}), timeout=60) as resp:
                iss = [{"number": it["number"], "title": it["title"],
                        "body": it.get("body"), "comments": []}
                       for it in json.load(resp) if "pull_request" not in it]
        except Exception as e:
            print(f"  issues fetch failed for {name}: {e}", flush=True); iss = []
    with open(OUT / f"{slug}.issues.txt", "w") as f:
        for it in iss:
            f.write(f"\n=== ISSUE #{it['number']}: {it['title']} ===\n{(it.get('body') or '')[:4000]}\n")
            for c in (it.get("comments") or [])[:5]:
                f.write(f"--- comment ---\n{(c.get('body') or '')[:2000]}\n")
    shutil.rmtree(dst, ignore_errors=True)   # keep text shards, drop working clone
    print(f"  {name}: code+{len(iss)} issues", flush=True)
print("HARVEST DONE", flush=True)
