#!/bin/bash
# harvest_prog.sh — 编程全域语料采集(2026-07-20 域放大第二刀, stage 1+2 驱动)。
# 钉死仓库列表(确定性, 不走搜索 API 方差): 每语言 2 个标杆真项目, depth-1 克隆抽文本
# shard 后即删克隆(盘账: 峰值≈单仓库<80MB, M4 剩 8G 安全)。issues 抓 top-20 高信号线程。
# 产物: corpus/raw/<lang>/ 文本 shard + corpus/build/<lang>_{simple,complex,mixed}.txt
# 用法: [ONLY="python rust"] ./harvest_prog.sh   # ONLY 指定子集, 缺省全 6 语言
# ⚠ macOS 系统 bash 3.2: 禁 declare -A/嵌套转义(2026-07-18 看门狗铁律), 用 case 表。
set -uo pipefail
# ⚠ harvest_repos/corpus_build 的 OUT 默认是 corpus/raw、corpus/build 相对路径 →
# 工作目录必须是 go-onebit/(不是 corpus/, 2026-07-20 首跑嵌套 corpus/corpus 教训)。
cd "$(dirname "$0")/.."           # → go-onebit/

repos_for() {
  case "$1" in
    python)     echo "pallets/flask,tiangolo/fastapi,TheAlgorithms/Python" ;;
    javascript) echo "expressjs/express,axios/axios,TheAlgorithms/JavaScript" ;;
    typescript) echo "honojs/hono,vuejs/core" ;;
    rust)       echo "serde-rs/serde,BurntSushi/ripgrep,TheAlgorithms/Rust" ;;
    c)          echo "memcached/memcached,jqlang/jq" ;;
    java)       echo "google/gson,square/okhttp" ;;
    go)         echo "TheAlgorithms/Go" ;;
    shell)      echo "dylanaraps/pure-bash-bible,acmesh-official/acme.sh" ;;
    *)          echo "" ;;
  esac
}

SEL="${ONLY:-python javascript typescript rust c java go shell}"; SEL="${SEL//,/ }"
for L in $SEL; do
  PIN="$(repos_for "$L")"
  [ -n "$PIN" ] || { echo "[harvest-prog] 未知语言 $L" >&2; exit 2; }
  echo "[harvest-prog] ==== $L ($PIN) ====" >&2
  KEYWORD="$L" LANG_Q="$L" REPOS="$PIN" ISSUES_PER_REPO=20 \
    python3 corpus/harvest_repos.py || { echo "[harvest-prog] $L 采集失败" >&2; exit 1; }
  KEYWORD="$L" python3 corpus/corpus_build.py \
    || { echo "[harvest-prog] $L 构建失败" >&2; exit 1; }
done
echo "[harvest-prog] 全部完成; 产物:" >&2
wc -c corpus/build/*_mixed.txt 2>/dev/null | tail -8 >&2
