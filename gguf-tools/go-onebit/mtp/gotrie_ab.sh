#!/bin/sh
# gotrie_ab.sh — Go trie 投机 drafter 的配对 A/B 判决脚本 (铁律: 配对+长样本).
#
# 交替跑 裸(n-gram copy-spec 基线) vs +go-trie, 每 prompt 两轮, 同缓存态; 报
# 每趟 gen t/s + trie summary + 逐位一致性 (greedy 无损的硬门). 短样本 t/s 是
# 页缓存噪声, 日间绝对基线漂移大 (实测 0.43→0.23), 只有轮内配对差有效.
#
# 用法:
#   ./gotrie_ab.sh                    # 默认: go1b+corr, 两个内置 Go prompt, -n 96
#   N=192 ./gotrie_ab.sh              # 更长样本
#   TRIE=/path/go_trie.bin ./gotrie_ab.sh
#   PROMPTS="/path/a.txt /path/b.txt" ./gotrie_ab.sh   # 自定 prompt(须含 BOS 前缀)
#
# trie 构建: /tmp/go_venv/bin/python ../pyfwd/build_go_trie.py \
#              --corpus ../gocorpus_big.txt --out $ROOT/gguf/go_trie.bin --stats
set -e
ROOT="${ROOT:-/Users/fodelf/git/ds4-main}"
M="${MODEL:-$ROOT/gguf/ds4-go1b.gguf}"
C="${CORR:-$ROOT/gguf/ds4-go1b-corr-rrr-partial.gguf}"
TRIE="${TRIE:-$ROOT/gguf/go_trie.bin}"
N="${N:-96}"
OUT="${OUT:-/tmp/gotrie_ab}"
mkdir -p "$OUT"
BOS='<｜begin▁of▁sentence｜>'

if [ -z "$PROMPTS" ]; then
  printf '%spackage main\n\nimport (\n\t"bufio"\n\t"fmt"\n\t"os"\n)\n\n// CountLines returns the number of lines in the file at path.\nfunc CountLines(path string) (int, error) {\n\tf, err := os.Open(path)\n' "$BOS" > "$OUT/p1.txt"
  printf '%spackage main\n\nimport (\n\t"encoding/json"\n\t"net/http"\n)\n\ntype User struct {\n\tID   int    `json:"id"`\n\tName string `json:"name"`\n}\n\nfunc handleUser(w http.ResponseWriter, r *http.Request) {\n\tvar u User\n\tif err := json.NewDecoder(r.Body).Decode(&u); err != nil {\n' "$BOS" > "$OUT/p2.txt"
  PROMPTS="$OUT/p1.txt $OUT/p2.txt"
fi

run() { # $1=tag $2=promptfile $3=trie(0/1)
  if [ "$3" = 1 ]; then TF="--go-trie $TRIE"; LOG=1; else TF=""; LOG=""; fi
  ( cd "$ROOT" && DS4_GO_TRIE_LOG=$LOG DS4_COPY_SPEC_LOG=$LOG \
      ./ds4 -m "$M" --corr "$C" $TF --prompt-file "$2" -n "$N" --temp 0 --metal \
      > "$OUT/$1.out" 2> "$OUT/$1.err" )
  printf '%-24s ' "$1"
  grep -oE "generation: [0-9.]+ t/s" "$OUT/$1.err" || echo "(no gen line)"
  grep "go-trie summary" "$OUT/$1.err" || true
}

pi=0
for P in $PROMPTS; do
  pi=$((pi+1))
  for r in 1 2; do
    run "p${pi}_bare_r${r}" "$P" 0
    run "p${pi}_trie_r${r}" "$P" 1
  done
done

echo "=== byte-exactness (greedy 无损硬门) ==="
pi=0
for P in $PROMPTS; do
  pi=$((pi+1))
  for r in 1 2; do
    if cmp -s "$OUT/p${pi}_bare_r${r}.out" "$OUT/p${pi}_trie_r${r}.out"; then
      echo "p${pi} r${r}: PASS"
    else
      echo "p${pi} r${r}: FAIL"
      diff "$OUT/p${pi}_bare_r${r}.out" "$OUT/p${pi}_trie_r${r}.out" | head -10
    fi
  done
done
echo "logs in $OUT/"
