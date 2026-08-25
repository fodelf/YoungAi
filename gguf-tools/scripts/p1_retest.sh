#!/bin/bash
# p1_retest.sh — P1 修正重测(消除两个混淆: 上下文域距 + 注释头分支):
# 相邻域真语料(awesome-go CI 的 map 去重惯用法, 与 twoSum seen-map 同构) +
# 无注释头裸签名探针(裸基线=正确暴力解)。判决: 续出 map 解=升级 / 正确循环=中性 / 崩=退化。
set -euo pipefail
cd "$(dirname "$0")/.."
CTX=$(cat <<'EOF'
func TestDuplicatedLinks(t *testing.T) {
	doc := goqueryFromReadme(t)
	links := make(map[string]bool, 0)
	doc.Find("body li > a:first-child").Each(func(_ int, s *goquery.Selection) {
		t.Run(s.Text(), func(t *testing.T) {
			href, ok := s.Attr("href")
			if !ok {
				t.Error("expected to have href")
			}
			if links[href] {
				t.Fatalf("duplicated link '%s'", href)
			}
			links[href] = true
		})
	})
}

EOF
)
DS4_RESIDUAL="${RESID:-gguf/sidecars/code-hot-res-v2.gguf}" \
PROMPT="<｜begin▁of▁sentence｜>${CTX}func twoSum(nums []int, target int) []int {" \
MODEL="${MODEL:-gguf/go-onebit/ds4-code1b-v2.gguf}" NPRED="${NPRED:-28}" TIMEOUT_S=420 \
  ./scripts/code1b_smoke.sh
