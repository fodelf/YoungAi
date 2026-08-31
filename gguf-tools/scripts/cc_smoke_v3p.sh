#!/bin/bash
# cc_smoke_v3p.sh — Claude Code 真场景冒烟(v3p 栈, M4 本机, P0 真任务门):
# 夹具=Clamp bug + 失败测试(07-12 同规格, /tmp 被 TTL 清后固化于此)。
# 栈: v3p+残差 + base-native + --nothink + --tool-primer + soul_v3 + 自由区(默认24)。
# 门: CC 完成 读文件→改 Clamp bug→跑测试 回路; 原始 stream-json 落 /tmp/cc_v3p_smoke.out。
set -uo pipefail
cd "$(dirname "$0")/../.."
PORT=8023
FIX=/tmp/ccsmoke_v3p

# ── 夹具 ──
rm -rf "$FIX"; mkdir -p "$FIX"
cat > "$FIX/go.mod" <<'EOF'
module ccsmoke

go 1.21
EOF
cat > "$FIX/clamp.go" <<'EOF'
package ccsmoke

// Clamp returns v limited to the range [lo, hi].
func Clamp(v, lo, hi int) int {
	if v < lo {
		return hi
	}
	if v > hi {
		return hi
	}
	return v
}
EOF
cat > "$FIX/clamp_test.go" <<'EOF'
package ccsmoke

import "testing"

func TestClampBelow(t *testing.T) {
	if got := Clamp(-5, 0, 10); got != 0 {
		t.Fatalf("Clamp(-5,0,10) = %d, want 0", got)
	}
}
EOF

# ── server(看门狗内嵌) ──
pkill -9 -x ds4-server 2>/dev/null; sleep 1
rm -rf /tmp/ds4-kv-ccsmoke   # 陈旧跨会话检查点塞满盘帽=攻7死因(hits=0全量冷灌)
# 批量注入+小自由区(2026-07-16 凌晨实证: 26k ctx 下 9s/token, 引导轮15-17min 撞客户端
# 流静默超时, 三次死在+68token; 批注入=结构token一次forward, 07-14预留的本场景最大杠杆)
# (env 大扫除 2026-08-31: PRIMER_BATCH_INJECT/PRIMER_FREE_BUDGET 已写死进 server, env 无读取者)
  ./ds4-server -m gguf/go-onebit/ds4-code1b-v3p.gguf --port "$PORT" --ctx 65536 \
    --base-native --residual gguf/sidecars/code-hot-res-v3p.gguf \
    --mem-budget-mb 12000 --prefill-chunk 512 \
    --max-output-tokens 64 \
    --nothink --tool-primer --soul gguf-tools/data/corpus/soul/soul_server_v3.txt \
    --kv-disk-dir /tmp/ds4-kv-ccsmoke --kv-disk-space-mb 8192 \
    > /tmp/cc_v3p_srv.log 2>&1 &
SP=$!
( while kill -0 "$SP" 2>/dev/null; do
    MB=$(footprint -p "$SP" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3;if(u=="GB")v*=1024;else if(u=="KB")v/=1024;printf "%d",v}')
    [ -n "${MB:-}" ] && [ "$MB" -gt 11776 ] && { echo "[watchdog] server ${MB}MB → kill" >&2; kill -9 "$SP"; break; }
    sleep 3
  done ) & WD=$!
for _ in $(seq 1 150); do
    curl --noproxy '*' -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
    kill -0 "$SP" 2>/dev/null || { echo "[cc-smoke] server 未起(日志尾: $(tail -2 /tmp/cc_v3p_srv.log))" >&2; exit 2; }
    sleep 2
done
echo "[cc-smoke] server 就绪, 起 claude CLI(max-turns 6)" >&2

# ── Claude Code 回路(绕代理; 07-12 缺口②: MAX_THINKING_TOKENS=0) ──
cd "$FIX"
env ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" NO_PROXY='*' no_proxy='*' \
    MAX_THINKING_TOKENS=0 API_TIMEOUT_MS=1800000 DISABLE_NON_ESSENTIAL_MODEL_CALLS=1 \
  claude -p "Run 'go test ./...'. One test fails because of a bug in clamp.go. Read clamp.go, fix the bug with a minimal edit, then run the test again to prove it passes." \
    --output-format stream-json --verbose --max-turns 6 --dangerously-skip-permissions \
    > /tmp/cc_v3p_smoke.out 2>&1
RC=$?
echo "[cc-smoke] claude 退出 rc=$RC" >&2
kill -9 "$SP" "$WD" 2>/dev/null
echo "[cc-smoke] 完成 → /tmp/cc_v3p_smoke.out (server日志 /tmp/cc_v3p_srv.log)" >&2
