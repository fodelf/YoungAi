#!/usr/bin/env bash
# 第五十五波 (路 A): 从官方 DeepSeek-V4-Flash HF 源量化一个更小的 MTP 草稿模型。
#
# 背景: 已发布的 MTP 草稿是 Q4K 3.8GiB, 在 M1 worker (10.67GiB GPU 预算) 上:
#   - wire 进 residency → OOM;  - 不 wire (DS4_MTP_NO_RESIDENCY=1) → embed/unembed 每步冷读 → decode 0.77。
# 草稿 3.8GiB 两头堵。把它量化到 Q2_K (~2.2GiB) 就能 wired 装下 → 驻留 → 快 → MTP 净正 (尤其 smoke 非回显)。
# HF 仓库无更小变体可下, 故从官方 safetensors 源 (gated) 重量化。草稿被 verify 纠错, 低精度只降接受率不破正确性。
#
# 用法:  HF_TOKEN=hf_xxx tools/make_small_mtp.sh
#        (或 --token hf_xxx)。只下载含 MTP 张量的 shard, 不下整模型。
#
# 产物:  gguf/DeepSeek-V4-Flash-MTP-Q2K.gguf  (mtp_pipe 脚本的 MTP_GGUF 默认会优先用它)
set -uo pipefail

SRC_REPO=${SRC_REPO:-deepseek-ai/DeepSeek-V4-Flash}   # 官方 safetensors 源 (gated)
TEMPLATE=${TEMPLATE:-gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}  # 现有 Q4K MTP 作 metadata/张量序模板
OUT=${OUT:-gguf/DeepSeek-V4-Flash-MTP-Q2K.gguf}
MTP_TYPE=${MTP_TYPE:-q2_k}                            # 目标精度 (q2_k 最稳; iq2_xxs 更小但搜索慢)
MTP_FILTER=${MTP_FILTER:-mtp}                         # 张量名筛子 (官方命名若不同改这里, 如 "nextn"/"eh_proj")
WORK=${WORK:-/tmp/ds4_mtp_src}                        # 临时 HF 源目录 (只放 MTP shard + 裁剪 index)
TOKEN=${HF_TOKEN:-}
[ "${1:-}" = "--token" ] && TOKEN=${2:-}

BASE="https://huggingface.co/$SRC_REPO/resolve/main"
log(){ printf '[make-small-mtp] %s\n' "$*"; }
die(){ printf '[make-small-mtp] ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "$TOKEN" ] || die "需要 HF token (源 gated): HF_TOKEN=hf_xxx tools/make_small_mtp.sh"
[ -f "$TEMPLATE" ] || die "缺模板 $TEMPLATE (现有 Q4K MTP)。先 ./download_model.sh mtp"
[ -x gguf-tools/deepseek4-quantize ] || { log "构建量化器…"; make -C gguf-tools deepseek4-quantize || die "量化器构建失败"; }
command -v curl >/dev/null || die "需要 curl"
command -v python3 >/dev/null || die "需要 python3"

mkdir -p "$WORK"
AUTH=(-H "Authorization: Bearer $TOKEN")

# 1) 下 index.json (小), 筛出 MTP 张量所在的最小 shard 集, 写裁剪后的 index (只含 MTP 张量)。
log "下载 index.json …"
http=$(curl -sSL --max-time 120 -w '%{http_code}' "${AUTH[@]}" \
       "$BASE/model.safetensors.index.json" -o "$WORK/_full_index.json" 2>/dev/null)
log "HTTP $http, 下载 $(wc -c < "$WORK/_full_index.json" 2>/dev/null) 字节"
# gated 源即使认证失败也可能回 HTTP 200 + HTML 登录/条款页 → 校验确实是 JSON。
if [ "$http" != 200 ] || ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$WORK/_full_index.json" 2>/dev/null; then
  log "index.json 不是有效 JSON。返回内容前 500 字节:"
  head -c 500 "$WORK/_full_index.json" 2>/dev/null; echo
  die "源未取到 (HTTP $http)。检查: ① token 有效且已 huggingface.co/$SRC_REPO 点 Agree 接受条款; \
② SRC_REPO 仓库名对 (现=$SRC_REPO, 可设 SRC_REPO=deepseek-ai/DeepSeek-V4-Flash-Base 试); ③ 网络可达 HF。"
fi

# 把过滤脚本写到临时文件再跑 (避免 heredoc 嵌在 <(...) 进程替换里, bash 解析不了 → "no closing )")。
cat > "$WORK/_filter_mtp.py" <<'PY'
import sys, json
full, trimmed, filt = sys.argv[1], sys.argv[2], sys.argv[3].lower()
d = json.load(open(full))
wm = d.get("weight_map", {})
mtp = {k: v for k, v in wm.items() if filt in k.lower()}
if not mtp:
    sys.stderr.write("筛子 '%s' 没匹配到张量; 样例张量名:\n" % filt)
    for k in list(wm)[:20]:
        sys.stderr.write("  " + k + "\n")
    sys.exit(3)
shards = sorted(set(mtp.values()))
out = {"metadata": d.get("metadata", {}), "weight_map": mtp}
json.dump(out, open(trimmed, "w"))
for s in shards:
    print(s)
sys.stderr.write("MTP 张量 %d 个, 分布 %d 个 shard\n" % (len(mtp), len(shards)))
PY
shards_out=$(python3 "$WORK/_filter_mtp.py" "$WORK/_full_index.json" "$WORK/model.safetensors.index.json" "$MTP_FILTER") \
  || die "index 解析/筛选失败 (可能官方命名不是 '$MTP_FILTER'; 看上面样例张量名, 设 MTP_FILTER=... 重试)"
# macOS 自带 bash 3.2 无 mapfile/readarray; 用 while-read 兼容收集 (set -u 下先声明数组)。
SHARDS=()
while IFS= read -r _line; do
  [ -n "$_line" ] && SHARDS+=("$_line")
done <<< "$shards_out"

[ "${#SHARDS[@]}" -gt 0 ] || die "未定位到 MTP shard"
log "需下载 ${#SHARDS[@]} 个 shard: ${SHARDS[*]}"

# config.json 等小文件 (量化器可能要读 shape/dtype)
for f in config.json; do
  curl -fsSL --max-time 60 "${AUTH[@]}" "$BASE/$f" -o "$WORK/$f" 2>/dev/null || true
done

# 2) 只下 MTP shard。
# 完整性校验直接读 safetensors 头部 (8B 长度 + JSON header), 算出数据应到的字节数和文件实际大小比对。
# 不用 HTTP HEAD 的 Content-Length: HF resolve URL 会 302 重定向到 CDN, HEAD 返回的是重定向页大小
# (实测 1058 字节) 不是真文件大小 → 永远判"不完整"。本地头校验既不依赖网络头, 又真验证文件可用。
# 之前 `[ -f ]` 判存在即跳, 上次 curl 被打断 (truncated, 少 ~1GiB) 重跑永不补全 → 量化读 expert 撞 EOF。
# 只在文件缺失/残缺时才 curl -C - (断点续传), 完整文件不碰 (避免对全文件 -C - 触发 416)。
st_complete(){
  python3 - "$1" <<'PY'
import json, struct, os, sys
try:
    f = sys.argv[1]; sz = os.path.getsize(f)
    with open(f, 'rb') as fp:
        hlen, = struct.unpack('<Q', fp.read(8)); hdr = json.loads(fp.read(hlen))
    need = 8 + hlen + max(v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__')
    sys.exit(0 if sz == need else 1)
except Exception:
    sys.exit(1)
PY
}
for s in "${SHARDS[@]}"; do
  local=$(stat -f %z "$WORK/$s" 2>/dev/null || stat -c %s "$WORK/$s" 2>/dev/null || echo 0)
  if [ -f "$WORK/$s" ] && st_complete "$WORK/$s"; then
    log "已存在且完整 $s ($local 字节), 跳过"; continue
  fi
  log "$([ "${local:-0}" -gt 0 ] && echo 续传 || echo 下载) shard $s (本地 ${local:-0} 字节) …"
  curl -fL -C - --max-time 3600 "${AUTH[@]}" "$BASE/$s" -o "$WORK/$s" || die "shard $s 下载失败"
  st_complete "$WORK/$s" || die "shard $s 仍不完整 (safetensors 头声明的数据量 > 文件大小); 重跑本脚本继续续传"
done

# 3) 量化: 模板给张量序/shape, 权重从 MTP shard 读。
#    *** 只量化 routed experts (ffn_*_exps); backbone 一律保持模板类型。***
#    ds4 的 MTP validate (mtp_weights_validate_layout in ds4.c) 对 backbone 张量*硬编码*类型:
#      - attention (attn_*) / e_proj / h_proj / shared experts (ffn_*_shexp) 必须 Q8_0;
#      - 所有 norm/scale/base/sinks 必须 F32;
#      - hc_{head,attn,ffn}_fn 与 router gate ffn_gate_inp 必须 plain F16/F32。
#    把它们压成 q2_k → worker 加载时 mtp_weights_validate_layout 直接 exit(1):
#      "tensor mtp.0.<name> has type q2_k, expected F16 or F32 / Q8_0"。
#    故草稿*唯一*能压的是占大头的 routed experts (256 个); 其余保持模板
#    (已发布 Q4K-Q8_0-F32, 类型已合规)。实测产物 2.14 GiB, 仍可 wired 装进 worker。
#    第五十七波教训: 早期此处还传 --attention/--attention-proj/--shared/--dense/--embedding/--output
#    $MTP_TYPE → 整模型压 q2_k → 双机 worker 起不来 (卡"等就绪", 日志报 hc_head_fn 类型错)。
log "量化 → $OUT (routed experts → $MTP_TYPE, backbone 保持模板 Q8_0/F32/F16) …"
gguf-tools/deepseek4-quantize \
  --hf "$WORK" \
  --template "$TEMPLATE" \
  --out "$OUT" \
  --overwrite \
  --experts "$MTP_TYPE" \
  || die "量化失败 (若报张量名不匹配, 官方 MTP 命名与模板不同, 需调 --tensor-type 映射)"

SZ=$(stat -f %z "$OUT" 2>/dev/null || stat -c %s "$OUT" 2>/dev/null)
# 单独一行算 GiB, 不把 awk 嵌进 log 的 "...$(...)..." 里: 嵌套双引号 (awk 的 \" 套在 $() 套在 log 的 "")
# 会被 bash 误解析, awk 拿到残缺程序转而读 stdin(tty) 永久阻塞。-v 传值 + </dev/null 双保险。
GIB=$(awk -v b="$SZ" 'BEGIN{printf "%.2f", b/1073741824}' </dev/null)
log "完成: $OUT  ($GIB GiB)"
log "下一步: MTP_GGUF=$OUT NO_MTP=0 COPY_SPEC=0 PROMPT_PROFILE=smoke tools/mtp_pipe_q2_speed.sh"
log "(脚本 MTP_GGUF 默认已优先用此 Q2 版, 若存在)"
