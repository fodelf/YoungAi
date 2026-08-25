#!/bin/bash
# fetch_official_vectors.sh — 官方 API logprob 向量抓取/规整/紧凑夹具生成。
# (bash+curl+jq+awk, 2026-08-25 Python→C/bash 迁移; 取代 fetch_official_vectors.py)
# 金标: ①prompts/*.txt 与 py 逐字节 ②official.vec 从既有 official/*.json 重建=与库内
#   committed 版逐字节 ③fetch/normalize 结构等价(jq -S, 罐头响应对拍) — 记录 JSON 的
#   浮点回显与缩进为 jq 形态(与 py json.dumps 结构同、字节可能异, 归档件非判决件)。
# 用法: fetch_official_vectors.sh [--out DIR] [--only ID]...   (需 DEEPSEEK_API_KEY)
#       fetch_official_vectors.sh --rebuild-vec [--out DIR]    (离线重建 official.vec)
set -uo pipefail
MODEL="deepseek-v4-flash"
ENDPOINT="https://api.deepseek.com/chat/completions"
TOP_LOGPROBS=20
MAX_TOKENS=4

ctx_by_id() {
  case "$1" in
    short_italian_fact|long_memory_archive|long_code_audit) echo 16384 ;;
    short_code_completion|short_reasoning_plain) echo 4096 ;;
    *) echo "unknown id $1" >&2; exit 1 ;;
  esac
}

long_memory_prompt() {
  printf '%s' "You are checking a long technical archive. Read the repeated records and answer only the final question with one short sentence."
  printf '\n\n'
  local i
  for i in $(seq 0 71); do
    printf 'Record %03d: the archive entry says that component alpha keeps a compressed index, component beta keeps raw observations, and component gamma reports anomalies only after the checksum phrase appears. Do not summarize yet; retain the exact final question.\n' "$i"
  done
  printf '\nFinal question: which component reports anomalies after the checksum phrase appears?'
}

long_code_prompt() {
  printf '%s' "Review this generated C-code audit log. After the log, complete the sentence with the most likely next words."
  printf '\n\n'
  local i
  for i in $(seq 0 67); do
    printf 'Function f_%d validates a queue entry, calls normalize_path(), then appends a compact audit line. The invariant is that strlen() must not be recomputed when a trusted length returned by snprintf() is already available. Security note %d: reject negative sizes before casting.\n' "$i" "$i"
  done
  printf '\nCompletion target: The most important code quality issue is'
}

prompt_text() {
  case "$1" in
    short_italian_fact) printf '%s' 'Rispondi in italiano con una frase: chi era Ada Lovelace?' ;;
    short_code_completion) printf 'Complete the C statement with the next exact token only:\nreturn snprintf(buf, sizeof(buf), "%%d", value' ;;
    short_reasoning_plain) printf '%s' 'Answer with only the number: 2048 divided by 128 is' ;;
    long_memory_archive) long_memory_prompt ;;
    long_code_audit) long_code_prompt ;;
  esac
}
prompt_kind() { case "$1" in short_*) echo short ;; long_*) echo long ;; esac; }
ALL_IDS="short_italian_fact short_code_completion short_reasoning_plain long_memory_archive long_code_audit"

# normalize: 罐头/真响应 → 记录 JSON。token_bytes 回退=utf8 字节(jq 内实现码点→utf8)。
NORMALIZE_JQ='
  def utf8bytes: [explode[] | if . < 0x80 then [.]
    elif . < 0x800 then [0xC0 + (. / 64 | floor), 0x80 + (. % 64)]
    elif . < 0x10000 then [0xE0 + (. / 4096 | floor), 0x80 + ((. / 64 | floor) % 64), 0x80 + (. % 64)]
    else [0xF0 + (. / 262144 | floor), 0x80 + ((. / 4096 | floor) % 64), 0x80 + ((. / 64 | floor) % 64), 0x80 + (. % 64)] end] | flatten;
  def tokbytes(t; b): if (b != null) then (b | map(floor)) else (t // "" | utf8bytes) end;
  .choices[0] as $c |
  {
    schema: "ds4-official-logprobs-v1",
    source: "deepseek-official-api",
    model: $model, endpoint: $endpoint, created_at: $now,
    id: $pid, kind: $pkind, prompt: $prompt,
    request: {model: $model, temperature: 0, max_tokens: ($maxtok|tonumber),
              logprobs: true, top_logprobs: ($toplp|tonumber),
              thinking: {type: "disabled"},
              messages: [{role: "user", content: $prompt}]},
    usage: .usage, finish_reason: $c.finish_reason, message: ($c.message // {}),
    logits_available: false,
    steps: [ ($c.logprobs.content // [])[] as $item | {
        step: 0,
        token: {text: ($item.token // ""), bytes: tokbytes($item.token; $item.bytes)},
        logprob: $item.logprob,
        top_logprobs: [ ($item.top_logprobs // [])[] | {
            token: {text: (.token // ""), bytes: tokbytes(.token; .bytes)},
            logprob: .logprob } ]
    } ] | to_entries | map(.value + {step: .key})
  }'

rebuild_vec() {   # 从 manifest+official/*.json 重建 official.vec(与 py write_compact_fixture 同式)
  local root="$1" tmp
  tmp=$(mktemp)
  {
    printf '# ds4-official-logprob-vectors-v1\n'
    printf '# case <id> <ctx> <steps> <prompt-file>\n'
    printf '# step <index> <selected-hex> <top-count>\n'
    printf '# top <token-hex> <official-logprob>\n\n'
    jq -r '.prompts[] | [.id, .official_file, .prompt_file] | @tsv' "$root/manifest.json" | \
    while IFS=$'\t' read -r vid ofile pfile; do
      ctx=$(ctx_by_id "$vid")
      # 每步: selected-hex 全 top; top 行滤 lp<=-1000 与空 hex; %.9g 用 awk 复刻 py 的 :.9g
      jq -r --arg vid "$vid" --arg ctx "$ctx" --arg pfile "$root/$pfile" '
        def hexb: map(. as $b | "0123456789abcdef"[$b/16|floor:($b/16|floor)+1] + "0123456789abcdef"[$b%16:($b%16)+1]) | join("");
        "case \($vid) \($ctx) \(.steps|length) \($pfile)",
        (.steps[] |
          ([.top_logprobs[] | select((.logprob // -9999) > -1000) | select((.token.bytes|length) > 0)
              | "top \(.token.bytes|hexb) \(.logprob)"]) as $tops |
          ("step \(.step) \(.token.bytes|hexb) \($tops|length)", $tops[])),
        "end", ""
      ' "$root/$ofile" | awk '{ if ($1=="top") { printf "top %s %.9g\n", $2, $3 } else print }'
    done
  } > "$tmp"
  # py 用 "\n".join(lines) — 末尾无换行; 上面每 case 尾多一空行, 需去掉最后一个换行
  perl -0pi -e 's/\n\z//' "$tmp"
  mv "$tmp" "$root/official.vec"
}

main() {
  local out="tests/test-vectors" only=() rebuild=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --out) out="$2"; shift 2 ;;
      --only) only+=("$2"); shift 2 ;;
      --rebuild-vec) rebuild=1; shift ;;
      *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
  done
  if [ "$rebuild" = 1 ]; then rebuild_vec "$out"; echo "rebuilt $out/official.vec"; exit 0; fi
  [ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 2; }
  mkdir -p "$out/prompts" "$out/official"
  local mprompts="[]" id
  for id in $ALL_IDS; do
    if [ ${#only[@]} -gt 0 ]; then
      local hit=0 w; for w in "${only[@]}"; do [ "$w" = "$id" ] && hit=1; done
      [ $hit = 1 ] || continue
    fi
    prompt_text "$id" > "$out/prompts/$id.txt"
    local prompt payload resp now
    prompt=$(cat "$out/prompts/$id.txt")
    payload=$(jq -n --arg model "$MODEL" --arg p "$prompt" --argjson mt "$MAX_TOKENS" --argjson tl "$TOP_LOGPROBS" \
      '{model:$model,temperature:0,max_tokens:$mt,logprobs:true,top_logprobs:$tl,thinking:{type:"disabled"},messages:[{role:"user",content:$p}]}')
    resp=$(curl -sf --max-time 120 "$ENDPOINT" -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
      -H 'Content-Type: application/json' -d "$payload") || { echo "DeepSeek API 请求失败 ($id)" >&2; exit 1; }
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s' "$resp" | jq --arg model "$MODEL" --arg endpoint "$ENDPOINT" --arg now "$now" \
      --arg pid "$id" --arg pkind "$(prompt_kind "$id")" --arg prompt "$prompt" \
      --arg maxtok "$MAX_TOKENS" --arg toplp "$TOP_LOGPROBS" "$NORMALIZE_JQ" \
      > "$out/official/$id.official.json"
    mprompts=$(jq -n --argjson acc "$mprompts" --arg id "$id" --arg kind "$(prompt_kind "$id")" \
      --arg pf "prompts/$id.txt" --arg of "official/$id.official.json" \
      --argjson pc "$(printf '%s' "$prompt" | wc -c | tr -d ' ')" \
      --argjson st "$(jq '.steps|length' "$out/official/$id.official.json")" \
      '$acc + [{id:$id,kind:$kind,prompt_file:$pf,official_file:$of,prompt_chars:$pc,steps:$st}]')
    echo "wrote $out/official/$id.official.json"
  done
  jq -n --arg model "$MODEL" --arg endpoint "$ENDPOINT" --argjson tl "$TOP_LOGPROBS" \
    --argjson mt "$MAX_TOKENS" --argjson prompts "$mprompts" \
    '{schema:"ds4-test-vector-manifest-v1",source:"deepseek-official-api",model:$model,endpoint:$endpoint,top_logprobs:$tl,max_tokens:$mt,prompts:$prompts}' \
    > "$out/manifest.json"
  [ ${#only[@]} -gt 0 ] || rebuild_vec "$out"
}
main "$@"
