#!/bin/bash
# g8_genprobe.sh — 生成侧验证器(2026-07-29 用户"先验证"): 回放态自回归贪心, 词汤/复读直检。
# 用法(M1): ./g8_genprobe.sh prompt | fp [N] | mix [N] | decode "<ids>"
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
QDIR="$ROOT/gguf-tools/go-onebit/quant"; G8="$ROOT/gguf/go-onebit/g8"; mkdir -p "$G8"
case "${1:-}" in
prompt)
python3 - <<'PY'
from tokenizers import Tokenizer
tok=Tokenizer.from_file("/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base/tokenizer.json")
code='''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if any two numbers in the list are closer than threshold.
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
'''
ids=tok.encode(code,add_special_tokens=False).ids
open("/Users/fodelf/ds4-main/gguf/go-onebit/g8/prompt_py.ids","w").write("".join(f"{i}\n" for i in [0]+ids))
print("prompt tok=",len(ids)+1)
PY
;;
fp|mix)
N="${2:-24}"; M="$1"
cd "$QDIR"
if [ "$M" = mix ]; then
  DS4_GENPROBE=$N DS4_THREADS=6 DS4_LAYER_DIR="$ROOT/gguf/go-onebit/g7/out/layers" \
    ./ds4quant_run "$G8/prompt_py.ids" "$(grep -c . "$G8/prompt_py.ids")" 2>"$G8/${M}_run.log" | tee "$G8/${M}_out.txt" >/dev/null
else
  DS4_GENPROBE=$N DS4_THREADS=6 \
    ./ds4quant_run "$G8/prompt_py.ids" "$(grep -c . "$G8/prompt_py.ids")" 2>"$G8/${M}_run.log" | tee "$G8/${M}_out.txt" >/dev/null
fi
grep '^GENIDS' "$G8/${M}_out.txt" | head -1
;;
decode)
shift; python3 - "$@" <<'PY'
import sys
from tokenizers import Tokenizer
tok=Tokenizer.from_file("/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base/tokenizer.json")
print(tok.decode([int(x) for x in sys.argv[1].split()]))
PY
;;
esac
