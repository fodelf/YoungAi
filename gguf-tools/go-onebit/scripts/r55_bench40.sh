#!/bin/bash
# r55_bench40.sh — r55 40 题批+报告(r55_overnight ④⑤ 段修正版: 补 MODEL)
set -uo pipefail
ROOT=/Users/fodelf/git/ds4-main
cd $ROOT
export MODEL=gguf/go-onebit/ds4-r30.gguf
LOG(){ echo "[r55b $(date +%H:%M:%S)] $*"; }
SUITE=humaneval      TAG=r55 N=20 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Py 批异常(继续 Go)★"
SUITE=humaneval-x-go TAG=r55 N=20 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Go 批异常★"
pkill -f ds4-server 2>/dev/null
python3 - <<'PYEOF'
import json, glob, re, datetime, os
R='/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/reports/pubbench'
def suite(pat):
    rows=[]
    for f in sorted(glob.glob(f'{R}/{pat}'), key=lambda x:int(re.search(r'_t(\d+)\.jsonl',x).group(1)) if re.search(r'_t(\d+)\.jsonl',x) else 0):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows
py=suite('pubbench_humaneval_r55_t*.jsonl'); go=suite('pubbench_humaneval-x-go_r55_t*.jsonl')
def s(x): return sum(1 for r in x if r.get('pass'))
gb=os.path.getsize('/Users/fodelf/git/ds4-main/gguf/go-onebit/ds4-r30.gguf')/1e9
rep=f"""# r55 战役 40 题测试报告({datetime.date.today()})

## 模型
55GB 落地战役: 热49 vq4x512 + 冷 signref 1bit + RB 路由烘焙 α2.5(6661 槽); 语料 v5mini 同前。
实测落地 {gb:.2f} GB。五指标(反修终态, VERDICT 口径): PPL ratio 1.597 / Σmin 0.719 /
KL 0.891 / top1一致 73.4%(r64 热108 对照: 1.185 / 0.847 / 0.255 / 84.7%)。

## 总分
- Python: {s(py)}/{len(py)}
- Go: {s(go)}/{len(go)}
- **总计: {s(py)+s(go)}/{len(py)+len(go)}**

## 对表
| 版本 | 落地体积 | Python | Go | 总分 |
|---|---|---|---|---|
| 冠军 v4bf | ~58 GB | 15/20 | 10/20 | 25/40 |
| r64 全局版(热108) | 67.5 GB | 19/20 | 11/20 | 30/40 |
| **r55(热49+RB)** | {gb:.2f} GB | {s(py)}/{len(py)} | {s(go)}/{len(go)} | {s(py)+s(go)}/{len(py)+len(go)} |

## 逐题
### Python
""" + "\n".join(f"- {r['task_id']}: {'PASS' if r.get('pass') else 'FAIL'}" for r in py) + "\n\n### Go\n" + \
"\n".join(f"- {r['task_id']}: {'PASS' if r.get('pass') else 'FAIL'}" + (" (TODO 弃权)" if '// TODO' in (r.get('response_text') or '') and len(r.get('response_text',''))<30 else "") for r in go)
open(f'{R}/r55_report.md','w').write(rep)
print(rep[:400])
PYEOF
LOG "★40 题+报告收官 → reports/pubbench/r55_report.md★"
