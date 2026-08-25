#!/bin/bash
# r55_overnight.sh — 2026-08-05 深夜用户令: 55GB 落地新战役, 明早交 40 题报告。
# 链: M1[锚重建→top49 热表→plan(HOT=49)→量化→反修(RB FIT)→合并(烘焙+55.0GB 闸)→五指标]
#     → 回传 M4 → 40 题 serial(Py20+Go20, TAG=r55) → 报告 md。
# 语料=v5mini 不变(deadline 内不加新变量); RB 工序保留; 体积口径=落地十进制 GB。
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
SC_M1=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[r55 $(date +%H:%M:%S)] $*"; }

# ① 清场
pgrep -f ds4-server >/dev/null && { pkill -f ds4-server; sleep 2; }
ssh $M1 "pkill -f ds4-server; pkill -f ds4quant_run" 2>/dev/null; sleep 1
LOG "清场 ✓"

# ② M1: 锚重建(~1h) → top49 热表 → plan(HOT=49) → 量化 → 反修 → 合并 → 指标
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh anchor" || { LOG "★锚重建失败★"; exit 3; }
LOG "锚 ✓"
ssh $M1 "cd $M1DIR && python3 $SC_M1/anchor_top_experts.py \
    gguf/go-onebit/r30/anchor_r30_s1716.bin 49 gguf-tools/go-onebit/corpus/prog_active_top49.txt \
    && head -1 gguf-tools/go-onebit/corpus/prog_active_top49.txt" || { LOG "★top49 表生成失败★"; exit 3; }
LOG "top49 热表 ✓"
ssh $M1 "cd $M1DIR && PLAN_HOT=49 bash $SC_M1/r30_campaign.sh plan"  || { LOG "★plan 失败★"; exit 3; }
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh quant"             || { LOG "★量化失败★"; exit 3; }
LOG "量化 43 层 ✓"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=1 DS4_GS_CONV_PCT=0 \
         bash $SC_M1/r30_campaign.sh backfit"                        || { LOG "★反修失败★"; exit 3; }
LOG "反修+回扫 ✓"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         bash $SC_M1/r30_campaign.sh student"                        || { LOG "★学生回放失败★"; exit 3; }
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh merge"             || { LOG "★合并/烘焙失败★"; exit 3; }
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh metrics"           || { LOG "★指标失败★"; exit 3; }
LOG "合并+烘焙+五指标 ✓"

# ③ 回传 M4(RB 版保留为 ds4-r30-rb64.gguf 前代, 新 55G 模型顶 ds4-r30.gguf 主位)
[ -f $ROOT/gguf/go-onebit/ds4-r30.gguf ] && mv $ROOT/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30-rb64.gguf
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓ $(ls -l $ROOT/gguf/go-onebit/ds4-r30.gguf | awk '{printf "%.2f GB", $5/1e9}')"

# ④ 40 题(逐题隔离, Python 20 + Go 20)
cd $ROOT
SUITE=humaneval      TAG=r55 N=20 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Py 批异常(继续 Go)★"
SUITE=humaneval-x-go TAG=r55 N=20 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Go 批异常★"
pkill -f ds4-server 2>/dev/null

# ⑤ 报告
python3 - <<'PYEOF'
import json, glob, re, datetime
R='/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/reports/pubbench'
def suite(pat):
    rows=[]
    for f in sorted(glob.glob(f'{R}/{pat}'), key=lambda x:int(re.search(r'_t(\d+)\.jsonl',x).group(1)) if re.search(r'_t(\d+)\.jsonl',x) else 0):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows
py=suite('pubbench_humaneval_r55_t*.jsonl'); go=suite('pubbench_humaneval-x-go_r55_t*.jsonl')
def s(x): return sum(1 for r in x if r.get('pass'))
rep=f"""# r55 战役 40 题测试报告({datetime.date.today()})

## 模型
55GB 落地战役(热49 vq4x512 + 冷 signref 1bit + RB 路由烘焙 α2.5; 语料 v5mini 同前)

## 总分
- Python: {s(py)}/{len(py)}
- Go: {s(go)}/{len(go)}
- **总计: {s(py)+s(go)}/{len(py)+len(go)}**

## 对表
| 版本 | 落地体积 | Python | Go | 总分 |
|---|---|---|---|---|
| 冠军 v4bf | ~58 GB | 15/20 | 10/20 | 25/40 |
| r64 全局版(热108) | 67.5 GB | 19/20 | (11/20, 部分批) | 30/40 |
| **r55(热49+RB)** | {'{:.1f}'.format(__import__('os').path.getsize('/Users/fodelf/git/ds4-main/gguf/go-onebit/ds4-r30.gguf')/1e9)} GB | {s(py)}/{len(py)} | {s(go)}/{len(go)} | {s(py)+s(go)}/{len(py)+len(go)} |

## 逐题
### Python
""" + "\n".join(f"- {r['task_id']}: {'PASS' if r.get('pass') else 'FAIL'}" for r in py) + "\n\n### Go\n" + \
"\n".join(f"- {r['task_id']}: {'PASS' if r.get('pass') else 'FAIL'}" + (" (TODO 弃权)" if '// TODO' in (r.get('response_text') or '') and len(r.get('response_text',''))<30 else "") for r in go)
open('/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/reports/pubbench/r55_report.md','w').write(rep)
print(rep[:600])
PYEOF
LOG "★r55 战役收官: 报告 → reports/pubbench/r55_report.md★"
