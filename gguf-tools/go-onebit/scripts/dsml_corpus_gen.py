#!/usr/bin/env python3
"""dsml_corpus_gen.py — P2 DSML 工具域校准语料批量生成器 (步骤 1/6, 秒级可跑).

目标能力 (Gate v2 判决): 长上下文里检索用户真实值(路径/命令)填入工具参数,
而非复读模板占位符。因此每条样本 = 真实感上下文(可变长) + 用户指令(含真值)
+ 金标 DSML 轨迹(真值必须逐字来自用户指令)。

产物: corpus/dsml/gen_v1/NNN.txt  (=== PROMPT ===/=== GOLD === 对)
     corpus/dsml/gen_v1/prompts.txt (捕获用纯 prompt 列表, 每行一个文件路径)
用法: python3 dsml_corpus_gen.py [N=200] [--ctx-pad TOKENS≈0]
      --ctx-pad 用无关代码填充上下文到指定规模(模拟 23k 场景, 默认不填充)
"""
import os, random, sys

random.seed(20260707)
OUT = os.path.join(os.path.dirname(__file__), '..', 'corpus', 'dsml', 'gen_v1')
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
PAD = 0
for a in sys.argv:
    if a.startswith('--ctx-pad='): PAD = int(a.split('=')[1])

DIRS = ['/tmp/gotask', '/Users/dev/proj', '/home/ci/build', './internal/api', './pkg/utils',
        '/srv/app', './cmd/server', '/opt/tools/x']
FILES = ['main.go', 'main_test.go', 'handler.go', 'utils.go', 'parse_test.go', 'router.go',
         'config.yaml', 'Makefile', 'go.mod']
TOOLS = {
    'Read':  ('file_path', lambda d, f: f'{d}/{f}'),
    'Bash':  ('command',  lambda d, f: random.choice([
                  f'go test {d}/...', f'cat {d}/{f}', f'go build {d}/...',
                  f'grep -n TODO {d}/{f}', f'go vet {d}/...'])),
    'Write': ('file_path', lambda d, f: f'{d}/{f}'),
    'Glob':  ('pattern',   lambda d, f: f'{d}/**/*.go'),
}
ASKS = {
    'Read':  ['{v} 里的 {fn} 函数有 bug, 先读这个文件看看代码。',
              '请打开 {v} 检查实现。', 'Read {v} and inspect the logic.'],
    'Bash':  ['运行 `{v}` 看看结果。', '先执行 {v} 确认现状。', 'Run {v} first.'],
    'Write': ['把修好的代码写回 {v}。', 'Save the fixed version to {v}.'],
    'Glob':  ['列出匹配 {v} 的所有源文件。', 'Find files matching {v}.'],
}
FUNCS = ['isPalindrome', 'twoSum', 'parseConfig', 'reverseString', 'mergeSort', 'httpRetry']

TOOLS_HEADER = '''## Tools

You have access to a set of tools to help answer the user question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block.

### Available Tool Schemas

{"name":"Read","description":"Reads a file from the local filesystem.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}
{"name":"Bash","description":"Executes a bash command.","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}
{"name":"Write","description":"Writes a file to the local filesystem.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}
{"name":"Glob","description":"Fast file pattern matching.","input_schema":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
'''

PAD_SNIPPET = '''func helper%d(xs []int) int {
\tacc := 0
\tfor i, x := range xs {
\t\tif x%%2 == 0 { acc += x } else { acc -= i }
\t}
\treturn acc
}
'''

os.makedirs(OUT, exist_ok=True)
plist = []
for n in range(N):
    tool = random.choice(list(TOOLS))
    pname, vgen = TOOLS[tool]
    d, f = random.choice(DIRS), random.choice(FILES)
    v = vgen(d, f)
    ask = random.choice(ASKS[tool]).format(v=v, fn=random.choice(FUNCS))
    pad = ''
    if PAD:
        chunks = []
        approx = 0
        k = 0
        while approx < PAD * 3:          # ~3 chars/token 粗算
            chunks.append(PAD_SNIPPET % k); approx += len(PAD_SNIPPET); k += 1
        pad = '这是当前项目的部分代码(与任务无关的背景):\n```go\n' + ''.join(chunks) + '```\n\n'
    prompt = f"<｜begin▁of▁sentence｜>{TOOLS_HEADER}\n<｜User｜>{pad}{ask}<｜Assistant｜>"
    gold = (f'<｜DSML｜tool_calls>\n<｜DSML｜invoke name="{tool}">\n'
            f'<｜DSML｜parameter name="{pname}" string="true">{v}</｜DSML｜parameter>\n'
            f'</｜DSML｜invoke>\n</｜DSML｜tool_calls>')
    path = os.path.join(OUT, f'{n:03d}.txt')
    with open(path, 'w') as fp:
        fp.write("=== PROMPT ===\n" + prompt + "\n=== GOLD ===\n" + gold + "\n")
    plist.append(path)
with open(os.path.join(OUT, 'prompts.txt'), 'w') as fp:
    fp.write('\n'.join(plist) + '\n')
print(f"generated {N} samples -> {OUT} (pad≈{PAD} tok)")
