"""把训练器的留出段归回原题。eval_n 会整题跳过，不能把留下的篇号重新编号。

篇号从 data 的第一条记录开始；eval_kl 在 data 后面，因此原题下标 = 篇号 − data 记录数。
缺篇号、来源对不上或前后两份评估不一致时直接失败，不再按长度猜题。
"""
import json
import math
from pathlib import Path
import re


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fam_of_prefix(prefix, questions):
    matches = [(q["id"], q["src"]) for q in questions if q.get("q") and q["q"] in prefix]
    if len(matches) > 1:
        raise ValueError("同一前缀匹配多道题，不能确定题目归属")
    return matches[0] if matches else (None, None)


def read_config(tag, directory):
    stem = "run" if Path(directory).name.startswith("vstep-") else "pt"
    path = Path("gguf/go-onebit/evo/vstep") / f"{stem}_{tag}.cfg"
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    if int(config.get("expand", "4")):
        raise ValueError(f"{path}: expand 非零，缺少扩展后的篇号清单")
    if not config.get("data") or not config.get("eval_kl"):
        raise ValueError(f"{path}: 缺 data 或 eval_kl，无法确认留出来源")
    return config


def selected_epoch(tag, directory, ledger):
    """账本记录的实际判决件可能是第二遍，也可能退回第一遍，不能拿最新文件猜。"""
    matches = []
    if Path(ledger).exists():
        for line in Path(ledger).read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 5 or fields[1] != tag:
                continue
            checkpoint = Path(fields[4])
            match = re.fullmatch(r"ckpt_e(\d+)(?:v17)?", checkpoint.name)
            if match and checkpoint.parent.name == Path(directory).name:
                matches.append(int(match[1]))
    if matches:
        return matches[-1]
    epochs = [int(m[1]) for p in Path(directory).glob("heldout_e*.tsv")
              if (m := re.fullmatch(r"heldout_e(\d+)\.tsv", p.name)) and int(m[1]) > 0]
    if len(epochs) == 1:
        return epochs[0]
    raise ValueError(f"{tag}: 有多遍评估但账本未指明判决件；用 scan --epoch 指定，不能猜")


def read_heldout(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"{path}: 缺真实篇号；旧的按长度猜测映射已停用")
        sample, tokens, doc = int(fields[1]), int(fields[2]), int(fields[5])
        kl, decision = float(fields[3]), float(fields[4])
        if tokens <= 0 or doc < 0 or not math.isfinite(kl) or not math.isfinite(decision):
            raise ValueError(f"{path}: 无效的留出记录")
        if records and sample <= records[-1][4]:
            raise ValueError(f"{path}: 样本号重复或乱序")
        records.append((tokens, kl, decision, doc, sample))
    if not records:
        raise ValueError(f"{path}: 留出记录为空")
    return records


def validate_pair(before, after):
    if len(before) != len(after) or any((a[0], a[3], a[4]) != (b[0], b[3], b[4]) for a, b in zip(before, after)):
        raise ValueError("轮前、轮后的篇号/样本号/token 数不一致，不能配对")


def map_segments(records, training, heldout, questions, families):
    # 当前循环的每条 ids/text 记录恰好占一个篇号。别把新的输入形态默默当成同一约定。
    if any(not isinstance(row.get("ids"), list) and not row.get("text") for row in training):
        raise ValueError("训练料包含非 ids/text 行，不能用记录数确定篇号起点")
    if any(not row.get("text") for row in heldout):
        raise ValueError("留出不是非空 text 行，不能用记录数确定原题")
    offset = len(training)
    mapping = {}
    for i, record in enumerate(records):
        row = record[3] - offset
        if row < 0 or row >= len(heldout):
            raise ValueError(f"篇号 {record[3]} 不在留出范围 {offset}..{offset + len(heldout) - 1}")
        qid, family = fam_of_prefix(heldout[row].get("prefix", ""), questions)
        if qid is None or family not in families:
            raise ValueError(f"留出第 {row + 1} 行不能唯一归入已知题族")
        mapping[i] = (row, qid, family)
    return mapping


def paired_gain(before, after):
    """与训练器相同的 token 加权降幅；标准误先把同一篇的段合并。"""
    validate_pair(before, after)
    docs = {}
    for a, b in zip(before, after):
        item = docs.setdefault(a[3], [0.0, 0])
        item[0] += a[1] - b[1]
        item[1] += a[0]
    tokens = sum(x[1] for x in docs.values())
    gain = sum(x[0] for x in docs.values()) / tokens
    n = len(docs)
    se = math.sqrt(sum((delta - gain * count) ** 2 for delta, count in docs.values()) * n / (n - 1)) / tokens if n > 1 else None
    return {"gain": gain, "se": se, "z": gain / se if se else None, "documents": n, "segments": len(before)}
