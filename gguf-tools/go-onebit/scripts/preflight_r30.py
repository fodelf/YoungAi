#!/usr/bin/env python3
"""preflight_r30.py — R30 起跑前置闸(2026-08-02, 源模型换 DeepSeek-V4-Flash-0731)。

换源风险: 合并模型的 KV 元数据(tokenizer/超参)来自 template_head.gguf(published GGUF 头,
上一代模型的)。若 0731 改了 tokenizer 或超参, 合并模型 = 新权重 × 旧元数据 = 静默错误。
本脚本做两件事, 任一不过 = 拒跑战役:

  ① tokenizer 逐词对比: GGUF tokenizer.ggml.tokens[i] ↔ 0731 tokenizer.json vocab(id→token)
  ② 超参对比: GGUF KV(deepseek4.*) ↔ 0731 config.json 能对上的字段全对

用法: preflight_r30.py <template_head.gguf> <0731目录>
"""
import json, struct, sys

GGUF_T = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
SCALAR = {0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),6:('<f',4),7:('<B',1),10:('<Q',8),11:('<q',8),12:('<d',8)}


def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')


def rd_val(f, t):
    if t in SCALAR:
        fmt, n = SCALAR[t]
        return struct.unpack(fmt, f.read(n))[0]
    if t == 8:
        return rd_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        return [rd_val(f, et) for _ in range(cnt)]
    raise SystemExit(f'未知 GGUF KV type {t}')


def read_kv(path):
    kv = {}
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit('不是 GGUF')
        ver, n_t, n_kv = struct.unpack('<IQQ', f.read(20))
        for _ in range(n_kv):
            k = rd_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            kv[k] = rd_val(f, t)
    return ver, n_t, kv


def main():
    tmpl, hf = sys.argv[1], sys.argv[2].rstrip('/')
    ver, n_t, kv = read_kv(tmpl)
    print(f'template: GGUF v{ver} n_tensors={n_t} n_kv={len(kv)}')

    # ---- ① tokenizer 逐词 ----
    tj = json.load(open(f'{hf}/tokenizer.json'))
    vocab = {int(v): k for k, v in tj['model']['vocab'].items()}
    for at in tj.get('added_tokens', []):
        vocab[int(at['id'])] = at['content']
    gt = kv.get('tokenizer.ggml.tokens')
    if gt is None:
        raise SystemExit('★template 无 tokenizer.ggml.tokens — 模板坏★')
    bad = 0
    if len(gt) != len(vocab):
        print(f'★tokenizer 词表长度不一致: GGUF {len(gt)} vs 0731 {len(vocab)}★')
        bad += 1
    n = min(len(gt), len(vocab))
    diff = [(i, gt[i], vocab.get(i)) for i in range(n) if gt[i] != vocab.get(i)]
    if diff:
        bad += 1
        print(f'★tokenizer 逐词不一致 {len(diff)}/{n}, 前 5:')
        for i, a, b in diff[:5]:
            print(f'   id={i} GGUF={a!r} 0731={b!r}')
    else:
        print(f'tokenizer ✓ {n} 词逐一相同')
    # merges 同样要一致(BPE 切分路径)
    gm = kv.get('tokenizer.ggml.merges')
    hm = tj['model'].get('merges')
    if gm is not None and hm is not None:
        hm2 = [' '.join(m) if isinstance(m, list) else m for m in hm]
        if list(gm) != hm2:
            k = sum(1 for a, b in zip(gm, hm2) if a != b) + abs(len(gm) - len(hm2))
            print(f'★merges 不一致({k} 处, len {len(gm)} vs {len(hm2)})★')
            bad += 1
        else:
            print(f'merges ✓ {len(gm)} 条相同')

    # ---- ② 超参: GGUF KV 里能和 config.json 对上的全对 ----
    cfg = json.load(open(f'{hf}/config.json'))
    # GGUF key 尾段 → config 字段(能直接对数值的)
    pairs = [
        ('embedding_length', 'hidden_size'), ('block_count', 'num_hidden_layers'),
        ('attention.head_count', 'num_attention_heads'), ('expert_count', 'n_routed_experts'),
        ('expert_used_count', 'num_experts_per_tok'), ('expert_shared_count', 'n_shared_experts'),
        ('expert_feed_forward_length', 'moe_intermediate_size'), ('vocab_size', 'vocab_size'),
        ('attention.q_lora_rank', 'q_lora_rank'), ('expert_weights_scale', 'routed_scaling_factor'),
        ('indexer.head_count', 'index_n_heads'), ('indexer.head_dim', 'index_head_dim'),
        ('indexer.top_k', 'index_topk'), ('rope.dimension_count', 'qk_rope_head_dim'),
    ]
    mismatch = []
    for gk_tail, ck in pairs:
        gv = next((v for k, v in kv.items() if k.endswith(gk_tail)), None)
        cv = cfg.get(ck)
        if gv is None or cv is None:
            continue
        ok = abs(float(gv) - float(cv)) < 1e-6 if isinstance(cv, (int, float)) else gv == cv
        tag = '✓' if ok else '★不一致★'
        print(f'  {gk_tail:34s} GGUF={gv}  config={cv}  {tag}')
        if not ok:
            mismatch.append(gk_tail)
    if mismatch:
        bad += 1
        print(f'★超参不一致: {mismatch}★')

    # 数值型 KV 全 dump(人工兜底看一眼, 数组截断)
    print('\n-- GGUF 全 KV(数组截断) --')
    for k, v in kv.items():
        if isinstance(v, list):
            print(f'  {k} = [{len(v)}]{v[:3] if len(v) <= 200 else "…"}')
        else:
            print(f'  {k} = {v!r}' if not isinstance(v, str) or len(v) < 80 else f'  {k} = <str {len(v)}>')

    print('\n★preflight PASS★' if bad == 0 else f'\n★preflight FAIL({bad} 项)— 拒跑战役★')
    sys.exit(0 if bad == 0 else 1)


if __name__ == '__main__':
    main()
