#!/usr/bin/env python3
"""bpw_audit.py — 可复现的 bits-per-weight 审计(对外主张的证据脚本)。

为什么需要它: 文献报 bpw 口径不统一(BiMoE 报"1-bit"但 attention/shared experts
是 4-bit)。任何 "N.NN bit" 的对外主张必须给出 ①口径定义 ②可复现的算法。
本脚本只读 GGUF header/KV/tensor-info 区(绝不读张量数据), 内存安全。

统计口径(三档, 全部输出, 不许只报最好看的那个):
  A. routed-expert bpw  = routed 专家权重的编码字节 / 专家权重数
  B. 全模型 bpw         = 文件总字节 / 全模型参数数
  C. 无损叠加 VQ overlay 后的 A/B (--vq-dir 指定侧车目录)

用法:
  bpw_audit.py BASE.gguf [--vq-dir DIR] [--hot-experts N]
"""
import struct, sys, os, glob

SCALAR_FMT = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),
              6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}


def read_gguf(path):
    """返回 (kv, tensors, data_off, file_size)。tensors = [(name, dims, type, off)]"""
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', 'bad magic'
    struct.unpack('<I', f.read(4))  # version
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))

    def rstr():
        ln, = struct.unpack('<Q', f.read(8))
        return f.read(ln).decode('utf-8', 'replace')

    def rval(t):
        if t in SCALAR_FMT:
            fmt, sz = SCALAR_FMT[t]
            return struct.unpack('<' + fmt, f.read(sz))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, = struct.unpack('<I', f.read(4))
            cnt, = struct.unpack('<Q', f.read(8))
            if et == 8:
                return [rstr() for _ in range(cnt)]
            fmt, sz = SCALAR_FMT[et]
            return list(struct.unpack('<' + fmt * cnt, f.read(sz * cnt)))
        raise ValueError(f'unknown type {t}')

    kv = {}
    for _ in range(n_kv):
        key = rstr()
        t, = struct.unpack('<I', f.read(4))
        kv[key] = rval(t)

    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack('<I', f.read(4))
        dims = list(struct.unpack('<' + 'Q' * nd, f.read(8 * nd)))
        ttype, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        tensors.append((name, dims, ttype, off))

    align = kv.get('general.alignment', 32)
    pos = f.tell()
    data_off = (pos + align - 1) // align * align
    size = os.path.getsize(path)
    f.close()
    return kv, tensors, data_off, size


def tensor_bytes(tensors, data_off, file_size):
    """用相邻 offset 差反推每个张量的真实字节数(不依赖 type 表, 对自定义 type 也准)。"""
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][3])
    out = {}
    for k, i in enumerate(order):
        name, dims, ttype, off = tensors[i]
        end = tensors[order[k + 1]][3] if k + 1 < len(order) else (file_size - data_off)
        out[name] = end - off
    return out


def classify(name):
    if 'ffn_gate_exps' in name or 'ffn_up_exps' in name or 'ffn_down_exps' in name:
        return 'routed_experts'
    if 'exps' in name or 'tid2eid' in name:
        return 'moe_routing'
    if 'shexp' in name or 'shared' in name:
        return 'shared_expert'
    if 'token_embd' in name or 'output.weight' in name:
        return 'embed_output'
    if name.startswith('blk.'):
        return 'attention_norm'
    return 'other'


def fmt_gib(b):
    return f'{b / (1 << 30):9.3f} GiB'


def main():
    path = sys.argv[1]
    vq_dir = None
    vq_bytes_arg = None
    hot = 64
    for i, a in enumerate(sys.argv):
        if a == '--vq-dir':
            vq_dir = sys.argv[i + 1]
        if a == '--vq-bytes':
            # base 与侧车不在同一台机器时用: 直接传侧车总字节(可由 ls -la 求和复现)
            vq_bytes_arg = int(sys.argv[i + 1])
        if a == '--hot-experts':
            hot = int(sys.argv[i + 1])

    kv, tensors, data_off, fsize = read_gguf(path)
    tb = tensor_bytes(tensors, data_off, fsize)

    n_layer = kv['deepseek4.block_count']
    d_model = kv['deepseek4.embedding_length']
    n_exp = kv['deepseek4.expert_count']
    n_ff = kv['deepseek4.expert_feed_forward_length']

    print(f'== {os.path.basename(path)}')
    print(f'   arch: n_layer={n_layer} d_model={d_model} n_expert={n_exp} n_ff_exp={n_ff}')
    print(f'   file={fmt_gib(fsize)}  data_off={data_off}')
    print()

    # 按类别汇总 参数数 / 字节数
    agg = {}
    for name, dims, ttype, off in tensors:
        cls = classify(name)
        nparam = 1
        for d in dims:
            nparam *= d
        p, b = agg.get(cls, (0, 0))
        agg[cls] = (p + nparam, b + tb[name])

    print('--- 分类账(base GGUF) ---')
    print(f'{"class":<18}{"params":>16}{"bytes":>16}{"size":>14}{"bpw":>9}')
    tot_p = tot_b = 0
    for cls in sorted(agg, key=lambda c: -agg[c][1]):
        p, b = agg[cls]
        tot_p += p
        tot_b += b
        print(f'{cls:<18}{p:>16,}{b:>16,}{fmt_gib(b):>14}{b * 8 / p:>9.4f}')
    print(f'{"TOTAL":<18}{tot_p:>16,}{tot_b:>16,}{fmt_gib(tot_b):>14}{tot_b * 8 / tot_p:>9.4f}')
    print()

    rp, rb = agg['routed_experts']
    print(f'[A] routed-expert bpw (base 1-bit)  = {rb * 8 / rp:.4f}')
    print(f'[B] 全模型 bpw (base)               = {fsize * 8 / tot_p:.4f}'
          f'   (文件总字节/总参数, 含 tokenizer 等元数据)')
    print()

    if not vq_dir and vq_bytes_arg is None:
        return

    # --- VQ overlay 口径 ---
    if vq_dir:
        vq_files = sorted(glob.glob(os.path.join(vq_dir, 'dql_vq_L*.bin')))
        if not vq_files:
            print(f'!! 未找到 dql_vq_L*.bin @ {vq_dir}')
            return
        vq_bytes = sum(os.path.getsize(p) for p in vq_files)
        n_vq = len(vq_files)
    else:
        vq_bytes, n_vq = vq_bytes_arg, n_layer

    # VQ 侧车覆盖: 热 hot 个专家的全三矩阵 + 冷 (n_exp-hot) 个专家的 w1/w3。
    # 冷 w2 仍从 base 的 1-bit signref 展开 → 那部分 base 字节必须留在账里。
    cold = n_exp - hot
    w_per_mat = n_ff * d_model
    # base 每层 routed 字节(三张量)
    base_layer_b = rb / n_layer
    base_per_mat_b = base_layer_b / (n_exp * 3)          # 1-bit+scale 每矩阵字节
    kept_cold_w2_b = cold * base_per_mat_b * n_layer     # 仍在用的 base 字节

    vq_routed_b = vq_bytes + kept_cold_w2_b
    print(f'--- VQ overlay 口径 (hot={hot} cold={cold}, 共 {n_vq} 层侧车) ---')
    print(f'  VQ 侧车总字节        = {vq_bytes:>16,}  {fmt_gib(vq_bytes)}')
    print(f'  仍用的 base 冷w2     = {int(kept_cold_w2_b):>16,}  {fmt_gib(kept_cold_w2_b)}')
    print(f'  routed 合计          = {int(vq_routed_b):>16,}  {fmt_gib(vq_routed_b)}')
    print()
    print(f'[A-vq] routed-expert bpw            = {vq_routed_b * 8 / rp:.4f}')
    nonexp_b = fsize - rb
    full_b = vq_routed_b + nonexp_b
    print(f'[B-vq] 全模型 bpw                   = {full_b * 8 / tot_p:.4f}'
          f'   (VQ routed + base 非专家 {fmt_gib(nonexp_b)})')
    print()
    print(f'  非专家部分占比: {nonexp_b / full_b * 100:.1f}% 字节 / '
          f'{(tot_p - rp) / tot_p * 100:.2f}% 参数'
          f'   ← 这是 "routed bpw" 与 "全模型 bpw" 的差源, 对外必须两个都报')


if __name__ == '__main__':
    main()
