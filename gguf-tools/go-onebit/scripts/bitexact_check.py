#!/usr/bin/env python3
"""bitexact_check.py — Bit-exact weights 指标(R30, 2026-08-02 用户令)。

口径: 合并 GGUF 里所有【骨架来源】张量必须与骨架逐字节一致 —— 合并管线只许"搬运+追加",
不许碰权重。唯一合法差异 = blk.*.exp_probs_b.bias(路由偏置 α·Δb 烘焙, 设计内改写)。

输出: bit-exact 张量数/骨架张量数 + 合法差异清单 + ★非法差异(有则 FAIL)★ + 新增(blob)统计。
用法: bitexact_check.py <skeleton.gguf> <merged.gguf>
"""
import hashlib, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'quant'))
from vq_merge_v4 import parse_header, sizes_by_offset


def load(path):
    f = open(path, 'rb')
    n_kv, kv_raw, tens, data0 = parse_header(f)
    sizes_by_offset(tens, os.path.getsize(path), data0)
    return f, {t['name']: t for t in tens}, data0


def sha(f, off, n):
    h = hashlib.sha256()
    f.seek(off)
    left = n
    while left:
        b = f.read(min(1 << 24, left))
        if not b:
            raise SystemExit('★读穿文件尾 — 偏移表坏★')
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def main():
    fs, skel, d0s = load(sys.argv[1])
    fm, mrg, d0m = load(sys.argv[2])
    exact, legal, illegal, missing = 0, [], [], []
    for nm, t in skel.items():
        m = mrg.get(nm)
        if m is None:
            missing.append(nm)
            continue
        same = (m['bytes'] == t['bytes']
                and sha(fs, d0s + t['off'], t['bytes']) == sha(fm, d0m + m['off'], m['bytes']))
        if same:
            exact += 1
        elif nm.endswith('.exp_probs_b.bias'):
            legal.append(nm)
        else:
            illegal.append(nm)
    added = [nm for nm in mrg if nm not in skel]
    added_b = sum(mrg[nm]['bytes'] for nm in added)
    print(f'骨架张量 {len(skel)}: bit-exact {exact}, 合法差异(路由烘焙 bias) {len(legal)}, '
          f'非法差异 {len(illegal)}, 合并缺失 {len(missing)}')
    print(f'合并新增(量化载荷) {len(added)} 张量 {added_b/2**30:.3f} GiB')
    if legal:
        print(f'  合法差异层: {sorted(int(n.split(".")[1]) for n in legal)}')
    for nm in illegal[:10]:
        print(f'  ★非法差异★ {nm}')
    for nm in missing[:10]:
        print(f'  ★缺失★ {nm}')
    ok = not illegal and not missing
    print(f'Bit-exact weights = {exact}/{len(skel)} '
          f'({exact/len(skel)*100:.2f}%)  ⇒ {"PASS(其余全为设计内烘焙)" if ok else "★FAIL★"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()


def vq_blob_invariants(f, data0, tensors):
    """★量化载荷结构不变量(2026-08-03 用户纠: 骨架逐位恒过没守门价值, blob 损毁时照报 PASS):
    每层 vq blob 槽表全部偏移必须 < blob 长度(今日事故一秒即抓)。"""
    import struct
    bad = []
    for L in range(43):
        nm = f"blk.{L}.ffn_exps_vq.blob"
        if nm not in tensors:
            continue
        blen, off = tensors[nm][0], tensors[nm][1]
        f.seek(data0 + off + 16)
        tab = struct.unpack("<768Q", f.read(768 * 8))
        if max(tab) >= blen:
            bad.append(L)
    if bad:
        print(f"★VQ blob 槽表越界层: {bad} ⇒ FAIL★")
        return False
    print("VQ blob 槽表不变量 = 43/43 界内 ⇒ PASS")
    return True
