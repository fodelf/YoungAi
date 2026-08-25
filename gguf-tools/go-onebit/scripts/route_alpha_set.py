#!/usr/bin/env python3
# route_alpha_set.py — 幂等地把模型的 exp_probs_b 设成 [裸态 + α·Δb]。
#
# 为什么要幂等(2026-07-31): r36_rebake_bias.py 是**增量**的(原位 -= / +=), 连着调两次
# 就叠加两份, 扫 α 时会越扫越偏。本工具改成绝对写: 首次运行把当前 bias 存成快照
# <model>.bias0.bin(裸态 = 已减掉冠军 Δb、尚未加自己 Δb 的状态), 之后每次都从快照重算,
# 所以 α 可以来回扫多少次都不累积。
#
# 口径(与 vq_merge_v4.load_route_bias / rb_save 对齐): R28 侧车由 rb_save 落盘的是
# RB_APPLY —— 已经 /cnt 的均值, 且不满 mincnt 门的槽写 0, 故消费端直接 ×α, 不再除 cnt。
#
# 用法: route_alpha_set.py <model.gguf> <rb.bin> <alpha> [--snapshot-only]
import os, struct, sys

NEXP = 256


def parse(f):
    f.read(8)
    n_t, n_kv = struct.unpack('<QQ', f.read(16))

    def rs():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n)

    def sv(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz:
            f.seek(sz[t], 1); return
        if t == 8:
            rs(); return
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            for _ in range(n):
                sv(et)

    align = 32
    for _ in range(n_kv):
        k = rs(); t = struct.unpack('<I', f.read(4))[0]
        if k == b'general.alignment':
            align = struct.unpack('<I', f.read(4))[0]
        else:
            sv(t)
    tgt = {}
    for _ in range(n_t):
        name = rs().decode()
        nd = struct.unpack('<I', f.read(4))[0]
        f.seek(8 * nd, 1)
        ty, off = struct.unpack('<IQ', f.read(12))
        if name.endswith('.exp_probs_b.bias'):
            assert ty == 0, f'{name} 非 f32'
            tgt[int(name.split('.')[1])] = off
    data_start = (f.tell() + align - 1) // align * align
    return tgt, data_start


def main():
    model, rbp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    snap_only = '--snapshot-only' in sys.argv

    with open(rbp, 'rb') as g:
        hd = struct.unpack('<4I', g.read(16))
        assert hd[0] == 0x41494252, f'{rbp} 非 RBIA'
        nl, ne = hd[1], hd[2]
        acc = struct.unpack(f'<{nl*ne}f', g.read(nl * ne * 4))

    f = open(model, 'r+b')
    tgt, data_start = parse(f)
    snap = model + '.bias0.bin'

    if not os.path.exists(snap):
        # 首次: 当前状态即裸态, 存快照
        with open(snap, 'wb') as s:
            s.write(struct.pack('<I', len(tgt)))
            for L in sorted(tgt):
                f.seek(data_start + tgt[L])
                s.write(struct.pack('<I', L))
                s.write(f.read(NEXP * 4))
        print(f'[alpha] 裸态快照 → {snap} ({len(tgt)} 层)')
        if snap_only:
            f.close(); return

    base = {}
    with open(snap, 'rb') as s:
        n, = struct.unpack('<I', s.read(4))
        for _ in range(n):
            L, = struct.unpack('<I', s.read(4))
            base[L] = list(struct.unpack(f'<{NEXP}f', s.read(NEXP * 4)))

    nset = 0
    for L in sorted(tgt):
        if L not in base:
            continue
        vals = list(base[L])
        if L < nl:
            for e in range(min(NEXP, ne)):
                a = acc[L * ne + e]
                if a != 0.0:            # RB_APPLY 已含 mincnt 门(不满门=0)
                    vals[e] += alpha * a
                    nset += 1
        f.seek(data_start + tgt[L])
        f.write(struct.pack(f'<{NEXP}f', *vals))
    f.close()
    print(f'[alpha] α={alpha} 写入 {nset} 槽(从裸态快照绝对重写, 不累积)')


if __name__ == '__main__':
    main()
