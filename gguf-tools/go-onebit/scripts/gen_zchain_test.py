#!/usr/bin/env python3
# gen_zchain_test.py — 合成 DQZ2 侧车(引擎插桩验证用, 非真实校准产物)。
# 全 NL 层各一条 GL(type1) op:
#   g=1.0 → λ≡1 恒等: 引擎带/不带 --zchain 输出必须逐字一致(插桩正确性判据)
#   g≠1.0 → 机制生效证明: 输出必须改变
# 用法: gen_zchain_test.py OUT.bin [g=1.0] [nl=43]
import struct, sys

out = sys.argv[1]
g = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
nl = int(sys.argv[3]) if len(sys.argv) > 3 else 43
with open(out, "wb") as f:
    f.write(struct.pack("<II", 0x325A5144, nl))          # "DQZ2", n_layer
    for L in range(nl):
        f.write(struct.pack("<II", L, 1))                # L, n_ops=1
        f.write(struct.pack("<IIf", 1, 4, g))            # type=GL, paysz=4, g
print(f"wrote {out}: {nl} layers x GL g={g}")
