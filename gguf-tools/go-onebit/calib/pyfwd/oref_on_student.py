#!/usr/bin/env python3
# oref_on_student.py — corr 快校准数据面(2026-07-27): error-feedback 语义的 teacher 参考。
# 对每层: 取学生引擎轨迹的真实 FFN 输入 x̂(DS4_CAP_DIR raw_ffn_in, f16), 喂 teacher
# 原始权重的 MoE(F.moe_all) → o_ref = "该层在学生输入上本应输出什么"。
# corr 目标 E = o_ref − o_student(raw_ffn_out)。(承 precompute_obase.py 形态)
# 用法: DS4_HF=... CAP=/tmp/capV3_all IDS=/tmp/rr_calib_prog_v3.withbos.ids \
#       OUT=/tmp/orefV3 NROWS=932 python3 oref_on_student.py
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
import dsv4_fwd as F

CAP = os.environ["CAP"]
OUT = os.environ["OUT"]
NROWS = int(os.environ["NROWS"])           # 只取 prefill 行(去掉尾部 decode 行)
ids = np.array([int(l) for l in open(os.environ["IDS"]) if l.strip()], dtype=np.int64)
assert len(ids) >= NROWS, f"ids {len(ids)} < NROWS {NROWS}"
ids = ids[:NROWS]
os.makedirs(OUT, exist_ok=True)
print(f"oref_on_student: NROWS={NROWS} CAP={CAP} -> {OUT}", flush=True)

for L in range(43):
    x = np.fromfile(f"{CAP}/raw_ffn_in_L{L}", dtype=np.float16).reshape(-1, 4096)[:NROWS].astype(np.float32)
    W = F.load_layer(L)
    o_ref, _ = F.moe_all(x, ids, W, L)
    np.save(f"{OUT}/oref_L{L}.npy", o_ref.astype(np.float16))
    del W
    print(f"L{L:2d} oref {o_ref.shape}", flush=True)
print("ALL OREF SAVED", flush=True)
