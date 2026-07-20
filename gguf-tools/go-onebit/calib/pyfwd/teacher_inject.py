#!/usr/bin/env python3
# teacher_inject.py — error-feedback TARGETS: for each layer, run the TEACHER's
# routed MoE on the STUDENT's captured layer inputs (x̂ from student_traj) and
# overwrite routed_L{L}.npy in the student cap dir with teacher(x̂). After this
# the student cap has exactly what calib_run's rrr assembly needs to re-solve
# the layer ON THE STUDENT TRAJECTORY:
#   ffn_in = x̂ (student), route/route_w = student routing (what the runtime corr
#   will actually see), routed = teacher-on-x̂ (the restoration target).
# Learned-routing layers only (L>=3): teacher selection+weights come from the
# teacher gate on x̂ inside moe_all; token ids are irrelevant there.
# Usage: DS4_HF=... venv/bin/python teacher_inject.py --cap STUDENT_CAP --layers 25,26,...
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
import dsv4_fwd as F

def arg(k, d=None):
    return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d

cap = arg("--cap")
layers = [int(t) for t in arg("--layers").split(",")]
for L in layers:
    x = np.load(f"{cap}/ffn_in_L{L}.npy").astype(np.float32)
    if os.environ.get("PAIRED"):
        # 部署配对模式: 教师专家由采集态路由+采集态 gate 权重驱动 (EF 口径).
        # 重路由版 targets 在深层与部署配对错位 (权重值漂移+~5%换专家 → L38 上界 cos 0.37), 修尺用此模式.
        idx = np.load(f"{cap}/route_L{L}.npy").astype(np.int64)
        wt = np.load(f"{cap}/route_w_L{L}.npy").astype(np.float32)
        n = min(len(x), len(idx)); xs, idx, wt = x[:n], idx[:n], wt[:n]
        y = np.zeros((n, xs.shape[1]), dtype=np.float32)
        Kp = idx.shape[1]
        flat_e = idx.reshape(-1); tok = np.repeat(np.arange(n), Kp); flat_w = wt.reshape(-1)
        for e in np.unique(flat_e):
            sel = flat_e == e; t = tok[sel]; ww = flat_w[sel][:, None]
            w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
            w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
            w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
            np.add.at(y, t, F.expert_fp(xs[t], w1, w3, w2, ww))
        np.save(f"{cap}/routed_paired_L{L}.npy", y.astype(np.float16))
        print(f"L{L}: PAIRED teacher routed saved (n={n})", flush=True)
        continue
    W = F.load_layer(L)
    if L >= 3:
        dummy_ids = np.zeros(x.shape[0], dtype=np.int64)   # unused when tid2eid is None
        F.moe_all(x, dummy_ids, W, L)
        np.save(f"{cap}/routed_L{L}.npy", F.CAP['routed'].astype(np.float16))
        # teacher RAW router logits on the SAME x̂ — δ (L_cls) aligns these against
        # the student's captured route_logits positionally within this cap dir.
        np.save(f"{cap}/route_logits_teacher_L{L}.npy", F.CAP['route_logits'].astype(np.float16))
        print(f"L{L}: teacher routed+logits on student x̂ saved", flush=True)
    else:
        # Hash layers: selection is token-id-driven (input-independent) and the
        # router is never quantized, so the CAPTURED student ids+weights on the
        # same tokens/x̂ ARE the teacher's — drive the teacher experts directly
        # with them instead of re-routing (no token ids in the cap). δ does not
        # apply (runtime skips router bias on hash layers).
        idx = np.load(f"{cap}/route_L{L}.npy").astype(np.int64)      # [Nt,6]
        wt  = np.load(f"{cap}/route_w_L{L}.npy").astype(np.float32)  # [Nt,6]
        Nt, nact = idx.shape
        y = np.zeros((Nt, x.shape[1]), dtype=np.float32)
        flat_e = idx.reshape(-1); flat_w = wt.reshape(-1)
        tok = np.repeat(np.arange(Nt), nact)
        for e in np.unique(flat_e):
            sel = flat_e == e
            t = tok[sel]; ww = flat_w[sel][:, None]
            w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
            w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
            w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
            np.add.at(y, t, F.expert_fp(x[t], w1, w3, w2, ww))
        np.save(f"{cap}/routed_L{L}.npy", y.astype(np.float16))
        print(f"L{L}: teacher routed (hash: captured ids/w) on student x̂ saved", flush=True)
    del W
print("TEACHER-INJECT DONE", flush=True)
