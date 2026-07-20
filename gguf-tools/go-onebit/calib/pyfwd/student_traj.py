#!/usr/bin/env python3
# student_traj.py — STUDENT (1-bit experts + optional corr sidecar z) full-model
# SELF-propagated trajectory over the Go corpus, captured with the same eight
# signals as the teacher capture. Feeds:
#   - L_cls delta solve: teacher route_logits (cap v2) − student route_logits,
#     position-aligned (same corpus/chunking, teacher-forced token stream)
#   - error-feedback rounds: student ffn_in_hat → teacher targets via injection
#   - end-to-end student top1 acc / final_topk vs teacher
#
# Mechanics: patches ds4reader.read_weight to the EXACT go1b_blk semantics for
# routed experts (per-row mean|w| scale → fp16 roundtrip → sign×scale; backbone
# stays fp32 — matches the GGUF where only experts are go1b), and replicates
# moe_all with the runtime corr semantics on top:
#   raw router logits += delta  (pre-sqrtsoftplus, L>=3)
#   y_routed += Σ_{e∈sel} U·(C_e⊙(V·φ)) + topk·b + Σ beta   (φ = x | ŷ_routed)
#
# Env: DS4_HF, CORPUS, OUT (capture dir), CHUNK_LO/HI, ZDIR (optional z_L*.bin
# dir → corr on), FEAT=x|yhat (must match how z was solved), S=512.
import os, sys, struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
import dsv4_fwd as F

ZDIR = os.environ.get("ZDIR", "")
FEAT = os.environ.get("FEAT", "x")

# ---- exact go1b_blk 1-bit round-trip (row scale, fp16 roundtrip) ----------
def onebit(w):
    s = np.abs(w).mean(1, keepdims=True).astype(np.float16).astype(np.float32)
    return np.where(w >= 0, s, -s).astype(np.float32)

_orig_read = R.read_weight
def read_weight_1bit(name, *a, **kw):
    w = _orig_read(name, *a, **kw)
    if ".ffn.experts." in name and name.endswith((".w1.weight", ".w2.weight", ".w3.weight")):
        return onebit(w)
    return w
R.read_weight = read_weight_1bit

# ---- z sidecar --------------------------------------------------------------
def zload(L):
    p = os.path.join(ZDIR, f"z_L{L}.bin")
    if not ZDIR or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        d, ne, dl = struct.unpack("<iii", f.read(12))
        def rd(n): return np.frombuffer(f.read(4 * n), "<f4").reshape(-1).copy()
        U = rd(d * dl).reshape(d, dl); V = rd(dl * d).reshape(dl, d)
        C = rd(ne * dl).reshape(ne, dl); b = rd(d); beta = rd(ne); delta = rd(ne)
    return dict(U=U, V=V, C=C, b=b, beta=beta, delta=delta, dl=dl)

_Z = {}
def zget(L):
    if L not in _Z: _Z[L] = zload(L)
    return _Z[L]

# ---- moe_all with runtime corr semantics ------------------------------------
def moe_all_student(Fin, Ids, W, L):
    Z = zget(L)
    raw = (Fin.astype(np.float32) @ W['gate'].T)
    if Z is not None and W['tid2eid'] is None:
        raw = raw + Z['delta'][None]          # runtime: delta pre-sqrtsoftplus, learned-routing layers only
    scores = np.sqrt(np.log1p(np.exp(raw)))
    if W['tid2eid'] is not None:
        idx = W['tid2eid'][Ids]
    else:
        sc = scores + W['gbias'][None]
        idx = np.argpartition(-sc, F.NACT - 1, axis=1)[:, :F.NACT]
    wt = np.take_along_axis(scores, idx, 1)
    wt = wt / wt.sum(1, keepdims=True) * F.ROUTE_SCALE

    y = F.expert_fp(Fin, W['s1'], W['s3'], W['s2'])      # shared (fp backbone)
    shared = y.copy()
    flat_e = idx.reshape(-1); flat_w = wt.reshape(-1)
    tok = np.repeat(np.arange(Fin.shape[0]), F.NACT)
    for e in np.unique(flat_e):
        sel = flat_e == e
        t = tok[sel]; ww = flat_w[sel][:, None]
        w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
        np.add.at(y, t, F.expert_fp(Fin[t], w1, w3, w2, ww))

    if Z is not None:
        routed = y - shared
        phi = routed if FEAT == "yhat" else Fin.astype(np.float32)
        v = phi @ Z['V'].T                                # [n, dl]
        sC = Z['C'][idx].sum(1)                           # [n, dl] Σ_{e∈sel} C_e
        corr = (sC * v) @ Z['U'].T                        # [n, d]
        corr += F.NACT * Z['b'][None] + Z['beta'][idx].sum(1, keepdims=True)
        y = y + corr

    F.CAP['route_logits'] = raw                           # student raw (incl. delta)
    F.CAP['route_w'] = wt
    F.CAP['routed'] = (y - shared)                        # routed(+corr): student corr-target space
    return y.astype(np.float32), idx

F.moe_all = moe_all_student

if __name__ == "__main__":
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
    CORPUS = os.environ["CORPUS"]; OUT = os.environ["OUT"]
    LO = int(os.environ.get("CHUNK_LO", "0")); HI = int(os.environ.get("CHUNK_HI", "48"))
    S = int(os.environ.get("S", "512"))
    ids_all = tk.encode(open(CORPUS).read()).ids
    chunks = [np.array(ids_all[i*S:(i+1)*S], dtype=np.int64) for i in range(LO, HI)]
    chunks = [c for c in chunks if len(c) >= 64]
    print(f"student_traj chunks[{LO}:{HI}]={len(chunks)} z={'ON:'+ZDIR if ZDIR else 'OFF'} feat={FEAT}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    F.run(chunks, chunks, OUT, capture=True)
    print("STUDENT TRAJ DONE", flush=True)
