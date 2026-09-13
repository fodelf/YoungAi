"""v41_amp_hooks — 放大器的挂钩层: 只做编排, 一次矩阵乘都不在这儿算。

★为什么这个文件里没有数值★(铁律 feedback_c_not_python_shared_impl)
放大器的解算和应用全在 gguf-tools/amp/v41_amp_solve.cu 里, 本模块只负责
"什么时候调、拿哪块显存去调、结果写哪儿"。同一个公式 Z = X·(B·A) 曾经在
C 自检里写一遍、Python 里又手抄两遍, 布局差一点就是一次静默错位 ——
方向写反那次端到端 PPL 从 1.66 崩到 265, 光看日志根本看不出来。
现在 Python 侧只递 GPU 指针。

★挂点: MoE 整块(ffn 模块)★
输入 = MoE 输入 x(post-ffn_norm, router 消费的那个向量), 靶 = MoE 输出误差。
两端在同一个因果切面上, 所以靶是 x 的真函数, 低秩线性逼近才有依据。
不能挂整层: 层输出要过 attn, 而 attn 带 KV cache —— 当前 token 的输出依赖
历史所有 token 的 KV, 同一个层输入在不同位置对应不同的层输出误差, 靶就不
是函数了, 放大器多大的秩也对不上。
"""
import ctypes
import struct
from pathlib import Path

import numpy as np
import torch

from v41_amp_split import layout_split, _permute   # noqa: F401 (teacher/diag 经本模块取用)

_LIB = None


def _lib():
    """libv41amp.so: 解算 + 应用都在里面。缺了直接炸, 不做 Python 兜底 ——
    兜底会让"走了参考路"这种失真悄悄发生(memory: CPU 参考路失真 40× 且自洽)。"""
    global _LIB
    if _LIB is None:
        so = Path(__file__).resolve().parent.parent / "amp" / "libv41amp.so"
        lib = ctypes.CDLL(str(so))
        lib.v41_amp_solve_layer_gpu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + \
                                               [ctypes.c_float] + \
                                               [ctypes.c_void_p] * 2 + [ctypes.POINTER(ctypes.c_float)] + \
                                               [ctypes.c_int]                       # whiten
        lib.v41_amp_solve_layer_gpu.restype = ctypes.c_int
        lib.v41_amp_apply_gpu.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + \
                                         [ctypes.POINTER(ctypes.c_float)]
        lib.v41_amp_apply_gpu.restype = ctypes.c_int
        lib.v41_amp_scan_k_gpu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + \
                                          [ctypes.c_void_p, ctypes.c_int] + \
                                          [ctypes.c_void_p, ctypes.c_int] + \
                                          [ctypes.POINTER(ctypes.c_float)] * 3 + \
                                          [ctypes.c_int] + [ctypes.POINTER(ctypes.c_float)] * 2   # whiten, train_w, val_w
        lib.v41_amp_scan_k_gpu.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _same_x_target(mod, args, out, qmode, busy, li=-1):
    """靶的两端喂同一个 x: 关掉量化开关, 在【这一次的 x】上重跑一遍 ffn 拿 y_fp(x_q)。
    router gate 不量化 + x 相同 ⇒ 两遍选中的专家完全一致, 严格可比。
    qmode["swap"](ffn, li) 是 teacher 给的回调: 落盘学生时 shared_experts 是随层装好的 FP4,
    切到 FP 态要重装一次, 跑完再装回学生态 —— 否则靶里少了 shared 的误差(2026-09-12)。
    ★必须显式 float32★: teacher 为对齐官方代码设了 set_default_dtype(bfloat16), 不写 dtype
    的缓冲就是 bf16(半个大小), 而 C 库按 float32 往里写 ⇒ 越界写显存, 解出来的放大器还自洽。
    返回 (n, d, xf, qf, yfp), 后三个是 [n,d] 的 f32 连续张量。"""
    x = args[0]
    n, d = x.reshape(-1, x.shape[-1]).shape
    xf = x.detach().float().reshape(n, d).contiguous()
    qf = out.detach().float().reshape(n, d).contiguous()
    swap = qmode.get("swap")
    busy["v"] = True
    qmode["on"] = False
    try:
        if swap and li >= 0:
            swap(mod, li)
        with torch.no_grad():
            yfp = mod(*args).detach().float().reshape(n, d).contiguous()
    finally:
        qmode["on"] = True
        if swap and li >= 0:
            swap(mod, li)
        busy["v"] = False
    return n, d, xf, qf, yfp


def _moe_layers(net):
    """带 routed 专家的层 —— 只有这些层有放大器。"""
    for i, layer in enumerate(net.layers):
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and getattr(ffn, "experts", None) is not None:
            yield i, ffn


def install_online(net, dev, out_dir, K, qmode, lam=10.0):
    """在线序贯反修: 一次前向里逐层【解出放大器 + 立刻应用】, 解出来就落盘。

    ★为什么必须序贯★ 并行解算(所有层用同一次前向的状态)已被三次端到端实测证死:
    PPL 265 → 10311 → 99972。机理 = 正反馈耦合: L0 一修 x₁ 就变, 而 L1 的放大器是按
    未修正的 x₁ 解的 ⇒ 失配; 失配的修正让 x₂ 偏更多 ⇒ 逐层放大。
    序贯解 L_i 时 L0..L_{i-1} 已修正并传播, 此刻看到的 x 就是部署时的 x。

    ★为什么要在 hook 里重跑一遍 FP★ 靶的两端必须喂同一个 x:
        正确  y_fp(x_q) − y_q(x_q)      只有专家权重不同
        错的  y_fp(x_fp) − y_q(x_q)     混进了 x 漂移导致的输出变化
    后者那部分不是 x_q 的函数, 放大器拟合不掉(实测每层只吃到 13.6% 能量), 还会被当
    噪声背下来 ⇒ 拟合料 PPL 比值 1.051 而判决料崩到 55837。
    序贯时 x 逐层变, 没法预先 dump, 所以只能当场算: 关掉量化开关再调一次 ffn。
    router gate 不量化, x 又是同一个 ⇒ 两遍选中的专家完全一致, 严格可比。
    代价 = 每层多装载一次 FP 专家(用完即弃, 内存峰值不变)。

    ★lam★: ridge 强度, 默认 10.0 = held-out K×λ 扫描的峰值(3 层平均 λ=10~30)。
    原来硬编码 1e-3(V4 候选下端的 1/300, 近乎无正则), 实测泛化 −66.5%; 改对后 +1.8%。
    λ 是压过拟合的唯一旋钮, 别再动它之前先跑 scank 档。

    qmode: teacher 持有的 {"on": bool} 开关, materialize/do_vq 看它决定要不要量化。
    返回统计 dict(逐层结果在里面, 调用方跑完打总账)。
    """
    lib = _lib()
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    st = {"ok": 0, "skip": 0, "ratio": [], "dz": []}
    busy = {"v": False}

    def mk(li):
        def h(mod, args, out):
            if busy["v"]:
                return None              # FP 重跑自己触发的递归, 放行
            n, d, xf, qf, yfp = _same_x_target(mod, args, out, qmode, busy, li)
            qnorm = float(qf.norm())     # 必须在 apply 原地改 qf 之前量

            dA = torch.zeros(K, d, device=dev, dtype=torch.float32)
            dB = torch.zeros(d, K, device=dev, dtype=torch.float32)
            r = ctypes.c_float(0)
            rc = lib.v41_amp_solve_layer_gpu(
                ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(yfp.data_ptr()),
                ctypes.c_void_p(qf.data_ptr()), n, d, K, ctypes.c_float(lam),
                ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()),
                ctypes.byref(r), 0)
            if rc != 0:
                st["skip"] += 1
                print(f"  [L{li:02d}] ★残差没降(比 {r.value:.3f}), 跳过不挂★", flush=True)
                return None

            # dB 由 C 按列主序 D×K 写入 ⇒ 内存上就是 [K,D] 行主序(= Bᵀ), 与离线格式一致
            Bt = dB.reshape(K, d)
            with open(od / f"amp_L{li:02d}.bin", "wb") as fh:
                fh.write(struct.pack("<iii", d, K, 1))
                dA.cpu().numpy().astype("float32").tofile(fh)
                Bt.cpu().numpy().astype("float32").tofile(fh)

            dz = ctypes.c_float(0)
            lib.v41_amp_apply_gpu(ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(qf.data_ptr()),
                                  ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()),
                                  n, d, K, ctypes.byref(dz))
            # ★信任域诊断★: 不加 clip 兜底, 先把 ‖Δ‖/‖y_q‖ 量出来。p99 都很小 ⇒ 夹持是
            # 死代码, 删掉; 出长尾 ⇒ 是解算有问题要去挖, 不是夹一下了事。
            rel = dz.value / qnorm if qnorm > 0 else 0.0
            st["ok"] += 1
            st["ratio"].append(r.value)
            st["dz"].append(rel)
            print(f"  [L{li:02d}] 放大器已解: 残差/靶 = {r.value:.3f} "
                  f"(吃掉 {100 * (1 - r.value):.1f}% 能量)  ‖Δ‖/‖y_q‖ = {rel:.4f}", flush=True)
            return qf.reshape(out.shape).to(out.dtype)
        return h

    n = 0
    for i, ffn in _moe_layers(net):
        ffn.register_forward_hook(mk(i))
        n += 1
    print(f"[在线序贯反修] {n} 层挂钩, K={K}, ★λ={lam:g}★, 靶=当场重算的 y_fp(x_q) → {od}",
          flush=True)
    return st


def parse_layers(spec):
    """'0' / '0-9' / '0,5,7-9' → 层号集合; 空串 → None(全部)。给消融用: 只挂一部分层的放大器。"""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def install_apply(net, dev, amp_dir, layers=None):
    """判决态: 加载解好的放大器, 每层 MoE 输出加 X·(B·A)。应用走 C 库, 与解算同一份代码。
    layers: 只挂这些层(消融: 单挂 L00 看一层的修正是不是端到端就负); None=目录里有的全挂。"""
    lib = _lib()
    ad = Path(amp_dir)
    narm, mb_sum, ks_seen = 0, 0.0, []

    def mk(dA, dB, K):
        def h(mod, args, out):
            x = args[0]
            n, d = x.reshape(-1, x.shape[-1]).shape
            xf = x.detach().float().reshape(n, d).contiguous()
            yf = out.detach().float().reshape(n, d).contiguous()
            lib.v41_amp_apply_gpu(ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(yf.data_ptr()),
                                  ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()),
                                  n, d, K, None)
            return yf.reshape(out.shape).to(out.dtype)
        return h

    for i, ffn in _moe_layers(net):
        f = ad / f"amp_L{i:02d}.bin"
        if not f.exists() or (layers is not None and i not in layers):
            continue
        with open(f, "rb") as fh:
            D_, K_, _ = struct.unpack("<iii", fh.read(12))
            Aw = np.frombuffer(fh.read(K_ * D_ * 4), dtype=np.float32).reshape(K_, D_).copy()
            Bw = np.frombuffer(fh.read(D_ * K_ * 4), dtype=np.float32).reshape(K_, D_).copy()
        # Bw 内存是 [K,D] 行主序 = C 侧要的列主序 D×K, 指针直接递过去即可
        dA = torch.from_numpy(Aw).to(dev).contiguous()
        dB = torch.from_numpy(Bw).to(dev).contiguous()
        ffn.register_forward_hook(mk(dA, dB, K_))
        narm += 1
        ks_seen.append(K_)
        mb_sum += 2 * D_ * K_ * 2 / 1e6          # 逐层 K 可不同(择优路), 按层累加
    if narm:
        print(f"[反修] 挂上 {narm} 层放大器 (K {min(ks_seen)}~{max(ks_seen)}, "
              f"f16 部署合计 {mb_sum:.1f} MB{'' if layers is None else f', 只挂层 {sorted(layers)}'}) ← {ad}", flush=True)
    return narm


SCAN_KS = (16, 32, 64, 128, 256, 512, 1024, 2048, 5120)
# ★λ 网格★: V4 的 zloss_solve.c 用的是 {3e-3, 3e-2, 3e-1} 三档配 ranks[] 二维择优。
# V4.1 首版把 λ 焊死在 1e-3(比 V4 候选上端小 300×)只扫 K, 量出来的"train 涨 val 跌"
# 是正则缺席的形态, 被我误读成"低秩不可学"。这里扫宽一些 —— V4.1 的量化误差量级
# 比 V4 大(靶/‖y_q‖ 深层到 0.62), 可能需要比 V4 更强的正则。
# 2026-09-12 二扫: 首扫 8 档最优顶在最右端 λ=3(边界最优=没扫到头), 三层全一样。
# 往上扫到 1000 找峰值 —— λ→∞ 时解趋近 0(等于不修), val 必然先升后回落到 0%,
# 峰值在哪、多高, 就是低秩反修在 V4.1 上的真实上限。留 1e-3 作修复前的对照锚。
SCAN_LAMS = (1e-3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def _scan(lib, dev, xf, yfp, qf, nf, ks, lams, whiten=0):
    """一层的 K×λ held-out 扫描(C 库), 返回 (train%, val%, 靶/‖y_q‖[, train_w%, val_w%]) ——
    [nλ, nK] 数组, 值 = 100·(1 − ‖残差‖/‖靶‖)(幅值比, 正=有效; ★不是平方能量比★)。
    whiten≠0 时再返回白化空间(每通道除以 σ)的同口径两表 —— 择优要看白化表。
    输入行必须已按拟合在前/val 在后排好。"""
    n, d = xf.shape
    kk = (ctypes.c_int * len(ks))(*ks)
    ll = (ctypes.c_float * len(lams))(*lams)
    nkl = len(ks) * len(lams)
    tr, va, trw, vaw = ((ctypes.c_float * nkl)() for _ in range(4))
    rel = ctypes.c_float(0)
    rc = lib.v41_amp_scan_k_gpu(
        ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(yfp.data_ptr()),
        ctypes.c_void_p(qf.data_ptr()), n, d, nf,
        ctypes.cast(kk, ctypes.c_void_p), len(ks),
        ctypes.cast(ll, ctypes.c_void_p), len(lams),
        tr, va, ctypes.byref(rel), int(whiten), trw, vaw)
    if rc != 0:
        return (None, None, rel.value) + ((None, None) if whiten else ())
    pct = lambda arr: np.array([100.0 * (1.0 - arr[i]) for i in range(nkl)]).reshape(len(lams), len(ks))
    out = (pct(tr), pct(va), rel.value)
    return out + ((pct(trw), pct(vaw)) if whiten else ())


def install_scan(net, dev, qmode, ks=SCAN_KS, lams=SCAN_LAMS, nfit=6144, perm=None, whiten=0):
    """★K 扫描诊断(held-out 口径)★: 每层把 token 切两段 —— 前 nfit 个解, 其余只评估。

    要回答的问题: K 取多少泛化最好?
      曲线有峰值   ⇒ 取峰值那个 K, 端到端确认
      单调下降     ⇒ 最小的 K 都没正收益, 低秩反修在 V4.1 1.5bpw 上不通, 收工

    ★为什么必须 held-out★ 首版在拟合集自己身上量残差, 据此选了 K=1024, 端到端
    Σmin 从 0.6711 崩到 0.2431。大 K 在拟合集上永远吃得更多 —— 那就是过拟合的定义。
    nfit 必须 > d_model(5120), 否则拟合那半欠定, 量的还是假账。

    ★不应用修正★: 纯诊断。序贯应用会改掉后面层的输入, 那是另一个变量;
    这里要每层都在同一个"裸量化态传播"的基准上, 曲线才可比。
    """
    lib = _lib()
    busy = {"v": False}
    rows, rows_w = [], []

    def mk(li):
        def h(mod, args, out):
            if busy["v"]:
                return None
            n, d, xf, qf, yfp = _same_x_target(mod, args, out, qmode, busy, li)
            xf, qf, yfp = _permute(perm, dev, xf, qf, yfp)
            nf = perm["nfit"] if perm else min(nfit, n - 256)   # 前缀切至少留 256 token 评估
            if nf <= d:
                print(f"  [L{li:02d}] ★拟合段 {nf} ≤ d_model {d}, 欠定, 跳过★", flush=True)
                return None
            sc = _scan(lib, dev, xf, yfp, qf, nf, ks, lams, whiten)
            et, ev, rel = sc[:3]
            if ev is None:
                print(f"  [L{li:02d}] ★K×λ 扫描失败★", flush=True)
                return None
            rows.append(ev)
            bj, bi = np.unravel_index(int(ev.argmax()), ev.shape)
            old = ev[0, ks.index(64)] if 64 in ks else ev[0, 0]   # λ=1e-3,K=64 = 修复前的配置
            print(f"  [L{li:02d}] 靶/‖y_q‖={rel:.3f} fit={nf}/{n} | "
                  f"★val 最优 λ={lams[bj]:g} K={ks[bi]} → {ev[bj, bi]:+.1f}%★ "
                  f"(train {et[bj, bi]:+.1f}%)   [修复前 λ=1e-3 K=64: {old:+.1f}%]", flush=True)
            if whiten:
                evw = sc[4]; rows_w.append(evw)
                wj, wi = np.unravel_index(int(evw.argmax()), evw.shape)
                print(f"         白化口径: val 最优 λ={lams[wj]:g} K={ks[wi]} → {evw[wj, wi]:+.1f}% "
                      f"(同格原始口径 {ev[wj, wi]:+.1f}%; 原始最优格的白化值 {evw[bj, bi]:+.1f}%)", flush=True)
            return None          # 不改输出
        return h

    n = 0
    for i, ffn in _moe_layers(net):
        ffn.register_forward_hook(mk(i))
        n += 1
    how = f"layout 分层 {perm['nfit']} 解 / {len(perm['perm']) - perm['nfit']} 评" if perm else f"前 {nfit} token 解 / 其余评"
    print(f"[K×λ 扫描·held-out] {n} 层挂钩; K = {list(ks)}; λ = {list(lams)}; {how}; 白化={whiten} —— ★只认 val★", flush=True)
    return {"ks": list(ks), "lams": list(lams), "rows": rows, "rows_w": rows_w}


# ★逐层择优的候选格★(2026-09-12 金融反修方案): λ 是主导旋钮(全域二扫 λ=10~30 峰), K 次要;
# K 上限 512 = 4.2% 解码带宽, 再大带宽翻倍而增量只剩零点几个点(全域 K 取舍表)。
SEL_KS = (64, 128, 256, 512)
SEL_LAMS = (1.0, 3.0, 10.0, 30.0, 100.0)


def _choose(ev, ks, lams, keep):
    """择优规则: val ≥ keep×最优 的格里取【最小 K】, 同 K 取【最大 λ】(最强正则)。
    为什么不直接取最优格: 全域二扫里 λ=10 K=5120 (+2.3%) 比 K=128 (+2.0%) 只多 0.3pp,
    带宽却贵 40 倍 —— 最优点不是经济点。keep=0.9 就是"最优的九成之内挑最便宜的"。"""
    best = float(ev.max())
    ok = [(ki, lj) for lj in range(len(lams)) for ki in range(len(ks)) if ev[lj, ki] >= keep * best]
    ki = min(k for k, _ in ok)
    lj = max(l for k, l in ok if k == ki)
    return best, ki, lj


def install_online_select(net, dev, out_dir, qmode, perm, ks=SEL_KS, lams=SEL_LAMS,
                          gate=0.5, keep=0.9, whiten=0):
    """★在线序贯反修·逐层择优★: 每层【扫 → 选 → 解 → 应用】, 一次前向走完 40 层。
      扫: K×λ 分层 held-out 扫描(拟合行解 / val 行只评)
      选: _choose 规则; 层最优 val < gate(%) ⇒ 收益闸, 该层不挂(V4 zlayer 同款)
      解: 用选中的 (λ,K) 在【拟合行】上解 —— val 行永不进解算, 逐层 val 就是诚实的 held-out
      用: 应用到全部行(含 val 行, 让后面层看到部署态的 x), 落 amp_L.bin + manifest 一行
    序贯的必要性与靶的口径见 install_online 的注释, 不重复。"""
    lib = _lib()
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mf = od / "manifest.txt"
    mf.write_text("# 层 λ K 靶/‖y_q‖ train% val% ‖Δ‖/‖y_q‖ [val_w%]  (val<gate 的层记 skip; 数值来自 C 库;"
                  " 百分比=1−‖残差‖/‖靶‖ 幅值比)\n"
                  f"# 候选 K={list(ks)} λ={list(lams)} gate={gate}% keep={keep} whiten={whiten}"
                  f"{'(择优看白化 val)' if whiten else ''}\n")
    st = {"ok": 0, "skip": 0, "ratio": [], "dz": [], "val": [], "sel": []}
    busy = {"v": False}
    if perm is None:
        raise SystemExit("★逐层择优要 layout 分层 held-out, <ids>.layout 缺失★")
    nf = perm["nfit"]

    def mk(li):
        def h(mod, args, out):
            if busy["v"]:
                return None
            n, d, xf, qf, yfp = _same_x_target(mod, args, out, qmode, busy, li)
            qnorm = float(qf.norm())
            xp, qp, yp = _permute(perm, dev, xf, qf, yfp)
            sc = _scan(lib, dev, xp, yp, qp, nf, ks, lams, whiten)
            et, ev, rel = sc[:3]
            if ev is None:
                print(f"  [L{li:02d}] ★扫描失败, 该层不挂★", flush=True)
                st["skip"] += 1
                return None
            # 白化时按白化 val 择优(原始能量 val 照记, 两口径并排才看得见分叉)
            evsel = sc[4] if whiten else ev
            best, ki, lj = _choose(evsel, ks, lams, keep)
            K, lam = ks[ki], lams[lj]
            if best < gate:
                st["skip"] += 1
                with mf.open("a") as fh:
                    fh.write(f"L{li:02d} skip - {rel:.4f} - {best:+.2f} -\n")
                print(f"  [L{li:02d}] 靶/‖y_q‖={rel:.3f} | val 最优 {best:+.2f}% < 闸 {gate}%, 不挂", flush=True)
                return None
            # 解: 只用拟合行(xp/yp/qp 的前 nf 行是连续的, 直接切片)
            dA = torch.zeros(K, d, device=dev, dtype=torch.float32)
            dB = torch.zeros(d, K, device=dev, dtype=torch.float32)
            r = ctypes.c_float(0)
            rc = lib.v41_amp_solve_layer_gpu(
                ctypes.c_void_p(xp[:nf].data_ptr()), ctypes.c_void_p(yp[:nf].data_ptr()),
                ctypes.c_void_p(qp[:nf].data_ptr()), nf, d, K, ctypes.c_float(lam),
                ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()), ctypes.byref(r), int(whiten))
            if rc != 0:
                st["skip"] += 1
                print(f"  [L{li:02d}] ★解算自检不过(残差比 {r.value:.3f}), 不挂★", flush=True)
                return None
            Bt = dB.reshape(K, d)
            with open(od / f"amp_L{li:02d}.bin", "wb") as fh:
                fh.write(struct.pack("<iii", d, K, 1))
                dA.cpu().numpy().astype("float32").tofile(fh)
                Bt.cpu().numpy().astype("float32").tofile(fh)
            dz = ctypes.c_float(0)
            lib.v41_amp_apply_gpu(ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(qf.data_ptr()),
                                  ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()),
                                  n, d, K, ctypes.byref(dz))
            relz = dz.value / qnorm if qnorm > 0 else 0.0
            tr_pct = 100 * (1 - r.value)
            st["ok"] += 1; st["ratio"].append(r.value); st["dz"].append(relz)
            st["val"].append(float(evsel[lj, ki])); st["sel"].append((li, lam, K))
            wtxt = f" {sc[4][lj, ki]:+.2f}" if whiten else ""
            with mf.open("a") as fh:
                fh.write(f"L{li:02d} {lam:g} {K} {rel:.4f} {tr_pct:+.2f} {ev[lj, ki]:+.2f} {relz:.4f}{wtxt}\n")
            print(f"  [L{li:02d}] 靶/‖y_q‖={rel:.3f} | 选 λ={lam:g} K={K}: train {tr_pct:+.1f}% "
                  f"(扫 {et[lj, ki]:+.1f}%) val {ev[lj, ki]:+.1f}%"
                  + (f" 白化val {sc[4][lj, ki]:+.1f}% [白化最优 {best:+.1f}%]" if whiten else f" [最优 {best:+.1f}%]")
                  + f"  ‖Δ‖/‖y_q‖={relz:.4f}", flush=True)
            return qf.reshape(out.shape).to(out.dtype)
        return h

    nl = 0
    for i, ffn in _moe_layers(net):
        ffn.register_forward_hook(mk(i))
        nl += 1
    print(f"[在线序贯反修·择优] {nl} 层挂钩; K={list(ks)} λ={list(lams)}; 分层 held-out "
          f"{nf} 解 / {len(perm['perm']) - nf} 评; 闸 {gate}% keep {keep}; 白化={whiten} → {od}", flush=True)
    return st


def report_select(st, out_dir):
    """择优路总账 + manifest 收尾行(judge 脚本认这一行判"解完了", 半成品目录会被重解)。"""
    report(st)
    if st["val"]:
        v = np.array(st["val"])
        ks = [k for _, _, k in st["sel"]]
        mb = sum(2 * 5120 * k * 2 for k in ks) / 1e6
        print(f"[择优] 挂 {st['ok']} 层 / 闸掉 {st['skip']} 层; val 平均 {v.mean():+.2f}% "
              f"(中位 {np.median(v):+.2f}%, 最好 {v.max():+.2f}%); K 分布 "
              + " ".join(f"{k}×{ks.count(k)}" for k in sorted(set(ks)))
              + f"; f16 部署 {mb:.0f} MB = {mb / 10050 * 100:.2f}% 解码带宽", flush=True)
    with (Path(out_dir) / "manifest.txt").open("a") as fh:
        fh.write(f"# 完成 挂{st['ok']} 闸{st['skip']}\n")


def report_scan(st):
    """跨层平均的 K↔质量曲线 + 体积/带宽账 —— 取舍要这两栏并排看才成立。"""
    if not st["rows"]:
        print("[K 扫描] 零层 —— 检查挂钩", flush=True)
        return
    a = np.array(st["rows"])                      # [层, λ, K]
    m = a.mean(axis=0)                            # [λ, K] 跨层平均
    ks, lams = st["ks"], st["lams"]
    print(f"\n[K×λ 扫描汇总] {a.shape[0]} 层平均吃掉的误差能量 ★held-out★ (正=有效):",
          flush=True)
    print("      λ\\K " + "".join(f"{k:>9}" for k in ks), flush=True)
    for j, lm in enumerate(lams):
        print(f"  {lm:>8g} " + "".join(f"{m[j, i]:>8.1f}%" for i in range(len(ks))), flush=True)
    bj, bi = np.unravel_index(int(m.argmax()), m.shape)
    mb = 2 * 5120 * ks[bi] * 2 / 1e6
    print(f"\n  ★跨层最优: λ={lams[bj]:g} K={ks[bi]} → {m[bj, bi]:+.1f}%★  "
          f"({mb*40:.0f} MB 全模型, +{mb*40*8e6/5.44e11:.4f} bpw, "
          f"{mb*40/10050*100:.2f}% 解码带宽)", flush=True)
    print(f"  修复前的配置 λ=1e-3 K=64: {m[0, ks.index(64)] if 64 in ks else float('nan'):+.1f}%",
          flush=True)
    if st.get("rows_w"):
        mw = np.array(st["rows_w"]).mean(axis=0)
        print(f"\n[K×λ 扫描汇总·白化口径] {len(st['rows_w'])} 层平均(每通道除以 σ 后的 1−‖残差‖/‖靶‖):", flush=True)
        print("      λ\\K " + "".join(f"{k:>9}" for k in ks), flush=True)
        for j, lm in enumerate(lams):
            print(f"  {lm:>8g} " + "".join(f"{mw[j, i]:>8.1f}%" for i in range(len(ks))), flush=True)
        wj, wi = np.unravel_index(int(mw.argmax()), mw.shape)
        print(f"  ★白化跨层最优: λ={lams[wj]:g} K={ks[wi]} → {mw[wj, wi]:+.1f}%★", flush=True)


def report(st):
    """跑完的总账: 吃掉多少能量 + 修正幅值分布(信任域该不该有的依据)。"""
    if not st["ok"]:
        print("[序贯反修] 零层解出 —— 检查靶/输入是不是喂错了", flush=True)
        return
    rr = np.array(st["ratio"])
    dz = np.array(st["dz"])
    print(f"[序贯反修] 解出 {st['ok']} 层 / 跳过 {st['skip']} 层; "
          f"平均吃掉 {100 * (1 - rr.mean()):.1f}% 能量 (最好 {100 * (1 - rr.min()):.1f}%)",
          flush=True)
    print(f"[信任域诊断] ‖Δ‖/‖y_q‖: 中位 {np.median(dz):.4f}  最大 {dz.max():.4f}  "
          f"p90 {np.percentile(dz, 90):.4f}", flush=True)
