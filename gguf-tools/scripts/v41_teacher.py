#!/usr/bin/env python3
"""v41_teacher.py — V4.1 FP 教师前向(流式权重), 五指标的教师端(2026-09-11)。

【定位】金标/教师夹具, 不是数值链的一部分。生产路(量化器、部署)一律 C；本文件的唯一
职责是产出**可信的参考 logits**, 因为官方 inference/ 是 V4.1 唯一的 ground truth 实现,
自己写一份 C 前向再拿它当判官, 等于用自己的近似给自己打分(本仓 08-24 "尺子事故"的教训)。

【为什么要流式】模型 510 GB, 机器 128 GB。但 Transformer.forward 是干净的层循环
(每层只传 h 和 pre_mix), 所以给每个加载单元挂 forward hook: 进层前把权重从盘搬上 GPU,
出层后释放。一层约 7 GB(专家 6.8 打包态 + attn 0.13), 峰值远低于内存。
★不重写 forward 逻辑★ —— 重写就会引入语义漂移, 而漂移会伪装成"量化损失"。

【为什么不用官方 tilelang kernel】官方 fp4/fp8 GEMM 会把**激活**也量化成 fp8。那是推理
优化, 对判决是额外噪声: 教师和学生必须走同一条前向路, 差异才纯粹来自权重量化。所以
两边都走 dequant + 全精度 matmul —— 与 V4 时代 caliper 参考前向同口径。

【engram 特殊处理】它的表 98 GB, 不能整层加载; 但每 token 只查 24 行, 所以换成 numpy
memmap 按行取。取到的行与 C 读器逐位对拍过(v41_dequant_parity.py 全绿)。

用法: v41_teacher.py <hf-dir> --ids <ids文件> [--layers N] [--out logits.bin]
      ids 文件: 一行一个 token id(与本仓 wt2.ids 同格式)
"""
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_hf_io import FP4_TABLE, build_index, load_raw, dq_fp8, dq_fp4, has_vq, load_vq, load_q4k
import v41_patches as P
from v41_cli import parse_args


def main():
    a = parse_args()              # 参数声明在 v41_cli.py

    hf = Path(a.hf)
    sys.path.insert(0, str(hf / "inference"))
    torch.set_grad_enabled(False)
    # ★官方 generate.py:118 就设这个★。model.py 里大量 buffer 用 torch.zeros(...) 不带 dtype
    # (KV cache / indexer 的 k_cache 等), 默认 float32 的话会在 Indexer 的
    # einsum("bshd,btd->bsht", q, index_k) 处报 "expected BFloat16 but found Float" ——
    # q 来自 Linear(bf16), index_k 来自那些 buffer。必须在建模前设。
    torch.set_default_dtype(torch.bfloat16)

    t0 = time.time()
    idx = build_index(hf)
    nvq = sum(1 for k in idx if k.endswith(".vq.cb"))
    print(f"[索引] {len(idx)} 个张量, {time.time()-t0:.1f}s" + (f"; 落盘 VQ 专家 {nvq} 个(量化目录)" if nvq else ""), flush=True)
    if nvq and (a.vq_nc or a.qnbit):
        raise SystemExit("★量化目录上不能再叠 --vq-nc/--qnbit: 那是对已量化的权重二次量化★")
    # ★两份索引★: idx = 学生态(量化目录或 HF), idx_fp = 教师态。反修的靶要在同一个 x 上
    # 重跑 FP 的 ffn —— 落盘学生的目录里没有 FP 权重, 所以 qmode 关时 ffn 前缀改从 idx_fp 装。
    # 原地量化路(HF 目录 + --vq-nc)两份就是同一份, qmode 只管要不要过 VQ, 行为不变。
    idx_fp = idx
    if a.fp_dir:
        idx_fp = build_index(Path(a.fp_dir))
        print(f"[索引·FP] {len(idx_fp)} 个张量 ← {a.fp_dir}(反修靶的教师侧)", flush=True)
    elif nvq and (a.amp_online or a.amp_select or a.amp_scan_k or a.amp_rowdiag or a.amp_t2diag):
        raise SystemExit("★量化目录上反修必须给 --fp-dir: 靶 y_fp(x_q) 的 FP 权重不在量化目录里★")

    import model as M
    # ★出口只要 logits, 不要官方 sample()★(2026-09-30 实撞): 官方 forward 末尾 `sample(logits, temperature)` 在温度 > 0 时做
    # `logits / t` 与 `softmax(dtype=float32)` 两份全词表副本 —— 47,500 位的判决料 logits 本身 24.6 GB, 三份 74 GB, 40 层刚跑完
    # 就把 121 GB 机器压到 available 5 GB, 看门狗把 31 分钟的教师前向杀在出口(24,400 位时三份 38 GB 才没撞)。
    # 这里把它换成 argmax(零副本): output_ids 教师端从不使用, logits 一个字节不变。
    M.sample = lambda logits, temperature=1.0: logits.argmax(dim=-1)

    dev = "cuda"
    tbl = torch.tensor(FP4_TABLE, dtype=torch.float32, device=dev)

    # VQ 库: ★算法在 C/CUDA★, 这里只递 GPU 指针
    vqlib = None
    if a.vq_nc:
        import ctypes
        so = Path(__file__).resolve().parent.parent / "quantize" / "libv41vq.so"
        vqlib = ctypes.CDLL(str(so))
        vqlib.v41_vq_expert_gpu.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                            ctypes.POINTER(ctypes.c_int),
                                            ctypes.POINTER(ctypes.c_int),
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int]
        vqlib.v41_vq_expert_gpu.restype = ctypes.c_int
        bits = max(1, int(np.ceil(np.log2(a.vq_nc))))
        cbbpw = a.vq_nc * a.vq_dim * 2 * 8 / (3 * 2304 * 5120)   # 每专家一份码本摊到它的三个矩阵
        print(f"[VQ] dim={a.vq_dim} nc={a.vq_nc} ⇒ {bits}bit/{a.vq_dim}元素 = {bits/a.vq_dim:.4f} bpw"
              f" + 码本 {cbbpw:.4f} + 行增益 0.0031 = ★{bits/a.vq_dim + cbbpw + 0.0031:.4f} bpw★"
              f"  (迭代 {a.vq_iters} 轮, 训练采样 1/{a.vq_stride})", flush=True)

    # 校准列权: f32[nlayer][dim], 每层一行。归一化到均值 1 —— 绝对尺度不影响 argmin,
    # 但归一化能让不同层的权重量级一致, 便于比对与排错。
    colw = None
    if a.act:
        with open(a.act, "rb") as f:
            nl, nd = struct.unpack("<ii", f.read(8))
            colw = np.frombuffer(f.read(nl * nd * 4), dtype=np.float32).reshape(nl, nd).copy()
        colw = colw / colw.mean(axis=1, keepdims=True)
        print(f"[校准] 列权 {nl} 层 × {nd} 通道 ← {a.act}; "
              f"峰均比 中位 {float(np.median(colw.max(1))):.1f}", flush=True)

    # 量化库: ★算法在 C/CUDA★, 这里只把 GPU 指针递进去(全仓零 Python 数值链)
    qlib = None
    if a.qnbit:
        import ctypes
        so = Path(__file__).resolve().parent.parent / "quantize" / "libv41quant.so"
        qlib = ctypes.CDLL(str(so))
        qlib.v41_quant_dequant_gpu.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
        qlib.v41_quant_dequant_gpu.restype = ctypes.c_int
        qlib.v41_set_codebook.argtypes = [ctypes.c_void_p, ctypes.c_int]
        qlib.v41_set_codebook.restype = ctypes.c_int
        if not a.qcb:
            raise SystemExit("★要量化必须给 --qcb(v41_codebook 求的码本)★: 高斯假设的码本"
                             " 在 V4.1 上会白吃一大截误差(4.25 bpw 都能到 KLD 0.41)")
        with open(a.qcb, "rb") as f:
            kk = struct.unpack("<i", f.read(4))[0]
            cb = np.frombuffer(f.read(kk * 4), dtype=np.float32).copy()
        if kk != (1 << a.qnbit):
            raise SystemExit(f"★码本 {kk} 点 != 2^{a.qnbit}★")
        if qlib.v41_set_codebook(cb.ctypes.data_as(ctypes.c_void_p), kk) != 0:
            raise SystemExit("★码本上传失败★")
        print(f"[码本] {kk} 点 ← {a.qcb}: " + " ".join(f"{v:.4f}" for v in cb[:8])
              + (" ..." if kk > 8 else ""), flush=True)
        print(f"[量化] 主干专家 {a.qnbit} bit + 每 {a.qblk} 元素 ue8m0 scale "
              f"= {a.qnbit + 8.0/a.qblk:.4f} bpw (MTP 专家按配方保 FP4 不动)"
              f"{'' if a.qpow2 else '  ★诊断模式: 精确 σ, 不是可落地格式★'}", flush=True)

    # ---- 官方 model.py 的三处补丁(linear fp32 / sparse_attn torch 版 / engram 流式), 见 v41_patches ----
    P.patch_linear(M)
    P.patch_sparse_attn(M)
    P.patch_indexer(M)   # 索引分数按块算(2026-09-22): 24k token 的判决料一次性物化 [s,h,t] 会吃干 121 GB

    # ---- 建模型: meta device, 不分配任何权重 ----
    cfg = json.loads((hf / "inference" / "config.json").read_text())
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})
    if a.layers:
        args.n_layers = a.layers
        args.dspark_block_size = 0        # 只跑前几层时不建 MTP
    # ★max_seq_len 必须盖住 --ntok★: ModelArgs 里它硬编码默认 4096, 而 config.json 根本
    # 没有这个字段(只有 max_position_embeddings=1048576), 所以建模拿到的一直是 4096。
    # engram / indexer / KV 三种 cache 都按它分配, 超了会在 engram.py 里报
    # "The expanded size of the tensor (4096) must match the existing size (8192)" ——
    # 而且是前向跑到一半才炸, 前面几分钟的装载全白费。
    if a.ntok > args.max_seq_len:
        args.max_seq_len = a.ntok
    print(f"[配置] {args.n_layers} 层 / dim {args.dim} / {args.n_routed_experts} 专家 "
          f"top-{args.n_activated_experts} / max_seq_len {args.max_seq_len}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
    with torch.device("meta"):
        net = M.Transformer(args, tok)
    print(f"[建模] meta 完成 {time.time()-t0:.1f}s", flush=True)

    # ---- 修 buffers(meta 建模后算出来的常量全丢了, 见 v41_patches.fix_buffers) ----
    nfix, nfreq = P.fix_buffers(M, net, args, tok, dev)
    print(f"[buffer] 修 {nfix} 个(其中 freqs_cis {nfreq} 个重算), engram 哈希态重建", flush=True)

    # --no-engram(对拍夹具): 直接摘模块, Transformer.forward 里 `if layer.engram is not None` 自然跳过
    if a.no_engram:
        ne = sum(1 for l in net.layers if getattr(l, "engram", None) is not None)
        for l in net.layers:
            l.engram = None
        print(f"[★no-engram★] 摘掉 {ne} 个 engram 层 —— 对拍口径, 非生产", flush=True)
    # engram 表换成流式版(在 meta 上建好后替换, 它自带真实数据)
    for i, layer in enumerate(net.layers):
        if getattr(layer, "engram", None) is not None:
            nw = f"layers.{i}.engram.embed.weight"
            _, _, _, shp = idx[nw]
            # fp8_block_size 是 model.py 的模块级全局(=32), 不是 ModelArgs 字段
            layer.engram.embed = P.StreamEngramEmbed(idx, dev, nw, f"layers.{i}.engram.embed.scale",
                                                     shp[0], shp[1], M.fp8_block_size)
            print(f"  L{i} engram 表 {shp[0]}×{shp[1]} → 流式", flush=True)

    # ---- 流式 hook: 进模块前装权重, 出模块后卸 ----
    stat = [0, 0, 0]  # [量化过的矩阵数, 参数量, 被路由到的专家矩阵数] —— 收尾时打, 证明真的量化了而不是静默跳过
    # ★专家装载时要不要量化★: 序贯反修要在同一个 x 上再跑一遍 FP 专家拿靶
    # (v41_amp_hooks.install_online), 那一遍必须装原始权重。关掉时 stat 也不计,
    # 否则"被路由到的专家矩阵数"会翻倍, 路由对照就废了。
    qmode = {"on": True}

    def pick(full):
        """这个张量从哪份索引装: qmode 关(= 反修在算靶的 FP 那一遍)时, ffn 前缀走教师索引。
        ★只有 ffn★: 靶的挂点是 MoE 块出口, 块内 routed(VQ) 与 shared(FP4) 两处误差都要算进靶
        (部署时块出口就是这个差); attn/norm/gate 不在块内, 两遍都用学生态, 保证 x 同一个。"""
        if not qmode["on"] and full.startswith("layers.") and ".ffn." in full:
            return idx_fp
        return idx

    def names_of(mod, prefix, skip):
        """这个加载单元自己负责的参数全名。skip 用来把 engram 摘出去 —— ★它在
        Transformer.forward 里是先于 layer(...) 单独调用的★, 所以必须自带 hook,
        否则 layer 的 pre-hook 还没触发, engram 的权重还在 meta(实撞: 报
        'Tensor on device meta is not on the expected device cuda:0')。"""
        out = []
        for n, _ in mod.named_parameters(recurse=True):
            full = f"{prefix}.{n}" if prefix else n
            if skip and n.startswith(skip):     # skip 是前缀元组, startswith 直接吃元组
                continue
            ix = pick(full)
            if full in ix or has_vq(ix, full):     # has_vq: 量化目录里专家只有 .vq.* 三件
                out.append(full)
        return out

    def materialize(mod, prefix, skip):
        """★装载即 dequant 成 bf16★。原本只 patch linear() 就够, 但 model.py 里还有直接
        拿权重做 einsum 的地方(Attention 的 o_groups 分组投影: einsum("bsgd,grd->bsgr", o, wo_a)),
        einsum 不认 fp8 dtype ⇒ 报 "expected scalar type BFloat16 but found Float8_e4m3fn"。
        与其逐个追着 patch, 不如在入口就统一成 bf16 —— 而且这一步【无损】: fp8 e4m3(3 位尾数)
        和 fp4 e2m1 的值乘上 2 的幂 scale, bf16(8 位尾数)都能精确表示。
        代价是一层专家从 6.8 GB(打包态)涨到 27 GB(bf16), 128 GB 统一内存扛得住。"""
        for full in names_of(mod, prefix, skip):
            if full.endswith(".scale"):
                continue                      # scale 只跟着 weight 用, 不单独装
            ix = pick(full)
            sn = full.rsplit(".", 1)[0] + ".scale"
            is_vq = full not in ix                    # 只剩 VQ 三件 ⇒ 落盘 VQ 产物, 解码核在 C
            dt = "VQ" if is_vq else ix[full][2]
            # q4_K 骨架(2026-09-19 的 100 GB 档): 块自带 f16 主 scale/min, 没有 sibling .scale ——
            # 所以它进不了下面那条"有 scale 才 dequant"的路, 得单独解, 但之后与 fp4/fp8 同等对待。
            is_q4k = dt == "Q4_K"
            if is_vq:
                t = load_vq(ix, full, dev)            # 当作量化权重: 下面 dtype/统计口径与 fp4 专家同
            elif is_q4k:
                t = load_q4k(ix, full, dev)
            else:
                t = load_raw(ix, full, dev)
            if sn in ix or is_vq or is_q4k:
                if not is_vq and not is_q4k:
                    sc = load_raw(ix, sn, dev)
                    t = dq_fp4(t, sc, tbl) if dt == "I8" else dq_fp8(t, sc)
                # ★只量化主干 routed 专家★: MTP 的 128 专家按 108 GB 配方保 FP4(草稿质量
                # 直接决定投机接受率, 而它只占专家参数 2.4%), engram/attn/骨架另有档位。
                if qmode["on"] and full.startswith("layers.") and ".ffn.experts." in full:
                    stat[2] += 1          # 被路由到的专家矩阵数(教师态也统计, 作路由对照)
                if qlib is not None and qmode["on"] and full.startswith("layers.") and ".ffn.experts." in full:
                    t = t.contiguous()
                    rc = qlib.v41_quant_dequant_gpu(ctypes.c_void_p(t.data_ptr()),
                                                    t.numel(), a.qnbit, a.qblk, a.qpow2)
                    if rc != 0:
                        raise RuntimeError(f"量化失败 rc={rc} @{full}")
                    stat[0] += 1
                    stat[1] += t.numel()
                if not (a.exact_weights and ".wo_a." not in sn):   # --exact-weights: 精确值留 f32(wo_a 例外, 见下)
                    t = t.to(torch.bfloat16)
            sub, _, leaf = full.rpartition(".")
            owner = net.get_submodule(sub) if sub else net
            # ★保持模型建立时声明的 dtype★: 量化 dtype(fp8/fp4)的必须 dequant 成 bf16, 其余
            # 照原样 —— ParallelHead 声明的是 float32 且 forward 里写死 F.linear(x.float(), w),
            # 装成 bf16 会报 "expected mat1 and mat2 to have the same dtype"。
            old = getattr(owner, leaf, None)
            if old is not None and old.dtype in (torch.float32, torch.bfloat16, torch.float16):
                # --exact-weights: 量化权重(fp4/fp8/VQ)留 f32 精确值; linear_dq 本就 f32 算, 输出 dtype 由
                # _was_quant 决定(bf16), 所以模型其余 dtype 流不变, 只是权重不再被多舍一次 bf16
                # wo_a 走 einsum(官方 model.py "bsgd,grd->bsgr"), 没有 linear_dq 接它, 必须留 bf16 否则 dtype 不配
                if not (a.exact_weights and (sn in ix or is_vq or is_q4k) and ".wo_a." not in sn):
                    t = t.to(old.dtype)
            par = torch.nn.Parameter(t, requires_grad=False)
            par._was_quant = sn in ix or is_vq or is_q4k   # 原本是 fp4/fp8/VQ/q4_K => linear 按 kernel 口径出 bf16
            setattr(owner, leaf, par)

    def swap_ffn(ffn, li):
        """反修在 hook 里切 qmode 后调用: 把 ffn 里【层级装的】权重(shared_experts / gate)按当前
        qmode 重装。routed 专家是逐专家 hook 现装现卸, 不用管; 但 shared 是随 layer pre-hook
        一次装好的, 不重装的话 FP 那一遍跑的还是学生态 FP4 shared, 靶就少了 shared 的误差。"""
        materialize(ffn, f"layers.{li}.ffn", ("experts.",))
    qmode["swap"] = swap_ffn

    def release(mod, prefix, skip):
        for full in names_of(mod, prefix, skip):
            sub, _, leaf = full.rpartition(".")
            owner = net.get_submodule(sub) if sub else net
            setattr(owner, leaf, torch.nn.Parameter(
                torch.empty(0, device="meta"), requires_grad=False))

    def do_vq(m, lid):
        """一个 Expert 的 w1/w3/w2 共享一份码本 —— 三个矩阵一起送进去。
        权重在 materialize 里已是 bf16, VQ kernel 吃 f32 ⇒ 这里转一次(47 MB×3, 用完即弃)。"""
        mats, ts = [], []
        for nm in ("w1", "w3", "w2"):
            lin = getattr(m, nm, None)
            if lin is None or lin.weight.numel() == 0:
                return
            t = lin.weight.float().contiguous()
            mats.append((lin, t)); ts.append(t)
        ptrs = (ctypes.c_void_p * len(ts))(*[t.data_ptr() for t in ts])
        rr = (ctypes.c_int * len(ts))(*[t.shape[0] for t in ts])
        cc = (ctypes.c_int * len(ts))(*[t.shape[1] for t in ts])
        cwp, cwn = None, 0
        if colw is not None and lid < colw.shape[0] and colw.shape[1] == ts[0].shape[1]:
            cwv = np.ascontiguousarray(colw[lid])
            cwp = cwv.ctypes.data_as(ctypes.c_void_p); cwn = cwv.shape[0]
        rc = vqlib.v41_vq_expert_gpu(ptrs, rr, cc, len(ts), a.vq_dim, a.vq_nc, a.vq_iters,
                                     a.vq_stride, cwp, cwn)
        if rc != 0:
            raise RuntimeError(f"VQ 失败 rc={rc}")
        for lin, t in mats:
            lin.weight = torch.nn.Parameter(t.to(torch.bfloat16), requires_grad=False)
            stat[0] += 1
            stat[1] += t.numel()

    def hook_unit(mod, prefix, skip=None, tag=None, is_expert=False):
        def pre(m, _a):
            m._t0 = time.time()
            materialize(m, prefix, skip)
            if vqlib is not None and qmode["on"] and is_expert and prefix.startswith("layers."):
                do_vq(m, int(prefix.split(".")[1]))

        def post(m, _a, _o):
            release(m, prefix, skip)
            if tag:
                print(f"  [{tag}] {time.time() - m._t0:.1f}s", flush=True)

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    hook_unit(net.embed, "embed")
    for i, layer in enumerate(net.layers):
        # ★专家单独挂 hook★: MoE.forward 里 `if counts[i]==0: continue` 会跳过没被路由到的
        # 专家, 但 layer 级 hook 会把 384 个全装全 dequant。32 token 只碰到 ~150 个 ⇒ 白干一倍多,
        # 而且内存峰值白涨(一层 384 专家 bf16 = 27 GB)。挂到 Expert 上, 谁被调谁才装。
        skips = ["engram."] if getattr(layer, "engram", None) is not None else []
        if getattr(layer, "engram", None) is not None:
            hook_unit(layer.engram, f"layers.{i}.engram")
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and getattr(ffn, "experts", None) is not None:
            skips.append("ffn.experts.")
            for j, ex in enumerate(ffn.experts):
                if ex is not None:
                    hook_unit(ex, f"layers.{i}.ffn.experts.{j}", is_expert=True)
        hook_unit(layer, f"layers.{i}", skip=tuple(skips) if skips else None, tag=f"L{i}")
    hook_unit(net.norm, "norm")
    hook_unit(net.head, "head")

    # ---- 反修原料: dump 每层 MoE 的 (x, y) / 或用教师的 x 强制喂入 ----
    # 【z 靶的语义】"教师 − 当层全部量化计算"(V4 时代 zlayer 同义): 必须用【同一个 x】
    # 分别过 FP 与量化权重, 差值才是**当层量化引入的误差**。若各自用自己传播来的 x,
    # 差里会混进上游误差, 解出来的放大器是在补别人的账。
    # 所以量化态那一趟要 --force-x: 每层入口把输入换成教师的 x ⇒ 层间隔断。
    # (代价: 那一趟的最终 logits 无意义 —— 本来也只取每层的 y_q。)
    moed = Path(a.dump_moe) if a.dump_moe else None
    if moed:
        moed.mkdir(parents=True, exist_ok=True)
    forced = Path(a.force_x) if a.force_x else None
    if moed or forced:
        for i, layer in enumerate(net.layers):
            ffn = getattr(layer, "ffn", None)
            if ffn is None or getattr(ffn, "experts", None) is None:
                continue

            def mk_pre(li):
                def h(mod, args):
                    if forced:
                        f = forced / f"x_L{li:02d}.bin"
                        if f.exists():
                            x = args[0]
                            v = np.fromfile(f, dtype=np.float32).reshape(x.shape)
                            return (torch.from_numpy(v).to(x.device, x.dtype),) + args[1:]
                    return None
                return h

            def mk_post(li):
                def h(mod, args, out):
                    if moed:
                        args[0].detach().float().cpu().numpy().tofile(moed / f"x_L{li:02d}.bin")
                        out.detach().float().cpu().numpy().tofile(moed / f"y_L{li:02d}.bin")
                return h
            ffn.register_forward_pre_hook(mk_pre(i))
            ffn.register_forward_hook(mk_post(i))
        if moed:   # engram 对拍: hash id([1,n,n_eng,24] int64) + 每个 engram 模块输出(hc 栈 [n][hc][E])
            def mk_eng(lid):
                def h(mod, args, out):
                    out.detach().float().cpu().numpy().tofile(moed / f"hce_L{lid:02d}.bin")
                return h
            for name, mod in net.named_modules():
                cn = type(mod).__name__
                if cn == "Engram":
                    mod.register_forward_hook(mk_eng(int(mod.layer_id)))
                elif cn == "NgramHashState":
                    mod.register_forward_hook(lambda m_, a_, o_: o_.detach().cpu().numpy().astype(np.int64).tofile(moed / "hash_ids.bin"))
        print(f"[反修原料] dump={a.dump_moe or '-'} force-x={a.force_x or '-'}", flush=True)

    # ---- 放大器: 解算与应用都在 v41_amp_hooks, 数值一律走 C 库 ----
    # 【为什么搬出去】同一个公式 Z = X·(B·A) 原来在 C 自检里一遍、这里手抄两遍,
    # 布局差一点就是静默错位(方向写反那次 PPL 1.66→265, 日志上看不出来)。
    # 【靶的口径见 v41_amp_hooks.install_online】y_fp 必须是同一个 x 上当场重算的,
    # 不能读 FP 自然跑 dump 的那份 —— --amp-teacher 因此退役。
    seqstat = None
    scanstat = None
    if a.amp_scan_k or a.amp_select or a.amp_rowdiag or a.amp_t2diag:
        import v41_amp_hooks as amph
        # held-out 按 <ids>.layout 的窗分层(每域各留 1/4 窗); 没有 layout 就退回前缀切
        perm = amph.layout_split(a.ids, a.ntok)
    if a.amp_t2diag:
        import v41_amp_diag as ampd
        kk, ll, dd = a.amp_t2diag.split(":", 2)
        ampd.install_t2diag(net, dev, qmode, perm, int(kk), float(ll), dd)
    elif a.amp_rowdiag:
        import v41_amp_diag as ampd
        kk, ll = a.amp_rowdiag.split(":")
        ampd.install_rowdiag(net, dev, qmode, perm, int(kk), float(ll), whiten=a.amp_whiten)
    elif a.amp_scan_k:
        scanstat = amph.install_scan(net, dev, qmode, nfit=a.amp_scan_split, perm=perm, whiten=a.amp_whiten)
    elif a.amp_select:
        seqstat = amph.install_online_select(net, dev, a.amp_select, qmode, perm, whiten=a.amp_whiten)
    elif a.amp_online:
        import v41_amp_hooks as amph
        seqstat = amph.install_online(net, dev, a.amp_online, a.amp_k, qmode, lam=a.amp_lam)
    if a.amp:
        import v41_amp_hooks as amph
        amph.install_apply(net, dev, a.amp, layers=amph.parse_layers(a.amp_layers))

    # ---- 激活捕获: 每层 MoE 输入的列能量 E[x²] ----
    # 【干什么用】量化校准。★这不是训练★: 只统计校准语料在每层 MoE 入口的逐通道能量,
    # 得到列权 c_j = E[x_j²], 让量化误差优先落在不重要的通道上。权重语义不动, 无梯度。
    # 【口径】w1/w3 吃的是 x(dim=5120), w2 吃的是 SwiGLU 中间态 h(2304) —— 本版只收 x,
    # w2 的列权是后续欠账(V4 时代是"w1/w3 用 H_x, w2 用 H_h 抽 8 个专家")。
    act = {}
    if a.dump_act:
        for i, layer in enumerate(net.layers):
            ffn = getattr(layer, "ffn", None)
            if ffn is None or getattr(ffn, "experts", None) is None:
                continue

            def mk(li):
                def h(mod, args):
                    x = args[0].detach().float()
                    e = (x * x).reshape(-1, x.shape[-1]).sum(0)
                    act[li] = e if li not in act else act[li] + e
                return h
            ffn.register_forward_pre_hook(mk(i))
        print(f"[捕获] {len(net.layers)} 层 MoE 入口列能量 → {a.dump_act}", flush=True)

    # ---- DSpark 夹具·main_hidden 取法验证(--dump-mainh-variants): 目标层的四种候选各落一份 ----
    # 官方 model.py 取的是层输入的 hc 均值(h.mean(dim=2)); 但 hc 模型里注意力真正吃的是 hc_pre(h, pre_mix)
    # (按上一层给的 pre_mix 折叠四路), 均值只是它的一个特例。哪一种是草稿器训练时的口径, 只能喂夹具看命中率。
    mhv = {}
    if a.dump_mainh_variants:
        tl = list(args.dspark_target_layer_ids)
        mhv = {k: [None] * len(tl) for k in ("inmean", "inpre", "outmean", "outpre")}

        def mk_pre(si):
            def h(mod, fargs):          # Block.forward(x, start_pos, pre_mix, image_mask): 位置参数
                x, pm = fargs[0].float(), fargs[2].float()
                mhv["inmean"][si] = x.mean(2)[0].cpu()
                mhv["inpre"][si] = torch.sum(pm.unsqueeze(-1) * x, dim=2)[0].cpu()
            return h

        def mk_post(si):
            def h(mod, fargs, out):     # 返回 (h_out, ffn_pre): ffn_pre 就是下一层折叠输入用的 pre_mix
                y, fp = out[0].float(), out[1].float()
                mhv["outmean"][si] = y.mean(2)[0].cpu()
                mhv["outpre"][si] = torch.sum(fp.unsqueeze(-1) * y, dim=2)[0].cpu()
            return h
        for si, li in enumerate(tl):
            net.layers[li].register_forward_pre_hook(mk_pre(si))
            net.layers[li].register_forward_hook(mk_post(si))
        print(f"[取法验证] 目标层 {tl}: 四种 main_hidden 候选 → {a.dump_mainh_variants}.*.bin", flush=True)

    # ★强制全位置 logits★: ParallelHead.forward 默认 full_logits=False, 只算最后一个位置
    # (生成时的优化)。五指标(PPL 比/Σmin/KLD/same-top/top1)是逐位置 teacher-forcing 的,
    # 只拿末位等于没有尺。
    _head_fwd = net.head.forward
    net.head.forward = lambda x, full_logits=True: _head_fwd(x, True)

    # ---- 跑 ----
    # ★官方代码依赖默认 device 是 cuda★(generate.py 里 load_model 之后就设): 模型内部
    # 有若干 torch.arange/zeros 不带 device 参数, 默认 cpu 的话会在 sparse_attn 那里报
    # "topk_idxs device_type expected cuda, but got cpu"。放在建模之后, 免得干扰 meta 建模。
    torch.set_default_device(dev)
    ids = [int(x) for x in Path(a.ids).read_text().split()][: a.ntok]
    x = torch.tensor([ids], device=dev)
    print(f"[前向] {len(ids)} token", flush=True)
    t1 = time.time()
    out_ids, logits, mainh = net(x)
    torch.cuda.synchronize()
    print(f"[完成] {time.time()-t1:.1f}s, logits {tuple(logits.shape)}", flush=True)
    if a.dump_mainh:
        # DSpark 夹具的 FP 侧原料: 官方 forward 里 main_hiddens.append(h.mean(dim=2)) 在目标层拼出来的那份,
        # 第 p 行 = 位置 p(与引擎 --dspark-capture 落的 <out>.fix 第 p 对吃的 main_hidden(p) 同一个位置)。
        if mainh is None:
            raise SystemExit("★--dump-mainh: 这趟没建 MTP(--layers 截了层?), 拿不到 main_hidden★")
        arr = mainh.view(-1, mainh.shape[-1]).float().cpu().numpy()
        arr.astype("float32").tofile(a.dump_mainh)
        print(f"  [main_hidden] {arr.shape[0]} 位置 × {arr.shape[1]} → {a.dump_mainh}", flush=True)
    for k, parts in mhv.items():
        if any(p is None for p in parts):
            raise SystemExit(f"★取法验证: {k} 有目标层没钩到(hook 没触发?)★")
        arr = torch.cat(parts, dim=-1).numpy()          # [S][目标层数×dim], 槽序 = 目标层升序, 与引擎同
        arr.astype("float32").tofile(f"{a.dump_mainh_variants}.{k}.bin")
        print(f"  [取法验证] {k}: {arr.shape[0]} × {arr.shape[1]} → {a.dump_mainh_variants}.{k}.bin", flush=True)
    lg = logits.view(-1, logits.shape[-1])
    print(f"  首位 top5: {lg[0].float().topk(5).indices.tolist()}")
    print(f"  末位 top5: {lg[-1].float().topk(5).indices.tolist()}")
    # teacher-forcing 的 PPL: 第 i 位的 logits 预测第 i+1 个 token
    tgt = torch.tensor(ids[1:], device=lg.device)
    # 分块算 CE(2026-09-30): 整段一次 cross_entropy 内部再开一份 [S][V] 的 log_softmax 临时张量(47,500 位 = 24.6 GB); 分块求和
    # 再除以位数, 数值同一个公式(reduction="sum" 逐块加), 峰值只多一块。
    CHUNK = 2048
    ce_sum = 0.0
    for i0 in range(0, lg.shape[0] - 1, CHUNK):
        i1 = min(lg.shape[0] - 1, i0 + CHUNK)
        ce_sum += F.cross_entropy(lg[i0:i1].float(), tgt[i0:i1], reduction="sum").item()
    ppl = math.exp(ce_sum / (lg.shape[0] - 1))
    print(f"  ★PPL(本段 {len(ids)} token) = {ppl:.4f}★")
    if a.dump_act and act:
        ks = sorted(act)
        arr = torch.stack([act[k] for k in ks]).cpu().numpy()
        with open(a.dump_act, "wb") as f:
            f.write(struct.pack("<ii", arr.shape[0], arr.shape[1]))
            arr.astype("float32").tofile(f)
        r = arr.max(axis=1) / (arr.mean(axis=1) + 1e-30)
        print(f"  [捕获] {arr.shape[0]} 层 × {arr.shape[1]} 通道 → {a.dump_act}; "
              f"峰均比 中位 {float(np.median(r)):.1f} 最大 {float(r.max()):.1f} "
              f"(★峰均比高 = 存在巨值通道, 平权量化会把它们抹平★)")
    if a.vq_nc:
        note = f"; 其中 VQ {stat[0]} 个 / {stat[1]/1e9:.3f} B 参数 @dim{a.vq_dim}×nc{a.vq_nc}"
    elif a.qnbit:
        note = f"; 其中量化 {stat[0]} 个 / {stat[1]/1e9:.3f} B 参数 @{a.qnbit}bit blk{a.qblk}"
    elif nvq:
        note = f" (落盘 VQ 模型: {nvq} 个量化专家从文件读回)"
    else:
        note = " (教师态, 未量化)"
    print(f"  [路由] 本次被激活的专家矩阵 {stat[2]} 个" + note)
    if seqstat is not None:
        if a.amp_select:
            amph.report_select(seqstat, a.amp_select)
        else:
            amph.report(seqstat)
    if scanstat is not None:
        amph.report_scan(scanstat)
    if a.out:
        # ★格式 = 本仓现役判决尺 anchor_metrics 的 <i32 S><i32 V><f32 logits[S][V]>★
        # (它的 --ref-raw 与 --student 吃同一种)。这样五指标不用另写一份, 口径与 V4 时代
        # 逐式同源 —— 判决器只许有一份。
        with open(a.out, "wb") as f:
            f.write(struct.pack("<ii", lg.shape[0], lg.shape[1]))
            # 分块落盘(2026-09-30): `lg.float().cpu().numpy()` 是整份 [S][V] 的主机副本(47,500 位 = 24.6 GB), 与出口那份加起来正是
            # 看门狗停车的那一下; 按 2048 行一块搬, 文件字节逐位同(同一份 float32, 同一个顺序)。
            for i0 in range(0, lg.shape[0], 2048):
                lg[i0:i0 + 2048].float().cpu().numpy().tofile(f)
        print(f"  → {a.out} (S={lg.shape[0]} V={lg.shape[1]})")


if __name__ == "__main__":
    main()
