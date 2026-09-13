"""v41_patches — 官方 inference/model.py 在 GB10 教师端要打的三处补丁 + meta 建模后的 buffer 修复。
从 v41_teacher.py 拆出(2026-09-12, 主文件超 500 行仓规)。逐字搬家, 语义零改动。

① linear 走 dequant + fp32 matmul(不做激活量化) —— 教师与学生必须同一条前向路
② sparse_attn 换 torch 版 —— 官方 kernel 要 141312 B 动态 shared memory, GB10 上限 101376
③ engram 表按行流式取 —— 98 GB 不能整层驻留, 每 token 只查 24 行
④ meta 建模后修 buffer —— 非零常量(freqs_cis / ngram 哈希表)清零不报错只静默出错
"""
import numpy as np
import torch
import torch.nn.functional as F

from v41_hf_io import e8m0_to_f32


def patch_linear(M):
    """权重在 materialize 时已 dequant 成 bf16, 这里只负责【用 fp32 算】。
    ★为什么不图省事用 bf16 GEMM★: 本仓实测过 fp16 层缓存让判决 KL 从 0.4742 劣化到
    0.4775(fable5 09-03), 教师端掺这种损失会把它算进"量化代价"里。官方 kernel 更低
    (激活压 fp8), 那是推理优化, 不该进判决。"""
    def linear_dq(x, weight, bias=None):
        assert bias is None
        y = F.linear(x.float(), weight.float())
        # 两个分支的返回 dtype 必须跟官方对齐: 官方对量化权重走 fp4/fp8_gemm, kernel 出 bf16
        # (下游 act_quant 就是按 bf16 收的); 对普通权重走 F.linear 保持输入 dtype -- Gate 正是
        # 靠后者拿 fp32 分数做 topk, 一刀切成 bf16 会改变专家选择。
        return y.to(torch.bfloat16) if getattr(weight, "_was_quant", False) else y.to(x.dtype)
    M.linear = linear_dq


def patch_sparse_attn(M):
    """★不是嫌它慢, 是 GB10 上物理跑不了★: 官方 kernel 要 141312 字节动态 shared memory
    (q_shared 64KB + o_shared 64KB, h=64/d=512 定死), 而 GB10 的 opt-in 上限是 101376
    (实测 shared_memory_per_block_optin)。那 kernel 是照数据中心 Blackwell(227KB)写的。
    逐条对应 kernel 语义, 一条都不能差:
      ① idx == -1 ⇒ 该位置 kv 取 0 且 score 取 -inf
      ② scores_max 初值 -1e30(有限下界) ⇒ 全无效的行得到全零输出而不是 NaN
      ③ ★attn_sink 只进分母, 不贡献 value★
    数值上本版全程 fp32(kernel 是 bf16 gemm + fp32 累加), 教师端取更准的一侧。"""
    def sparse_attn_torch(q, kv, attn_sink, topk_idxs, scale):
        b, m, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        safe = idx.clamp_min(0)
        out = torch.empty(b, m, h, d, dtype=torch.float32, device=q.device)
        step = max(1, int(2e8 // (idx.shape[-1] * d * 4)))   # 控 gather 的峰值内存
        for s0 in range(0, m, step):
            s1 = min(m, s0 + step)
            si, sv = safe[:, s0:s1], valid[:, s0:s1]
            k = kv.gather(1, si.reshape(b, -1, 1).expand(b, si.shape[1] * si.shape[2], d))
            k = k.view(b, s1 - s0, -1, d).float() * sv.unsqueeze(-1)
            sc = torch.einsum("bmhd,bmkd->bmhk", q[:, s0:s1].float(), k) * scale
            sc = sc.masked_fill(~sv.unsqueeze(2), float("-inf"))
            mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
            p = (sc - mx).exp()
            den = p.sum(-1) + (attn_sink.view(1, 1, h).float() - mx.squeeze(-1)).exp()
            out[:, s0:s1] = torch.einsum("bmhk,bmkd->bmhd", p, k) / den.unsqueeze(-1)
        return out.to(q.dtype)
    M.sparse_attn = sparse_attn_torch


class StreamEngramEmbed(torch.nn.Module):
    """engram 表换成按行 memmap 取(在 meta 上建好后替换, 它自带真实数据)。
    取到的行与 C 读器逐位对拍过(v41_dequant_parity.py 全绿)。"""
    def __init__(self, idx, dev, name_w, name_s, num_emb, dim, blk):
        super().__init__()
        pw, ow, _, _ = idx[name_w]
        ps, os_, _, _ = idx[name_s]
        self.w = np.memmap(pw, dtype=np.uint8, mode="r", offset=ow, shape=(num_emb, dim))
        self.s = np.memmap(ps, dtype=np.uint8, mode="r", offset=os_, shape=(num_emb, dim // blk))
        self.blk = blk
        self.dev = dev

    def forward(self, indices):
        flat = indices.reshape(-1).cpu().numpy()
        w = torch.from_numpy(np.ascontiguousarray(self.w[flat])).to(self.dev)
        s = torch.from_numpy(np.ascontiguousarray(self.s[flat])).to(self.dev)
        v = w.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.blk))
        v = (v * e8m0_to_f32(s).unsqueeze(-1)).flatten(-2)
        return v.to(torch.bfloat16).view(*indices.shape, -1)


def fix_buffers(M, net, args, tok, dev):
    """【为什么必须修】参数可以留在 meta(hook 按需装), 但 buffer 不行: 它们是**算出来的**,
    在 meta context 下建等于值当场丢失。分两类:
      ① KV cache 类(window_kv_cache/compress_kv_cache/kv_state/...): 本来就是 zeros, 重建即可
      ② ★非零常量★: Attention.freqs_cis(YaRN 频率, 还分层不同) 和 NgramHashState 的
         token_map/primes/multipliers —— 这类清零【不会报错, 只会静默出错】, 必须重算。
    返回 (修的 buffer 数, 其中重算的 freqs_cis 数)。"""
    M.precompute_freqs_cis.cache_clear()   # ★lru_cache 在 meta 下已缓存过 meta 结果, 不清会拿回 meta★
    nfix = nfreq = 0
    with torch.device(dev):
        for mod in net.modules():
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if buf is None or buf.device.type != "meta":
                    continue
                if bname == "freqs_cis":
                    # 复原 model.py Attention.__init__ 的分支: 有压缩比才开 YaRN
                    if getattr(mod, "compress_ratio", 0):
                        osl, theta = args.original_seq_len, args.compress_rope_theta
                    else:
                        osl, theta = 0, args.rope_theta
                    nb = M.precompute_freqs_cis(mod.rope_head_dim, args.max_seq_len, osl, theta,
                                                args.rope_factor, args.beta_fast, args.beta_slow)
                    nfreq += 1
                else:
                    nb = torch.zeros(buf.shape, dtype=buf.dtype)
                mod.register_buffer(bname, nb, persistent=False)
                nfix += 1
    # NgramHashState 整个重建: 它全是算出来的 buffer(token_map 要遍历整个词表), 没有参数
    if net.engram_hash is not None:
        with torch.device(dev):
            net.engram_hash = M.NgramHashState(args, net.engram_layout, tok)
    return nfix, nfreq
