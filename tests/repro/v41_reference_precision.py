"""Explicit precision ablations for a same-GGUF implementation comparison.

These are never silently enabled for the official-dtype reference. Each changes
one documented rounding boundary shared by the existing Metal/CUDA GGUF paths.
"""
import types
import torch
import torch.nn.functional as F
import v41_patches as P


def install(M, choices):
    if 'hc' in choices:
        def hc_pre(self, x, pre_mix):
            return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=2).to(torch.bfloat16)
        M.Block.hc_pre = hc_pre
    if 'rowwise' in choices:
        proxy_f = types.ModuleType('torch.nn.functional')
        proxy_f.__dict__.update(F.__dict__)
        def linear(x, weight, bias=None):
            flat = x.reshape(-1, x.shape[-1])
            return torch.cat([F.linear(row[None], weight, bias) for row in flat], 0).reshape(*x.shape[:-1], weight.shape[0])
        proxy_f.linear = linear
        M.F = P.F = proxy_f
    if 'woa' in choices or 'rowwise' in choices:
        original = torch.einsum
        proxy = types.ModuleType('torch')
        proxy.__dict__.update(torch.__dict__)
        def einsum(equation, *args):
            if equation == 'bsgd,grd->bsgr':
                x, w = args
                if 'woa' in choices:
                    x, w = x.float(), w.float()
                if 'rowwise' in choices:
                    return torch.cat([original(equation, x[:, i:i+1], w) for i in range(x.shape[1])], 1).to(args[0].dtype)
                return original(equation, x, w).to(args[0].dtype)
            return original(equation, *args)
        proxy.einsum = einsum
        M.torch = proxy
    if 'routing' in choices:
        original_expert = M.Expert.forward
        def expert(self, x, weights=None):
            out = original_expert(self, x, None)
            return out if weights is None else out.float() * weights
        M.Expert.forward = expert

    if 'attention' in choices:
        def attention(q, kv, sink, indices, scale):
            # Explicit bf16 probability boundary and independent torch online
            # softmax. Only visible keys participate, in causal index order.
            out = torch.empty_like(q)
            for b in range(q.shape[0]):
                for row in range(q.shape[1]):
                    ids = indices[b, row].long(); ids = ids[ids >= 0]
                    keys = kv[b, ids].float()
                    scores = (q[b, row].float() @ keys.T) * scale
                    if not len(ids):
                        out[b, row].zero_(); continue
                    # Short causal fixture only: no window wrap or candidate pruning.
                    splits = max(1, min(32, (len(ids) + 31) // 32))
                    chunk = ((len(ids) + splits - 1) // splits + 7) // 8 * 8
                    parts = []
                    for lo in range(0, len(ids), max(8, chunk)):
                        mx = torch.full((q.shape[2],), -1e30, dtype=torch.float32)
                        den = torch.zeros_like(mx); acc = torch.zeros(q.shape[2:], dtype=torch.float32)
                        for k in range(lo, min(len(ids), lo+chunk), 8):
                            sc = scores[:, k:min(k+8, lo+chunk)]
                            new = torch.maximum(mx, sc.amax(-1)); rescale = (mx-new).exp()
                            prob = (sc-new[:,None]).exp()
                            den = den*rescale + prob.sum(-1)
                            acc = acc*rescale[:,None] + prob.to(torch.bfloat16).float() @ keys[k:k+sc.shape[-1]]
                            mx = new
                        parts.append((mx, den, acc))
                    maximum = torch.stack([p[0] for p in parts]).amax(0)
                    den = (sink.float()-maximum).exp(); acc = torch.zeros_like(parts[0][2])
                    for mx, part_den, part_acc in parts:
                        weight = (mx-maximum).exp()
                        den += part_den*weight; acc += part_acc*weight[:,None]
                    out[b, row] = (acc/den[:,None]).to(q.dtype)
            return out
        M.sparse_attn = attention
