"""Run the pinned official V4.1 forward with streamed GGUF weights on CPU.

The official Transformer/Block/Attention/Compressor/Indexer/MoE/Engram forwards
remain unchanged. The existing teacher's torch reference primitives replace
CUDA-only kernels; weights are independently unpacked with NumPy. This is a
same-weight implementation comparison, not an original-FP model quality score.
"""
import argparse
import hashlib
import json
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from v41_reference_weights import Weights, SCRIPTS, sidecar
import v41_patches as P
from v41_hf_io import build_index

OFFICIAL_MODEL_SHA = 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'


class TokenizerAdapter:
    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def install_cpu_kernels():
    kernel = types.ModuleType('kernel')

    def unsupported(*args, **kwargs):
        raise RuntimeError('CUDA-only operation reached the CPU reference')

    def act(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace or not scale_fmt:
            return unsupported()
        x.copy_(P.ref_fp8_inplace(x, block_size))
        return x

    def fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if not inplace:
            return unsupported()
        x.copy_(P.ref_fp4_inplace(x, block_size, scale_dtype == torch.float8_e4m3fn))
        return x

    kernel.act_quant, kernel.fp4_act_quant = act, fp4
    kernel.fp4_gemm = kernel.fp8_gemm = kernel.sparse_attn = unsupported
    kernel.hc_split_sinkhorn = P.ref_hc_split_sinkhorn
    sys.modules['kernel'] = kernel


def install_sidecars(net, directories):
    """Read explicit restore/posttrain files; never silently ignore unknown formats."""
    gains, biases, amps = {}, {}, {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        if list(directory.glob('ampr_L*.bin')):
            raise ValueError('routed low-rank sidecars are not supported by this reference')
        for i, layer in enumerate(net.layers):
            gp, rp, ap = [directory / f'{prefix}_L{i:02d}.bin' for prefix in ('gr', 'rb', 'amp')]
            if gp.exists():
                value, ty = sidecar(gp)
                assert value.shape == (1, layer.ffn.n_routed_experts, layer.ffn.dim)
                value = value[0] + (1 if ty == 43 else 0)
                gains[i] = gains.get(i, torch.ones_like(value)) * value
            if rp.exists():
                b = rp.read_bytes(); ne, ty = struct.unpack_from('<ii', b)
                assert ne == layer.ffn.n_routed_experts and ty == 1 and len(b) == 8 + 4 * ne
                value = torch.from_numpy(np.frombuffer(b, '<f4', offset=8).copy())
                biases[i] = biases.get(i, torch.zeros_like(value)) + value
            if ap.exists():
                b = ap.read_bytes(); d, k, ty = struct.unpack_from('<iii', b)
                assert ty in (1, 43), f'unsupported adapter type {ty}: {ap}'
                value, _ = sidecar(ap)
                assert value.numel() == 2 * d * k and d == layer.ffn.dim
                aa, bb = value.reshape(2, k, d)
                amps.setdefault(i, []).append((aa, bb))
    for i, layer in enumerate(net.layers):
        if i in gains:
            # The gain belongs to each expert's down-projection row. Folding it
            # into the independently decoded weight keeps the pre-bf16 placement.
            for e, expert in enumerate(layer.ffn.experts):
                expert.w2._reference_gain = gains[i][e]
        if i in biases:
            layer.ffn.gate._reference_bias = biases[i]
        if i in amps:
            aa = torch.cat([x[0] for x in amps[i]], dim=0)
            bb = torch.cat([x[1] for x in amps[i]], dim=0)

            def apply(mod, args, out, aa=aa, bb=bb):
                return (out.float() + F.linear(F.linear(args[0].float(), bb), aa.T)).to(out.dtype)
            layer.ffn.register_forward_hook(apply)
    return dict(gain_layers=len(gains), bias_layers=len(biases), amp_layers=len(amps))


def install_streaming(net, weights, dump, embedding_dtype, adaptations=(), replay=None):
    stats = dict(parameters=0, experts=0)

    def hook(mod, prefix, skip=(), layer=None):
        specs = [(name, tuple(p.shape), p.dtype) for name, p in mod.named_parameters()
                 if not (name == 'scale' or name.endswith('.scale')) and not any(name.startswith(s) for s in skip)]

        def owner(name):
            sub, _, leaf = name.rpartition('.')
            return (mod.get_submodule(sub) if sub else mod), leaf

        def pre(module, args):
            module._reference_started = time.monotonic()
            for name, shape, dtype in specs:
                full = f'{prefix}.{name}'
                value, quant = weights.get(full)
                target, leaf = owner(name)
                expected = shape[:-1] + (shape[-1] * 2,) if dtype == torch.float4_e2m1fn_x2 else shape
                if tuple(value.shape) != expected:
                    raise ValueError(f'{full}: shape {tuple(value.shape)} != {expected}')
                # Keep exact dequantized weights, as the upstream teacher's
                # --exact-weights does. Official wo_a uses bf16 einsum directly.
                if not quant or (prefix == 'embed' and embedding_dtype == 'bf16') or ('.wo_a.' in full and 'woa' not in adaptations):
                    value = value.to(dtype if dtype in (torch.float32, torch.bfloat16) else torch.bfloat16)
                if leaf == 'weight' and hasattr(target, '_reference_gain'):
                    value = value * target._reference_gain[:, None]
                if leaf == 'bias' and hasattr(target, '_reference_bias'):
                    value = value + target._reference_bias
                par = torch.nn.Parameter(value, requires_grad=False)
                par._was_quant = quant
                setattr(target, leaf, par)
                stats['parameters'] += 1
            if '.experts.' in prefix:
                stats['experts'] += 1

        def post(module, args, output):
            if dump and layer is not None:
                output[0].float().numpy().tofile(dump / f'h_L{layer:02d}.bin')
            for name, shape, dtype in specs:
                target, leaf = owner(name)
                setattr(target, leaf, torch.nn.Parameter(torch.empty(shape, dtype=dtype, device='meta'), requires_grad=False))
            if layer is not None:
                weights.discard_pages()
                print(f'layer={layer} seconds={time.monotonic() - module._reference_started:.3f} experts_total={stats["experts"]}', flush=True)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    hook(net.embed, 'embed')
    for i, layer in enumerate(net.layers):
        if layer.engram is not None:
            hook(layer.engram, f'layers.{i}.engram', ('embed.',))
        for e, expert in enumerate(layer.ffn.experts):
            hook(expert, f'layers.{i}.ffn.experts.{e}')
        hook(layer, f'layers.{i}', ('engram.', 'ffn.experts.'), i)
    hook(net.norm, 'norm'); hook(net.head, 'head')
    if replay:
        def replace_input(kind, layer):
            def replace(module, inputs):
                value = np.fromfile(f'{replay}.{kind}_L{layer:02d}.bin', dtype='<f4')
                value = torch.from_numpy(value.copy()).reshape(inputs[0].shape).to(inputs[0].dtype)
                return (value, *inputs[1:])
            return replace
        for i, layer in enumerate(net.layers):
            layer.attn.register_forward_pre_hook(replace_input('ax', i))
            layer.ffn.register_forward_pre_hook(replace_input('x', i))
    if dump:
        def capture(kind, layer, before=False):
            def save(module, inputs, output=None):
                value = inputs[0] if before else output
                value.detach().float().numpy().tofile(dump / f'{kind}_L{layer:02d}.bin')
            return save
        for i, layer in enumerate(net.layers):
            layer.register_forward_pre_hook(capture('hi', i, True))
            layer.attn.register_forward_pre_hook(capture('ax', i, True))
            layer.attn.register_forward_hook(capture('ao', i))
            layer.ffn.register_forward_pre_hook(capture('x', i, True))
            layer.ffn.register_forward_hook(capture('y', i))
            if layer.attn.is_kv_source:
                c = layer.attn.compressor
                c.wkv.register_forward_hook(capture('ckv', i))
                if c.compress_ratio > 1:
                    c.wgate.register_forward_hook(capture('csc', i))
                c.norm.register_forward_pre_hook(capture('pool', i, True))
                c.norm.register_forward_hook(capture('latent', i))
        import model as M
        current = [0]
        def set_layer(i):
            def set(module, inputs): current[0] = i
            return set
        for i, layer in enumerate(net.layers):
            layer.attn.register_forward_pre_hook(set_layer(i))
        original_sparse = M.sparse_attn
        def sparse(q, kv, sink, indices, scale):
            q.float().numpy().tofile(dump / f'q_L{current[0]:02d}.bin')
            kv.float().numpy().tofile(dump / f'kv_L{current[0]:02d}.bin')
            out = original_sparse(q, kv, sink, indices, scale)
            out.float().numpy().tofile(dump / f'o_L{current[0]:02d}.bin')
            return out
        M.sparse_attn = sparse
    return stats


def main():
    global F
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--gguf', type=Path, required=True)
    ap.add_argument('--engram-dir', type=Path, required=True)
    ap.add_argument('--sidecar', action='append', default=[])
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ntok', type=int, default=128)
    ap.add_argument('--layers', type=int, default=0, help='diagnostic only; zero means the full model')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--embedding-dtype', choices=('bf16', 'exact'), default='bf16',
                    help='bf16 preserves official dtype; exact matches the teacher --exact-weights and native Q4_K lookup')
    ap.add_argument('--precision-adaptation', action='append', default=[], choices=['hc', 'woa', 'routing', 'attention', 'rowwise'])
    ap.add_argument('--replay-stage-inputs', type=Path, help='diagnostic only: read native .ax/.x stage inputs')
    ap.add_argument('--dump-layers', type=Path)
    a = ap.parse_args()
    if a.ntok < 2 or a.threads < 1 or not 0 <= a.layers <= 40:
        ap.error('require ntok >= 2, threads >= 1, and layers between 0 and 40')
    if 'attention' in a.precision_adaptation and a.ntok > 128:
        ap.error('attention precision ablation supports only the short <=128-token causal fixture')
    if a.out.exists():
        raise FileExistsError(a.out)
    source_sha256 = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__), Path(__file__).with_name('v41_reference_weights.py'),
         Path(__file__).with_name('v41_reference_precision.py'), SCRIPTS/'v41_patches.py', SCRIPTS/'v41_file_parity.py']}
    model_file = a.reference / 'inference/model.py'
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == OFFICIAL_MODEL_SHA, 'official model revision differs'
    sources = json.loads(Path(__file__).with_name('v41_reference_sources.json').read_text())
    for name, entry in sources['files'].items():
        assert hashlib.sha256((a.reference/name).read_bytes()).hexdigest() == entry['sha256'], f'source differs: {name}'
    torch.set_grad_enabled(False); torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.bfloat16)
    install_cpu_kernels()
    sys.path.insert(0, str(a.reference / 'inference'))
    import model as M
    P.patch_linear(M); P.patch_sparse_attn(M)
    from v41_reference_precision import install
    install(M, a.precision_adaptation)
    F = M.F  # Include sidecar matmuls in the optional rowwise experiment.
    M.sample = lambda logits, temperature=1: logits.argmax(dim=-1)
    cfg = json.loads((a.reference / 'inference/config.json').read_text())
    cfg.update(max_batch_size=1, max_seq_len=max(128, a.ntok), vision_n_layers=0, dspark_block_size=0)
    if a.layers:
        cfg['n_layers'] = a.layers
    args = M.ModelArgs(**cfg)
    tok = TokenizerAdapter(a.reference / 'tokenizer.json')
    started = time.monotonic()
    with torch.device('meta'):
        net = M.Transformer(args, tok)
    P.fix_buffers(M, net, args, tok, 'cpu')
    weights = Weights(a.gguf)
    # Independently regenerated official hash constants must match the file.
    for name in ('token_map', 'primes', 'offsets', 'multipliers'):
        expected, _ = weights.dense(f'engram.{name}')
        assert torch.equal(getattr(net.engram_hash, name), expected), f'engram {name} mismatch'
    idx = build_index(a.engram_dir)
    for i, layer in enumerate(net.layers):
        if layer.engram is not None:
            key = f'layers.{i}.engram.embed.weight'
            shape = idx[key][3]
            layer.engram.embed = P.StreamEngramEmbed(idx, 'cpu', key, key[:-6] + 'scale', *shape, M.fp8_block_size)
    plugins = install_sidecars(net, a.sidecar)
    if a.dump_layers:
        a.dump_layers.mkdir(parents=True, exist_ok=False)
    stats = install_streaming(net, weights, a.dump_layers, a.embedding_dtype, a.precision_adaptation, a.replay_stage_inputs)
    original_head = net.head.forward
    net.head.forward = lambda x, full_logits=True: original_head(x, True)
    ids = [int(x) for x in a.ids.read_text().split()][:a.ntok]
    assert len(ids) == a.ntok and all(0 <= x < args.vocab_size for x in ids)
    print(json.dumps(dict(event='forward',tokens=len(ids),layers=args.n_layers,device='cpu',plugins=plugins)), flush=True)
    _, logits, _ = net(torch.tensor([ids], dtype=torch.int64))
    values = logits.float().squeeze(0).numpy()
    assert values.shape == (len(ids), args.vocab_size) and np.isfinite(values).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as f:
        f.write(struct.pack('<ii', *values.shape)); values.astype('<f4').tofile(f)
    nll = F.cross_entropy(logits[0, :-1].float(), torch.tensor(ids[1:], dtype=torch.int64), reduction='none')
    report = dict(tokens=len(ids),layers=args.n_layers,full_model=args.n_layers==40 and not a.replay_stage_inputs,
                  diagnostic=bool(a.layers or a.replay_stage_inputs or a.precision_adaptation),seconds=time.monotonic()-started,
                  mean_nll=nll.mean().item(),nll=nll.tolist(),stats=stats,plugins=plugins,
                  torch=torch.__version__,numpy=np.__version__,arguments=vars(a),official_model_sha256=OFFICIAL_MODEL_SHA)
    report['source_sha256'] = source_sha256
    a.out.with_suffix(a.out.suffix+'.json').write_text(json.dumps(report,default=str,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('nll','source_sha256','arguments')}),flush=True)


if __name__ == '__main__':
    main()
