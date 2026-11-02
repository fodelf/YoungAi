"""Validate the CPU reference's format readers and unavailable-kernel adapters."""
import argparse
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from v41_reference import install_cpu_kernels, install_sidecars, install_streaming
from v41_reference_weights import Weights, fp4_blocks, np_q4k


def test_primitives():
    install_cpu_kernels()
    import kernel
    # Known Q4_K block: all scales=1, minima=2, d=.5, dmin=.25,
    # alternating low/high nibble 3/7 -> exact 1/3 values.
    block = np.zeros((1, 144), np.uint8)
    block[0, :4] = np.frombuffer(struct.pack('<ee', .5, .25), np.uint8)
    block[0, 4:8] = 1; block[0, 8:12] = 2; block[0, 12:16] = 0x21
    block[0, 16:] = 0x73
    expected = np.tile(np.repeat([1., 3.], 32), 4).reshape(1, 256)
    np.testing.assert_array_equal(np_q4k(block, 1, 256), expected)
    # FP4 low nibble first, with independently stated E2M1 values.
    q = np.arange(16, dtype=np.uint8)
    b = np.zeros((1, 17), np.uint8); b[0, :16] = q | (q << 4); b[0, 16] = 128
    grid = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    np.testing.assert_array_equal(fp4_blocks(b), np.repeat(np.array(grid)*2, 2).reshape(1, 32))
    # RNE ties at unit scale; include 6 to keep the scale at exactly one.
    x = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 4)
    result = kernel.fp4_act_quant(x, 32, True)
    assert result is x
    torch.testing.assert_close(x, torch.tensor([0, 1, 1, 2, 2, 4, 4, 6] * 4, dtype=x.dtype), rtol=0, atol=0)
    x = torch.tensor([0, .5, 1, 2, 4, 8, 16, 32] * 4)
    old = x.clone(); result = kernel.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu, True)
    assert result is x
    torch.testing.assert_close(x, old, rtol=0, atol=0)
    try:
        kernel.fp4_gemm(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError('unsupported kernels must fail explicitly')


def test_real_vq(weights):
    """Scalar bit addressing cross-checks vectorized decoding of 12/13-bit files."""
    checked = 0
    for layer in (0, 14, 39):
        _, ne, off = weights.tensors[f'blk.{layer}.ffn_exps_vq.blob']
        b = weights.raw(weights.data0 + off, ne[0])
        for expert, which in ((0, 0), (1, 1), (383, 2)):
            value, _ = weights.expert(layer, expert, which)
            p = struct.unpack_from('<Q', b, 16 + (expert*3 + which)*8)[0]
            _, dim, nc, rows, cols, flags, bits, cb = struct.unpack_from('<IHHIIIIQ', b, p)
            gains = b[p+32:p+32+rows*2].view('<f2')
            stream = p+32+rows*2; ni=cols//dim; row_bytes=ni*bits//8
            ext = stream+rows*row_bytes
            table = torch.from_numpy(b[cb:cb+nc*dim].copy()).view(torch.float8_e4m3fn).float().numpy()
            for row in (0, 1, rows-1):
                for column in (0, 7, 8, cols//2, cols-1):
                    slot, lane = divmod(column, dim); bit=slot*bits
                    word = int.from_bytes(b[stream+row*row_bytes+bit//8:stream+row*row_bytes+bit//8+3], 'little')
                    code = (word >> (bit%8)) & ((1<<bits)-1)
                    if flags & 2:
                        code |= ((int(b[ext+row*((ni+7)//8)+slot//8]) >> (slot%8)) & 1) << bits
                    expected = np.float32(table[code*dim+lane] * np.float32(gains[row]))
                    assert value[row,column].item() == expected, (layer,expert,which,row,column)
                    checked += 1
    # Returned weights must own storage: unmapping the file cannot invalidate
    # a parameter still in use by the official forward.
    snapshot = value.clone()
    del b
    weights.discard_pages()
    assert torch.equal(value, snapshot)
    print(f'VQ scalar/vector checks: {checked} exact values')


def test_sidecars():
    class FFN(torch.nn.Module):
        dim, n_routed_experts = 2, 2

        def __init__(self):
            super().__init__()
            self.experts = [SimpleNamespace(w2=SimpleNamespace()) for _ in range(2)]
            self.gate = SimpleNamespace()

        def forward(self, x):
            return torch.tensor([[.5, 1.5]], dtype=x.dtype)

    ffn = FFN(); net = SimpleNamespace(layers=[SimpleNamespace(ffn=ffn)])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'gr_L00.bin').write_bytes(struct.pack('<iii4f', 2, 2, 1, 1, 2, 3, 4))
        (root/'rb_L00.bin').write_bytes(struct.pack('<ii2f', 2, 1, .25, -.5))
        # A=[3,4], B=[5,6], x=[1,2]: x B^T = 17, correction=[51,68].
        (root/'amp_L00.bin').write_bytes(struct.pack('<iii4f', 2, 1, 1, 3, 4, 5, 6))
        assert install_sidecars(net, [root]) == dict(gain_layers=1, bias_layers=1, amp_layers=1)
        torch.testing.assert_close(ffn(torch.tensor([[1., 2.]])), torch.tensor([[51.5, 69.5]]), rtol=0, atol=0)
        torch.testing.assert_close(ffn.experts[1].w2._reference_gain, torch.tensor([3., 4.]), rtol=0, atol=0)
        torch.testing.assert_close(ffn.gate._reference_bias, torch.tensor([.25, -.5]), rtol=0, atol=0)
        (root/'ampr_L00.bin').write_bytes(b'unsupported')
        try:
            install_sidecars(net, [root])
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported routed adapters must fail')


def test_embedding_policy():
    values = torch.tensor([[1.001, 2.003], [3.005, 4.007]], dtype=torch.float32)
    weights = SimpleNamespace(get=lambda name: (values.clone(), True))
    for policy, dtype in [('bf16', torch.bfloat16), ('exact', torch.float32)]:
        net = SimpleNamespace(embed=torch.nn.Embedding(2, 2, dtype=torch.bfloat16), layers=[],
                              norm=torch.nn.Linear(2, 2, bias=False), head=torch.nn.Linear(2, 2, bias=False))
        install_streaming(net, weights, None, policy)
        actual = net.embed(torch.tensor([1]))
        assert actual.dtype == dtype and net.embed.weight.device.type == 'meta'
        torch.testing.assert_close(actual, values[1:].to(dtype), rtol=0, atol=0)


def test_precision_controls():
    # Invalid key slots must contribute neither value nor denominator; the sink
    # contributes only to the denominator. This closed-form case is independent
    # of any engine kernel or floating-point reduction order.
    from v41_reference_precision import install
    import v41_patches as patches
    old_f = patches.F
    model = SimpleNamespace()
    try:
        install(model, ['attention', 'rowwise'])
        q = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
        kv = torch.stack([torch.ones(8), torch.full((8,), 3.)])[None].to(torch.bfloat16)
        ids = torch.tensor([[[0, -1], [0, 1]]])
        got = model.sparse_attn(q, kv, torch.zeros(2), ids, 1.)
        expected = torch.stack([torch.full((2, 8), .5), torch.full((2, 8), 4/3)])[None].to(q.dtype)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        ids.fill_(-1)
        assert torch.count_nonzero(model.sparse_attn(q, kv, torch.zeros(2), ids, 1.)) == 0
        # Rowwise controls must retain prefix values when unrelated future rows
        # are added, including the grouped projection used outside M.linear.
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(1, 9, 3, 32, generator=generator)
        w = torch.randn(3, 8, 32, generator=generator)
        short = model.torch.einsum('bsgd,grd->bsgr', x[:, :2], w)
        full = model.torch.einsum('bsgd,grd->bsgr', x, w)
        torch.testing.assert_close(short, full[:, :2], rtol=0, atol=0)
        weight = w.flatten(0, 1)
        short = model.F.linear(x[:, :2], weight)
        full = model.F.linear(x, weight)
        torch.testing.assert_close(short, full[:, :2], rtol=0, atol=0)
    finally:
        patches.F = old_f


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gguf',required=True)
    a=ap.parse_args();test_primitives();test_sidecars();test_embedding_policy();test_real_vq(Weights(a.gguf));test_precision_controls();print('CPU reference tests: PASS')
