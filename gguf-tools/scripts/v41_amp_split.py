"""v41_amp_split — 反修 held-out 的【窗分层】切分与行置换(2026-09-12, 从 v41_amp_hooks 拆出)。
只搬行号, 零算术: 数值全在 C 库。"""
from pathlib import Path

import torch


def layout_split(ids_path, ntok, hold_every=4):
    """held-out 按 <ids>.layout 的窗【分层】切: 每域内窗号 k%4==3 的窗作 val, 其余作拟合。
    ★为什么不用前缀切(前 6144 解 / 后 2048 评)★: idshalf 把各域【连续】铺进 8192 行
    (金融 layout: fin_flash 0-2048 / announce / article / exam / news 6656-8192), 前缀切的 val
    全是最后一个域, 而拟合段根本没见过它 —— 量的是"跨域泛化", 不是"同域 held-out", 假账。
    分层后 64 窗 → 48 拟合(6144 > d_model 5120, 满秩) + 16 val(每域都有: flash 4 + 其余各 3)。
    返回 {"perm": 行置换(拟合行在前, val 行在后), "nfit": 拟合行数} 或 None(没 layout, 前缀切)。
    这里只搬行号, 一次算术都没有 —— 数值仍全在 C 库。"""
    lp = Path(str(ids_path) + ".layout")
    if not lp.exists():
        print(f"[held-out] ★{lp.name} 不存在, 退回前缀切(域错位风险)★", flush=True)
        return None
    win, rows = 128, []
    for ln in lp.read_text().splitlines():
        p = ln.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "win":
            win = int(p[1])
        else:
            rows.append((p[0], int(p[1]), int(p[2])))
    fit, val, desc = [], [], []
    for dom, off, n in rows:
        nw = n // win
        dv = 0
        for k in range(nw):
            r = range(off + k * win, off + (k + 1) * win)
            if k % hold_every == hold_every - 1:
                val.extend(r); dv += 1
            else:
                fit.extend(r)
        fit.extend(range(off + nw * win, off + n))          # 不满一窗的尾巴归拟合
        desc.append(f"{dom} {nw - dv}+{dv}窗")
    fit = [i for i in fit if i < ntok]
    val = [i for i in val if i < ntok]
    print(f"[held-out·分层] 窗宽 {win}: 拟合 {len(fit)} / val {len(val)} 行 ({', '.join(desc)})", flush=True)
    return {"perm": torch.tensor(fit + val, dtype=torch.long), "nfit": len(fit)}


def _permute(perm, dev, *ts):
    """按 perm 把 [n,d] 张量的行重排(拟合行在前); perm 为 None 时原样返回。"""
    if perm is None:
        return ts
    p = perm["perm"].to(dev)
    return tuple(t.index_select(0, p).contiguous() for t in ts)
