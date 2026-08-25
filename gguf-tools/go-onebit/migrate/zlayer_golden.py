#!/usr/bin/env python3
"""zlayer_golden.py — zlayer.py ↔ calib/zlayer 的二期四模式对拍驱动(夹具版)。

先跑 zlayer_fixture.py 造夹具, 再用同一套输入分别跑 .py 和 C, 逐项比:
  ① stdout 的每一行(★行 / k曲线 / 分域 / ADDON / ERF 行)——【逐字符】;
  ② zcache 里的 prow/pe/pw ——【逐位】(整数与纯搬运的 f32);
  ③ zcache 里的 dH/pYQ ——【相对 1e-5】: 两边都走 sgemm 但不是同一份 BLAS,
     f32 求和顺序不同, 末位必然差(见 zlayer_transcription_notes.md B 组);
  ④ 注入到 dql 的字节 ——【逐位】(ADDON/ERF 用; ERF 的 U/V 载荷例外, 见下)。

用法: python3 zlayer_golden.py [夹具目录]
退出码 0 = 全过。
"""
import os, shutil, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "zlayer_fixtures")
PY = os.path.join(HERE, "..", "zlever", "zlayer.py")
CC = os.path.join(HERE, "..", "calib", "zlayer")
HF = os.path.join(FIX, "hf")
AP = os.path.join(FIX, "anchor.bin")
AP2 = os.path.join(FIX, "anchor2.bin")

BASE_ENV = {"DS4_ZL_NTOK": "96", "DS4_ZL_NFIT": "64", "DS4_ZL_FTA": "0",
            "DS4_ZL_GE": "1", "DS4_ZL_ERF": "0", "DS4_ZL_SWLIM": "10", "DS4_ZL_GATE": "0"}


def fresh(tag, with_dql):
    d = os.path.join(FIX, "work_" + tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(os.path.join(FIX, "layers", "dql_vq_L00.bin"), d)
    if with_dql:
        shutil.copy(os.path.join(FIX, "layers", "dql_L00.bin.tpl"), os.path.join(d, "dql_L00.bin"))
    return d


def run(cmd, env, cwd):
    e = dict(os.environ)
    e.update(env)
    e["PYTHONPATH"] = e.get("PYTHONPATH", "")
    p = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        print("  ★命令失败★", " ".join(cmd))
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        return None
    return p.stdout


def load_zc(d):
    p = os.path.join(d, "zcache_L00.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def cmp_case(name, env, extra_args, with_dql=False):
    print(f"\n=== {name} ===")
    e = dict(BASE_ENV)
    e.update(env)
    dpy, dc = fresh(name + "_py", with_dql), fresh(name + "_c", with_dql)
    opy = run([sys.executable, "-u", PY, HF, dpy, AP, "0"] + extra_args, e, FIX)
    oc = run([CC, HF, dc, AP, "0"] + extra_args, e, FIX)
    if opy is None or oc is None:
        return False
    ok = True
    # ① stdout 逐字符(去掉计时字段: "缓存 3s 解算 12s 总 15s" 两边必然不同)
    import re
    scrub = lambda s: re.sub(r"缓存 \d+s 解算 \d+s 总 \d+s", "缓存 -s 解算 -s 总 -s", s)
    lpy, lc = scrub(opy).strip().splitlines(), scrub(oc).strip().splitlines()
    lpy = [x for x in lpy if "缓存 …" not in x]
    lc = [x for x in lc if "缓存 …" not in x]
    if lpy != lc:
        ok = False
        print("  ✗ stdout 不一致:")
        for a, b in zip(lpy + [""] * (len(lc) - len(lpy)), lc + [""] * (len(lpy) - len(lc))):
            print(f"    py| {a}\n    C | {b}" if a != b else f"    ok| {a}")
    else:
        for x in lc:
            print("    ok|", x)
    # ②③ zcache
    zpy, zc = load_zc(dpy), load_zc(dc)
    if (zpy is None) != (zc is None):
        print(f"  ✗ zcache 一边有一边没有(py={zpy is not None} C={zc is not None})")
        return False
    if zpy is not None:
        for k in ("prow", "pe", "pw"):
            if not np.array_equal(zpy[k], zc[k]):
                ok = False
                print(f"  ✗ zcache {k} 不逐位")
            else:
                print(f"    ok| zcache {k} 逐位相同 {zpy[k].shape}")
        for k in ("dH", "pYQ"):
            a, b = zpy[k].astype(np.float64), zc[k].astype(np.float64)
            rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            flag = "ok" if rel < 1e-5 else "✗ "
            if rel >= 1e-5:
                ok = False
            print(f"    {flag}| zcache {k} 相对偏差 {rel:.2e}(闸 1e-5)")
        for k in ("yqe", "xcap"):
            if (k in zpy) != (k in zc):
                ok = False
                print(f"  ✗ zcache 字段 {k} 一边有一边没有")
    # ④ 注入字节
    if with_dql:
        bpy = open(os.path.join(dpy, "dql_L00.bin"), "rb").read()
        bc = open(os.path.join(dc, "dql_L00.bin"), "rb").read()
        tpl = os.path.getsize(os.path.join(FIX, "layers", "dql_L00.bin.tpl"))
        print(f"    ..| dql 长度 py={len(bpy)} C={len(bc)}(模板 {tpl})")
        if len(bpy) != len(bc):
            ok = False
            print("  ✗ 注入长度不一致")
        elif bpy[:12] != bc[:12]:
            ok = False
            print("  ✗ dql 头(含 nrec)不逐位")
        else:
            print("    ok| dql 头(含 nrec)逐位相同")
            ok = cmp_records(recs(bpy), recs(bc)) and ok
    return ok


def cmp_both_fail(name, env, extra_args):
    """两边都该拒跑的格子: 只要求 py 与 C 都非零退出, 且 C 的错话能 grep 到(不是段错误)。"""
    print(f"\n=== {name} ===")
    e = dict(BASE_ENV)
    e.update(env)
    dpy, dc = fresh(name + "_py", True), fresh(name + "_c", True)
    ee = dict(os.environ)
    ee.update(e)
    ppy = subprocess.run([sys.executable, "-u", PY, HF, dpy, AP, "0"] + extra_args,
                         env=ee, cwd=FIX, capture_output=True, text=True)
    pc = subprocess.run([CC, HF, dc, AP, "0"] + extra_args,
                        env=ee, cwd=FIX, capture_output=True, text=True)
    ok = True
    if ppy.returncode == 0:
        ok = False; print("  ✗ py 居然跑通了 — 这一格的前提没了, 改夹具")
    else:
        tail = [l for l in ppy.stderr.strip().splitlines() if l.strip()][-1:]
        print(f"    ok| py 拒跑(rc={ppy.returncode}): {tail[0] if tail else ''}")
    if pc.returncode == 0:
        ok = False; print("  ✗ C 居然跑通了 — 与 py 不一致")
    elif pc.returncode < 0 or pc.returncode > 128:
        ok = False; print(f"  ✗ C 是被信号打死的(rc={pc.returncode}) — 应该是带话的停车不是段错误")
    else:
        tail = [l for l in pc.stderr.strip().splitlines() if "zlayer Error" in l]
        if not tail:
            ok = False; print(f"  ✗ C 的错话里没有 'zlayer Error'(战役脚本 grep 不到): {pc.stderr[-300:]}")
        else:
            print(f"    ok| C 拒跑(rc={pc.returncode}): {tail[0]}")
    return ok


def recs(buf):
    """dql 记录流: [12B 头(nrec 在 offset 8)] + nrec×(116B 头 + 载荷)"""
    n, = struct.unpack_from("<I", buf, 8)
    out, off = [], 12
    for _ in range(n):
        nm = buf[off:off + 16].split(b"\0")[0].decode("ascii", "replace")
        psz, = struct.unpack_from("<Q", buf, off + 88)
        out.append((nm, buf[off:off + 116], buf[off + 116:off + 116 + psz]))
        off += 116 + psz
    return out


def f16(b):
    return np.frombuffer(b, dtype=np.float16).astype(np.float64)


def cmp_records(rpy, rc):
    """记录级对拍。结构/头/标量字段【逐位】; U/V 载荷因 QR/rSVD 数值路径不同, 只比数值。"""
    ok = True
    if [r[0] for r in rpy] != [r[0] for r in rc]:
        print(f"  ✗ 记录名序列不同 py={[r[0] for r in rpy]} C={[r[0] for r in rc]}")
        return False
    for i, ((nm, hpy, ppy), (_, hc, pc)) in enumerate(zip(rpy, rc)):
        tag = f"记录{i} {nm}"
        if hpy != hc:
            ok = False
            print(f"  ✗ {tag}: 116B 头不逐位")
        else:
            print(f"    ok| {tag}: 116B 头逐位相同(含 psz={len(ppy)})")
        if nm.startswith("bf.GE"):
            if ppy == pc:
                print(f"    ok| {tag}: 载荷逐位相同")
            else:
                d = np.abs(f16(ppy) - f16(pc)).max()
                ok = ok and d == 0
                print(f"    {'ok' if d == 0 else '✗ '}| {tag}: max|Δ|={d:.3e}")
        elif nm.startswith("zl.RRR"):
            a = struct.unpack_from("<IfII", ppy, 0)
            b = struct.unpack_from("<IfII", pc, 0)
            if a != b:
                ok = False
                print(f"  ✗ {tag}: 头字段(k,tr,din,dout) py={a} C={b}")
            else:
                print(f"    ok| {tag}: 头字段(k,tr,din,dout)={a} 逐位相同")
            k, _, di, do = a
            def split(p):
                h = f16(p[16:16 + 2 * (k + do * k + di * k)])
                return h[:k], h[k:k + do * k].reshape(do, k), h[k + do * k:].reshape(di, k)
            z1, U1, V1 = split(ppy); z2, U2, V2 = split(pc)
            # 单个 U/V 因基不同不可比; 可比的是它们张成的那个低秩矩阵 V·diag(z)·Uᵀ
            M1 = (V1 * z1) @ U1.T
            M2 = (V2 * z2) @ U2.T
            rel = np.abs(M1 - M2).max() / max(np.abs(M1).max(), 1e-30)
            good = rel < 2e-2
            ok = ok and good
            print(f"    {'ok' if good else '✗ '}| {tag}: 重构积 V·diag(z)·Uᵀ 相对偏差 {rel:.2e}"
                  f"(闸 2e-2; f16 载荷本身量化误差就在 1e-3 量级)")
            print(f"    ..| {tag}: 奇异值 z 相对偏差 {np.abs(z1-z2).max()/max(np.abs(z1).max(),1e-30):.2e}")
        elif nm.startswith("zl.ERF"):
            ne1, r1, _ = struct.unpack_from("<IHH", ppy, 0)
            ne2, r2, _ = struct.unpack_from("<IHH", pc, 0)
            if (ne1, r1) != (ne2, r2) or ppy[:8] != pc[:8]:
                ok = False
                print(f"  ✗ {tag}: 头 (ne,r) py=({ne1},{r1}) C=({ne2},{r2})")
            else:
                print(f"    ok| {tag}: 头 <IHH>(ne={ne1}, r={r1}, 0) 逐字节相同")
            Dd = 4096
            F = (len(ppy) - 8 - ne1 * (8 + Dd * r1 * 2)) // (ne1 * r1 * 2)
            one = 8 + Dd * r1 * 2 + r1 * F * 2
            for e in range(ne1):
                o = 8 + e * one
                e1, t1 = struct.unpack_from("<If", ppy, o)
                e2, t2 = struct.unpack_from("<If", pc, o)
                if e1 != e2:
                    ok = False
                    print(f"  ✗ {tag}: 第 {e} 条专家号 py={e1} C={e2}(专家序必须逐位)")
                    continue
                dt = abs(t1 - t2) / max(abs(t1), 1e-30)
                good = dt < 1e-3
                ok = ok and good
                U1 = f16(ppy[o + 8:o + 8 + Dd * r1 * 2]).reshape(Dd, r1)
                V1 = f16(ppy[o + 8 + Dd * r1 * 2:o + one]).reshape(r1, F)
                U2 = f16(pc[o + 8:o + 8 + Dd * r1 * 2]).reshape(Dd, r1)
                V2 = f16(pc[o + 8 + Dd * r1 * 2:o + one]).reshape(r1, F)
                M1, M2 = U1 @ V1, U2 @ V2
                rel = np.abs(M1 - M2).max() / max(np.abs(M1).max(), 1e-30)
                gm = rel < 5e-2
                ok = ok and gm
                print(f"    {'ok' if good and gm else '✗ '}| {tag}: e={e1} τ 相对偏差 {dt:.2e}"
                      f"(闸 1e-3), 重构积 U·V 相对偏差 {rel:.2e}(闸 5e-2)")
    return ok


def main():
    if not os.path.exists(os.path.join(FIX, "model.gguf")):
        print("先跑 zlayer_fixture.py")
        return 1
    res = []
    res.append(("A 非XCAP+VQ", cmp_case("A", {}, ["64", "0"])))
    res.append(("B 非XCAP+GGUF", cmp_case("B", {"DS4_ZL_GGUF": os.path.join(FIX, "model.gguf")}, ["64", "0"])))
    res.append(("C XANCHOR", cmp_case("C", {"DS4_ZL_XANCHOR": AP2}, ["64", "0"])))
    res.append(("D XANCHOR+ADDON", cmp_case("D", {"DS4_ZL_XANCHOR": AP2, "DS4_ZL_ADDON": "1"},
                                            ["64", "1"], with_dql=True)))
    res.append(("E ERF", cmp_case("E", {"DS4_ZL_ERF": "1", "DS4_ZL_ERF_BAR": "1.0", "DS4_ZL_ERF_R": "4",
                                        "DS4_ZL_GATE": "99"}, ["64", "1"], with_dql=True)))
    # ★G★ 上面 D 的 k 曲线是负的 ⇒ K=0 ⇒ 合并退化成"只有旧记录"这一块, 没真拼过。
    # 把闸压到 -100 逼 K=64 出来, 才走到 [旧 | 新] 两块拼接 + 重新截断那条路;
    # 顺带把 ftA 打开: 新块 din=3D 而旧块 din=D, 于是还走到 _lift 的零填充分支。
    res.append(("G ADDON拼接+ftA", cmp_case("G", {"DS4_ZL_XANCHOR": AP2, "DS4_ZL_ADDON": "1",
                                                 "DS4_ZL_FTA": "1", "DS4_ZL_GATE": "-100"},
                                            ["64", "1"], with_dql=True)))
    # ★H★ .py 的潜伏坑: 关 ftA + 线性 k 曲线全负 + 闸比它更低 ⇒ py 选形态选成 ftA 但
    # Af/Sf/Bf 是 None → TypeError。这一格判的是"两边都拒跑", 不是两边都出结果。
    res.append(("H py潜伏坑(两边都该拒跑)", cmp_both_fail(
        "H", {"DS4_ZL_XANCHOR": AP2, "DS4_ZL_ADDON": "1", "DS4_ZL_GATE": "-100"}, ["64", "1"])))
    print("\n===== 汇总 =====")
    for n, o in res:
        print(f"  {'通过' if o else '不过'}  {n}")
    return 0 if all(o for _, o in res) else 1


if __name__ == "__main__":
    sys.exit(main())
