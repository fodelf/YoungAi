#!/usr/bin/env python3
"""punch_holes.py — ★原地★把文件全零块打成 APFS 稀疏洞(macOS fcntl F_PUNCHHOLE)。
与 sparsify_zeros.py 的区别: 那个是复制重写(需 ~文件大小 临时空间, 42G 模型会写爆盘, 07-24 险情);
本工具零额外空间。v2: ctypes 变参 fcntl 在 arm64 静默失败(打洞0B 实证) → 改 python fcntl 模块
(arg 为 24B struct fpunchhole{u32 flags,u32 rsv,i64 off,i64 len}, 失败抛 OSError 不静默)。
用法: punch_holes.py FILE...
"""
import os, sys, fcntl, struct

F_PUNCHHOLE = 99   # <sys/fcntl.h>
CH = 1 << 18       # 256K 扫描粒度(APFS 洞粒度 4K)
Z = bytes(CH)

def punch(path):
    fd = os.open(path, os.O_RDWR)
    sz = os.fstat(fd).st_size
    saved = 0; errs = 0; first_err = None
    def do_punch(a, b):
        nonlocal saved, errs, first_err
        a2 = (a + 4095) // 4096 * 4096; b2 = b // 4096 * 4096   # 4K 对齐收缩
        if b2 <= a2: return
        try:
            fcntl.fcntl(fd, F_PUNCHHOLE, struct.pack("<IIqq", 0, 0, a2, b2 - a2))
            saved += b2 - a2
        except OSError as e:
            errs += 1
            if first_err is None: first_err = e
    off = 0; run_start = -1
    while off < sz:
        b = os.pread(fd, min(CH, sz - off), off)
        if not b: break
        if b == Z[:len(b)]:
            if run_start < 0: run_start = off
        else:
            if run_start >= 0: do_punch(run_start, off); run_start = -1
        off += len(b)
    if run_start >= 0: do_punch(run_start, off)
    os.close(fd)
    msg = f"{path}: 逻辑{sz>>20}MiB 打洞{saved>>20}MiB"
    if errs: msg += f" ★{errs}次打洞失败 首错={first_err}★"
    print(msg, flush=True)

for p in sys.argv[1:]:
    punch(p)
