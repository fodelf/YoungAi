#!/bin/bash
# lldb_qrun.sh — 量化器 lldb 包装(SIGSEGV 归因诊断, 2026-08-03 探针 rc=139)。
# 用法: QBIN_OVERRIDE=本脚本 r30_campaign.sh probe1bf; 崩溃时打印 bt+寄存器后退出。
REAL="$(cd "$(dirname "$0")/../amp" && pwd)/ds4quant_run.r30"
exec lldb --batch -o run -o "bt 20" -o "register read x0 x1 x2 x8 x9" -o "quit" -- "$REAL" "$@"
