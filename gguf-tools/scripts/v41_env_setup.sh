#!/bin/bash
# v41_env_setup.sh — V4.1 教师端环境(2026-09-11, spark 本机跑)。
#
# 【为什么要它】V4.1 的五指标需要一个可信的 FP 教师。官方 inference/ 是唯一的 ground truth
# 实现, 但它的 fp4/fp8 GEMM 走 tilelang kernel, **没有 CPU 路径** —— spark 上现装的是
# torch 2.13.0+cpu, 跑不了; 纯 CPU 硬跑一次 2653 token 前向要 ~2 小时(瓶颈在 543B 参数的
# FP4 解包), 每次量化都重跑一遍不现实。
#
# 【为什么用 venv 隔离】现有 CPU torch 是别的活在用的, 直接 pip install 会就地覆盖。
# 铁律"破坏前先保全": 装进 ~/v41env, 失败了删目录即可, 系统环境零改动。
#
# 【硬件】GB10 = compute_cap 12.1(sm_121, Blackwell), CUDA 13.0, 驱动 580.173.02。
# cu130 的 wheel 通常编译到 sm_120; sm_121 同代, 靠 PTX JIT 兼容 —— 装完必须实测
# 一次真实 GPU 矩阵乘, ★只看 cuda.is_available() 会骗人★(能报 True 但一算就
# "no kernel image is available for execution on the device")。
#
# 用法: v41_env_setup.sh [venv路径]   默认 ~/v41env
set -uo pipefail
VENV="${1:-$HOME/v41env}"
HF="$HOME/ds4-main/hf/DeepSeek-V4.1-Flash"
LOG(){ echo "[v41env $(date '+%m-%d %H:%M:%S')] $*"; }

if [ ! -x "$VENV/bin/python" ]; then
    LOG "建 venv → $VENV"
    python3 -m venv "$VENV" || { LOG "★venv 建不起来★"; exit 1; }
fi
P="$VENV/bin/python"
"$P" -m pip install -q --upgrade pip || { LOG "★pip 升级失败★"; exit 1; }

LOG "装 torch(cu130) —— 几个 GB, 慢"
"$P" -m pip install torch --index-url https://download.pytorch.org/whl/cu130 || {
    LOG "★cu130 wheel 装不上★"; exit 2; }
LOG "装官方 inference 的其余依赖"
"$P" -m pip install -q safetensors tokenizers numpy sympy tqdm Pillow transformers || {
    LOG "★依赖装不上★"; exit 3; }
# tilelang 提供官方的 sparse_attn / hc_split_sinkhorn / fp4_act_quant(KV 的 FP4 量化)。
# ★apache-tvm-ffi 必须钉 0.1.6★: pip 默认拉 0.1.13, 它改了对象注册机制, tilelang 0.1.8
# 内嵌的 TVM 一 import 就死在 "attribute '__dict__' of 'type' objects is not writable"。
# tilelang 自己声明的 ">=0.1.2,~=0.1.0" 拦不住 0.1.13(同属 0.1.x), 所以必须显式钉。
"$P" -m pip install -q "tilelang==0.1.8" && "$P" -m pip install -q "apache-tvm-ffi==0.1.6" || {
    LOG "★tilelang 装不上 —— 退路是自己用 torch 实现那三个 kernel, 但那样就没有对拍基准了★"; exit 3; }
"$P" -c "import tilelang" || { LOG "★tilelang import 失败★"; exit 3; }

LOG "=== 验证 ==="
"$P" - <<'PYEOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("★CUDA 不可用, 这个 wheel 白装★")
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
# ★真算一次★: available() 为 True 但 kernel image 不匹配时, 只有真跑才会炸
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
c = (a @ b).float()
ref = (a.float().cpu() @ b.float().cpu())
err = (c.cpu() - ref).abs().max().item() / ref.abs().max().item()
print(f"GPU 矩阵乘实测 OK, 相对最大误差 {err:.3e} (bf16 量级正常)")
print("fp4 dtype:", hasattr(torch, "float4_e2m1fn_x2"))
PYEOF
rc=$?
[ $rc = 0 ] || { LOG "★验证失败 rc=$rc★"; exit 4; }
LOG "环境就绪: $P"
LOG "下一步: tilelang(官方 kernel 需要) 或改 linear() 走 dequant+matmul"
