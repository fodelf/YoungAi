"""v41_cli — v41_teacher.py 的命令行定义(2026-09-12 从主文件拆出, 主文件超 500 行仓规)。
只有参数声明, 零逻辑。每个开关的"为什么"写在 help 里, 主文件的用法段不再重复。"""
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf", help="模型目录: FP 的 HF 目录, 或 v41_quantize 产出的量化目录(学生从文件读回)")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--layers", type=int, default=0, help="只跑前 N 层(0=全部)")
    ap.add_argument("--out", default="")
    ap.add_argument("--qnbit", type=int, default=0,
                    help="主干专家量化位宽(0=不量化, 走 FP 教师); 落地 bpw = nbit + 8/blk")
    ap.add_argument("--qblk", type=int, default=32, help="每几个元素一个 ue8m0 scale")
    ap.add_argument("--dump-act", default="",
                    help="把每层 MoE 输入的 E[x^2] 存成 f32[nlayer][dim](金融语料校准用)")
    ap.add_argument("--vq-nc", type=int, default=0,
                    help="VQ 码字数(0=不走 VQ); 配 --vq-dim, bpw = ceil(log2 nc)/dim + 行增益")
    ap.add_argument("--vq-dim", type=int, default=8)
    ap.add_argument("--vq-iters", type=int, default=8)
    ap.add_argument("--vq-stride", type=int, default=8, help="训练码本的采样步长(最终 assign 仍全量)")
    ap.add_argument("--dump-moe", default="",
                    help="dump 每层 MoE 的输入 x 与输出 y(f32) 到目录: 解放大器的原料")
    ap.add_argument("--force-x", default="",
                    help="每层 MoE 的输入强制替换成该目录里教师的 x(层内解算, 隔断上游误差传播)")
    ap.add_argument("--amp-online", default="",
                    help="★在线序贯反修★: 一次前向里逐层解出放大器并立刻应用, 放大器存到该目录。"
                         "靶当场重算(关量化开关再跑一遍 ffn), 不需要预先 dump 的教师 y")
    ap.add_argument("--amp-k", type=int, default=128)
    ap.add_argument("--amp-lam", type=float, default=10.0,
                    help="★ridge 强度★: held-out K×λ 扫描的峰值是 10~30。曾硬编码 1e-3"
                         "(近乎无正则) ⇒ 泛化 −66.5%, 端到端 Σmin 0.6711→0.2431")
    ap.add_argument("--amp-scan-split", type=int, default=6144,
                    help="K 扫描的拟合/评估切分(无 <ids>.layout 时的前缀切): 前 N 个 token 解, 其余只评估(必须 > 5120)")
    ap.add_argument("--amp-scan-k", action="store_true",
                    help="★K 扫描诊断★: 每层对一串候选秩各算一次残差, 给 K↔质量曲线"
                         "(末位=完整最小二乘解=低秩这条路的天花板); 只诊断不应用修正")
    ap.add_argument("--amp", default="",
                    help="放大器目录(v41_amp_solve 产物): 每层 MoE 输出加 B·(A·x)")
    ap.add_argument("--amp-layers", default="",
                    help="配 --amp: 只挂这些层的放大器, 如 '0' / '0-9' / '0,5,7-9'(消融用); 空=全挂")
    ap.add_argument("--amp-select", default="",
                    help="★在线序贯反修·逐层择优★: 每层先 K×λ 分层 held-out 扫描, 按规则选 (λ,K) 再解、"
                         "再应用; 放大器 + manifest 存到该目录。held-out 按 <ids>.layout 的窗分层")
    ap.add_argument("--amp-rowdiag", default="",
                    help="★逐行诊断★ 形如 K:λ(如 256:3): 每层按 (λ,K) 在拟合行解一次, 报能量增益 vs 逐行增益、"
                         "顶 1% 行能量份额 —— 查'层内能量正 / 端到端负'的分叉在哪。不应用修正")
    ap.add_argument("--amp-t2diag", default="",
                    help="★靶口径诊断★ 形如 K:λ:<教师dump目录>: 每层解同 x 放大器, 对照教师 --dump-moe 的"
                         " y_fp(x_fp), 报 T2/T1 能量比、cos(T1,T2)、修掉 T1 后总误差的涨跌。不应用修正")
    ap.add_argument("--exact-weights", action="store_true",
                    help="★对拍夹具★: fp4/fp8/VQ 解出的权重留 f32 不舍 bf16(官方 fp4/fp8 kernel 就是用精确权重算的;"
                         " 舍 bf16 是本夹具为了喂官方 bf16 模型声明做的近似)。引擎用的是文件精确值, 对拍用这个口径")
    ap.add_argument("--no-engram", action="store_true",
                    help="★对拍夹具★: 摘掉 engram 层(L1/L14 的 n-gram 查表加法), 给引擎 P2 分段对拍用 —— 引擎先不带"
                         " engram 跑通其余链路, 两边同口径比 logits。不是生产配置")
    ap.add_argument("--amp-whiten", type=int, default=0,
                    help="★输出通道白化★ 0=原始(按能量截断) 1=按靶通道 RMS 白化 2=按 y_fp 通道 RMS 白化。"
                         "只改 SVD 截断保留哪些方向(满秩解不变): 防止秩全花在巨值通道上。"
                         "扫描/择优/解/逐行诊断统一吃这个开关")
    ap.add_argument("--fp-dir", default="",
                    help="★量化目录当学生时必给★: FP 原始 HF 目录。反修的靶 y_fp(x_q) 要在同一个 x 上"
                         "重跑 FP 的 ffn, 权重从这里装(量化目录里只有 VQ/FP4, 没有 FP)")
    ap.add_argument("--act", default="",
                    help="校准激活(--dump-act 的产物): 给 VQ 的最近邻加列权 E[x^2], 误差优先落在低能量通道")
    ap.add_argument("--qcb", default="",
                    help="码本文件(v41_codebook 产出); 不给则拒绝量化 —— 码本必须匹配源分布")
    ap.add_argument("--qpow2", type=int, default=1,
                    help="1=scale 取 2 的幂(真格式 ue8m0); 0=精确 f32 σ(仅诊断, 不可落地)")
    return ap.parse_args()
