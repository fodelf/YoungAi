# -*- coding: utf-8 -*-
"""问题定义文档配图：
图1 三联 —— ①等体积对照(我方 vs 官方q2 vs FP) ②损失归属(两个oracle上界互证) ③度量陷阱
图2     —— 探索环境接口与一轮最小闭环
所有数字均为本项目实测，口径见正文。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
import os

FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Hiragino Sans GB"
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2933"
MUTED = "#6b7280"
BLUE = "#2563eb"
RED = "#dc2626"
AMBER = "#b45309"
GREEN = "#15803d"
GRID = "#e5e7eb"


# ==================== 图 1 ====================
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.45))

# ---- ① 等体积对照：三项指标 我方 vs 官方 q2（各自归一到官方=1.0 便于同图）----
ax = axes[0]
names = ["top-1 一致率\n(越高越好)", "KL 散度\n(越低越好)", "困惑度比\n(越低越好)"]
ours = [78.78, 0.4522, 1.430]
offi = [77.92, 0.4207, 1.3575]
# 统一成"相对官方的百分比优劣"，正=更好
rel = [(ours[0] - offi[0]) / offi[0] * 100,
       -(ours[1] - offi[1]) / offi[1] * 100,
       -(ours[2] - offi[2]) / offi[2] * 100]
cols = [GREEN if v > 0 else RED for v in rel]
ax.bar(range(3), rel, width=0.5, color=cols, zorder=3)
ax.axhline(0, color="#111827", lw=1.2, zorder=4)
for i, (v, o, f) in enumerate(zip(rel, ours, offi)):
    va = "bottom" if v > 0 else "top"
    off = 1.4 if v > 0 else -1.4
    ax.text(i, v + off, f"{v:+.1f}%", ha="center", va=va, fontsize=10,
            weight="bold", color=INK)
    ax.text(i, -29.0, f"我方 {o:g}\n社区 {f:g}", ha="center", fontsize=8,
            color=MUTED, linespacing=1.4)
ax.text(0.02, 0.96, "社区 q2 件 = 0 基准线", transform=ax.transAxes,
        fontsize=8.2, color=MUTED, va="top")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=8.3)
ax.set_ylabel("相对社区 q2 件的优劣 (%)", fontsize=9)
ax.set_ylim(-34, 14)
ax.set_title("① 等体积对照：赢了主峰，输在尾部", fontsize=10.3, color=INK,
             weight="bold", pad=7)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8.2)

# ---- ② 损失归属：两个 oracle 上界互证 ----
ax = axes[1]
labs = ["当前最好\n(M10 全域)", "输出残差侧\n作弊上界", "路由选择侧\n作弊上界"]
vals = [0.4522, 0.4522, 0.3823]
cols2 = ["#9aa5b1", "#9aa5b1", GREEN]
ax.bar(range(3), vals, width=0.5, color=cols2, zorder=3)
ax.axhline(0.4207, color=RED, ls="--", lw=1.5, zorder=4)
ax.text(-0.44, 0.4232, "社区 q2 件 = 0.4207", fontsize=8.2, color=RED,
        ha="left", va="bottom")
for i, v in enumerate(vals):
    if i == 2:
        ax.text(i, v - 0.008, f"{v:.4f}", ha="center", fontsize=9.4,
                weight="bold", color="white")
    else:
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9.4,
                weight="bold", color=INK)
ax.text(1, 0.4335, "回收 ≈ 0\n(容量再大也\n 永不转正)", ha="center", fontsize=8.0,
        color=MUTED, linespacing=1.45)
ax.annotate("", xy=(2, 0.3900), xytext=(2, 0.4480),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7))
ax.text(2.02, 0.4245, "-15.5%\n全面越线", ha="center", fontsize=8.4,
        color=GREEN, weight="bold", linespacing=1.4)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=8.3)
ax.set_ylabel("KL 散度 (对未量化模型)", fontsize=9)
ax.set_ylim(0.355, 0.475)
ax.set_title("② 损失藏在哪：作弊上界定位", fontsize=10.3, color=INK,
             weight="bold", pad=7)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8.2)

# ---- ③ 度量陷阱 ----
ax = axes[2]
x = [0, 1]
keep = [0.90, 0.52]
ax.bar(x, keep, width=0.46, color=[BLUE, RED], zorder=3)
ax.axhline(1.0, color=MUTED, ls="--", lw=1.0, zorder=2)
ax.text(1.5, 1.015, "未量化模型 = 1.00", fontsize=8.2, color=MUTED,
        ha="right", va="bottom")
for xi, v, tag in zip(x, keep, ["劣化 0.96×", "劣化 3.03×"]):
    ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=11.5,
            weight="bold", color=INK)
    ax.text(xi, v / 2, tag, ha="center", fontsize=8.6, color="white",
            weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["容易文本\n(高度可预测)", "高难度多样文本\n(叙述+推理+混合体裁)"],
                   fontsize=8.3)
ax.set_ylabel("能力保留率", fontsize=9)
ax.set_ylim(0, 1.22)
ax.set_title("③ 度量陷阱：同方案，换判据集", fontsize=10.3, color=INK,
             weight="bold", pad=7)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8.2)

plt.tight_layout(pad=0.8, w_pad=2.2)
fig.savefig(os.path.join(OUT, "fig1_evidence.png"), dpi=250,
            bbox_inches="tight", facecolor="white")
plt.close(fig)


# ==================== 图 2：探索闭环 ====================
def box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.45)


def arrow(ax, p1, p2, color=MUTED, rad=0.0, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(9.0, 3.75))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, 0.008, 0.235, 0.205, 0.615,
    "不可改的规则\n\n"
    "① 体积冻结\n   与社区 q2 件同档\n"
    "② 零训练\n   只做后训练量化\n"
    "③ 判据集冻结\n   与校准语料零重叠\n"
    "④ 全层全专家平权\n   不做任何裁剪",
    "#f2f4f7", "#cbd2d9", fs=8.2)

y1, h1 = 0.615, 0.205
xs = [0.245, 0.425, 0.605, 0.785]
w1 = 0.158
for x, t in zip(xs, [
        "① 提出假设\n(等体积改动)",
        "② 落成配置\n量化出候选件",
        "③ 分钟级层扫描\n取参数甜点",
        "④ 冻结判据集\nKL / top-1 / PPL比"]):
    box(ax, x, y1, w1, h1, t, "#e8effc", BLUE, fs=8.5)
for i in range(3):
    arrow(ax, (xs[i] + w1, y1 + h1 / 2), (xs[i + 1], y1 + h1 / 2))

y2, h2 = 0.285, 0.205
box(ax, 0.785, y2, w1, h2, "⑤ 通用尺终判\nwikitext-2 对 FP", "#e6f4ea", GREEN, fs=8.5)
box(ax, 0.545, y2, 0.205, h2, "⑥ 判决 vs 社区件\n成功 / 失败同等入档", "#e6f4ea", GREEN, fs=8.5)
box(ax, 0.245, y2, 0.265, h2, "⑦ 更新假设 · 关闭分支\n· 修正问题定义", "#f6ecff", "#7c3aed", fs=8.5)
arrow(ax, (0.785 + w1 / 2, y1), (0.785 + w1 / 2, y2 + h2), color=GREEN)
arrow(ax, (0.785, y2 + h2 / 2), (0.750, y2 + h2 / 2), color=GREEN)
arrow(ax, (0.545, y2 + h2 / 2), (0.510, y2 + h2 / 2), color="#7c3aed")
arrow(ax, (0.3235, y2 + h2), (0.3235, y1), color="#7c3aed", ls="--", lw=1.15)
ax.text(0.335, (y2 + h2 + y1) / 2,
        "闭环：每轮必产出一个可陈述结论（正向发现 / 上界与归属 / 稳定负结果 / 问题修正）",
        fontsize=7.9, color="#7c3aed", ha="left", va="center")

box(ax, 0.008, 0.045, 0.984, 0.115,
    "记录与预算：每次实验留存 假设 / 配置 / 原始输出 / 判决 / 结论   ｜   单次验证 ≤ 10 分钟   ｜   量化全程内存看门狗强制",
    "#fff8e6", "#d9a441", fs=8.3)

ax.text(0.008, 0.905, "图 2  探索环境接口与一轮最小闭环",
        fontsize=10.5, color=INK, weight="bold", ha="left")
plt.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "fig2_loop.png"), dpi=250,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print("figs done:", sorted(os.listdir(OUT)))
