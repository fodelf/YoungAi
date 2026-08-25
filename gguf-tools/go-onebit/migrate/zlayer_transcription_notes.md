# zlayer.py→zlayer.c 转录歧义点清单(zlayer-port agent, 2026-08-25)
# 对拍偏差归因速查在文末。金标纪律: z%/k曲线 对 py-GPU 路金标; GE/组合 对 py-CPU 路金标
# (CUDA_VISIBLE_DEVICES= 强制)。

A 影响数值:
1★ py 自身 GPU/CPU 路在 XCAP 下语义分裂: GPU路跳学生块→pYQ空→GE恒1.0(战役里 GE均值
   1.0000 的真相); C 转录 CPU 路(GE 真解)。amp_campaign 走 GE=0 不受影响。
2★ colw/Ra 实为 f32(dH.var 不升精度), 解算时才升 f64 — C 照抄精度阶梯(最易被"顺手改好")。
3  ftA 支线 Gram 在 f32(sgemm+f32加脊→再升f64); 线性支线全程 f64 — 照抄。
4  swiglu 双侧 clip(py) vs dq_expert_fp 单侧 — 按 py; SWLIM=0 时 cupy/CPU 又分叉, 按 CPU。
5  路由权重在 w2 gemm 之后乘(py) — 自带 zl_expert_fwd, 未用 dq_expert_fp。
6  zl_phi 两个精度变体(f64入/f32入) — 双实现。
7  numpy 标量提升版本依赖(NEP50): gf[pe]-1.0 取 f64(numpy1.x 语义); (pw*pYQ) 保持先f32舍入。
8  SVD=子空间迭代12轮+Rayleigh-Ritz, 过采样 kmax+64; k曲线读数 vs numpy 差 ~0.001pp;
   偏差>0.1pp 先查 2/3 别赖 SVD。
9  solve 后端: 对偶ridge=Cholesky, GE 256×256=部分主元LU; 有/无 BLAS 构建间也不逐位。
B 只影响打印: 10 norm=f32 sdot(C用f64累加,%.1f可能末位差) 11 median/percentile末位
   12 pairwise sum 亚ULP 13 ge.mean 用 f32 累加器(照抄, 否则 %.4f 翻末位)。
C 照抄的坑: 14 sign(0)=0→inf(PREV兜底) 15 _dom_recov 恒叠GE效应 16 tr=0.5(已同步)。
D 范围: 17 INJ=2 zrec外挂已实现(zside 需要, 不实现会破坏性追加) 18 PREV 实现(ftA+PREV
   拒跑) 19 env 白名单9+GE_LAM+CACHE_ONLY; GGUF/XANCHOR/ADDON 硬拒; ERF 一期跳过打印
   20 zcache 写 ZIP32(np.load 实测可读; >4GB 拒写) 21 新增只读 guard(锚magic/形状/账本
   空行/越界), 致命信息带 Error/assert 前缀保 grep 可见; macOS posix_fadvise no-op。
归因速查: GE/组合不对→1; ≤0.1pp→8/9/12 正常; >0.1pp→2/3; 只 ftA 不对→3/6; 末位→B组。
