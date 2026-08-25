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

# ── 二期(非XCAP / DS4_ZL_GGUF / XANCHOR / ADDON / ERF), zlayer-port 第二棒, 2026-08-25 ──
# 一期条目 1-21 全部不变。二期只增分支, 没动一期已金标路径的数值语义 —— 只有一处例外:
# 条目 13 是错的, 见下面 22。

A 影响数值(接着一期编号):
22★【一期修正】numpy 对 f16 数组的 .mean() 是 f32 累加/相除【再舍回 f16】
   (_methods._mean 结尾 `if is_float16_result: ret = arr.dtype.type(ret)`)。一期条目 13
   只写了"f32 累加器", 漏了最后这一步舍回。夹具金标撞出来: 只有 3 个专家被路由到时
   py 打 GE均值 1.0010 而 C 打 1.0005 —— 差的正是 f16 在 1.0 附近的一格(2⁻¹⁰)。
   一期是在 256 个专家全被路由到的 XCAP 层上对的拍, 两边恰好落同一格所以没露。
   ★只影响 ★行那一个打印字段, 不碰任何载荷字节★ —— 一期的产物不用重做。
23★ iq2_xxs 的【解码网格 ≠ 编码器搜索网格】。gguf-tools/quants.c:714 那张 kgrid[256]
   是 2bit 打包表, 与 gguf-py 的 grid_hex 逐位相同(已核对), 可以直接用; 但 quants.c 里
   把码展开成 2*l+1 ∈ {1,3,5,7} 的那一步是【IQ2_XXS 编码器的搜索空间】。真正的解码值是
   {0x08, 0x19, 0x2b}(metal/moe.metal:23 的 ds4_metal_iq2xxs_grid = llama.cpp
   iq2xxs_grid = gguf-py IQ2_XXS.grid_map)。照 2*l+1 写与 gguf-py 对拍 max|Δ|=15 ——
   数量级都不对但代码一声不吭。码 3 在这张表里从不出现, C 里留了 assert。
24  ADDON 的合并 = 对 M=Pc·Qcᵀ 做截断 SVD。py 走 np.linalg.qr(LAPACK Householder) 两次
   + 小核 SVD; C 手写 Householder(hh_qr)。两者的 Q/R 符号约定与舍入都不同 ⇒ 载荷 U/V
   【不逐位】, 但 V·diag(z)·Uᵀ 这个乘积逐位无关(数学唯一)。夹具实测相对偏差 1.9e-4。
25  ERF 的 randomized SVD: py 的 np.linalg.svd 吃 f32 出 f32; C 升 f64 算完再落 f32。
   合理(更稳), 但 τ 与 U/V 由此不逐位。夹具实测 τ 相对偏差 ≤6.7e-8, U·V 乘积 ≤1.2e-4。
26  ERF 里 np.argsort(-_en) 是 numpy 默认 quicksort(不稳定); C 用稳定排序(同值按下标升序)。
   _en 是连续浮点能量实测不撞值; 真撞了两边 τ 也一样(同值)。
27  ERF 的 x 取 XSOLVE(= XQ0 if XAP else X0), 与 .py 的 `_X = XQ0 if XAP else X0` 同 ——
   注意这跟教师侧的 X0 不是一回事, XANCHOR 下会分叉。
28  GGUF 专家切片 = 张量【原始字节】按专家数均分(py: per=db.size//nexp, db 是 uint8),
   不是按元素数。C 里 nbytes = 元素数/块大小×块字节, 与 gguf-py 的 n_bytes 同式;
   除不尽直接停车(py 那边 // 会静默截断, 出来的是错位的专家)。

B 只影响打印: (一期 10-12 不变; 13 被 22 取代)

C 照抄的坑(接着一期):
29★【.py 的缩进事实, 不是笔误】ADDON 的读取整块嵌在 `if XAP:` 里 —— 也就是说
   【只设 DS4_ZL_ADDON 不设 DS4_ZL_XANCHOR 时 ADDON 完全不生效】。C 照抄, 另加一行提示。
   同理 PREV(第 8 个位置参数)与 XCAP 对齐自检整块嵌在 `if XCAP:` 里, 非 XCAP 时是死的。
30★【.py 的潜伏坑, C 改成停车】关了 ftA 时 py 把 curvef 全填 0 ⇒ rz_fta=0.0。若线性 k
   曲线【全负】而 DS4_ZL_GATE 比它更低(如 -100), 选形态会走成 `elif rz_fta>rz_lin:
   FORM="ftA"`, 而 Af/Sf/Bf 是 None ⇒ TypeError: 'NoneType' object is not subscriptable。
   C 照抄会是空指针段错误(比 py 还难查), 所以改成一句带 "assert 失败" 的停车, 判据一模一样。
   产线用的 GATE 是 0 或 99, 撞不到。
31  XCAP 口径下 dH 最后统一减 YQE; 非 XCAP 口径是逐专家减 wq·Yq —— 两条不能同时做。
   zcache 的 yqe/xcap 两个字段也只有 XCAP 才写(py: `**({...} if XCAP else {})`)。
32  ADDON 落地那一支在 py 里排在 `elif L in done` 【前面】: 叠加式不看"已注入过",
   它本来就是要覆盖上一轮贪心记录的。有账就 truncate 回裸底座再写, 且不重复记账。
33  GGUF 模式下 py 压根不开 dql_vq(`blob=None if _GG else open(...)`), 所以 C 也只在
   非 GGUF 时才要求侧车存在。但 ERF 段【无条件】重开 dql_vq —— GGUF 底座上它必然
   FileNotFoundError, 被 py 的 try/except 兜住打一行"ERF死层部件异常"。C 同样打这一行继续走。

D 范围(接着一期 17-21):
34  二期实现: 非XCAP / DS4_ZL_GGUF / DS4_ZL_XANCHOR / DS4_ZL_ADDON / ERF 死层部件。
   env 白名单加 DS4_ZL_GGUF / DS4_ZL_XANCHOR / DS4_ZL_ADDON / DS4_ZL_ERF_BAR / DS4_ZL_ERF_R
   (全是 .py 已有的, 没新增)。仍不做: cupy/GPU 路径。
35  GGUF 标量 dequant 只实现这条产线出现过的类型: q2_K / iq2_xxs(全q2 底座 = 专家
   w1/w3 IQ2_XXS + w2 Q2_K, 见 scripts/quant_allq2_spark.sh) + f32/f16/bf16/q8_0/q4_K。
   别的类型直接停车 —— py 那边 gguf-py 也是抛异常, 不静默出垃圾。
   来源(不重写数值表): q2_K ← ds4.c:4352 deq_q2K_row_f32; iq2_xxs 打包表 ← quants.c:714
   kgrid + 解码值表 ← metal/moe.metal:23; q4_K/q8_0 ← llama.cpp 标准式。
36  VQ 解码沿用一期的 vq_dequant(转录自 probe_behavior_spectrum.vq_dequant)。它与
   quant/vq_qc.h:265 的 vq_unpack_dequant(ds4quant_run.c 的 bytes_moe 用的那个)数值同式:
   索引位流、码本×行乘子都一样, 只有 `r=(i*dim)/cols` vs `r=i/(cols/dim)` 的写法差异
   (cols 整除 dim 时恒等)。一期版另加了 nc 越界钳位与 blob 尾部边界检查, 是只读 guard。
37  新增两个自检入口(与一期 --selftest-rng 同类, 都是 CLI 开关不是 env):
     ./zlayer --selftest-deq <ggml类型号> <块数> < blocks.bin > out.f32
     ./zlayer --selftest-qr <m> <n>        # 报 ‖QR−A‖inf 与 ‖QᵀQ−I‖inf

# ── 金标 ──────────────────────────────────────────────────────────
G1【已在本机跑通, 可复现】夹具对拍。造夹具→两边各跑一遍→逐项比:
     cd gguf-tools/go-onebit/migrate
     python3 zlayer_fixture.py          # 造 D=4096/MOEI=128/专家3/S=96 的最小夹具
     python3 zlayer_golden.py           # 退出码 0 = 全过
   需要 gguf-py(只夹具生成用: pip install --target <某处> gguf, 再 PYTHONPATH=<某处>)。
   夹具落 migrate/zlayer_fixtures/(不入库, 约 16MB)。覆盖的格子:
     A 非XCAP+VQ   B 非XCAP+GGUF   C XANCHOR   D XANCHOR+ADDON
     E ERF(注入)   G ADDON 真拼接(旧k8+新k64→km=72)+ftA(_lift 零填充, DIW=12288)
     H .py 潜伏坑(条目 30): 判两边都拒跑, 且 C 的错话能 grep 到
   比什么: stdout 逐字符(去掉计时字段) / zcache 的 prow·pe·pw 逐位 / dH·pYQ 相对 1e-5
   (两边都走 sgemm 但不是同一份 BLAS, f32 求和顺序不同, 末位必然差) /
   注入到 dql 的记录: 116B 头逐位、zl.RRR 头字段(k,tr,din,dout)逐位、
   zl.ERF 头 <IHH>(ne,r,0) 逐位、逐专家 <If> 的专家号逐位、τ 相对 1e-3、
   载荷 U/V 只比重构积(理由见 24/25)。
G2【已在本机跑通】GGUF 标量 dequant 对 gguf-py 逐位:
     PYTHONPATH=<装了 gguf 的目录> python3 migrate/zlayer_deq_check.py
   q2_K / q4_K / q8_0 / iq2_xxs / f16 / bf16 / f32 七种各 64 块, 全部【逐位相同】
   (max|Δ|=0)。条目 23 那个错就是这支抓出来的。
G3【已在本机跑通】Householder QR 自检 `./zlayer --selftest-qr <m> <n>`:
   512×64 / 4096×16 / 200×200 三档, ‖QR−A‖inf ≤ 4.4e-13, ‖QᵀQ−I‖inf ≤ 7.5e-15。
   一期的 `--selftest-rng` 也复跑过, 仍与 np.random.RandomState(1).randn(8) 逐位同。
G4【要真数据, 我没跑】非 XCAP: L20 K=64 (FP,FP) 与 py CPU 路对拍(★行 + k曲线 + 分域
   全逐字符)。命令两边一样, 只换可执行:
     env DS4_ZL_NTOK=… DS4_ZL_NFIT=… DS4_ZL_FTA=0 DS4_ZL_GE=1 \
       {python3 zlever/zlayer.py | calib/zlayer} <hf> <layers> <anchor> 20 64 0
   注意 py 侧要 CUDA_VISIBLE_DEVICES= 强制走 CPU 路(见一期条目 1)。
G5【要真数据, 我没跑】ERF: L40(amp_clean 实测 "ERF死层注入(+1记录 11.7MB, held+1.2%,
   专家125)")。对齐 ★行 的专家数/挽回%/记录长度。记录长度可以先算死:
     116 + 8 + ne×(8 + D·r·2 + r·F·2)
     = 116 + 8 + 125×(8 + 4096·8·2 + 8·2048·2) = 12,289,124 B = 11.7 MB ✓
   (反推出 r=8=默认 ERF_R、F=MOEI=2048, 与那条日志自洽。)
   载荷 U/V 不逐位(24/25), τ/专家序/记录结构逐位。
G6【没拍到, 写明】① GGUF 的 q2_K/iq2_xxs 只做了【解码器】对 gguf-py 的逐位对拍(G2),
   没在真 86G 底座上跑过整层 —— 夹具的 GGUF 用的是 Q8_0(gguf-py 只能 quantize 到它)。
   ② XANCHOR/ADDON 只有夹具级金标, 没在真链态锚上跑过。
   ③ PREV(第二轮反修)一期就实现了, 二期没动, 也没重新对拍。
