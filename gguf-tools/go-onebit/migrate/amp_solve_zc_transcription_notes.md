# amp_solve.py→amp_solve_zc.c 转录歧义点清单(2026-08-25)
# 目标文件: calib/amp_solve_zc.c(zcache 口径)。★别和 calib/amp_solve.c 混★ —— 那是换代
# 工具(AMPD/capnpy 口径, 四损失, 无 SVD, Cholesky), 两者输入/目标函数/选基都不同, 数值不可互拍。
# 金标纪律: 赢家/λ/k 必须完全相同; held% 差 ≤0.05pp; 载荷长度逐位同; 赢家是 rand/rnd2 时
# A、V 两块可直接 cmp 逐字节。整文件 md5 不作判据(理由见 A1/A7/A8)。

## A 影响数值(全部照抄, 无一处"顺手改好")

1★ **SVD 是唯一的算法替换。** numpy `np.linalg.svd`(LAPACK gesdd) → C 侧走 行 Gram
   `C = Y·Yᵀ` + 对称特征分解(Householder 三对角化 + 隐式 QL) + `Vt_i = Yᵀw_i/σ_i`。
   数学等价, 但**奇异向量的符号约定**和**简并子空间内的旋转**由算法决定, 与 numpy 不同。
   实测(夹具, PCA 当选): A 块 768 列里 377 同号 391 反号、V 块 411/357, **无一列"都不是"**
   —— 说明子空间与向量本身都对上了, 只差符号。符号对结果无影响(tanh 是奇函数, 基列翻号被
   U 对应行翻号抵消), 但**载荷字节必然不同**。
   附带代价: λ=σ² 把条件数平方, 尾部小奇异值相对精度掉一半(实测数据 σ_2048/σ_0 ~1e-2..1e-3,
   平方后仍远在 f64 余量内)。
   ⚠ 这条只在 V₀=PCA 或 门=pca2 当选时才落盘 —— 而 .py 的 215 条历史解算记录里
   **PCA 从未当选**(V₀=rand 100% 胜、门=rnd2 98.6% 胜), 所以实战中这条不进产物。

2★ **精度阶梯**(最容易被"顺手改好"的地方, 与 zlayer 清单 #2 同款坑):
   `dH`/`yq`/`yfp`/`eps` 全是 f32; `R = (dH*yq/(yq²+eps²)).astype(f64)` **括号里整段 f32**,
   只在最后升一次 f64。`colw = sqrt(var₀(R[tr])+1e-12) * sqrt(mean₀(yq[tr]²)+1e-12)` 左半 f64
   (R 已升)、右半 f32(m2 还是 f32)、乘积 f64。`1e-2`/`1e-12`/`1e-6` 这些 Python 标量按
   NEP50 弱标量降到 f32 参与运算, 不提升数组精度。

3★ **`yfp - yq` ≠ `dH`。** `yfp = yq + dH` 是 f32 加法, 所以 `e0` 里的 `(yfp[ev]-yq[ev])`
   带着一次 f32 往返舍入, 不等于直接用 dH。少走这一步 e0 会差, 整条 held% 跟着漂。
   同理 `rec` 里 `yh = yq*(1+g)` 的 g 必须先 `.astype(float32)` 再进乘法(矩阵乘本身是 f64)。

4  **numpy 的两套归约都要复刻。** 归约轴连续(整块 `.sum()`、`.mean(axis=1)`)走 pairwise
   求和(块 128、8 个累加器、`n2 -= n2%8` 的切法); 归约轴不连续(`.mean(axis=0)`、`.var(axis=0)`)
   走朴素逐行累加。累加类型也照抄: f32 数组的 mean 用 **f32** 累加器。`np.trace` 是跨步视图
   ⇒ 朴素求和。文件里 `pw_sq_f64/pw_sq_f32/pw_sqdiff/pw_sqres` 是前者, 手写循环是后者。

5  `np.add.at(yq, prow, pw[:,None]*pYQ)`: 乘积**先在 f32 舍入**再无缓冲累加, 累加顺序 = 对序
   (不是按 token 分组)。与 zlayer 清单 #7 同款。

6  `_phi` 里 `(M*M)` 算了两遍(一遍算 rms、一遍做中间块), 值相同; `np.maximum(M,0)` 用
   `v>0?v:0`(−0.0 → +0.0, 两者数值都是零, 且只作为乘法操作数, 无影响)。

7  `np.linalg.solve` = LAPACK **dgesv(部分主元 LU)**, 不是 Cholesky。G 虽是对称正定,
   换 Cholesky 更稳但求和顺序不同 = 改数。★换代 amp_solve.c 用的是 Cholesky, 别抄过来。★
   C 侧右端项按**转置布局**(nrhs 行 × n 列)解, 前代/回代全连续访问 —— 只是内存布局,
   不改任何一次浮点运算的操作数。

8  GEMM/LU 的求和顺序与 BLAS 不同(BLAS 分块+向量化), f64 尾位差 ~1e-13 相对。**有/无
   `-DDQ_BLAS` 两种构建之间也不逐位**(zlayer 清单 #9 同款既判)。U 落 f16(11 位尾数)后
   绝大多数条目相同, 但边界值会翻末位 ⇒ 整文件 md5 不作判据。

9  **FMA 合并必须关掉。** `#pragma STDC FP_CONTRACT OFF` 放在 include 之后(clang 认;
   实测默认构建与 `-ffp-contract=off` 构建输出逐字节相同, 说明 pragma 生效)。gcc 只认命令行
   开关, 但任务指定的构建行不带 `-march`, 基线 x86-64 无 FMA, 默认安全。**加 `-march=native`
   时务必同时加 `-ffp-contract=off`。**

10★ **macOS 上别拿本机 numpy 当 RNG 金标。** arm64 的 numpy 2.5.1 把 legacy_gauss 里
   `r2 = x1*x1 + x2*x2` 编成了一条 FMA, r2 差 1 ULP。实测 200 万次抽样 **13.79%** 的高斯值
   与非 FMA 写法差 1 ULP —— Mac 上 `RandomState(1).randn(8)` 会有一个(第 3 个)对不上,
   **这不是转录错**: spark(Linux x86-64)上 numpy 不合并, zlayer.c 同款 RNG 段已在那边过了
   逐位金标。而且这 1 ULP 到不了产物: 同样 200 万样本 `×1/√DIN` 落 f16 后 **0 个不同**;
   生产尺寸 12288×1024 的 seed 7/11 两条基与 numpy **12582912/12582912 逐位全同**。
   要复核 RNG 请在 spark 上做, 或直接比 f16 基。

11 四个投影 `tanh(X·V/s)` / `tanh(X·A/s)` 提到 (nm,gn) 双循环外面。.py 在每个组合里重算一遍,
   值完全一样 —— 纯省 4× 矩阵乘, **不是数值改动**(每个元素的操作数与舍入逐位相同)。

12 `best` 里存的是 `U/colw`(已除)而不是 .py 的裸 U。.py 在网格里除一次、出载荷时再除一次,
   两次是同一个 f64 除法, 逐位等价; C 侧只除一次并复用。

13 平局保先: 判据是 `rec > best` **严格大于**, 所以迭代顺序决定平局归属 ——
   `PCA→rand` × `pca2→rnd2` × λ `1000,300,100,30,10,3,1` × k `16,32,64,128,256,384,512,768,1024`,
   一个都不能换。

14 层闸分支写的记录名是 **`zl.AMP`**(不是 `zl.AMPD`), 载荷长 0。.py 两个分支名字就不同,
   照抄不统一 —— 合并器按 psz<16 判层闸, 名字只是标记。

## B 只影响打印 / 退出

15 `λ={lam}` 是 Python `repr(float)`。C 用 `%.1f` —— 只因为那七个字面量的 repr 恰好都是
   一位小数("1000.0"…"1.0")。**改 λ 表要重核这行**, 否则打印会和 .py 不一样。

16 stderr 上的 `[amp_solve_zc]` 进度行是 C 侧新增(铁律"进度可观测强制": 8-12 分钟/层的活
   不能是黑盒)。**stdout 仍与 .py 逐字符可比**, 金标只看 stdout。

17 参数不足时 .py 抛 IndexError(退出码 1), C 打一行用法后退出 1。`int()` 的严格语义
   (前后空白可去、整串必须是十进制)已用 `py_int()` 复刻, 报 `ValueError: invalid literal...`。

18 `--selftest` 是 C 侧新增入口, 只在 `argc==2 && argv[1]=="--selftest"` 时拦截(.py 会把它
   当锚路径去 open)。自检内容: RandomState(1).randn 前 8 值 / f16 舍入抽查 / 小矩阵特征分解
   的奇异值与正交性。不读任何输入文件。

## C 范围与硬拒(响亮失败, 不静默兜底)

19 无 xcap 时 .py 的 `anchor_layer` 还会读 ridx/rw 两段(读完丢弃) —— 锚文件短了 **.py 崩**
   (`ValueError: cannot reshape array of size 0`), C 只读 fin 段不崩。真锚三段齐全, 这条只在
   夹具上暴露(夹具已补齐三段)。属"C 比 py 宽容"的行为差, 不改。

20 `2*KMAX > min(na, DIN)` 时 .py 在 pca2 门上广播失败(ValueError), C 显式 die。
   实战 S=8192/NFIT=1638 → na=3276、DIN=12288, 满足。

21 `NFIT ≥ S` 时 .py 的 e0=0 会除零出 inf/nan, C 显式 die。

22 行 Gram 路要求 `na ≤ DIN`(否则 O(na³) 不成立/爆), 由 #20 的检查覆盖。

23 `prow`/`pw`/`pYQ` **只查在不物化**(有 yqe 时用不上; S=8192 的 pYQ 有 GB 量级)。缺键时
   与 .py 的 KeyError 行为一致。.py 第 18 行会真读进内存, 这条只差内存不差结果。

24 `DS4_ZL_XANCHOR` 是 .py 既有 env, 照抄**不算新增**(本项目禁新增 env)。C 侧没有引入任何
   新 env。

25 **成本欠账**: 纯 CPU 约 8-12 min/层(生产尺寸, 无 BLAS)、3-4 min/层(`-DDQ_BLAS`),
   43 层 ≈ 7h / 2.5h。.py 走 cupy GPU 快得多。按"spark 重计算必须 GPU 化"铁律, 若要上产线
   批量重解需补 CUDA 路 —— 与 zlayer.c 一期同款欠账。

## 对拍覆盖(Mac 合成夹具, 与 .py 同机对跑)

- **出载荷路**(S=1300 D=700 DIN=2100 NFIT=1024, 赢家 PCA×pca2, k=768):
  赢家 / λ=30.0 / k / held 1.17% / 文件长度 7526532B 全同; 头部 `scale` 逐位同;
  A、V 两块每列只差全局符号(见 A1); **独立回放复评 held: .py 1.167461% vs C 1.167444%
  (差 1.7e-5 pp)**; 整体放大量 g 的相对差 2.7e-4(= f16 量化底噪量级)。
- **层闸路 + 两条退路**(无 xcap 读锚 / 无 yqe 走 np.add.at): held 0.04% ≤0.5%,
  116B 记录**逐字节相同**(md5 69d6ccff78fb556f0b267f570e7440dc)。
- **随机基逐位**: 生产尺寸 DIN=12288 × KMAX=1024, seed 7(V₀=rand)与 seed 11(门=rnd2),
  落 f16 后与 numpy **12582912/12582912 元素全部逐位相同**。
- **构建**: 任务指定的 `gcc -O3 -o amp_solve_zc amp_solve_zc.c -lm -lpthread` 零 error 零
  warning(`-Wall -Wextra` 亦净); 默认构建与 `-ffp-contract=off` 构建输出逐字节相同。

## 未覆盖 / 待 spark 补

- **赢家是 rand/rnd2 的出载荷夹具**没构造出来(合成数据 DIN≫na 时 held 压不过 0.5% 层闸线,
  DIN≈na 时 PCA 反而赢)。但该分支的两个风险点已被别的证据钉死: 基字节由上面的逐位对拍覆盖,
  载荷写出时的转置朝向由 PCA 夹具"每列只差符号"覆盖(写反了会是乱数而不是符号对齐)。
  **真数据首跑时请直接 `cmp` A、V 两块** —— rand+rnd2 当选时那 2/3 载荷应逐字节同。
- 真 zcache(S=8192/NFIT=1638/DIN=12288)未在 Mac 上跑(盘上无存量 zcache)。
- 夹具生成与对拍脚本在本会话 scratchpad(`mkfix.py`/`mkfix2.py`/`cmprec.py`/`replay.py`),
  未入库(任务限定只交 .c + 本清单)。要长期保留请挪到 `gguf-tools/go-onebit/migrate/`。
