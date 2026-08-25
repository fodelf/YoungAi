# vq_merge_v4.py→vq_merge_v4.c 转录歧义点清单(2026-08-25)
# 金标纪律: 同参数同输入下输出 GGUF 逐字节 md5 相同(--extract-skeleton / --merge 本地 dql 目录模式)。
# 已在 Mac 本地小号夹具上跑过 md5 对拍, 通过项见文末"对拍覆盖"。

A 影响字节(全部照抄, 无一处"顺手改好"):
1★ gate.bias 烘焙的舍入点: .py 的 rb 行是 f64(alpha 与 acc 都升 f64 再乘), 加法 v+d 在 f64,
   只在 struct.pack 落一次 f32。C 侧 g_rb 必须是 double —— 我第一版写成 float[] 会多一次
   舍入, 已改回 f64(实测 α=1e-8/0.1/1e-30 双舍入敏感区逐位一致)。
2★ sorted(tens, key=off) 是稳定排序: 同 off 的零尺寸张量保持原序, 否则 sizes_by_offset 分出
   的 bytes 会串。C 用 qsort + (off, 原始下标) 双键复现稳定性。
3  f16→f32(bf.GE): 用位精确转换(次正规归一化 + inf/nan 尾数左移 13 位保位), 不用
   "算成 double 再截"。夹具里随机字节含 NaN f16, numpy 会喊 RuntimeWarning 但结果位一致。
4  GLhc 折进 ge: numpy 2.x NEP50 下 `f32数组 *= python float` 是全程 f32 乘(弱标量先降 f32),
   C 侧同样 float 乘。若将来 py 侧改 numpy 1.x 语义需重核。
5  build_opt_tensors 的记录判定顺序照抄 if/elif 链(GLdyn2 → GLdyn8 → bf.GE → zl.RRR → GLhc
   → .GL → TREF/xlayer)。".GL" 排在 GLdyn* 之后, 顺序换了会改归类。
6  GLdyn8 无 V8 块时 `continue`(不进 chain), 有 V8 时 row[15]=append 前的 len —— 两条都保。
7  zl.RRR 回退读 dql 主文件时, `break` 在 `if "zl.RRR" in nm and psz>=16` 块内、内层 if 之后:
   即"遇到第一条 zl.RRR 就停", 不管 zk 是否合法。照抄。
8  py 切片 `pay = raw[off+116 : off+116+psz]` 会自动截断到文件尾, 所以 plen=min(psz, 剩余),
   但 `off += 116+psz` 用的是未截断的 psz。C 侧 plen 截断 + 饱和加(溢出必然触发上界 break,
   与 py 大整数判定等价)。
9  write_gguf 用 seek 打洞, 末尾张量不补对齐 —— 输出文件尾不是 32 对齐, 照抄(C 用
   fseeko+写, 同样产生稀疏洞)。

B 只影响打印/退出:
10 `α={alpha}` 是 Python repr(float) 而非 %g: 自写 py_float_repr(最短往返 + 整数补 ".0" +
   decpt<=-4||decpt>16 走指数)。已对 2.5/1/0.1/1e-5/1e16/17位 等 11 个值逐字符对齐。
11 assert/异常文本: C 统一 `die()` 打 "AssertionError: ..."/"FileNotFoundError: ..." 前缀,
   不复刻 Python traceback。退出码同 py(assert/sys.exit=1, argparse 错=2)。
12 argparse 的 usage/-h 文本按同样选项顺序手写, 但不逐字符等同 py 的换行位置(py 按终端宽度
   折行)。错误措辞("unrecognized arguments"/"expected one argument"/"the following arguments
   are required: --out")保持一致。
13 不支持 argparse 的前缀缩写(py 允许 `--extract-sk`), 也不支持 `--` 终止符。脚本调用一律
   写全名, 无影响。

C 无法在 Mac 侧对拍的:
14 ssh 远端分支: py 是 Popen(list) 不过本地 shell, C 走 popen 必过 /bin/sh —— 用单引号把
   host 与远端命令串各包一个 shell 词(内嵌单引号转 '\''), 保证 ssh 收到的 argv 与 py 一致,
   远端 shell 看到的仍是 `tail -c +{skip+1} {path} | head -c {nbytes}`(路径不加引号, 同 py)。
   路径含空格/单引号时两边都会坏, 属既有行为, 未"修好"。
15 pclose 的退出码没检查(py 也只 p.wait() 不看 returncode, 只靠字节数断言)。照抄。

D 范围与前提:
16 假定小端主机(py 全用 struct "<"), 大端不支持 —— 头注释已写死前提。
17 DQL_HDR=35104 在 py 里是死常量(只出现在注释), C 侧保留为文档常量, 真 D 偏移一律走
   --down-offsets。SZ_G 同理(--gud 分支里 py 有个同值的局部同名量, C 复用全局值)。
18 blob_sizes/down_offsets 用 dict 语义实现(同键覆盖、len() 数唯一键、缺键 KeyError);
   行解析要求恰好两列且两列都是纯整数(与 py 的 split()+int() 同严格度)。单行上限 1023 字符。

对拍覆盖(全部 md5 逐字节相同, 且 stdout 除文件名外逐行相同):
- --extract-skeleton(88 张量, 含 gate/up/down/blob/opt 五类丢弃规则)
- --merge 本地模式: route-bias 烘焙(43 层 5420 武装槽) + opt 内嵌 8 张量(GLdyn2/GLdyn8+V8/
  bf.GE/GLhc 折叠/.GL/TREF/zl.RRR 正位 + dql 回退) + blob 修剪 + ds4.zchain.present
- --merge --down-offsets(非零 skip 的 D 段, 3×285MiB) + --blob-layers 切片存根(40 层 6160B)
- --merge --gud(g_off 回推 skip=128, G/U/D 三段, 2.57GiB 到失败点)
- --merge --consume(down+blob 双删路, 消费后剩余文件数一致)
- --extract-blobs(43 个 dql_vq_L*.bin 全量)
- opt 零内嵌分支; α ∈ {0.1, 1e-8, 3.7, 1e-30, 123456.789} 双舍入敏感区
未覆盖: ssh 远端分支(需双机), --dql-host 非本地 + --consume 的 sys.exit 拒绝(纯字符串路径)。
