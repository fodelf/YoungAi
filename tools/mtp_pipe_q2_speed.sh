#!/usr/bin/env bash
# 双机 *层切分(layer-pipeline) + MTP 跨机投机* q2 smoke/测速脚本 (mtp.md Phase 1 方案A)。
#
# 拓扑 (本机 M4 扛大部分层, M1 扛 MTP + 少部分末段层):
#   本机 M4 = coordinator: 持前段 *大部分* 层 (0:N) + token_embd, 做 tokenize/sample/编排,
#             跑 -p 一次性生成 (生成文本+计时在本机, 直接打印)。
#   M1      = worker: 持末段 *少部分* 层 (N:output) + output head + MTP drafter (--mtp 草稿模型)。
#             MTP 必须跟 output head 同机 (mtp_for_worker_draft 硬约束), 故 MTP 落 M1=worker。
#
# *** reverse-connect (DS4_DIST_REVERSE_CONNECT=1) ***
#   默认 layer-pipeline 是 worker 主动拨 coordinator。本拓扑反过来:
#   M1 worker 只 listen/accept (零出站), M4 coordinator 主动拨 M1 + 主动拨 M1 的数据通道。
#   原因: macOS 本地网络隐私拒绝 M1 上 ds4 走雷电桥 bridge0 出站 (connect EHOSTUNREACH, nc 却通);
#   ssh 无头启动弹不出授权框。reverse 后 M1 零出站, 不需要该权限; M4 (可交互弹框授权) 全部出站。
#   ⟹ 一劳永逸绕开权限问题, 且正好是用户要的 M4 大层 / M1 MTP 布局。
#   注: 首次在 M4 上跑会弹"本地网络"授权框, 点允许即可 (M4 是你正在用的机器)。
#
# 数据流: M4(embed+层0:N) → hidden → M1(层N:output + output head + MTP draft) → logits → 回 M4 采样。
#
# 启动顺序 (reverse): 先起 M1 worker (listen 等待) → 再起本机 coordinator (主动拨 M1)。
#
# 安全闸 (任一触发"两边同杀, 只杀进程不删文件"): Ctrl+C / 本机 coordinator RSS 超 LOCAL_MAX_GB /
#   M1 worker RSS 超 REMOTE_MAX_GB。DS4_MEM_BUDGET_MB 和 L1 resident-budget gate 负责模型常驻硬预算；
#   可设 MEM_WATCH_MODE=footprint 做慢速诊断，但 footprint 每秒扫描 VM 会明显拖慢测速。
# !! 本机 M4 现扛大部分层 (~8G), 若 M4 被其他程序占满会 GPU OOM —— 先腾内存再跑。!!
set -uo pipefail

# ---------------- 配置 (均可 env 覆盖) ----------------
REMOTE=${REMOTE:-192.168.1.2}                  # M1 (worker), ssh 目标
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
LOCAL_DIR=${LOCAL_DIR:-/Users/fodelf/git/ds4-main}
WORKER_IP=${WORKER_IP:-192.168.1.2}            # M1 雷电 IP: worker 在此 listen 控制端口, M4 coordinator 拨此
PORT=${PORT:-5599}
# 档位: 默认 q2-imatrix 完整模型；脚本只做双机层切分 + A3 按需专家加载 smoke。
# 分层等待诊断默认：本机 0:22，M1 23:output。这样故意把更多后段层放到 M1，
# 用 --debug telemetry 观察是否 coordinator 在等 worker；若 M1 8GB 预算拒绝或变慢，再回 0:33/34:output。
MODEL=${MODEL:-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
MTP_GGUF=${MTP_GGUF:-gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf} # 草稿模型, 仅 M1 worker 加载
# 层切分 (block_count=43, layers 0..42)。本机 M4 扛大部分前段, M1 扛少部分末段 + output + MTP。
SPLIT_COORD=${SPLIT_COORD:-0:19}               # 本机 M4 coordinator 层切片 (实测 >0.79 t/s 档)
SPLIT_WORKER=${SPLIT_WORKER:-20:output}        # M1 worker 层切片 (后 23 层 + output)
CTX=${CTX:-200000}
# ---- project.md PC.5: PROMPT_PROFILE 档位 ----
# smoke (默认): 现行短问句, 历史速度可比基线; PC.1 复制投机在它身上系统性无收益
#   (无可抄结构, 第二十波实测 fire 率 0%) —— 它只测"不赔钱"。
# code-edit:   编辑型负载 (给一段代码 + 改名指令, 输出 ≈ 大段回显输入) —— Claude Code
#   真实回合的缩影, PC.1 的收益档。生成长 (NPRED 224) ⇒ 超时放宽。两档都该跑:
#   PROMPT_PROFILE=code-edit tools/mtp_pipe_q2_speed.sh
#   验收看 generation t/s (= 有效 t/s, 含投机接受) 与 dist-mtp 行 tok/call。
PROMPT_PROFILE=${PROMPT_PROFILE:-smoke}
if [ "$PROMPT_PROFILE" = "code-edit" ]; then
  NPRED=${NPRED:-224}
  RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-600}
  if [ -z "${PROMPT:-}" ]; then
    PROMPT=$(cat <<'PEOF'
下面是一段 Python 代码：
```python
def process_data(items):
    result = []
    for item in items:
        if item is None:
            continue
        value = item.strip().lower()
        if not value:
            continue
        if value in result:
            continue
        result.append(value)
    return result

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return process_data(lines)
```
请把函数名 process_data 改名为 clean_items（包括所有调用处），其他逻辑和格式保持完全不变，输出修改后的完整代码，不要任何解释。
PEOF
)
  fi
fi
NPRED=${NPRED:-48}
DRAFT=${DRAFT:-2}                              # --mtp-draft N: 分布式 MTP 默认 2；4 在当前 M1 worker 上实测负收益
NO_MTP=${NO_MTP:-1}                            # 默认关 MTP：当前 M1 drafter 尽管接受率高但端到端负收益；设 NO_MTP=0 做实验
if [ "$NO_MTP" = 1 ]; then
  COORD_MTP_ARGS=""
  WORKER_MTP_ARGS=""
else
  COORD_MTP_ARGS="--mtp-role worker --mtp-draft $DRAFT"
  WORKER_MTP_ARGS="--mtp $MTP_GGUF --mtp-role worker --mtp-draft $DRAFT"
fi
PROMPT=${PROMPT:-"写一个 Python 函数判断字符串是否回文，并解释它的原理。"}
SEED=${SEED:-1}
# 第三十二波: 设备限制更新 —— 两台都是 12G (总 24G)。worker 杀线 8→12、硬预算 8000→12000;
# 多出的 ~4G 给 worker 侧 source cache (128→2048) 和 worker backbone mlock (切片 23 层 ~4.7GiB)。
LOCAL_MAX_GB=${LOCAL_MAX_GB:-12}
REMOTE_MAX_GB=${REMOTE_MAX_GB:-12}
LOCAL_BUDGET_MB=${LOCAL_BUDGET_MB:-12000}
REMOTE_BUDGET_MB=${REMOTE_BUDGET_MB:-12000}
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-180}          # coordinator 运行超过该秒数就杀两边并保留日志摘要
# Metal prefill scratch chunk。默认 4096 会让 M1 worker 单个命令缓冲的 wired working set
# (模型常驻 ~6.88G + scratch 池) 越过 M1 Pro GPU 的 ~10.67G 工作集天花板 → kIOGPU OOM
# (= 曾经"本机爆了"的真因; 实测 currentAllocated 11.23G > recommendedMax 10.67G)。prompt 短时
# 4096 的 scratch 几乎全浪费; 512 缩小 scratch 池, worker 不再 OOM (实测 coordinator prefill 21.5 t/s)。
# 分布式 layer-slice 现在只给本机/worker 各自层切片分配 KV/压缩前沿；200K 下可在预算内把
# prefill scratch 降到真实 prompt 长度。
# 要更大上下文/吞吐再上调并实测。
PREFILL_CHUNK=${PREFILL_CHUNK:-512}
DIST_PREFILL_CAP=${DIST_PREFILL_CAP:-128}
# gather 工作单元已细化到张量级 (每活跃专家 3 个 pread 单元, decode 6 专家=18 单元),
# 8 线程才能吃满 QD; 回退 A/B 用 GATHER_THREADS=4。
GATHER_THREADS=${GATHER_THREADS:-8}
# ---- project.md P0.1/P1.1/P1.2 落地开关（默认开新路径；任意一项设 0 即回旧基线做 A/B）----
# EXPERT_PREAD=1        P1.1: 冷专家绕过 mmap 缺页+memcpy 双拷贝, pread() 单拷贝直读进 Shared scratch。
# EXPERT_PREAD_NOCACHE=1  冷读走独立 F_NOCACHE fd, 防止 1.7GiB/token 冷流冲掉热页缓存（默认 0, 单独 A/B）。
# EXPERT_SORT_IDS=1     gather 槽位按专家 id 升序 = 文件偏移升序 → 冷读顺序化（bit-exact, 不改数值）。
# EXPERT_STREAM=1       P1.2: prefill 大 chunk 激活 ≥THRESHOLD% 专家时, 整层 gate/up/down 顺序流式读,
#                       专家 id 不重映射（selected 缓冲不动）, 阈值/分块可调。
# EXPERT_IO_PROFILE=1   P0.1: 每个 routed-MoE 层调用打一行 ds4-io 分解计时
#                       (cold/hit 字节, wall/fault/memcpy/pread/drain ms, 等效带宽)。
EXPERT_PREAD=${EXPERT_PREAD:-1}
EXPERT_PREAD_NOCACHE=${EXPERT_PREAD_NOCACHE:-0}
# 第三十五波: 批 NOCACHE 阈值 24→64。24 是 K=16 时代定的 (verify ≤17 行永远不触发);
# K=64 注梯后 kc=25/33 两个大 verify 轮被误判成 prefill 冷流, 丢掉相邻轮 union 重叠
# 与 decode 热专家的页缓存复用 (wave-25 判决: verify 走 NOCACHE 输 1.68→1.48)。
# 64 让所有 verify 批 (协议上限 64 行) 走 cached fd; prefill 128-token 帧仍 NOCACHE。
EXPERT_BATCH_NOCACHE_MIN=${EXPERT_BATCH_NOCACHE_MIN:-64}
EXPERT_SORT_IDS=${EXPERT_SORT_IDS:-1}
EXPERT_STREAM=${EXPERT_STREAM:-1}
EXPERT_STREAM_THRESHOLD_PCT=${EXPERT_STREAM_THRESHOLD_PCT:-60}
EXPERT_STREAM_CHUNK_MB=${EXPERT_STREAM_CHUNK_MB:-16}
EXPERT_IO_PROFILE=${EXPERT_IO_PROFILE:-1}
# ---- project.md P2.1: 跨层路由预测 + 异步专家预取（decode 把 L+1 层 IO 藏进 L 层计算）----
# EXPERT_PREFETCH=1     drain 后用本层 hidden 在 CPU 复算下一层 router (F16 matvec, 后台线程),
#                       top-N 预测专家经 F_RDADVISE 拉进页缓存; 预测错只浪费带宽, 零正确性风险。
#                       命中率看 ds4-io 行的 pf=hits/total (P2.1 立项门槛: ≥85%)。
# EXPERT_PREFETCH_TOP   预测余量 (实际 top-6, 多预取 N-6 个对冲预测误差; 越大额外 SSD 流量越多)。
# v2 礼让式预取: 只在前台 gather 不占 SSD 的空闲窗发 1MiB 分块 readahead, mincore 跳过已缓存段,
# 新预测覆盖旧任务。EXPERT_PREFETCH_DEPTH=D: 每层用同一 hidden 预测 L+1..L+D 层 (最近的先发),
# coordinator (慢盘 2.5GB/s) 的空闲窗 (~drain 3.7ms≈9MiB) 单层吃不满 40MiB, D=2 给每层预测
# 双倍窗口。P0.3 实测: mini 盘随机/顺序都顶 ~2.4-2.6GB/s; MacBook 盘 5.5-6.5GB/s (见 ssd_bench)。
EXPERT_PREFETCH=${EXPERT_PREFETCH:-1}
# TOP=8: 链路字节预算 = 层周期 ~14ms × 4.5GB/s ≈ 63MiB; TOP=10 (69MiB) 实测超额 ⇒ 取数
# 迟到、命中反降 (1.99→1.94)。TOP=8 (55MiB) 在预算内; 迟到的少数由 STAGE_WAIT_US 在途等待
# (≤2.5ms) 转成命中。
EXPERT_PREFETCH_TOP=${EXPERT_PREFETCH_TOP:-8}
# DEPTH=2 实测回退 (1.71→1.66: 远层预测的算力/字节挤占了本就饱和的空闲窗) —— 默认回 1。
EXPERT_PREFETCH_DEPTH=${EXPERT_PREFETCH_DEPTH:-1}
# ---- project.md P2.2 低成本变体: 远程专家字节服务 ----
# P0.3 实测: mini 盘 ~2.5GB/s 全模式封顶, MacBook 盘 5.5-6.7GB/s 且在 coordinator 干活的
# ~350ms/token 里完全空闲。两机 gguf 字节相同 ⇒ coordinator 的 gather 单元一部分走本地盘、
# 一部分经雷电从 worker 盘拉 (聚合 ≈ 本地 SSD + TB 链路)。链路失败自动永久回本地, 正确性不依赖对端。
# A/B 回退: EXPERT_REMOTE_FETCH=0。CONNS = 拉取线程数 (每线程一条 TCP)。
# v2: 每连接 2 个在途请求 (服务端 pread 与网络传输重叠, RTT 出关键路径); 远程工人对本地
# 页缓存已有的单元直接 RAM 拷贝, 雷电带宽只花在真正的冷字节上。
# v3 (隔离实测: 生产协议在真实 gguf 上 4 连接 4.69GB/s, 1.17GB/s/连接):
#   - 尾部禁抢: 层尾最后 ~GATHER_THREADS 个单元只许本地 (远程 ~2ms 在途请求拖住整层收尾,
#     是 v2 实测无提升的真因); DS4_METAL_EXPERT_REMOTE_TAIL_RESERVE 可调。
#   - efetch socket 显式 4MB 缓冲 (128MB 请求超 macOS maxsockbuf 被静默拒绝)。
#   - 连接数 6: 用连接并行掩盖服务端 pread 串行 (worker 盘 QD6 仍 5.5GB/s)。
# v4 预测驱动暂存 (EXPERT_STAGE=1): decode 不再层内抢单 (18 单元被 8 本地线程秒抢,
# 远程 ~2ms 往返只会站到层尾) —— 改为: L 层 gather 期间, 远程线程用整个 ~14ms 窗口把
# L+1 层预测专家 (pf~80%) 拉进 RAM 暂存; L+1 层 gather 命中即 RAM memcpy, 仅预测错走本地盘。
# 专家权重不可变 ⇒ 暂存永不过期; 部分完成也按专家粒度安全生效。层内抢单仅保留给 prefill 大批量。
EXPERT_REMOTE_FETCH=${EXPERT_REMOTE_FETCH:-1}
EXPERT_REMOTE_FETCH_CONNS=${EXPERT_REMOTE_FETCH_CONNS:-6}
EXPERT_STAGE=${EXPERT_STAGE:-1}
EXPERT_FETCH_PORT=${EXPERT_FETCH_PORT:-$((PORT+7))}
# P0.1 屏障税回收: A3 每层 drain 从 commit+waitUntilCompleted 改 MTLSharedEvent 快路径
# host wait (同 TP rendezvous 先例), CB 状态检查推迟一拍。43 层/token 的逐 CB 调度开销直接砍掉。
# A/B 回退: EXPERT_EVENT_DRAIN=0。
EXPERT_EVENT_DRAIN=${EXPERT_EVENT_DRAIN:-1}
# ---- project.md §3.5 PC.1: 零成本复制式投机 (prompt-lookup drafting) ----
# 单 token decode 已到 ~2.1 t/s 地板 (两机 drain GPU 计算 ~175ms/token 不可再藏), M2 ≥3 必须
# 一次前向产出多个 token。drafter 不是模型: 在 transcript 上找尾部 NGRAM-gram 的最近一次
# 早先出现, 直接"抄"其后续 ≤DRAFT-1 个 token 当草稿, 连同目标 argmax 一并经既有 VERIFY 批
# (mtp.md Phase 1 协议原样复用) 一次验证; 接受前缀提交, 尾部跨机 KV 回滚。
# 匹配失败 = 不发 Round 2, 零开销退化为普通 decode; 正确性纯由目标 argmax 把关 (greedy-only,
# 脚本本就 --temp 0)。零草稿内存/算力, 不占 worker 一字节 (仅 worker 端 spec_logits ~8MiB)。
# 验证批 K token 的专家并集天然去重 ⇒ IO 按 round 摊薄 (W2 墙下投机的正确打开方式)。
# 第十九波教训 (1.82): 固定 3-gram 锚 + 固定抄 7 个 = 低精度高赌注 —— 唯一一次 verify
# (~1.5s, n_active 19-45/层) 7 个抄注全敗只换 1 token。v2 改 SuffixDecoding 式:
#   - 锚 = 最长公共后缀 (NGRAM 变最小锚长, 默认 4; 锚越长精度越高);
#   - 自适应抄长: 初始 INIT=3, 全接受翻倍 (≤DRAFT-1), (近)全拒回 INIT; 抄长 <MIN=2 不发批。
# A/B 回退: COPY_SPEC=0。接受率看 coordinator 日志 dist-mtp 行 (tok/call, verify 次数)。
COPY_SPEC=${COPY_SPEC:-1}
# DRAFT=64 (协议上限, 第三十波): 引擎已扩 K≤64 (drafts[64]/spec_logits 64 行/toks[65])。
# MAX=32 (押注上限, 第三十一波): 第三十波实测 48 注赔钱 —— r2 每 token 成本在 kc≈25
# 即饱和 (228→216ms/tok, 专家并集去重耗尽), 而抄写源分叉点之后的每个 token 都是纯损
# (sent=49 只中 29, r2=10.6s; wave29 同样 29 个 token 只花 7.2s)。
# 自适应抄长 3→6→12→24→32 (全接受翻倍, 顶 MAX)。
# smoke 档永不触发, 不影响基线。A/B: COPY_SPEC_MAX=63 复第三十波。
COPY_SPEC_DRAFT=${COPY_SPEC_DRAFT:-64}         # 验证批协议上限 (argmax + ≤DRAFT-1 个抄来的 token)
COPY_SPEC_MAX=${COPY_SPEC_MAX:-32}             # 自适应押注上限 (经济最优注型, 见上)
COPY_SPEC_NGRAM=${COPY_SPEC_NGRAM:-4}          # 最小锚长 (最长后缀匹配须 ≥ 此值才信)
COPY_SPEC_INIT=${COPY_SPEC_INIT:-3}            # 自适应抄长初始值/重置值
COPY_SPEC_MIN=${COPY_SPEC_MIN:-2}              # 抄长低于此不发验证批 (赔不起往返)
COPY_SPEC_LOG=${COPY_SPEC_LOG:-1}              # 每次 verify 打一行 anchor/sent/accepted
COPY_SPEC_ENV=""
[ "$COPY_SPEC" = 1 ] && COPY_SPEC_ENV="DS4_DIST_COPY_SPEC=1 DS4_DIST_COPY_SPEC_DRAFT=$COPY_SPEC_DRAFT DS4_DIST_COPY_SPEC_MAX=$COPY_SPEC_MAX DS4_DIST_COPY_SPEC_NGRAM=$COPY_SPEC_NGRAM DS4_DIST_COPY_SPEC_INIT=$COPY_SPEC_INIT DS4_DIST_COPY_SPEC_MIN=$COPY_SPEC_MIN DS4_DIST_COPY_SPEC_LOG=$COPY_SPEC_LOG"
IO_ENV="DS4_METAL_EXPERT_PREAD=$EXPERT_PREAD DS4_METAL_EXPERT_PREAD_NOCACHE=$EXPERT_PREAD_NOCACHE DS4_METAL_EXPERT_BATCH_NOCACHE_MIN=$EXPERT_BATCH_NOCACHE_MIN DS4_METAL_EXPERT_SORT_IDS=$EXPERT_SORT_IDS DS4_METAL_EXPERT_FULL_LAYER_STREAM=$EXPERT_STREAM DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT=$EXPERT_STREAM_THRESHOLD_PCT DS4_METAL_EXPERT_STREAM_CHUNK_MB=$EXPERT_STREAM_CHUNK_MB DS4_METAL_EXPERT_IO_PROFILE=$EXPERT_IO_PROFILE DS4_METAL_EXPERT_PREFETCH_AHEAD=$EXPERT_PREFETCH DS4_METAL_EXPERT_PREFETCH_TOP=$EXPERT_PREFETCH_TOP DS4_METAL_EXPERT_PREFETCH_DEPTH=$EXPERT_PREFETCH_DEPTH DS4_METAL_EXPERT_EVENT_DRAIN=$EXPERT_EVENT_DRAIN DS4_METAL_EXPERT_STAGE=$EXPERT_STAGE"
# 源专家 LRU（A3 fallback 用）：direct per-tensor read 虽避免了 30+GiB view OOM，
# 但实测每 token 15–16s（随机 mmap/页故障/每层 command drain），比 A3 慢一个数量级；默认回 A3。
# 可手动设置 DS4_METAL_EXPERT_OFFLOAD_DIRECT=1 做 direct 实验。
# LRU 只保活/复制专家源字节；每 token 仍走 compact A3 scratch，保持连续 GPU 读。
# HARD_COPY=1 时 cache 真正占用匿名内存保存专家副本，命中后从热 DRAM 副本拷到 scratch，避免 mmap 冷页/页缓存抖动。
LOCAL_EXPERT_SOURCE_CACHE_MB=${LOCAL_EXPERT_SOURCE_CACHE_MB:-2048}
# 第三十二波: worker 升 12G 后 source cache 128→2048。此前 worker decode 全程 hit_mib=0.0
# (128MiB 摊 23 层等于没有); coordinator 同设计 2048MiB 命中 50-100%, 全命中层 wall 1.1ms vs 冷层 10.8ms。
REMOTE_EXPERT_SOURCE_CACHE_MB=${REMOTE_EXPERT_SOURCE_CACHE_MB:-2048}
LOCAL_EXPERT_SOURCE_CACHE_LAYER_START=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_START:-3}
LOCAL_EXPERT_SOURCE_CACHE_LAYER_END=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_END:-19}
REMOTE_EXPERT_SOURCE_CACHE_LAYER_START=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_START:-20}
REMOTE_EXPERT_SOURCE_CACHE_LAYER_END=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_END:-42}
EXPERT_SOURCE_CACHE_ADMIT_AFTER=${EXPERT_SOURCE_CACHE_ADMIT_AFTER:-2}
EXPERT_SOURCE_CACHE_MLOCK=${EXPERT_SOURCE_CACHE_MLOCK:-0}
EXPERT_SOURCE_CACHE_HARD_COPY=${EXPERT_SOURCE_CACHE_HARD_COPY:-0}
EXPERT_SOURCE_CACHE_ASYNC=${EXPERT_SOURCE_CACHE_ASYNC:-1}
# 真实 GPU expert pool 目前命中率指标是“单专家命中”，但只有整层 active set 全命中才会更快。
# 实测默认开启会让 generation 从约 0.62/0.72 t/s 降到 0.40-0.49 t/s；保留为显式实验开关。
LOCAL_EXPERT_POOL_MB=${LOCAL_EXPERT_POOL_MB:-0}
REMOTE_EXPERT_POOL_MB=${REMOTE_EXPERT_POOL_MB:-0}
LOCAL_EXPERT_POOL_LAYER_START=${LOCAL_EXPERT_POOL_LAYER_START:-16}
LOCAL_EXPERT_POOL_LAYER_END=${LOCAL_EXPERT_POOL_LAYER_END:-22}
REMOTE_EXPERT_POOL_LAYER_START=${REMOTE_EXPERT_POOL_LAYER_START:-29}
REMOTE_EXPERT_POOL_LAYER_END=${REMOTE_EXPERT_POOL_LAYER_END:-42}
EXPERT_POOL_PREFETCH_LOOKAHEAD=${EXPERT_POOL_PREFETCH_LOOKAHEAD:-1}
EXPERT_POOL_PREFETCH_TOP=${EXPERT_POOL_PREFETCH_TOP:-16}
EXPERT_POOL_PREFETCH_SELF=${EXPERT_POOL_PREFETCH_SELF:-1}
EXPERT_POOL_PREFETCH_ADJACENT=${EXPERT_POOL_PREFETCH_ADJACENT:-0}
EXPERT_POOL_WAIT_INFLIGHT=${EXPERT_POOL_WAIT_INFLIGHT:-1}
EXPERT_POOL_FOREGROUND_FILL=${EXPERT_POOL_FOREGROUND_FILL:-1}
EXPERT_POOL_WARM_BATCH=${EXPERT_POOL_WARM_BATCH:-1}
EXPERT_POOL_PREFETCH_EVICT=${EXPERT_POOL_PREFETCH_EVICT:-1}
EXPERT_POOL_HIT_ONLY=${EXPERT_POOL_HIT_ONLY:-1}
EXPERT_POOL_ADMIT_AFTER=${EXPERT_POOL_ADMIT_AFTER:-1}
EXPERT_POOL_HOTLOCK_TOP=${EXPERT_POOL_HOTLOCK_TOP:-16}
EXPERT_POOL_AUTO_PIN_TOP=${EXPERT_POOL_AUTO_PIN_TOP:-16}
EXPERT_POOL_AUTO_PIN_MIN_REQ=${EXPERT_POOL_AUTO_PIN_MIN_REQ:-8}
EXPERT_POOL_AUTO_PIN_INTERVAL=${EXPERT_POOL_AUTO_PIN_INTERVAL:-64}
EXPERT_POOL_PIN_RESERVE=${EXPERT_POOL_PIN_RESERVE:-10}
EXPERT_POOL_HOTLIST_TOP=${EXPERT_POOL_HOTLIST_TOP:-24}
EXPERT_POOL_HOTLIST_INTERVAL=${EXPERT_POOL_HOTLIST_INTERVAL:-64}
EXPERT_POOL_PREFETCH_QUEUE=${EXPERT_POOL_PREFETCH_QUEUE:-8192}
EXPERT_POOL_MIN_LAYER_SLOTS=${EXPERT_POOL_MIN_LAYER_SLOTS:-16}
EXPERT_POOL_PIN_TOP=${EXPERT_POOL_PIN_TOP:-16}
EXPERT_POOL_PIN_FROM_LOGS=${EXPERT_POOL_PIN_FROM_LOGS:-0}
EXPERT_POOL_PINNED=${EXPERT_POOL_PINNED:-}
LOCAL_EXPERT_POOL_PINNED=${LOCAL_EXPERT_POOL_PINNED:-}
REMOTE_EXPERT_POOL_PINNED=${REMOTE_EXPERT_POOL_PINNED:-}

COORD_LOG=/tmp/mtp_pipe_coord.log              # 本机 M4 路径 (coordinator stderr: ds4 日志/速度/profile)
COORD_OUT=/tmp/mtp_pipe_coord.out              # 本机 M4 路径 (coordinator stdout: 纯生成文本)
WORKER_LOG=/tmp/mtp_pipe_worker.log            # M1 上的路径 (worker)
WORKER_PID_FILE=/tmp/mtp_pipe_worker.pid        # M1 上本脚本启动的 worker pid
COORD_PID=""

log(){ echo "[mtp-pipe] $*"; }

# 从上一轮 coordinator/worker 日志里的 expert-profile/expert-pool top 行生成静态钉住白名单。
# 格式传给 DS4_METAL_EXPERT_POOL_PINNED: L23:1,2,3/L24:...（用 / 避免 shell 分号转义）。
make_pinned_from_log(){
  local log_path=$1 layer_start=$2 layer_end=$3 top_n=$4
  [ -f "$log_path" ] || return 0
  perl -Mstrict -Mwarnings -e '
    my ($ls, $le, $top, $path) = @ARGV;
    my %count;
    open my $fh, "<", $path or exit 0;
    while (<$fh>) {
      next unless /expert-(?:profile|pool)\s+L(\d+)\s+E(\d+)\s+.*?req=(\d+)/;
      my ($layer, $expert, $req) = ($1 + 0, $2 + 0, $3 + 0);
      next if $layer < $ls || $layer > $le || $req <= 0;
      $count{$layer}{$expert} += $req;
    }
    my @parts;
    for my $layer ($ls .. $le) {
      next unless exists $count{$layer};
      my @experts = sort { $count{$layer}{$b} <=> $count{$layer}{$a} || $a <=> $b } keys %{$count{$layer}};
      splice @experts, $top if @experts > $top;
      next unless @experts;
      push @parts, sprintf("L%02d:%s", $layer, join(",", @experts));
    }
    print join("/", @parts);
  ' "$layer_start" "$layer_end" "$top_n" "$log_path"
}

pin_count(){
  awk -v s="$1" 'BEGIN{n=split(s,parts,"/"); c=0; for(i=1;i<=n;i++){split(parts[i],kv,":"); if(length(kv[2])==0) continue; m=split(kv[2],e,","); for(j=1;j<=m;j++) if(e[j] ~ /^[0-9]+$/) c++;} print c}'
}

if [ "$EXPERT_POOL_PIN_FROM_LOGS" = 1 ]; then
  [ -z "$LOCAL_EXPERT_POOL_PINNED" ] && [ -z "$EXPERT_POOL_PINNED" ] && \
    LOCAL_EXPERT_POOL_PINNED=$(make_pinned_from_log "$COORD_LOG" "$LOCAL_EXPERT_POOL_LAYER_START" "$LOCAL_EXPERT_POOL_LAYER_END" "$EXPERT_POOL_PIN_TOP")
  if [ -z "$REMOTE_EXPERT_POOL_PINNED" ] && [ -z "$EXPERT_POOL_PINNED" ]; then
    REMOTE_EXPERT_POOL_PINNED=$(ssh "$REMOTE" "perl -Mstrict -Mwarnings -e '
      my (\$ls, \$le, \$top, \$path) = @ARGV; my %count;
      open my \$fh, q{<}, \$path or exit 0;
      while (<\$fh>) { next unless /expert-(?:profile|pool)\\s+L(\\d+)\\s+E(\\d+)\\s+.*?req=(\\d+)/; my (\$l, \$e, \$r)=(\$1+0,\$2+0,\$3+0); next if \$l<\$ls || \$l>\$le || \$r<=0; \$count{\$l}{\$e} += \$r; }
      my @parts; for my \$l (\$ls .. \$le) { next unless exists \$count{\$l}; my @e = sort { \$count{\$l}{\$b} <=> \$count{\$l}{\$a} || \$a <=> \$b } keys %{\$count{\$l}}; splice @e, \$top if @e > \$top; push @parts, sprintf(q{L%02d:%s}, \$l, join(q{,}, @e)) if @e; } print join(q{/}, @parts);
    ' '$REMOTE_EXPERT_POOL_LAYER_START' '$REMOTE_EXPERT_POOL_LAYER_END' '$EXPERT_POOL_PIN_TOP' '$WORKER_LOG'" 2>/dev/null || true)
  fi
fi
[ -n "$EXPERT_POOL_PINNED" ] && LOCAL_EXPERT_POOL_PINNED=$EXPERT_POOL_PINNED
[ -n "$EXPERT_POOL_PINNED" ] && REMOTE_EXPERT_POOL_PINNED=$EXPERT_POOL_PINNED
LOCAL_PIN_ENV=""
REMOTE_PIN_ENV=""
[ -n "$LOCAL_EXPERT_POOL_PINNED" ] && LOCAL_PIN_ENV="DS4_METAL_EXPERT_POOL_PINNED=$LOCAL_EXPERT_POOL_PINNED"
[ -n "$REMOTE_EXPERT_POOL_PINNED" ] && REMOTE_PIN_ENV="DS4_METAL_EXPERT_POOL_PINNED=$REMOTE_EXPERT_POOL_PINNED"
# Expert offload 命中率模拟日志：不改变推理，只在 A3 offload 每次 CPU-gather 后统计
# 「如果有一个 LRU 专家池」的命中率/省下的拷贝量，并打印每层单专家 slot 内存。
# coordinator 跑的层多，1GiB 连 34 层单 token 活跃集都放不下；默认给本机模拟池 4GiB。
# 这是 profiler 的模拟 cache，不实际分配这些 expert 常驻内存；真实 expert pool 后续再按预算实现。
# EXPERT_PROFILE_ALL=1 会在进程正常退出时打印所有出现过的专家；默认只打印每层 top-N。
EXPERT_PROFILE=${EXPERT_PROFILE:-0}
LOCAL_EXPERT_PROFILE_CACHE_MB=${LOCAL_EXPERT_PROFILE_CACHE_MB:-4096}
REMOTE_EXPERT_PROFILE_CACHE_MB=${REMOTE_EXPERT_PROFILE_CACHE_MB:-1024}
# 兼容旧变量：如果只设置 EXPERT_PROFILE_CACHE_MB，则两边都用这个值。
if [ -n "${EXPERT_PROFILE_CACHE_MB:-}" ]; then
  LOCAL_EXPERT_PROFILE_CACHE_MB=$EXPERT_PROFILE_CACHE_MB
  REMOTE_EXPERT_PROFILE_CACHE_MB=$EXPERT_PROFILE_CACHE_MB
fi
EXPERT_PROFILE_TOP=${EXPERT_PROFILE_TOP:-8}
EXPERT_PROFILE_INTERVAL=${EXPERT_PROFILE_INTERVAL:-0}
EXPERT_PROFILE_ALL=${EXPERT_PROFILE_ALL:-0}
LOCAL_PROFILE_ENV=""
REMOTE_PROFILE_ENV=""
# 第二十五波: coordinator backbone mlock —— verify 批冷读不再驱逐 mmap 常驻 backbone 热页
# (r1 548→2361ms 爬升的根治; NOCACHE 收窄到 prefill 量级批, verify 批回 cached fd)。
# BACKBONE_MLOCK=0 关闭; BACKBONE_MLOCK_BUDGET_MB 调钉住预算 (coordinator 切片 ~4.07GiB)。
BACKBONE_MLOCK=${BACKBONE_MLOCK:-1}
BACKBONE_MLOCK_BUDGET_MB=${BACKBONE_MLOCK_BUDGET_MB:-4608}
if [ "$BACKBONE_MLOCK" = 1 ]; then
  LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_BACKBONE_MLOCK=1 DS4_METAL_BACKBONE_MLOCK_BUDGET_MB=$BACKBONE_MLOCK_BUDGET_MB"
fi
# 第三十二波: worker 也 mlock backbone (12G 预算下才放得下)。worker 切片 23 层 ~4.7GiB, 预算 5120。
# 动机: worker GPU drain 74ms/层 (kc=25, 占 r2 的 ~45%), 与 wave 25 coordinator 修前的
# backbone 被 verify 冷流驱逐→命令执行中 refault 同病; coordinator mlock 后 drain 42ms/层。
# WORKER_BACKBONE_MLOCK=0 单独关闭做 A/B。
WORKER_BACKBONE_MLOCK=${WORKER_BACKBONE_MLOCK:-1}
WORKER_BACKBONE_MLOCK_BUDGET_MB=${WORKER_BACKBONE_MLOCK_BUDGET_MB:-5120}
if [ "$WORKER_BACKBONE_MLOCK" = 1 ]; then
  REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_BACKBONE_MLOCK=1 DS4_METAL_BACKBONE_MLOCK_BUDGET_MB=$WORKER_BACKBONE_MLOCK_BUDGET_MB"
fi
# 第三十三波尝试 worker staging, 第三十四波回滚默认: 实测净亏 (code-edit 3.00→2.83,
# smoke 2.14→1.75)。worker 端 completion 只有 64%、dropped 33%、decode 命中仅 ~26%,
# decode wall 反升 (2-6→4-9.6ms/层) 且未暂存层 racing 复活 (125/1104), r1 全线 +20%。
# 不对称根因: mini 盘慢 (stream 1.9GB/s) 且要兼任 efetch server, 喂不饱 worker 的暂存窗口;
# coordinator 侧 staging (从 worker 快盘拉) completion 97% 才是赚的。保留 =1 可复现实验。
WORKER_EXPERT_STAGE=${WORKER_EXPERT_STAGE:-0}
if [ "$WORKER_EXPERT_STAGE" = 0 ]; then
  REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_STAGE=0"
fi
if [ "$EXPERT_PROFILE" = 1 ]; then
  LOCAL_PROFILE_ENV="DS4_METAL_EXPERT_OFFLOAD_PROFILE=1 DS4_METAL_EXPERT_PROFILE_CACHE_MB=$LOCAL_EXPERT_PROFILE_CACHE_MB DS4_METAL_EXPERT_PROFILE_TOP=$EXPERT_PROFILE_TOP DS4_METAL_EXPERT_PROFILE_INTERVAL=$EXPERT_PROFILE_INTERVAL"
  REMOTE_PROFILE_ENV="DS4_METAL_EXPERT_OFFLOAD_PROFILE=1 DS4_METAL_EXPERT_PROFILE_CACHE_MB=$REMOTE_EXPERT_PROFILE_CACHE_MB DS4_METAL_EXPERT_PROFILE_TOP=$EXPERT_PROFILE_TOP DS4_METAL_EXPERT_PROFILE_INTERVAL=$EXPERT_PROFILE_INTERVAL"
  [ "$EXPERT_PROFILE_ALL" = 1 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_PROFILE_ALL=1"
  [ "$EXPERT_PROFILE_ALL" = 1 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_PROFILE_ALL=1"
fi
[ "$LOCAL_EXPERT_POOL_MB" != 0 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_POOL_MB=$LOCAL_EXPERT_POOL_MB DS4_METAL_EXPERT_POOL_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_POOL_LAYER_START=$LOCAL_EXPERT_POOL_LAYER_START DS4_METAL_EXPERT_POOL_LAYER_END=$LOCAL_EXPERT_POOL_LAYER_END DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS=$EXPERT_POOL_MIN_LAYER_SLOTS DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD=$EXPERT_POOL_PREFETCH_LOOKAHEAD DS4_METAL_EXPERT_POOL_PREFETCH_TOP=$EXPERT_POOL_PREFETCH_TOP DS4_METAL_EXPERT_POOL_PREFETCH_SELF=$EXPERT_POOL_PREFETCH_SELF DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT=$EXPERT_POOL_PREFETCH_ADJACENT DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT=$EXPERT_POOL_WAIT_INFLIGHT DS4_METAL_EXPERT_POOL_FOREGROUND_FILL=$EXPERT_POOL_FOREGROUND_FILL DS4_METAL_EXPERT_POOL_WARM_BATCH=$EXPERT_POOL_WARM_BATCH DS4_METAL_EXPERT_POOL_PREFETCH_EVICT=$EXPERT_POOL_PREFETCH_EVICT DS4_METAL_EXPERT_POOL_HIT_ONLY=$EXPERT_POOL_HIT_ONLY DS4_METAL_EXPERT_POOL_ADMIT_AFTER=$EXPERT_POOL_ADMIT_AFTER DS4_METAL_EXPERT_POOL_HOTLOCK_TOP=$EXPERT_POOL_HOTLOCK_TOP DS4_METAL_EXPERT_POOL_AUTO_PIN_TOP=$EXPERT_POOL_AUTO_PIN_TOP DS4_METAL_EXPERT_POOL_AUTO_PIN_MIN_REQ=$EXPERT_POOL_AUTO_PIN_MIN_REQ DS4_METAL_EXPERT_POOL_AUTO_PIN_INTERVAL=$EXPERT_POOL_AUTO_PIN_INTERVAL DS4_METAL_EXPERT_POOL_PIN_RESERVE=$EXPERT_POOL_PIN_RESERVE DS4_METAL_EXPERT_POOL_HOTLIST_TOP=$EXPERT_POOL_HOTLIST_TOP DS4_METAL_EXPERT_POOL_HOTLIST_INTERVAL=$EXPERT_POOL_HOTLIST_INTERVAL DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE=$EXPERT_POOL_PREFETCH_QUEUE $LOCAL_PIN_ENV"
[ "$REMOTE_EXPERT_POOL_MB" != 0 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_POOL_MB=$REMOTE_EXPERT_POOL_MB DS4_METAL_EXPERT_POOL_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_POOL_LAYER_START=$REMOTE_EXPERT_POOL_LAYER_START DS4_METAL_EXPERT_POOL_LAYER_END=$REMOTE_EXPERT_POOL_LAYER_END DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS=$EXPERT_POOL_MIN_LAYER_SLOTS DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD=$EXPERT_POOL_PREFETCH_LOOKAHEAD DS4_METAL_EXPERT_POOL_PREFETCH_TOP=$EXPERT_POOL_PREFETCH_TOP DS4_METAL_EXPERT_POOL_PREFETCH_SELF=$EXPERT_POOL_PREFETCH_SELF DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT=$EXPERT_POOL_PREFETCH_ADJACENT DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT=$EXPERT_POOL_WAIT_INFLIGHT DS4_METAL_EXPERT_POOL_FOREGROUND_FILL=$EXPERT_POOL_FOREGROUND_FILL DS4_METAL_EXPERT_POOL_WARM_BATCH=$EXPERT_POOL_WARM_BATCH DS4_METAL_EXPERT_POOL_PREFETCH_EVICT=$EXPERT_POOL_PREFETCH_EVICT DS4_METAL_EXPERT_POOL_HIT_ONLY=$EXPERT_POOL_HIT_ONLY DS4_METAL_EXPERT_POOL_ADMIT_AFTER=$EXPERT_POOL_ADMIT_AFTER DS4_METAL_EXPERT_POOL_HOTLOCK_TOP=$EXPERT_POOL_HOTLOCK_TOP DS4_METAL_EXPERT_POOL_AUTO_PIN_TOP=$EXPERT_POOL_AUTO_PIN_TOP DS4_METAL_EXPERT_POOL_AUTO_PIN_MIN_REQ=$EXPERT_POOL_AUTO_PIN_MIN_REQ DS4_METAL_EXPERT_POOL_AUTO_PIN_INTERVAL=$EXPERT_POOL_AUTO_PIN_INTERVAL DS4_METAL_EXPERT_POOL_PIN_RESERVE=$EXPERT_POOL_PIN_RESERVE DS4_METAL_EXPERT_POOL_HOTLIST_TOP=$EXPERT_POOL_HOTLIST_TOP DS4_METAL_EXPERT_POOL_HOTLIST_INTERVAL=$EXPERT_POOL_HOTLIST_INTERVAL DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE=$EXPERT_POOL_PREFETCH_QUEUE $REMOTE_PIN_ENV"
[ "$LOCAL_EXPERT_SOURCE_CACHE_MB" != 0 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_SOURCE_CACHE_MB=$LOCAL_EXPERT_SOURCE_CACHE_MB DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START=$LOCAL_EXPERT_SOURCE_CACHE_LAYER_START DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END=$LOCAL_EXPERT_SOURCE_CACHE_LAYER_END DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER=$EXPERT_SOURCE_CACHE_ADMIT_AFTER DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK=$EXPERT_SOURCE_CACHE_MLOCK DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY=$EXPERT_SOURCE_CACHE_HARD_COPY DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC=$EXPERT_SOURCE_CACHE_ASYNC"
[ "$REMOTE_EXPERT_SOURCE_CACHE_MB" != 0 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_SOURCE_CACHE_MB=$REMOTE_EXPERT_SOURCE_CACHE_MB DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START=$REMOTE_EXPERT_SOURCE_CACHE_LAYER_START DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END=$REMOTE_EXPERT_SOURCE_CACHE_LAYER_END DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER=$EXPERT_SOURCE_CACHE_ADMIT_AFTER DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK=$EXPERT_SOURCE_CACHE_MLOCK DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY=$EXPERT_SOURCE_CACHE_HARD_COPY DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC=$EXPERT_SOURCE_CACHE_ASYNC"
DIST_DEBUG=${DIST_DEBUG:-0}
DEBUG_ARGS=""
[ "$DIST_DEBUG" = 1 ] && DEBUG_ARGS="--debug"
# reverse-connect 是本拓扑的核心 (见顶部注释)。两机都要带。
# DS4_METAL_PREFILL_CHUNK 限制每次预填充 work 的 token 数；DS4_DIST_PREFILL_CAP 进一步把
# graph 内部 batch scratch cap 钳到脚本真实 smoke prompt 量级（200K KV 仍由 -c 控制）。
# DS4_METAL_EXPERT_OFFLOAD 让 q2 routed experts 走 A3 按需 scratch；EXPERT_GATHER_THREADS
# 并行复制同层 active experts，降低 A3 memcpy 墙。per-tensor DIRECT 已验证不 OOM 但 15s/token，
# 默认关闭；需要实验时覆盖 DS4_METAL_EXPERT_OFFLOAD_DIRECT=1。
# NO_MODEL_WARMUP 避免启动时扫冷 expert views。
BASE_RUN_ENV=${BASE_RUN_ENV:-"DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_PREFILL_CHUNK=$PREFILL_CHUNK DS4_DIST_PREFILL_CAP=$DIST_PREFILL_CAP DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_EXPERT_OFFLOAD_DIRECT=0 DS4_METAL_EXPERT_GATHER_THREADS=$GATHER_THREADS DS4_METAL_NO_MODEL_WARMUP=1 $IO_ENV $COPY_SPEC_ENV"}
# 远程专家字节服务: coordinator (本机) 当客户端拉 worker 盘; worker 当服务端。
# 第三十波反向 efetch 改 accept 模式 (EXPERT_REMOTE_FETCH_REVERSE=1, 默认开):
#   目的不变 (二十六波): verify 批的 worker 半程 (~2.3s, 23 层×~270MiB 冷读) 期间
#   mini SSD 完全闲置 —— 让 worker 的 ≥96 单元批 gather racing 把部分单元经雷电拉
#   mini 盘 (聚合 ≈5.5+~2 GB/s)。
#   但 worker 进程内主动外连 mini 一律瞬时 EHOSTUNREACH (26-29 波 31 连败, 含静默窗;
#   shell 的 nc/ping/未签名测试二进制全通, 根因未明) —— 所以只反转 TCP 建立方向:
#   worker 起 ACCEPT_PORT 监听 (逻辑客户端), mini 引擎起 SERVE_DIAL 后台线程拨入
#   (逻辑 pread 服务端), 线上协议字节不变。
#   关键不变量: worker 端不设 FETCH_HOST ⇒ staging 自然不激活 (decode 期 worker
#   staging 拉 mini 慢忙盘是负收益; racing 只动 ≥96 单元批, decode 18 单元不沾)。
LOCAL_FETCH_ENV=""
REMOTE_FETCH_ENV=""
EXPERT_REMOTE_FETCH_REVERSE=${EXPERT_REMOTE_FETCH_REVERSE:-1}
if [ "$EXPERT_REMOTE_FETCH" = 1 ]; then
  LOCAL_FETCH_ENV="DS4_DIST_EXPERT_FETCH_HOST=$WORKER_IP DS4_DIST_EXPERT_FETCH_PORT=$EXPERT_FETCH_PORT DS4_DIST_EXPERT_FETCH_CONNS=$EXPERT_REMOTE_FETCH_CONNS"
  REMOTE_FETCH_ENV="DS4_DIST_EXPERT_FETCH_SERVE=1 DS4_DIST_EXPERT_FETCH_PORT=$EXPERT_FETCH_PORT"
  if [ "$EXPERT_REMOTE_FETCH_REVERSE" = 1 ]; then
    LOCAL_FETCH_ENV="$LOCAL_FETCH_ENV DS4_DIST_EXPERT_FETCH_SERVE_DIAL_HOST=$WORKER_IP DS4_DIST_EXPERT_FETCH_SERVE_DIAL_PORT=$((EXPERT_FETCH_PORT+1)) DS4_DIST_EXPERT_FETCH_SERVE_DIAL_CONNS=$EXPERT_REMOTE_FETCH_CONNS"
    REMOTE_FETCH_ENV="$REMOTE_FETCH_ENV DS4_DIST_EXPERT_FETCH_ACCEPT_PORT=$((EXPERT_FETCH_PORT+1)) DS4_DIST_EXPERT_FETCH_CONNS=$EXPERT_REMOTE_FETCH_CONNS"
  fi
fi
# RUN_ENV 仍可一把覆盖两边；LOCAL_RUN_ENV/REMOTE_RUN_ENV 可分别覆盖。
LOCAL_RUN_ENV=${LOCAL_RUN_ENV:-${RUN_ENV:-"$BASE_RUN_ENV $LOCAL_PROFILE_ENV $LOCAL_FETCH_ENV"}}
REMOTE_RUN_ENV=${REMOTE_RUN_ENV:-${RUN_ENV:-"$BASE_RUN_ENV $REMOTE_PROFILE_ENV $REMOTE_FETCH_ENV"}}

# ---------------- 清理: 两边同杀 (幂等, 只杀进程) ----------------
cleanup(){
  trap - INT TERM EXIT
  echo
  log "cleanup: 杀两边 ds4 进程 (只杀进程, 不删任何文件)"
  [ -n "$COORD_PID" ] && kill "$COORD_PID" 2>/dev/null || true
  ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null || true; pkill -f 'ds4 -m' 2>/dev/null || true" 2>/dev/null || true
  pkill -f 'ds4 -m' 2>/dev/null || true
  log "done."
}
trap cleanup INT TERM EXIT

# ---------------- 内存看门狗 (GiB) ----------------
# 默认用 ps RSS，开销小、适合测速；DS4_MEM_BUDGET_MB/L1 gate 仍是模型常驻硬预算。
# 诊断 mmap/Metal 真实 footprint 时设置 MEM_WATCH_MODE=footprint，但 footprint 很慢，会拖低 t/s。
MEM_WATCH_MODE=${MEM_WATCH_MODE:-rss}
rss_gb_local(){ local kb; kb=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0; }
rss_gb_remote(){
  local kb; kb=$(ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -z \"\$pid\" ] && pid=\$(pgrep -f 'ds4 -m' | head -1); [ -n \"\$pid\" ] && ps -o rss= -p \$pid 2>/dev/null | tr -d ' '" 2>/dev/null)
  [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0
}
footprint_gb_local(){
  local bytes; bytes=$(footprint -pid "$1" -f bytes --noCategories 2>/dev/null | awk '/phys_footprint:/ {print $2; exit}')
  [ -n "$bytes" ] && awk "BEGIN{printf \"%.2f\",$bytes/1073741824}" || rss_gb_local "$1"
}
footprint_gb_remote(){
  local bytes; bytes=$(ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -z \"\$pid\" ] && pid=\$(pgrep -f 'ds4 -m' | head -1); [ -n \"\$pid\" ] && footprint -pid \$pid -f bytes --noCategories 2>/dev/null | awk '/phys_footprint:/ {print \$2; exit}'" 2>/dev/null)
  [ -n "$bytes" ] && awk "BEGIN{printf \"%.2f\",$bytes/1073741824}" || rss_gb_remote
}
mem_gb_local(){ [ "$MEM_WATCH_MODE" = footprint ] && footprint_gb_local "$1" || rss_gb_local "$1"; }
mem_gb_remote(){ [ "$MEM_WATCH_MODE" = footprint ] && footprint_gb_remote || rss_gb_remote; }
mem_label(){ [ "$MEM_WATCH_MODE" = footprint ] && echo footprint || echo RSS; }
over(){ awk "BEGIN{a=$1+0;b=$2+0;exit !(a>b)}"; }

# ---------------- 0. 前置检查 ----------------
[ -f "$LOCAL_DIR/$MODEL" ] || { log "本机缺模型 $MODEL"; exit 1; }
[ "$NO_MTP" = 1 ] || ssh "$REMOTE" "[ -f '$REMOTE_DIR/$MTP_GGUF' ]" 2>/dev/null \
  || { log "M1 缺草稿模型 $MTP_GGUF。设 MTP_GGUF=... 覆盖, 或 NO_MTP=1 跳过"; exit 1; }
ssh "$REMOTE" "[ -f '$REMOTE_DIR/$MODEL' ]" 2>/dev/null || { log "M1 缺模型 $MODEL"; exit 1; }

# ---------------- 1. 同步代码 → M1 ----------------
log "同步源码 → $REMOTE:$REMOTE_DIR"
rsync -a --exclude '.git' --exclude '*.o' --exclude '*.gguf' --exclude 'gguf/' \
  --exclude '*.bin' --exclude 'ds4' --exclude 'ds4-server' --exclude 'ds4-bench' \
  --exclude 'ds4-eval' --exclude 'ds4-agent' --exclude 'e0-pingpong' --exclude 'ds4_test' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/ || { log "rsync 失败"; exit 1; }

# ---------------- 2. 两边 clean + build (共享 CORE_OBJS, 必须都重编) ----------------
log "本机: make clean && make ds4"
( cd "$LOCAL_DIR" && make clean >/dev/null 2>&1 && make ds4 >/tmp/mtp_pipe_build_local.log 2>&1 ) \
  || { log "本机编译失败:"; tail -8 /tmp/mtp_pipe_build_local.log; exit 1; }
log "M1:   make clean && make ds4"
ssh "$REMOTE" "cd '$REMOTE_DIR' && make clean >/dev/null 2>&1 && make ds4 >/tmp/mtp_pipe_build_remote.log 2>&1" \
  || { log "M1 编译失败:"; ssh "$REMOTE" "tail -8 /tmp/mtp_pipe_build_remote.log"; exit 1; }

# ---------------- 3. 清旧进程 + 清端口 ----------------
log "清理两边旧 ds4 进程与 $PORT 端口占用"
pkill -f 'ds4 -m' 2>/dev/null || true
ssh "$REMOTE" "pkill -f 'ds4 -m' 2>/dev/null; lsof -nP -iTCP:$PORT -t 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null || true
sleep 1

# ---------------- 4. 先起 M1 worker (control listen, 等 coordinator 来拨) ----------------
log "启动 M1 worker: --listen $WORKER_IP:$PORT --layers $SPLIT_WORKER ${WORKER_MTP_ARGS:-(无 MTP)} (reverse, 只 accept)"
ssh "$REMOTE" "cd '$REMOTE_DIR' && rm -f '$WORKER_LOG'; \
  $REMOTE_RUN_ENV DS4_MEM_BUDGET_MB=$REMOTE_BUDGET_MB \
  nohup ./ds4 -m '$MODEL' --role worker --listen '$WORKER_IP' '$PORT' \
  --layers '$SPLIT_WORKER' $DEBUG_ARGS $WORKER_MTP_ARGS \
  -c '$CTX' --temp 0 --nothink > '$WORKER_LOG' 2>&1 & echo \$! > '$WORKER_PID_FILE'; echo launched" 2>/dev/null

log "等 M1 worker backend 就绪并开始 control listen…"
wok=0
for _ in $(seq 1 90); do
  if ssh "$REMOTE" "grep -q 'waiting for coordinator\|control listen' '$WORKER_LOG'" 2>/dev/null; then wok=1; break; fi
  if ssh "$REMOTE" "grep -qiE 'refusing to load|fatal|Address already|invalid|Insufficient Memory' '$WORKER_LOG'" 2>/dev/null; then
    log "M1 worker 启动失败, 日志尾:"; ssh "$REMOTE" "tail -10 '$WORKER_LOG'"; cleanup; exit 1
  fi
  rg=$(mem_gb_remote); over "$rg" "$REMOTE_MAX_GB" && { log "M1 加载阶段 footprint ${rg}G 超 ${REMOTE_MAX_GB}G → 杀"; cleanup; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "M1 worker 90s 未就绪 → 放弃"; ssh "$REMOTE" "tail -10 '$WORKER_LOG'"; cleanup; exit 1; }
sleep 1

# ---------------- 5. 起本机 coordinator (主动拨 M1 worker, 跑一次性生成) ----------------
log "启动本机 coordinator: --coordinator $WORKER_IP:$PORT --layers $SPLIT_COORD ${COORD_MTP_ARGS:-(无 MTP)} (reverse, 主动拨)"
[ "$EXPERT_PROFILE" = 1 ] && log "专家命中率模拟 cache: coordinator=${LOCAL_EXPERT_PROFILE_CACHE_MB}MiB worker=${REMOTE_EXPERT_PROFILE_CACHE_MB}MiB (仅模拟)"
log "真实 expert pool: coordinator=${LOCAL_EXPERT_POOL_MB}MiB layers=${LOCAL_EXPERT_POOL_LAYER_START}:${LOCAL_EXPERT_POOL_LAYER_END} worker=${REMOTE_EXPERT_POOL_MB}MiB layers=${REMOTE_EXPERT_POOL_LAYER_START}:${REMOTE_EXPERT_POOL_LAYER_END} per-layer-LRU(min_slots=$EXPERT_POOL_MIN_LAYER_SLOTS,warm_batch=$EXPERT_POOL_WARM_BATCH) + 预测异步预取(lookahead=$EXPERT_POOL_PREFETCH_LOOKAHEAD top=$EXPERT_POOL_PREFETCH_TOP self=$EXPERT_POOL_PREFETCH_SELF adjacent=$EXPERT_POOL_PREFETCH_ADJACENT wait=$EXPERT_POOL_WAIT_INFLIGHT fg_fill=$EXPERT_POOL_FOREGROUND_FILL pf_evict=$EXPERT_POOL_PREFETCH_EVICT hit_only=$EXPERT_POOL_HIT_ONLY admit_after=$EXPERT_POOL_ADMIT_AFTER hotlock=$EXPERT_POOL_HOTLOCK_TOP auto_pin=${EXPERT_POOL_AUTO_PIN_TOP}/${EXPERT_POOL_AUTO_PIN_MIN_REQ}/${EXPERT_POOL_PIN_RESERVE} hotlist=${EXPERT_POOL_HOTLIST_TOP}/${EXPERT_POOL_HOTLIST_INTERVAL})"
log "静态钉住专家: coordinator=$(pin_count "$LOCAL_EXPERT_POOL_PINNED") worker=$(pin_count "$REMOTE_EXPERT_POOL_PINNED") from_logs=$EXPERT_POOL_PIN_FROM_LOGS"
log "源专家 LRU cache: coordinator=${LOCAL_EXPERT_SOURCE_CACHE_MB}MiB layers=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_START}:${LOCAL_EXPERT_SOURCE_CACHE_LAYER_END} worker=${REMOTE_EXPERT_SOURCE_CACHE_MB}MiB layers=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_START}:${REMOTE_EXPERT_SOURCE_CACHE_LAYER_END} admit_after=$EXPERT_SOURCE_CACHE_ADMIT_AFTER hard_copy=$EXPERT_SOURCE_CACHE_HARD_COPY async=$EXPERT_SOURCE_CACHE_ASYNC mlock=$EXPERT_SOURCE_CACHE_MLOCK"
log "  (首次可能弹 macOS 本地网络授权框 → 点允许)"
cd "$LOCAL_DIR"
rm -f "$COORD_LOG" "$COORD_OUT"
env $LOCAL_RUN_ENV DS4_MEM_BUDGET_MB=$LOCAL_BUDGET_MB \
  ./ds4 -m "$MODEL" --role coordinator --coordinator "$WORKER_IP" "$PORT" \
  --layers "$SPLIT_COORD" $DEBUG_ARGS $COORD_MTP_ARGS \
  -c "$CTX" -n "$NPRED" --temp 0 --seed "$SEED" --nothink \
  -p "$PROMPT" > "$COORD_OUT" 2> "$COORD_LOG" &
COORD_PID=$!

# ---------------- 6. 看门狗: 等本机 coordinator 一次性生成结束 ----------------
log "运行中… coordinator(本机) 生成完即退出。(Ctrl+C 两边同杀; 超过${RUN_TIMEOUT_SEC}s、本机>${LOCAL_MAX_GB}G 或 M1>${REMOTE_MAX_GB}G 也同杀)"
done_flag=0
timeout_flag=0
for _ in $(seq 1 "$RUN_TIMEOUT_SEC"); do
  if ! kill -0 "$COORD_PID" 2>/dev/null; then done_flag=1; break; fi
  lg=$(mem_gb_local "$COORD_PID"); rg=$(mem_gb_remote)
  mlabel=$(mem_label)
  printf "\r[mtp-pipe] %s 本机coord=%sG/%dG  M1worker=%sG/%dG  timeout=%ss    " "$mlabel" "$lg" "$LOCAL_MAX_GB" "$rg" "$REMOTE_MAX_GB" "$RUN_TIMEOUT_SEC"
  over "$lg" "$LOCAL_MAX_GB" && { echo; log "本机 coordinator ${mlabel} ${lg}G 超限 → 两边同杀"; cleanup; exit 2; }
  over "$rg" "$REMOTE_MAX_GB" && { echo; log "M1 worker ${mlabel} ${rg}G 超限 → 两边同杀"; cleanup; exit 2; }
  sleep 1
done
if [ "$done_flag" != 1 ] && kill -0 "$COORD_PID" 2>/dev/null; then
  timeout_flag=1
  echo
  log "运行超过 ${RUN_TIMEOUT_SEC}s → 杀两边 ds4 进程并保留日志摘要"
  kill "$COORD_PID" 2>/dev/null || true
  ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null || true; pkill -f 'ds4 -m' 2>/dev/null || true" 2>/dev/null || true
  pkill -f 'ds4 -m' 2>/dev/null || true
  wait "$COORD_PID" 2>/dev/null || true
fi
echo

# ---------------- 7. 结果 (在本机 coordinator 日志) ----------------
# 注: 生成文本可能含大量换行 token (尤其退化的 reduced-expert 模型), 末尾常是成片空行。
# 必须先滤掉纯空白行再 tail, 否则 tail 全抓到尾部空行 → 看着像"没输出文本"(其实内容在前面)。
log "本机 coordinator 生成文本 (stdout, 已滤纯空白行):"
gen_lines=$(grep -v -E '^[[:space:]]*$' "$COORD_OUT" 2>/dev/null || true)
if [ -n "$gen_lines" ]; then
  printf '%s\n' "$gen_lines" | tail -40
else
  log "(stdout 无非空文本行: 本次生成可能全是换行/空白 token；或 coordinator 未完成输出 flush)"
fi
echo "----------------------------------------"
if grep -qiE 'prefill:|generation:|t/s' "$COORD_LOG" 2>/dev/null; then
  log "速度 (本机 coordinator):"; grep -iE 'prefill:|generation:|t/s' "$COORD_LOG" | tail -2
else
  log "未拿到计时行, 本机 coordinator 日志尾:"; tail -12 "$COORD_LOG"
fi
if grep -qiE 'dist-mtp|adaptive-disable|copy-spec' "$COORD_LOG" 2>/dev/null; then
  echo "----------------------------------------"
  log "投机摘要 (本机 coordinator):"; grep -iE 'dist-mtp|adaptive-disable|copy-spec' "$COORD_LOG" | tail -30
fi
if grep -qiE 'expert-(profile|pool|source-cache)' "$COORD_LOG" 2>/dev/null; then
  echo "----------------------------------------"
  log "专家 LRU/pool/source 摘要 (本机 coordinator):"
  grep -iE 'expert-(profile|pool|source-cache) (summary|live|layer|  L|memory|enabled|enabled via|pinned)' "$COORD_LOG" | tail -100 || true
fi
if ssh "$REMOTE" "grep -qiE 'expert-(profile|pool|source-cache)' '$WORKER_LOG'" 2>/dev/null; then
  echo "----------------------------------------"
  log "专家 LRU/pool/source 摘要 (M1 worker):"
  ssh "$REMOTE" "grep -iE 'expert-(profile|pool|source-cache) (summary|live|layer|  L|memory|enabled|enabled via|pinned)' '$WORKER_LOG' | tail -100" 2>/dev/null || true
fi
[ "$timeout_flag" = 1 ] && log "(注: 本次达到 ${RUN_TIMEOUT_SEC}s 超时后被脚本杀停，上面是超时日志快照)"
[ "$timeout_flag" != 1 ] && [ "$done_flag" = 1 ] || [ "$timeout_flag" = 1 ] || log "(注: coordinator 未正常结束, 上面是当前日志快照)"
# 第二十五波: 按 profile 留档 —— 第二跑不再覆盖第一跑的取证日志 (第二十三/二十四波两次吃亏)。
cp -f "$COORD_LOG" "/tmp/mtp_pipe_coord.${PROMPT_PROFILE}.log" 2>/dev/null || true
cp -f "$COORD_OUT" "/tmp/mtp_pipe_coord.${PROMPT_PROFILE}.out" 2>/dev/null || true
scp -q "$REMOTE:$WORKER_LOG" "/tmp/mtp_pipe_worker.${PROMPT_PROFILE}.log" 2>/dev/null || true
log "日志留档: /tmp/mtp_pipe_coord.${PROMPT_PROFILE}.log|.out, /tmp/mtp_pipe_worker.${PROMPT_PROFILE}.log"
cleanup
