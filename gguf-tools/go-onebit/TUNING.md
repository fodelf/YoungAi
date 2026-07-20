# 单点微调面板（go-onebit 三件套：Θ_fix + z + 四损失）

主流程跑通后，一切优化 = 单变量点调。每个旋钮独立、代价明确、判决用短片配对 NLL（分钟级）。
铁律：一次一个旋钮；先短判（~120 tok 配对，双 heldout），赢了才上全锚（364 tok）。

## 旋钮总表

| 模块 | 旋钮 | 命令入口 | 单点成本 | 判决 |
|---|---|---|---|---|
| **Θ_fix 模型** | 专家类型/backbone 类型/imatrix | `cluster/quant_dual.sh OUT.gguf --experts go1b [--shared q8_0 ...]`（服务注册双机） | ~2h | bare 短NLL + z重解一轮 |
| **z（单层）** | 换目标/换语料重解某一层 | ① M1: `e5_pump.sh "L"`（CAP=采集目录）② `e5_consume.sh "L" ZDIR` | **~10 min/层** | 重 emit + 短NLL |
| **L_align** | `--w-align F`（逆范数重权） | e5_consume 的 CFG 里改 | ~10 min/层 | 同上 |
| **L_smooth** | `--w-smooth F`（跨token差分增广） | 同上 | 同上 | 同上 |
| **L_cls** | `--alpha-router F`（W_r 度量增广）；δ 基建在位（教师自轨迹 logits 待建） | 同上 | 同上 | 同上 |
| **L_fix** | imatrix 加权 scale（进模型不进 z） | quant_dual 的 IMATRIX env | 随模型 ~2h | bare 短NLL |
| **秩 k_L（逐层）** | `--rank N`（侧车体积↔质量） | e5_consume CFG | ~10 min/层 | 短NLL+侧车MB |
| **φ 特征** | `--feat x|yhat`（**x=速度免费，ŷ=+0.02质量但 2.4× 慢**——R3-e 终判决） | e5_consume CFG + emit `--phi` | ~10 min/层 | 短NLL+12tok速度臂 |
| **语料** | 换/混采集料（z=域适应方向盘，0.6-0.7 nats/10MB——2×2 判决） | `cap_ef2.sh OUT CORPUS CORR`(MODEL/CTX env) | ~1h 采集+重解 | 双 heldout 短NLL |
| **EF 轮次** | 学生底座换新侧车再采一轮 | cap_ef2.sh 第三参=侧车 | ~1h+重解 | 同上 |
| **MTP** | go-trie 词典/dense drafter（与 z 完全正交） | `--go-trie` / `--mtp FILE` | 独立 | 接受率+t/s |
| **运行时速度** | `DS4_METAL_EXPERT_PREAD=1 PREFETCH_AHEAD=1`（2.2×，token 无损，已默认进脚本） | env | 0 | 12tok 臂 |

## 单层 z 点调的标准三步（~10 分钟）
```sh
# 1. 重泵该层 sel（M1，CAP 指向持久化的采集目录）
ssh 192.168.1.2 'cd ~/ds4-main && env CAP=~/ds4-main/cap_v2r1 SPOOL=~/ds4-main/sel_spool_pt NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh "20"'
# 2. 本机重解（改任意损失/秩/φ 旗子）
env SPOOL_M1=/Users/fodelf/ds4-main/sel_spool_pt CFG="--solver rrr --rank 128 --lambda 1e-1 --feat x ..." \
  sh gguf-tools/go-onebit/cluster/e5_consume.sh "20" zdump_pt
# 3. 换装该层 z → 重 emit（秒级）→ 短判
cp zdump_pt/z_L20.bin zdump_v2r1/ && cd gguf-tools && ./emit_z --out ../gguf/ds4-go1b-v2-corr-r1.gguf \
  --zdir ../zdump_v2r1 --layers 43
sh go-onebit/scripts/nll_gate.sh ../gguf/ds4-go1b-v2-corr-r1.gguf   # HELDOUT 可换短片
```

## 当前基线（判决对照点）
- v1 时代：质量王 s1（φ=ŷ）2.186/0.40 t/s；速度王 A22（φ=x）2.209/0.97 t/s；bare 3.980
- v2 bare：短片 5.615（vs v1 5.569 持平）；速度 0.99 t/s
- v2+r1：主流程首件在跑（本文件生成时）

## 多域侧车插件（2026-07-04 用户定向：一基座 N 领域插件）
- 形态：单一 Θ_fix + 每领域一个 z 侧车（~90MB/域），插件即文件。
- 产插件：`KEYWORD=gin FRESH_CORPUS=1 sh factory.sh`（产 gguf/sidecars/gin.gguf 命名件）
- 用插件：`./ds4 -m <Θ_fix> --corr gguf/sidecars/<域>.gguf`（加载即切换）
- 热切换（排期，依赖 R3-g corr 模块抽出）：agent `/corr <域>` 命令 + server 请求级
  corr 选择 + 侧车注册表（目录扫描 gguf/sidecars/）。~90MB 重载秒级。

## 重复罚校准判决（2026-07-04）
- 机制修复后强度扫描：freq=3 长生成词汤（代码合法高频词被累罚压死）/ **freq=1 甜点**（无循环+词汇正常）/ 0.5 循环回归
- go1b 生成默认 DS4_REPEAT_FREQ=1 window=128；投机代价：罚改变 argmax → copy-spec 命中下降（freq=3 时归零）
