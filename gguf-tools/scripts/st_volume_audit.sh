#!/bin/bash
# st_volume_audit.sh — HF safetensors 体积/参数量账(2026-09-10, V4.1-Flash 量化发车前置)。
#
# 【干什么】把一个 HF 目录的全部张量按结构分桶, 每桶打印【真实参数量】和【HF 原始字节】,
# 再按给定的目标位宽算出量化后落地体积。
#
# 【为什么必须有这个脚本】徒手按 config.json 推参数量在本项目已经错过三次(fable5 bpw
# 口径注记)。V4.1 更容易错: ①专家权重是 FP4 两个打包进一个 I8, safetensors 的 shape
# 是打包后的形状(2304x2560), 直接乘会把专家参数量算成一半; ②engram 的 3.84 亿行查表
# 和 vision/mtp 三塔在 config 里根本看不出体积。漏一项 = 体积账差几十 GB。
#
# 用法: st_volume_audit.sh <manifest.tsv> [目标落地GB 骨架bpw engram-bpw]
#   manifest.tsv 由 gguf-tools/quantize/dump_st_meta <hf-dir> 产生
#   (一行一张量: name<TAB>dtype<TAB>bytes<TAB>numel<TAB>shape)
#   给了后三个参数就多打一段配方账: 骨架与 engram 按给定位宽定死, 反解【全部 MoE 专家】
#   还剩多少 bpw。目标体积是十进制 GB(Finder 口径, 铁律: 全字节永不摘项)。
#   例: st_volume_audit.sh m.tsv 100 4.5 1.0   # 落地 100 GB, 骨架 Q4_K, engram 1bit
#
# 【为什么专家桶要含 MTP 的专家】MTP 三塔各自带 128 个 routed 专家(dspark_n_routed_experts),
# 三层合计 13.59 B 参数 —— 占"MTP 三塔"那 14.2 B 的 96%。把它算进骨架, 骨架会虚胖 2.5 倍,
# 而它明明和主干专家是同一种东西(同尺寸 w1/w2/w3), 走同一套 VQ 量化。
set -uo pipefail
M="${1:?用法: st_volume_audit.sh <manifest.tsv>(dump_st_meta 的输出) [目标落地GB 骨架bpw engram-bpw [专家bpw]]}"
[ -s "$M" ] || { echo "★清单空或不存在: $M★" >&2; exit 2; }
TGT="${2:-}"; SKB="${3:-}"; ENB="${4:-}"; EXB="${5:-}"; MTB="${6:-}"
# 第 6 个参数: MTP 自带专家单独钉一个位宽(草稿质量决定接受率, 而它只占专家参数 2.4% ——
# 给它加宽几乎不吃主干预算)。不给就跟主干专家同档。
# 第 5 个参数给了专家位宽 = 反着算: 体积不再是约束, 而是结果; 同时算出"若硬守目标体积,
# 这个位宽下最多能留几个专家"。GB10 原生 FP4 想留住就得回答这个问题。

awk -F'\t' -v tgt="$TGT" -v skb="$SKB" -v enb="$ENB" -v exb="$EXB" -v mtb="$MTB" '
function bucket(n) {
    if (n ~ /^layers\.[0-9]+\.ffn\.experts\./)        return "1 routed专家(MoE)";
    if (n ~ /^layers\.[0-9]+\.ffn\.shared_experts\./) return "2 shared专家";
    if (n ~ /^layers\.[0-9]+\.engram\./)              return "3 engram查表";
    if (n ~ /^layers\.[0-9]+\.attn\./)                return "4 注意力(含indexer/compressor)";
    if (n ~ /^(embed|head)\.weight$/)                 return "5 词嵌入+输出头";
    if (n ~ /^(vision|aligner)\./)                    return "6 vision塔+aligner";
    if (n ~ /^mtp\./)                                 return "7 MTP三塔(投机)";
    return "8 其余(norm/gate/hc/sink)";
}
{
    name=$1; dt=$2; b=$3+0; ne=$4+0;
    # ★I8 = FP4 两两打包★ 真实参数是元素数的两倍。不还原 = 专家参数量少一半。
    p = (dt=="I8") ? ne*2 : ne;
    # scale/量化元数据不是"模型参数", 但占字节 —— 分开记, 否则 bpw 分母被污染。
    isS = (name ~ /\.scale$/);
    k = bucket(name);
    if (isS) { sb[k]+=b; sn[k]++ } else { pb[k]+=b; pp[k]+=p; pn[k]++ }
    keys[k]=1;
    # MTP 自带的 128 专家单独记一笔: 配方账里它跟主干专家同档, 不进骨架。
    if (!isS && name ~ /^mtp\.[0-9]+\.ffn\.experts\./) mtpexp += p;
    # 专家个数(按 w1 计一次): 反解"最多能留几个专家"要用
    if (!isS && name ~ /ffn\.experts\.[0-9]+\.w1\.weight$/) nexp++;
}
END {
    printf "%-32s %10s %14s %12s %12s\n", "桶", "张量数", "真实参数(B)", "权重GB", "scaleGB";
    tp=0; tb=0; ts=0;
    n=asorti(keys, sk);
    for (i=1;i<=n;i++) {
        k=sk[i];
        printf "%-32s %10d %14.3f %12.3f %12.3f\n", k, pn[k], pp[k]/1e9, pb[k]/1e9, sb[k]/1e9;
        tp+=pp[k]; tb+=pb[k]; ts+=sb[k];
    }
    printf "%-32s %10s %14.3f %12.3f %12.3f\n", "合计", "", tp/1e9, tb/1e9, ts/1e9;
    printf "\nHF 落地总字节 %.3f GB\n", (tb+ts)/1e9;
    printf "全模型真实参数 %.3f B\n", tp/1e9;
    printf "HF 现态平均位宽(含scale) %.4f bpw\n", (tb+ts)*8/tp;

    if (tgt == "") exit 0;

    # ---- 配方账 ----
    # 三分法: 全部 MoE 专家(主干 + MTP, 量化主体) / engram 查表 / 真骨架(其余全部)。
    # 骨架与 engram 位宽给定, 目标落地体积给定 ⇒ 专家位宽是唯一未知数, 闭式反解。
    Pexp = pp["1 routed专家(MoE)"] + mtpexp;
    Peng = pp["3 engram查表"];
    Pskel = tp - Pexp - Peng;
    Btot = tgt * 1e9;                       /* 目标是十进制 GB */
    Bskel = Pskel * skb / 8;
    Beng  = Peng  * enb / 8;
    Bexp  = Btot - Bskel - Beng;

    if (exb != "") {
        # ---- 反向: 专家位宽定死(如 GB10 原生 FP4=4bpw), 算真实体积, 并回答"守 100G 要砍到几个专家" ----
        BexpF = Pexp * exb / 8;
        Breal = Bskel + Beng + BexpF;
        printf "\n=== 反向账: 专家钉死 %s bpw / 骨架 %s bpw / engram %s bpw ===\n", exb, skb, enb;
        printf "%-30s %12s %12s %10s\n", "部件", "参数(B)", "落地GB", "bpw";
        printf "%-30s %12.3f %12.3f %10.4f\n", "真骨架(attn/embed/norm/vision)", Pskel/1e9, Bskel/1e9, skb;
        printf "%-30s %12.3f %12.3f %10.4f\n", "engram 查表", Peng/1e9, Beng/1e9, enb;
        printf "%-30s %12.3f %12.3f %10.4f\n", "MoE 专家(主干+MTP)", Pexp/1e9, BexpF/1e9, exb;
        printf "%-30s %12.3f %12.3f %10.4f\n", "合计", tp/1e9, Breal/1e9, Breal*8/tp;
        printf "\n★真实落地 %.3f GB, 是目标 %s GB 的 %.2f 倍★\n", Breal/1e9, tgt, Breal/Btot;
        # 守住目标体积的唯一出路: 少留几个专家。每专家参数量 = 专家总参数 / 专家个数。
        per = Pexp / nexp;
        keep = (Btot - Bskel - Beng) * 8 / exb / per;
        printf "若硬守 %s GB: 专家预算 %.3f GB ⇒ %s bpw 下只能留 %d / %d 个专家(%.1f%%),\n",
               tgt, (Btot-Bskel-Beng)/1e9, exb, int(keep), nexp, keep*100/nexp;
        printf "  即主干每层 384 → 约 %d 个。★丢专家在 V4 时代实测崩成乱码, 此处只是体积算术★\n",
               int(keep*384/nexp);
        act_report(exb, skb, exb);
        exit 0;
    }
    printf "\n=== 配方账: 落地 %s GB / 骨架 %s bpw / engram %s bpw ===\n", tgt, skb, enb;
    printf "%-30s %12s %12s %10s\n", "部件", "参数(B)", "落地GB", "bpw";
    printf "%-30s %12.3f %12.3f %10.4f\n", "真骨架(attn/embed/norm/vision)", Pskel/1e9, Bskel/1e9, skb;
    printf "%-30s %12.3f %12.3f %10.4f\n", "engram 查表", Peng/1e9, Beng/1e9, enb;
    if (Bexp <= 0) {
        printf "%-30s %12.3f %12s %10s\n", "MoE 专家(主干+MTP)", Pexp/1e9, "★负数★", "不可行";
        printf "\n★不可行: 骨架+engram 已占 %.3f GB, 超出目标 %.3f GB★\n", (Bskel+Beng)/1e9, Btot/1e9;
    } else if (mtb != "") {
        # MTP 专家钉死位宽, 主干专家吃剩下的
        Bmt = mtpexp * mtb / 8;
        Pmain = Pexp - mtpexp;
        Bmain = Bexp - Bmt;
        if (Bmain <= 0) { printf "\n★MTP @%s bpw 已吃光专家预算★\n", mtb; exit 1; }
        printf "%-30s %12.3f %12.3f %10.4f\n", "MTP 专家(草稿)", mtpexp/1e9, Bmt/1e9, mtb;
        printf "%-30s %12.3f %12.3f %10.4f\n", "主干专家(40层×384)", Pmain/1e9, Bmain/1e9, Bmain*8/Pmain;
        printf "%-30s %12.3f %12.3f %10.4f\n", "合计", tp/1e9, Btot/1e9, Btot*8/tp;
        printf "\n落地 %.3f GB, 整模平均 %.4f bpw\n", Btot/1e9, Btot*8/tp;
        printf "常驻内存(骨架+专家, engram 走盘 mmap): %.3f GB\n", (Bskel+Bexp)/1e9;
        act_report(Bmain*8/Pmain, skb, mtb);
    } else {
        printf "%-30s %12.3f %12.3f %10.4f\n", "MoE 专家(主干+MTP)", Pexp/1e9, Bexp/1e9, Bexp*8/Pexp;
        printf "%-30s %12.3f %12.3f %10.4f\n", "合计", tp/1e9, Btot/1e9, Btot*8/tp;
        printf "\n落地 %.3f GB, 整模平均 %.4f bpw\n", Btot/1e9, Btot*8/tp;
        printf "常驻内存(骨架+专家, engram 走盘 mmap): %.3f GB\n", (Bskel+Bexp)/1e9;
        act_report(Bexp*8/Pexp, skb, Bexp*8/Pexp);
    }
}

# ---- 解码每 token 读多少字节 ----
# 【为什么要单独算】体积决定装不装得下, 但 batch=1 解码的速度由【每 token 读的字节】决定,
# 两者不是一回事: routed 专家每 token 只碰 6/384, 而注意力是全读。GB10 实测读墙 237~241 GB/s
# (fable5 09-06), 所以 t/s 上限 = 读墙 / 每token字节。不算这一笔就会把"专家降位宽"的提速
# 幅度想当然放大 —— 实际大头在全读的注意力那边。
function act_report(expbpw, skelbpw, mtpbpw,   Bact_e, Bact_s, Bact) {
    # 每 token 激活的 routed 专家: num_experts_per_tok(6) / n_routed_experts(384)
    Bact_e = pp["1 routed专家(MoE)"] * (6/384) * expbpw / 8;
    # 全读部分: 注意力 + shared 专家 + 输出头 + 零碎(embed 只查一行, engram 稀疏查表, 都忽略)
    Pact_s = pp["4 注意力(含indexer/compressor)"] + pp["2 shared专家"];
    Pact_s = Pact_s + pp["5 词嵌入+输出头"]/2 + pp["8 其余(norm/gate/hc/sink)"];
    Bact_s = Pact_s * skelbpw / 8;
    Bact = Bact_e + Bact_s;
    printf "\n[解码激活账] 每 token 读: routed专家(6/384) %.3f GB + 全读部分(attn/shared/head) %.3f GB = %.3f GB\n",
           Bact_e/1e9, Bact_s/1e9, Bact/1e9;
    printf "  GB10 读墙 237 GB/s ⇒ 单流上限 %.1f t/s (纯解码, 不含投机)\n", 237e9/Bact;
    spec_report(Bact_e, Bact_s, mtpbpw);
}

# ---- 投机(MTP/DSpark)提速账 ----
# 【杠杆在哪】verify 一批 N 个 token 只读一次全读部分(attn/shared/head), 专家部分才随 N 长。
# V4.1 的全读部分占纯解码字节 79%, 所以这一摊薄是大头 —— 这也是 V4 时代投机红利小的原因
# (那边专家占大头, 摊不动)。
# 【保守在哪】专家按 N 个 token 全不重叠算(最坏情况); 真实路由有重叠, 实际字节只会更少。
function spec_report(Be, Bs, mtpbpw,   N, i, Bdraft, Bverify, Tround) {
    N = 5;                       /* config: dspark_block_size = 5 */
    # 草稿成本: MTP 三塔非专家部分(全读) + 每层 128 选 3 的专家, 一个草稿 token 一份
    # MTP 非专家部分随骨架档(它是全读的小矩阵), 128 选 3 的专家按 MTP 自己的位宽
    Bdraft = (pp["7 MTP三塔(投机)"] - mtpexp) * skb / 8 + mtpexp * (3/128) * mtpbpw / 8;
    Bverify = Bs + N * Be;
    Tround = (N * Bdraft + Bverify) / 237e9;
    printf "\n[投机账] DSpark block=%d: 草稿 %.3f GB + verify(%.3f 全读 + %d×%.3f 专家) = %.3f GB/轮\n",
           N, N*Bdraft/1e9, Bs/1e9, N, Be/1e9, (N*Bdraft+Bverify)/1e9;
    printf "  接受长度 → t/s: ";
    for (i = 1; i <= N; i++) printf "%d个=%.1f  ", i, i/Tround;
    printf "\n  (纯解码 %.1f t/s 作底; 接受 %.2f 个即打平)\n", 237e9/(Be+Bs), Tround*237e9/(Be+Bs);
}' "$M"
