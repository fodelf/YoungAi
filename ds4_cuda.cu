/* ds4_cuda.cu — CUDA 后端聚合根(2026-08-25 机械拆分)。
 * 实现按 kernel 族拆进 src/cuda/*.inc.cu, 本文件按原顺序纹理包含全部分片,
 * 保持单 TU nvcc 编译语义(static/__constant__/模板实例化全部不变, 行为零改动)。
 * 分片 ≤500 行; EXCEPTION(单函数体超限, 无安全切割点):
 *   src/cuda/cuda_moe_launch.inc.cu (884 行)
 * 跨分片共享的前奏(类型/表/前向声明)在 src/cuda/cuda_internal.cuh。 */

#include "src/cuda/cuda_internal.cuh"

#include "src/cuda/cuda_f16_shadow_1.inc.cu"
#include "src/cuda/cuda_f16_shadow_2.inc.cu"
#include "src/cuda/cuda_q8_repack_1.inc.cu"
#include "src/cuda/cuda_q8_repack_2.inc.cu"
#include "src/cuda/cuda_lifecycle.inc.cu"
#include "src/cuda/cuda_dspark.inc.cu"
#include "src/cuda/cuda_graphcap.inc.cu"
#include "src/cuda/cuda_decode_graph.inc.cu"   /* 解码整步 CUDA graph 原语(2026-09-18): 捕获/实例化/发射 + host 节点 + pinned 异步拷贝 */
#include "src/cuda/cuda_vq_align.inc.cu"   /* 启动缓存拷专家 blob 时把载荷挪到位流 128 B 对齐(2026-09-24); 在 modelmap 之前(被它调) */
#include "src/cuda/cuda_modelmap.inc.cu"
#include "src/cuda/cuda_embed_norm_kernels_1.inc.cu"
#include "src/cuda/cuda_embed_norm_kernels_2.inc.cu"
#include "src/cuda/cuda_embed_norm_kernels_3.inc.cu"
#include "src/cuda/cuda_embed_norm_kernels_4.inc.cu"
#include "src/cuda/cuda_compressor_kernels.inc.cu"
#include "src/cuda/cuda_attn_kernels_1.inc.cu"
#include "src/cuda/cuda_attn_kernels_2.inc.cu"
#include "src/cuda/cuda_attn_kernels_3.inc.cu"
#include "src/cuda/cuda_attn_kernels_4.inc.cu"
#include "src/cuda/cuda_attn_kernels_5.inc.cu"
#include "src/cuda/cuda_attn_kernels_6.inc.cu"
#include "src/cuda/cuda_router_kernels.inc.cu"
#include "src/cuda/cuda_indexer_kernels_1.inc.cu"
#include "src/cuda/cuda_indexer_kernels_2.inc.cu"
#include "src/cuda/cuda_indexer_kernels_3.inc.cu"
#include "src/cuda/cuda_indexer_kernels_4.inc.cu"
#include "src/cuda/cuda_indexer_kernels_5.inc.cu"
#include "src/cuda/cuda_indexer_kernels_6.inc.cu"
#include "src/cuda/cuda_api_embed_indexer_1.inc.cu"
#include "src/cuda/cuda_api_embed_indexer_2.inc.cu"
#include "src/cuda/cuda_api_matmul_1.inc.cu"
#include "src/cuda/cuda_api_matmul_2.inc.cu"
#include "src/cuda/cuda_api_matmul_3.inc.cu"
#include "src/cuda/cuda_q4k_gemm.inc.cu"
#include "src/cuda/cuda_api_compressor.inc.cu"
#include "src/cuda/cuda_api_attention_1.inc.cu"
#include "src/cuda/cuda_api_attention_2.inc.cu"
#include "src/cuda/cuda_api_attention_3.inc.cu"
#include "src/cuda/cuda_api_attention_4.inc.cu"
#include "src/cuda/cuda_api_attention_5.inc.cu"
#include "src/cuda/cuda_api_moe_corr_1.inc.cu"
#include "src/cuda/cuda_q4k_dot.inc.cu"     /* Q4_K 块点积/激活访问器/stage 助手(含多 token 预解 nibble 变体) */
#include "src/cuda/cuda_q4k_multi.inc.cu"   /* Q4_K 小批多 token 核 + 发射 */
#include "src/cuda/cuda_q4k_tile.inc.cu"
#include "src/cuda/cuda_qk_warp_1.inc.cu"
#include "src/cuda/cuda_qk_warp_2.inc.cu"
#include "src/cuda/cuda_qk_warp_3.inc.cu"
#include "src/cuda/cuda_qk_warp_4.inc.cu"
#include "src/cuda/cuda_moe_kernels_1.inc.cu"
#include "src/cuda/cuda_moe_kernels_2.inc.cu"
#include "src/cuda/cuda_moe_kernels_3.inc.cu"
#include "src/cuda/cuda_moe_kernels_4.inc.cu"
#include "src/cuda/cuda_moe_kernels_5.inc.cu"
#include "src/cuda/cuda_moe_q4k_tile.inc.cu"   /* Q4_K 专家 tile 核(drafter), 用 cuda_q4k_tile 的辅助 */
#include "src/cuda/cuda_moe_launch.inc.cu"
#include "src/cuda/cuda_vq.inc.cu"
#include "src/cuda/cuda_vq_prefill.inc.cu"
#include "src/cuda/cuda_v41_1.inc.cu"   /* DeepSeek V4.1 批前向原语 ①②③(2026-09-12): 稠密/hc/norm | rope/量化/indexer/attn | 路由/MoE */
#include "src/cuda/cuda_kv_pack.inc.cu"   /* 全局 KV 按官方格式打包(decode.md D1): 主 KV 288 B/组、索引 K 72 B/组; 在 v41_1 之后(用 bf16r/pow2_ceil_log2), 在三个注意力核之前(被它们解包) */
#include "src/cuda/cuda_v41_indexer.inc.cu"   /* indexer 打分/候选块/topk(2026-09-18 从 v41_2 拆出); 在 kv_pack 之后(解包索引键) */
#include "src/cuda/cuda_v41_indexer_mma.inc.cu"   /* 打分核的张量核版(2026-09-30, --idx-mma); 在 indexer 之后(用它的 v41_cand_ns, 被它的入口分发) */
#include "src/cuda/cuda_sparse_attn_mma.inc.cu"   /* 稀疏注意力张量核版(speed.md 段 4); 必须在 v41_1 之后(用 v41_bf16r)、v41_2 之前(被它调) */
#include "src/cuda/cuda_v41_hc.inc.cu"   /* mHC 一族(mix/sinkhorn/hc_pre/hc_post/合一核); 在 v41_1 之后(用它的 bf16r 与暂存槽) */
#include "src/cuda/cuda_v41_attn_split.inc.cu"   /* 解码路稀疏注意力 split-K(single.md S4); 在 v41_2 之前(被它调) */
#include "src/cuda/cuda_v41_attn_mma_decode.inc.cu"   /* 解码路稀疏注意力上张量核(decode.md D2): 按键分段的 mma 版; 必须在 sparse_attn_mma(借它的 gather/scores)与 attn_split(借它的合并核与暂存槽)之后 */
#include "src/cuda/cuda_kv_ring.inc.cu"   /* SWA 窗口的环维护(decode.md D1): 提交/快照/回滚; 在 v41_1 之后(用 v41_win_row 的同一套行号约定) */
#include "src/cuda/cuda_v41_2.inc.cu"
#include "src/cuda/cuda_v41_3.inc.cu"
#include "src/cuda/cuda_v41_fp4_planar.inc.cu"   /* fp4x32 权重的平面副本(single.md §2.6 A 路); 在 v41_4 之前(被它的 GEMV 调) */
#include "src/cuda/cuda_v41_4.inc.cu"   /* V4.1 解码小批融合核(fp4x32 GEMV / VQ 即乘 / 缩放舍入 / argmax) */
#include "src/cuda/cuda_v41_sample.inc.cu"   /* 设备采样核(2026-09-28): 温度/top-k/top-p/min-p + 投机拒绝采样, 替掉图末尾的 argmax */
#include "src/cuda/cuda_v41_q4k.inc.cu"   /* 骨架 q4_K(2026-09-19 的 100 GB 配方): 解码 GEMV / 预填解量化 / 嵌入取行; 在 v41_4 之后(用它的 v41_bf16r 与 V41_GEMV_* 常量) */
#include "src/cuda/cuda_v41_q4k_mma.inc.cu"   /* 合批 9~16 行的 q4_K 张量核乘法(2026-10-10): 借 q4k 的块常量与 v41_bf16r */
#include "src/cuda/cuda_vq_probe.inc.cu"    /* VQ 解码核的两个看门狗(--v41-prof 才跑); 在 v41_4 之后、vq_decode 之前 */
#include "src/cuda/cuda_vq_row.inc.cu"   /* VQ 载荷解析 + 一行点积(v2/v3 两版布局都在这一族); 必须在 cuda_vq_decode 之前 */
#include "src/cuda/cuda_vq_decode.inc.cu"   /* VQ 专家解码即乘(原在 v41_4 里); 在 v41_4 之后(用它的 bf16r/暂存槽), 在 draft 之前(草稿塔借本片的 reduce 核) */
#include "src/cuda/cuda_vq_group.inc.cu"   /* 多 token 分组解码即乘核(2026-09-22): 验证批/草稿塔 n≥2 时一 block 一个专家 × m 个 token; 在 decode 之后(用它的 shared 搬运/swiglu)、launch 之前(那里实例化 fused_moe_n) */
#include "src/cuda/cuda_vq_persist.inc.cu"   /* v3 纯解码常驻核(2026-09-23): 在 decode 之后(用它的码本搬运/swiglu/V41_VQ_WARPS)、launch 之前 */
#include "src/cuda/cuda_vq_decode_launch.inc.cu"   /* 上一片的发射器(按码本词数挑实例); 拆出去只为守 500 行 */
#include "src/cuda/cuda_v41_draft.inc.cu"   /* DSpark 草稿塔(speed.md 段 6 D1): 逐专家 FP4 dense MoE / hc 四路均值 / markov 行 gather; 必须在 v41_4 之后(用它的 reduce 核与暂存槽) */
#include "src/cuda/cuda_v41_gemv_highprec.inc.cu"   /* 非 FP4 权重(BF16 gate/compressor/indexer, F32 mHC)的小批 GEMV; clear.md C1 过门后整片删 */
#include "src/cuda/cuda_v41_nvfp4.inc.cu"   /* V4.1 预填稠密 GEMM 走 NVFP4 张量核(2026-09-15 speed.md S1) */
#include "src/cuda/cuda_vq_prefill_nvfp4.inc.cu"   /* 预填专家: VQ 解到 NVFP4 暂存 + 板子原生 FP4 张量核(speed.md 段 5); 在融合路之前, 被它调 */
#include "src/cuda/cuda_vq_prefill_fused.inc.cu"   /* 预填专家: VQ 解码即乘, 不落 f16 暂存(2026-09-15 第三轮) */
#include "src/cuda/cuda_vq_prefill_mma.inc.cu"   /* 预填专家 v3: VQ 解成 bf16 瓦片 + 张量核(2026-09-24); 在融合路之后(用它的 vqp_item, 被它的 vqp_fused_run 调) */
#include "src/cuda/cuda_vq_reg_mma.inc.cu"       /* 上一片的寄存器直解形态 vqs(2026-10-03, 逐位同、快 1.4~1.6 倍); 在它之后(用它的 g_vqm 暂存, 被它的 vqm_run_impl 调) */
#include "src/cuda/cuda_vq_fused2_0.inc.cu"
#include "src/cuda/cuda_vq_fused2_1.inc.cu"
#include "src/cuda/cuda_vq_fused2_2.inc.cu"
#include "src/cuda/cuda_vq_fused2_3.inc.cu"
#include "src/cuda/cuda_moe_api.inc.cu"
#include "src/cuda/cuda_hc.inc.cu"
#include "src/cuda/cuda_zchain_1.inc.cu"
#include "src/cuda/cuda_zchain_2.inc.cu"
#include "src/cuda/cuda_bwd_dense.inc.cu"   /* 后训练反传的稠密原语(2026-10-01): 借 q4k/fp4/fp8 → bf16 解码核, 必须在 v41_q4k / gemv_highprec 之后 */
#include "src/cuda/cuda_bwd_comp.inc.cu"    /* 后训练反传: 压缩行梯度 / 压缩器池化 / engram 门(分界层以下用; 借 v41_ckv_get), 必须在 bwd_attn 之前 */
#include "src/cuda/cuda_bwd_attn.inc.cu"    /* 后训练反传: 稀疏注意力 + RoPE(借 v41_rope_freq / KV 打包常量) */
#include "src/cuda/cuda_bwd_hc.inc.cu"      /* 后训练反传: mHC 混合系数(sinkhorn 重算倒推) */
#include "src/cuda/cuda_bwd_vq.inc.cu"      /* 后训练反传: routed 专家的 VQ 直读两种核(行点积 / 转置累加), 不落稠密阵 */
#include "src/cuda/cuda_bwd_moe.inc.cu"     /* 后训练反传: SwiGLU / 路由 / routed 专家(借 v41_vq_open·v41_vq_cw 解码与 g_v41_gr 增益覆盖) */
#include "src/cuda/cuda_draft_attn.inc.cu"  /* 草稿器蒸馏(2026-10-07): 批量全可见块注意力前向(借 sparse_attn_mma 的打分/统计积木与 attn_split 的分段公式)/反向、陪审团、批量取行 */

/* ★PDL 小核集中登记(2026-09-23)★: 这些核第一句就是 v41_pdl_wait()(见 cuda_internal.cuh 的 PDL 段), 所以前面的边可以改成程序化边 ——
 * 它们本身没有可预读的常量, 省的是发射与上线开销(与上一个核的收尾重叠)。放在聚合根末尾: 所有核都已定义, 这里才拿得到函数地址。
 * 新增一个小核想进来: 先在它第一句加 v41_pdl_wait(), 再加到这张表; 只加表不加 wait = 读到上一步的半成品。 */
static void v41_pdl_register_small(void) {
    const void *fns[] = {
        (const void *)v41_rms_norm_kernel, (const void *)v41_round_bf16_kernel, (const void *)v41_rope_kernel,
        (const void *)v41_compress_step_n_kernel, (const void *)v41_act_quant_kernel, (const void *)v41_kv_pack_kernel,
        (const void *)v41_hc_post_kernel, (const void *)v41_hc_fused_kernel, (const void *)v41_router_kernel,
        (const void *)v41_swiglu_kernel, (const void *)v41_engram_gate_kernel, (const void *)v41_vq_xpack_kernel,
        (const void *)v41_vq_tail_kernel, (const void *)v41_win_commit_kernel, (const void *)v41_attn_mma_seg_kernel,
        (const void *)v41_sparse_attn_merge_kernel, (const void *)v41_indexer_score_kernel, (const void *)v41_topk_kernel,
        (const void *)v41_candidate_kernel, (const void *)v41_scale_round_kernel, (const void *)v41_fp8blk_gemv_kernel<1u, 0u>,
    };
    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++) v41_pdl_register(fns[i]);
}
