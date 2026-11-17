/* core_v41_ckpt.h — V4.1 KV 检查点的磁盘离载模块(2026-11-15)。
 *
 * 为什么要: CK_PROMPT/CK_END 是进程级常驻 GPU 检查点。换 agent/会话时旧检查点对新提示无用，却继续占显存。
 * 本模块把整份检查点(显存缓冲 + 主机 tok/sig + 各层账)按对话边界写到磁盘，释放 GPU；回到旧对话再从磁盘装进请求态。
 * 磁盘预算与 kvstore 同一套 LRU: 文件头记 created/last_used，超过预算删最久没用的那几份。
 * 不做什么: 不改 ds4_v41_state 的分配式、不改 admission 公式 —— 磁盘只在请求态已开之后装，或换对话前离载。
 * 出错会怎样: 磁盘打不开/文件写失败 ⇒ 不离载(走 v41_kvreuse_release_all 的冷回收)；装不下 ⇒ 这一条照冷预填。 */
#ifndef DS4_CORE_V41_CKPT_H
#define DS4_CORE_V41_CKPT_H
#ifndef DS4_NO_GPU

#include "core_internal.h"

/* 检查点本体(core_v41_kvreuse.c 的两槽共用): 只存会被解码改掉或之后要读的那几样。
 * mainh_bytes 是检查点里 main_hidden 环的有效字节(缓冲可能更大, 磁盘只写有效那份)。 */
typedef struct {
    int valid;
    uint32_t n;                 /* 检查点位置 = 到这里为止的 token 数 */
    int32_t *tok; uint32_t tok_cap;
    uint64_t *sig;              /* [n] 每位所属图片的指纹(0 = 文本): 图片 span 的 token 全是同一个占位 id, 不比图片字节就会把别的图的 KV 接上(2026-10-13) */
    ds4_gpu_tensor *ckv[DS4_MAX_LAYER], *idk[DS4_MAX_LAYER], *win[DS4_MAX_LAYER], *cpk[DS4_MAX_LAYER], *cps[DS4_MAX_LAYER], *mainh;
    ds4_gpu_tensor *logits;     /* 末位的 logits 一行 [V] f32: 等长复用时不预填, 第一个 token 从它采 */
    uint32_t ng_src[DS4_MAX_LAYER], cpend[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
    int64_t mainh_end; uint32_t mainh_n;
    uint64_t mainh_bytes;       /* 检查点 main_hidden 有效字节(无 MTP = 0) */
    uint64_t bytes;             /* 检查点占的显存(日志用) */
} ckpt;
enum { CK_PROMPT = 0, CK_END = 1, CK_SLOTS = 2 };

/* 磁盘模块开/关: dir 为空 = 不开; budget_mb 0 = 默认 4096。开时扫目录建 LRU 索引并先压到预算内。 */
bool v41_ckpt_disk_open(const char *dir, uint64_t budget_mb, char *err, size_t errn);
void v41_ckpt_disk_close(void);
bool v41_ckpt_disk_enabled(void);

/* 整份检查点写到磁盘(槽 0/1)。成功 ⇒ 文件按 tok 序列的 sha 落盘、last_used=now; 失败 ⇒ 不动 GPU。 */
bool v41_ckpt_disk_store(const ckpt *k, uint32_t slot, char *err, size_t errn);

/* 回旧对话: 找磁盘里 tok 前缀与本条提示相同、sig 逐位同、且不短于检查点的那份，装进请求态。
 * 返回 0 = 没接上/装不下(冷预填); 成功返回检查点位置(已把 KV/环/余行/mainh/logits/账/tok/sig 装进 st)。 */
uint32_t v41_ckpt_disk_restore_state(ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                                     const uint64_t *sig, char *err, size_t errn);

/* 切插件/引擎关: 磁盘检查点是带老放大器算的，全部作废。 */
void v41_ckpt_disk_clear(void);
void v41_ckpt_disk_conversation_set(const char *conv);   /* agent 会话号: store 时写进 64B 尾 trailer */
int v41_ckpt_disk_erase_conversation(const char *conv);  /* 只删同 conv 的 CK 文件, 别的 agent 留 */

#endif /* !DS4_NO_GPU */
#endif /* DS4_CORE_V41_CKPT_H */
