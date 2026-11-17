/* server_generate.c — 机械拆分自 ds4_server.c (10732-12246 行): generate_job 主生成路径。 */

#include "server_internal.h"

/* worker 开工时按这条 job 自己的 conv 落归属, 不再用 parse 线程写共享 s->conv:
 * 并发解析把别人的 agent 号串进这里会让存盘 KB/CK 标错人, 删错别的 agent. */
static void job_conv_bind(server *s, job *j) {
    snprintf(s->conv, sizeof s->conv, "%s", j->req.conv);
    snprintf(s->kv.conv, sizeof s->kv.conv, "%s", s->conv);
#ifndef DS4_NO_GPU
    ds4_engine_v41_kv_disk_conversation_set(s->conv);
#endif
}

/* Execute one request on the worker-owned session.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
 * hits we build a fresh effective prompt from the checkpoint's exact token
 * history plus a newly tokenized string suffix; the canonical full-prompt
 * tokens are not sliced because BPE may merge across the byte boundary.  Cold
 * prompt caching is handled before generation: if the stable checkpoint is
 * shorter than the full prompt, we prefill to that boundary, store it, and
 * immediately continue to the real prompt.  The live graph therefore always
 * moves forward. */
void generate_job(server *s, job *j) {
    job_conv_bind(s, j);
    /* DeepSeek V4.1(2026-09-19): 没有 ds4_session, 走 server_generate_v41.c 的回调式生成路; 下面的会话路一行不动。 */
    if (ds4_engine_is_v41(s->engine)) { generate_job_v41(s, j); return; }

/* generate_job 函数体 (原 10733-12244 行): 单函数 1512 行、含 decode_again/guided_primer
 * goto 标签与函数内 PRIMER_KA 宏, 无法在不改控制流的前提下切成 <500 行的独立
 * 函数。为守住行为零变化, 用预处理分片保 token 流逐字节等价; 分片只被本文件
 * 按序包含, 不是独立编译单元。 */

#include "server_generate_body1.inc"

#include "server_generate_body2.inc"

#include "server_generate_body3.inc"

#include "server_generate_body4.inc"

}
