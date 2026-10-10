    bool dsml_recovery_attempted = false;
    uint64_t rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
decode_again:
    ;
    buf text = {0};
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    const char *finish = "length";
    int completion = 0;
    int max_tokens = j->req.max_tokens;
    int room = ds4_session_ctx(s->session) - ds4_session_pos(s->session);
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    size_t tool_scan_from = 0;
    int next_tool_progress = 128;
    int next_decode_log = 50;
    int next_alive_check = 16;   /* 每 16 个 token 探一次对端还在不在(与 V4.1 路同一条: server_http.c client_disconnected) */
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > room) max_tokens = room;
    /* --max-output-tokens: server-side hard cap regardless of the client's
     * requested limit (see server.max_output_tokens). */
    if (s->max_output_tokens > 0 && max_tokens > s->max_output_tokens)
        max_tokens = s->max_output_tokens;
    trace_event(s, trace_id, "prefill done; decode_max=%d ctx_room=%d", max_tokens, room);
    const double decode_t0 = now_sec();
    mon_first_token(s, j->mon);   /* 监控: 预填完, 进入生成 */
    double last_decode_log_t = decode_t0;
    int last_decode_log_completion = 0;
    thinking_state thinking = thinking_state_from_prompt(&j->req);
    const bool thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    bool tool_scan_waiting_for_think_close =
        thinking_gates_tool_markers && thinking.inside;
    dsml_decode_tracker dsml_tracker;
    dsml_decode_tracker_init(&dsml_tracker);

    /* --tool-primer: guided tool-call generation for base/continuation models.
     *
     * Measured ladder (2026-07-07, mono/go2b): from the bare template the
     * model ECHOES the prompt instead of acting (role turns mean nothing to a
     * continuation model); given the DSML opener as a prefix it continues
     * with the semantically correct tool, parameter and value -- but the
     * fragile 2-bit token stream misspells the DSML STRUCTURE tags
     * ("parameter" drifted to Chinese, closing tags collapsed).
     *
     * So split the work by strength: the SERVER injects every structure token
     * (it knows the DSML grammar and the tool schemas), the MODEL generates
     * only the free content (tool name, parameter values). The result is
     * parseable by construction. v0 scope: one invoke per turn, parameters
     * in schema order, values end at the first newline/'<' token (fine for
     * paths/commands/code-free args; multi-line values need the general
     * decoder and are documented in tiny-coder-plan P2).
     *
     * v1 copy-constrained values (2026-07-07 判决链): 自由贪心在参数值位置吐
     * schema 风占位符 ($FILE_PATH/$PATH_VALUE) —— 对 base 模型这是
     * `parameter name="file_path"` 上下文的高概率文档式延续, 与检索能力无关
     * (路径明写在短 prompt 里也照吐; per-layer 低秩侧车全幅度扫描无甜点,
     * fable5 判决)。编码域先验: 工具参数值几乎总是上下文的逐字 span。因此
     * 值区间解码时把候选 mask 到"能延续 prompt_text 中某个活跃 span 的
     * token"; 模型想停 (EOS/stopchar) 随时可停; 无可行延续 token 时回退
     * 自由生成并解除约束 (生成型参数如 Write.content 的逃生口)。工具名
     * 区间同机制, 源=声明工具名集合 (顺带修名字塌缩)。 */
    /* ★primer 自由区先行(2026-07-15, 根因②修复)★
     * 旧行为: primer 对每个带工具 chat 轮无条件开 DSML 帧 → 事实性问题(tool_choice=auto)
     * 结构性拿不到文本回答(行为门 8/8 全 finish=tool_calls 实证)。
     * 新语义(auto 档): 先走普通解码圈自由续写(server stops/EOS 全现成) —— 预算内自然
     * 收束 = 纯文本回答; 没收束 = 已出文本作 content 前缀, goto 回注帧(content+tool_calls
     * 双载荷, 协议合法)。tool_choice=none 已在 has_tools 处理; DS4_PRIMER_FREE_BUDGET=0
     * 回旧无条件注帧行为。 */
    /* ★生成段 SSE 心跳(2026-07-16)★: prefill 段的 keepalive 已有(server_progress_cb),
     * 但 primer 引导轮把全部文本缓冲到轮末才发 — 值区/注入 5-15min 流上零字节, 客户端
     * 静默超时重发→驱逐生成(CC 冒烟三攻实证: Edit 长轮永远到不了)。同款 `:` 注释行,
     * SSE 客户端忽略, 字节不进转录。 */
    double primer_last_ka = now_sec();
#define PRIMER_KA() do { \
        if (j->req.stream && j->fd >= 0 && progress.headers_sent && !progress.stream_failed) { \
            double _kanow = now_sec(); \
            if (_kanow - primer_last_ka >= 5.0) { \
                static const char _ka[] = ": generating\n\n"; \
                if (!send_all(j->fd, _ka, sizeof(_ka) - 1)) progress.stream_failed = true; \
                primer_last_ka = _kanow; \
            } \
        } \
    } while (0)
    bool primer_free_retry = false;
    /* 工具调用帧之前自由文本区的 token 预算。96 = 生产定版值(svc.sh 2026-07-16
     * CC 攻3 A/B: 批注入+小自由区, 引导轮 15min→5min); 旧代码默认 24 是陈值。 */
    const int primer_free_budget = 96;
    const bool primer_eligible = s->tool_primer && j->req.kind == REQ_CHAT &&
                                 j->req.has_tools && !dsml_recovery_attempted;
    /* Forced tool_choice (named or "any") skips the free region: the client has
     * already decided this turn is a tool call, so free text is not an answer. */
    const bool primer_free_first = primer_eligible && primer_free_budget > 0 &&
                                   !j->req.tool_force_name && !j->req.tool_force_any;
    /* 每请求采样策略接线(采样点全在下方: primer_copy_step 的 argmax 与解码圈的
     * ds4_session_sample):
     *   - 请求级 freq/presence 惩罚: 会话跨请求复用, 字段缺省=(0,0) 也必须落下去清掉;
     *   - 投机贪心门: argmax 接受(copy-spec/MTP)只在该请求实际生效 temperature==0
     *     时保分布 —— think 档强制 DS4_DEFAULT_TEMPERATURE, 与解码圈的覆盖一致;
     *   - 生成区边界: 到此为止的 checkpoint(prompt 及 decode_again 的恢复注入)都算
     *     prompt, anticycle 只扫 [这里, end); decode_again 重入会重新钉一次。 */
    ds4_session_set_request_penalties(s->session, j->req.frequency_penalty,
                                      j->req.presence_penalty);
    /* 客户端参数原样生效: 此前 think 档会把 temperature 静默换成 DS4_DEFAULT_TEMPERATURE,
     * 属引擎擅自改动模型行为(2026-08-21 铁律), 已删。 */
    ds4_session_set_spec_greedy(s->session, j->req.temperature <= 0.0f);
    ds4_session_mark_generation_start(s->session);
    /* 自由区起点 (prompt 注入后的 session 位置)。echo 回落必须 KV 回滚到这里:
     * 复读到自然收束的 KV 态本身已废 —— 模型视回合为已结束, 注帧后自由采样是碎片汤
     * (2026-07-17 三针对照: v1 无回滚=汤; v3 保留草稿=逐字节同款汤, 值区 fell back
     * 自由后与草稿无关; FORCE 针直接注帧=完美)。回滚后与 FORCE 形态同态。 */
    const int primer_prompt_pos = ds4_session_pos(s->session);
guided_primer:
    if (primer_eligible && (!primer_free_first || primer_free_retry)) {
        char perr[160];
        bool gok = true;
        int gen_used = 0;
        double inject_sec = 0.0, gen_sec = 0.0;   /* P6 分段计时 */
        /* inject: eval a literal text span into the session + text buf.
         *
         * P6 latency fix (2026-07-07 svc.log evidence): every eval on the
         * dual-host route is one serial cross-machine forward at the measured
         * decode cost of ~0.65 s (turns 2..8: decode-phase seconds / (free +
         * injected tokens) = 0.64-0.67 s/eval, with ~40 structure tokens
         * injected per guided turn = ~26 s/turn of pure injection).  The
         * structure tokens are all KNOWN up front, so evaluate each literal as
         * ONE batched span instead of token-by-token: build
         * (session tokens + literal tokens) and let ds4_session_sync() extend
         * the live checkpoint.  This is the exact entry the server prompt
         * prefill already uses; on the distributed route it becomes a single
         * dist_coordinator_eval_span() (one WORK frame, layer-major batch) and
         * on single-host Metal a metal_graph_prefill_chunked_range() resume.
         * Token stream, positions, and KV writes are identical to the
         * per-token loop (same tokenizer call per literal, batch-vs-single
         * prefill/decode equivalence is the invariant copy-spec already relies
         * on when it commits verified draft batches).  Expected: ~26 s/turn ->
         * 3 small spans (opener/param-head/closer), measured small-span cost
         * 6-10 s total, i.e. roughly -16..-20 s per guided tool-call turn. */
        /* P6 批量注入 (2026-07-14 重启; 2026-07-16 CC 攻3 A/B 胜出后定版恒开):
         *
         * 2026-07-07 曾回退为逐 token, 理由"batch 与 single 数值路径不同, 翻转了
         * 近平局贪心(参数值变 $FILE_PATH)"。重审: 注入的是 **已知字面 token**, 不采样
         * → 批量绝不可能改变注入的字节; 能变的只是注入后 KV 的数值细节, 进而影响下一个
         * 采样点。而今天的值区已由 copy 约束结构性兜底(占位符不再可达: 它不是上下文 span)
         * —— 当年那个失败模式的入口已被堵死。
         *
         * 成本证据(2026-07-14 双机 svc): 一个引导轮 gen=2(模型只吐 2 个自由 token)却要
         * 38 s ⇒ 0.05 t/s: 时间几乎全花在几十个 **已知** 结构 token 的逐 token forward 上
         * (每个都一次跨机 hop + ~0.85 GiB 专家字节)。批量走 prefill 路径 = 一次 forward
         * 吃掉整段, 是本场景最大的速度杠杆。
         *
         * 实现: 累积 (session 现有 token + 本段字面 token) 交给 ds4_session_sync() ——
         * 这正是 server prompt prefill 走的同一入口(分布式侧=一个 WORK 帧的 layer-major
         * 批)。 */
        /* ★紧凑 KV 注入 (2026-07-14, --primer-compact)★
         *
         * 成本实测(双机, 热轮): inject=32.8s / gen=3.6s / 总 48.4s —— 结构 token 的
         * forward 占 68%, 而模型真正的生成只有 3.6s。每个结构 token 都要跑一次完整
         * MoE forward(拉专家字节), 且 MoE 小批量无摊薄(批量注入实测无收益)。
         *
         * 洞察: 结构 token 进 KV 的唯一作用, 是让后续采样点"看到自己在填哪个工具的哪个
         * 参数"。模型不需要看到 DSML 的尖括号语法糖——它只需要语义锚点。而 replay 契约
         * 是 "live KV 是权威前缀"(server 自造自解自续), KV 里存什么由我们决定。
         *
         * 于是把 text(对外的标准 DSML 字节, 一字不改)与 KV(送进模型的 token 流)解耦:
         *   KV 看到:  bash(command=          ~5 tok
         *   text 输出: <｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">…  完整合法
         * edit(3参数): 68 → ~20 tok, 预期 inject 32.8s → ~10s, 单轮 48s → ~25s。
         * 采样语义(值区 copy 约束/停止 token)不受影响: 停止判据用字符与 par_close_tok,
         * 后者在紧凑模式下由 compact 尾串的首 token 承担。 */
        #define PRIMER_INJECT_KV(lit, kvlit) do { \
            const char *ktxt = (kvlit); \
            ds4_tokens itoks = {0}; \
            ds4_tokenize_rendered_chat(s->engine, ktxt, &itoks); \
            const double inj_t0 = now_sec(); \
            const bool batch = itoks.len > 1; \
            if (batch) { \
                const ds4_tokens *cur = ds4_session_tokens(s->session); \
                if (!cur || ds4_session_pos(s->session) + itoks.len > ds4_session_ctx(s->session)) gok = false; \
                if (gok) { \
                    ds4_tokens all = {0}; \
                    for (int pi = 0; pi < cur->len; pi++) ds4_tokens_push(&all, cur->v[pi]); \
                    for (int pi = 0; pi < itoks.len; pi++) ds4_tokens_push(&all, itoks.v[pi]); \
                    if (ds4_session_sync(s->session, &all, perr, sizeof(perr)) != 0) gok = false; \
                    ds4_tokens_free(&all); \
                } \
            } else { \
                for (int pi = 0; gok && pi < itoks.len; pi++) { \
                    if (ds4_session_pos(s->session) >= ds4_session_ctx(s->session) || \
                        ds4_session_eval(s->session, itoks.v[pi], perr, sizeof(perr)) != 0) gok = false; \
                    PRIMER_KA(); \
                } \
            } \
            if (gok) ds4_buf_append(&text, (lit), strlen(lit)); \
            inject_sec += now_sec() - inj_t0; \
            ds4_tokens_free(&itoks); \
        } while (0)
        /* 兼容入口: KV 与 text 同串(非紧凑模式) */
        #define PRIMER_INJECT(lit) PRIMER_INJECT_KV((lit), (lit))
        /* 紧凑模式选择器: compact 开时 KV 走 kvlit, 否则与 text 同串 */
        #define PRIMER_CK(lit, kvlit) \
            PRIMER_INJECT_KV((lit), (g_primer_compact ? (kvlit) : (lit)))
        /* generate: sample greedy tokens until a stop char appears or budget
         * runs out; the stop-char token itself is NOT committed (the session
         * then continues from the injected structure instead). */
        /* generate under the copy constraint (primer_copy_step): the model may
         * stop whenever it wants; what it emits while not stopping must extend
         * a live span of the copy source. cm=NULL 语义不存在 —— dead cm 即自由。
         * (oldp, oldl): new!=old contract (2026-07-17). Non-NULL oldp arms the
         * check: a stop attempt while the emitted value equals oldp[0..oldl) is
         * refused and decoding diverges via the best non-closer token. Known
         * limit on record: a budget-exhausted equal value is not caught (the
         * closer is injected structurally afterwards); at 384-token budgets
         * this path is not reachable for realistic old_string lengths. */
        /* 拷贝约束区 = COPY_EMISSION lane: 逐字复用上下文正是契约, 核心采样器在此
         * lane 下返回 raw 无惩罚结果(全局 anticycle 自抄禁令让位)。lane 在宏体内成对
         * 设置: 宏体是单一 do{} 出口, 内层 break(收束/gok=false/预算尽/发散)都落到
         * 尾部, 尾部恢复 FREE 覆盖所有退出路径(含 fell_back 后的自由续跑与外层
         * echo/rewind)。 */
        #define PRIMER_GEN_COPY_C(cmp, stopchars, stoptok, hard, budget, oldp, oldl) do { \
            const double gen_t0 = now_sec(); \
            ds4_session_set_lane(s->session, DS4_LANE_COPY_EMISSION); \
            int left = (budget); \
            bool fell_back = false; \
            const size_t nv_start = text.len; \
            while (gok && left-- > 0) { \
                if (ds4_session_pos(s->session) >= ds4_session_ctx(s->session)) { gok = false; break; } \
                int gt = primer_copy_step(s, (cmp), (stopchars), (stoptok), (hard), &fell_back); \
                if (gt < 0) { \
                    if ((oldp) != NULL && (oldl) > 0 && \
                        text.len - nv_start == (size_t)(oldl) && \
                        memcmp(text.ptr + nv_start, (oldp), (oldl)) == 0) { \
                        gt = primer_divergence_token(s, (stoptok)); \
                        if (gt < 0) break; \
                        trace_event(s, trace_id, \
                                    "new_string==old_string at closer -> forced divergence"); \
                    } else break; \
                } \
                size_t gl = 0; char *gp = ds4_token_text(s->engine, gt, &gl); \
                if (ds4_session_eval(s->session, gt, perr, sizeof(perr)) != 0) { gok = false; free(gp); break; } \
                if (gp) ds4_buf_append(&text, gp, gl); \
                free(gp); \
                gen_used++; \
                PRIMER_KA(); \
            } \
            ds4_session_set_lane(s->session, DS4_LANE_FREE); \
            if (fell_back) trace_event(s, trace_id, "guided copy-constraint fell back to free decode"); \
            gen_sec += now_sec() - gen_t0; \
        } while (0)
        #define PRIMER_GEN_COPY(cmp, stopchars, stoptok, hard, budget) \
            PRIMER_GEN_COPY_C(cmp, stopchars, stoptok, hard, budget, NULL, 0)

        /* copy sources: declared tool names (anchored) + the rendered prompt */
        buf names_src = {0};
        for (int ti = 0; ti < j->req.tool_orders.len; ti++) {
            const tool_schema_order *nts = &j->req.tool_orders.v[ti];
            if (!nts->name) continue;
            ds4_buf_append(&names_src, "\x01", 1);
            ds4_buf_append(&names_src, nts->name, strlen(nts->name));
        }
        primer_copy name_cm;
        primer_copy_init(&name_cm, names_src.ptr, names_src.len, true);
        /* 值拷贝源 = 首个 User 轮起的对话内容 (用户消息 + tool_result)。
         * 不能用整份 prompt: tools header 的 DSML 用法示例里字面含
         * "$PARAMETER_VALUE" (ds4_server.c 模板行), 全 prompt 作源时模型
         * 恰好合法地抄它 —— 占位符的真正出处 (2026-07-07 判决链终点)。 */
        const char *vsrc = j->req.prompt_text;
        size_t vlen = vsrc ? strlen(vsrc) : 0;
        /* 渲染器亲手记录的会话区起点优先(header 含 soul 示例/system 时 strstr 会被
         * 内容里的 "# User:" 劫持 — 2026-07-15 行为门实证: 无关问题抄 soul 命令)。 */
        if (vsrc && j->req.prompt_conv_off > 0 && j->req.prompt_conv_off < vlen) {
            vsrc += j->req.prompt_conv_off;
            vlen -= j->req.prompt_conv_off;
        } else
        if (vsrc) {
            /* 骨架无关: chat 帧用 <｜User｜>, base-native 用 "# User:" —— 两者都必须
             * 跳过 tools header(其 DSML 用法示例字面含 $PARAMETER_VALUE, 全 prompt
             * 作源时模型合法抄它 = 占位符真出处)。base-native 下找不到 <｜User｜>
             * 而不切源, 就是占位符在新骨架下复发的原因(2026-07-14)。 */
            const char *u = strstr(vsrc, "<｜User｜>");
            if (!u) u = strstr(vsrc, "\n# User:");
            if (u) { vlen -= (size_t)(u - vsrc); vsrc = u; }
        }
        /* ★本轮自由区文本并入值拷贝源(2026-07-16 Clamp 直测实证)★: 模型在自由区写出的
         * 计划("Edit ... to return lo instead of hi")是最贴任务的 span 库, 但值采样器
         * 原本只看渲染 prompt — 计划里明说 "return lo", 值区却抄不到, 只能沿文件 span
         * 贪婪续抄(new_string 抄成文件后续内容)。拼接缓冲生命周期=primer 块内。 */
        char *vsrc_own = NULL;
        if (text.len > 0) {
            vsrc_own = xmalloc(vlen + 1 + text.len + 1);
            memcpy(vsrc_own, vsrc ? vsrc : "", vlen);
            vsrc_own[vlen] = '\n';
            memcpy(vsrc_own + vlen + 1, text.ptr, text.len);
            vsrc_own[vlen + 1 + text.len] = '\0';
            vsrc = vsrc_own;
            vlen = vlen + 1 + text.len;
        }
        /* 参数关闭 tag 的首 token = 多行值的停止信号: 值里可含 '<' 和换行
         * (old_string 抄 "i <= len" 这类 Go 代码), 字符级 stop 会腰斩;
         * DSML 标记是特殊 token, 不受 BPE 上下文影响, token 级停可靠。 */
        int par_close_tok = -1;
        {
            ds4_tokens ct = {0};
            /* 紧凑模式下 KV 里的参数分隔是 "\n"(不是 DSML 关闭 tag) → 停止 token 随之改;
             * 多行值(content/new_string)在紧凑模式下靠字符集停止(见下 multiline 分支)。 */
            const char *close_lit =
                g_primer_compact ? "\n" : "</｜DSML｜parameter>";
            ds4_tokenize_rendered_chat(s->engine, close_lit, &ct);
            if (ct.len > 0) par_close_tok = ct.v[0];
            ds4_tokens_free(&ct);
        }

        PRIMER_CK("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"", "\ncall ");
        if (j->req.tool_force_name) {
            /* named tool_choice: the name region is a known literal — inject it
             * (same contract as every other structural token: known => injected) */
            PRIMER_INJECT(j->req.tool_force_name);
        } else {
            PRIMER_GEN_COPY(&name_cm, "\"\n<>", -1, false, 16);  /* tool name (约束=声明名集合) */
        }
        /* Snap the generated name to a declared tool, then inject its primary
         * parameter (schema property #0 -- file_path/command/pattern for the
         * standard agent tools) and let the model fill only the value. v0
         * limitation on record: secondary/optional parameters are not
         * emitted; the general schema-walking decoder is P2 work. */
        for (int ti = 0; gok && ti < j->req.tool_orders.len; ti++) {
            const tool_schema_order *ts = &j->req.tool_orders.v[ti];
            size_t name_from = 0;
            const char *inv = text.ptr ? strrchr(text.ptr, '"') : NULL;
            if (inv) name_from = (size_t)(inv + 1 - text.ptr);
            if (!ts->name || text.len <= name_from ||
                strncmp(text.ptr + name_from, ts->name, text.len - name_from) != 0)
                continue;
            text.len = name_from;               /* canonical name replaces prefix */
            ds4_buf_append(&text, ts->name, strlen(ts->name));
            /* 发射全部 required 参数 (schema 序; 无 required 声明时退回 v0
             * 首参行为)。每参数一个独立拷贝匹配器 —— span 状态不跨参数。 */
            bool opened = false;
            char *edit_old_val = NULL;   /* old_string 的生成值 (new!=old 契约用) */
            size_t edit_old_len = 0;
            for (int pk = 0; gok && pk < ts->len; pk++) {
                const char *pn = ts->prop[pk];
                bool need = false;
                if (ts->req_len > 0) {
                    for (int rk = 0; rk < ts->req_len && !need; rk++)
                        if (!strcmp(ts->req[rk], pn)) need = true;
                } else {
                    need = pk == 0;
                }
                if (!need || !pn) continue;
                /* 多行值参数 (编辑串/文件内容): 不用字符级 stop, 只认
                 * closer-token/EOS —— 值域含换行与 '<' 是正常代码。 */
                const bool multiline = !strcmp(pn, "old_string") ||
                                       !strcmp(pn, "new_string") ||
                                       !strcmp(pn, "content") ||
                                       !strcmp(pn, "body") || !strcmp(pn, "text");
                char head[256], khead[128];
                snprintf(head, sizeof(head),
                         "%s<｜DSML｜parameter name=\"%s\" string=\"true\">",
                         opened ? "" : "\">\n", pn);
                snprintf(khead, sizeof(khead), "\n%s=", pn);   /* 紧凑 KV: 只留语义锚(参数名) */
                PRIMER_CK(head, khead);
                opened = true;
                /* old_string 按合同必须逐字来自现存内容: 硬约束, 不可行即停,
                 * 宁可短 span 让 Edit 明确失败重试, 不给幻觉出口。 */
                const bool hard = !strcmp(pn, "old_string");
                /* ★new_string 解除 copy 约束走自由(2026-07-17 场景2 实证)★: new_string
                 * 按定义是"要写入的新内容", 修复后的行(如 i < n)从不以完整 span 出现在
                 * 上下文 → copy 约束物理上只能抄回旧行(i <= n)。生成型参数
                 * (content/body/text 同理)也解除。 */
                const bool gen_param = !strcmp(pn, "new_string") ||
                                       !strcmp(pn, "content") ||
                                       !strcmp(pn, "body") || !strcmp(pn, "text");
                const bool free_gen = gen_param;
                primer_copy pc;
                primer_copy_init(&pc, free_gen ? "" : vsrc, free_gen ? 0 : vlen, false);
                const size_t val_start = text.len;
                /* new!=old 契约只武装 new_string, 且仅当同一调用先产出了 old_string */
                const bool arm_neq = !strcmp(pn, "new_string") && edit_old_val != NULL;
                /* ★同调用禁重契约(2026-07-22)★: 上一调用同工具同参数的值字节在案时武装 —
                 * 闭合时值逐字节等于上一调用 → 拒闭合走分歧(多轮回路实证失败形态:
                 * 消化 tool_result 后原样重发同一命令)。neq 契约优先(同为闭合检查通道)。 */
                const bool arm_same = !arm_neq &&
                                      s->prev_call_tool && ts->name &&
                                      !strcmp(s->prev_call_tool, ts->name) &&
                                      s->prev_call_param && !strcmp(s->prev_call_param, pn) &&
                                      s->prev_call_val && s->prev_call_len > 0;
                PRIMER_GEN_COPY_C(&pc, multiline ? "" : "\n<", par_close_tok, hard,
                                  multiline ? 384 : 96,
                                  arm_neq ? edit_old_val : (arm_same ? s->prev_call_val : NULL),
                                  arm_neq ? edit_old_len : (arm_same ? s->prev_call_len : 0));
                if (gok && !strcmp(pn, "old_string") && text.len > val_start) {
                    free(edit_old_val);
                    edit_old_len = text.len - val_start;
                    edit_old_val = xmalloc(edit_old_len + 1);
                    memcpy(edit_old_val, text.ptr + val_start, edit_old_len);
                    edit_old_val[edit_old_len] = '\0';
                }
                /* 同调用禁重: 每个完成的参数值滚动暂存(跨请求, 跟 session 走);
                 * 单参工具(Bash.command)即主值, 多参工具落到最后一个必填参数。 */
                if (gok && text.len > val_start) {
                    free(s->prev_call_tool); free(s->prev_call_param); free(s->prev_call_val);
                    s->prev_call_tool = xstrdup(ts->name ? ts->name : "");
                    s->prev_call_param = xstrdup(pn);
                    s->prev_call_len = text.len - val_start;
                    s->prev_call_val = xmalloc(s->prev_call_len + 1);
                    memcpy(s->prev_call_val, text.ptr + val_start, s->prev_call_len);
                    s->prev_call_val[s->prev_call_len] = '\0';
                }
                PRIMER_CK("</｜DSML｜parameter>\n", "\n");
            }
            free(edit_old_val);
            PRIMER_CK(opened ? "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                             : "\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
                      "\n");
            break;
        }
        free(names_src.ptr);
        free(vsrc_own);
        #undef PRIMER_INJECT
        #undef PRIMER_GEN_COPY
        #undef PRIMER_GEN_COPY_C
        if (gok) {
            dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
            saw_tool_start = true;
            saw_tool_end = true;
            completion = gen_used > 0 ? gen_used : 1;
            max_tokens = completion;            /* skip the free-running loop */
            finish = "tool_calls";              /* same contract as the in-loop saw_tool_end path */
            trace_event(s, trace_id, "guided tool call generated (%d free tokens)", gen_used);
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: guided timing inject=%.1fs gen=%.1fs free=%d",
                       inject_sec, gen_sec, gen_used);
        }
    }

