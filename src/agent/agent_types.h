/* agent_types.h — ds4_agent.c 机械拆分出的跨文件共享类型/宏。
 * 块内容与原单文件逐字相同, 顺序保持原文件顺序; 只被单个 .c 用的类型仍留在那个 .c 里。
 * 不要直接 include 本文件: 统一经 agent_internal.h。 */
#ifndef DS4_AGENT_TYPES_H
#define DS4_AGENT_TYPES_H

typedef struct {
    const char *prompt;
    const char *system;
    const char *trace_path;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ds4_think_mode think_mode;
} agent_generation_options;

typedef struct {
    ds4_engine_options engine;
    agent_generation_options gen;
    const char *chdir_path;
    bool non_interactive;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_DRAINING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    int generated;
    double gen_tps;
    int ctx_used;
    int ctx_size;
    int power_percent;
    char error[256];
} agent_status;

typedef struct agent_bash_job agent_bash_job;

typedef struct {
    ds4_engine *engine;
    agent_config *cfg;
    ds4_session *session;
    ds4_tokens transcript;
    char *cache_dir;
    char *sysprompt_path;
    char session_sha[41];
    char *session_title;
    uint64_t session_created_at;
    char *legacy_session_path_to_delete;
    bool user_activity;
    bool session_dirty;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
    bool stop;
    bool interrupt;
    bool initialized;
    bool save_requested;
    bool compact_requested;
    bool power_requested;
    int requested_power;
    int progress_base;
    int last_system_prompt_reminder_at;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    ds4_web *web;
    bool web_approval_pending;
    bool web_approval_answered;
    bool web_approval_result;
    char web_approval_message[256];
    char web_approval_error[160];
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    char more_path[PATH_MAX];
    int more_next_line;
    bool more_bare;
    bool more_valid;
    agent_bash_job *bash_jobs;
    int next_bash_job_id;
} agent_worker;

typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_STAR,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    ds4_engine *engine;
    agent_worker *worker;
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    bool md_code_highlight_upto;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

typedef struct {
    char *name;
    char *value;
    bool is_string;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

typedef enum {
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
    agent_dsml_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    agent_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    agent_tool_calls calls;
    char error[160];
} agent_dsml_parser;

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

typedef struct {
    agent_token_renderer *renderer;
    agent_dsml_parser *parser;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    char think_dsml_tail[32];
    size_t think_dsml_len;
    bool dsml_in_think;
    bool dsml_in_think_reported;
    bool post_think_gap;
    bool tool_preflight_error;
    char tool_preflight_error_msg[256];
} agent_stream_renderer;

typedef struct {
    bool active;
    bool done;
} agent_edit_upto_forcer;

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
#define AGENT_HL_NORMAL 0

#define AGENT_HL_COMMENT 1

#define AGENT_HL_KEYWORD1 2

#define AGENT_HL_KEYWORD2 3

#define AGENT_HL_STRING 4

#define AGENT_HL_NUMBER 5

#define AGENT_SYNTAX_NUMBERS (1u<<0)

#define AGENT_SYNTAX_STRINGS (1u<<1)

#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)

#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)

struct agent_syntax {
    const char *name;
    const char *aliases;
    const char **keywords;
    const char *singleline_comments[3];
    const char *multiline_start;
    const char *multiline_end;
    unsigned flags;
};

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    bool truncated;
} agent_buf;

typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

#define AGENT_HISTORY_DEFAULT_TURNS 3

#define AGENT_HISTORY_MAX_TURNS 200

typedef enum {
    AGENT_HISTORY_MARK_NONE,
    AGENT_HISTORY_MARK_USER,
    AGENT_HISTORY_MARK_ASSISTANT,
    AGENT_HISTORY_MARK_EOS,
} agent_history_mark;

#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024

#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64

#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2

#define AGENT_COMPACT_SOFT_PERCENT 85

#define AGENT_COMPACT_MIN_FREE_TOKENS 8192

#define AGENT_COMPACT_TAIL_DIVISOR 10

#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000

#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

struct agent_bash_job {
    int id;
    pid_t pid;
    int pipe_fd;
    int tmp_fd;
    char path[PATH_MAX];
    char *cmd;
    double start_time;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    size_t observed_bytes;
    int observed_display_lines;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    struct agent_bash_job *next;
};

#define AGENT_INPUT_INITIAL_BUFLEN 4096

#define AGENT_INPUT_MAX_BUFLEN (1024*1024)

#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"

#define AGENT_STATUS_STYLE_END "\x1b[0m"

#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

typedef enum {
    AGENT_YES_NO_AUTO_NONE,
    AGENT_YES_NO_AUTO_NO,
    AGENT_YES_NO_AUTO_YES,
} agent_yes_no_auto;

typedef struct {
    int timeout_sec;
    agent_yes_no_auto timeout_answer;
} agent_yes_no_options;

typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

#endif /* DS4_AGENT_TYPES_H */
