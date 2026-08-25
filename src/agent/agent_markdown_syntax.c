/* agent_markdown_syntax.c — 机械拆分自 ds4_agent.c: Assistant Markdown Rendering (1/3: renderer 基础写 + 语法高亮表)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Assistant Markdown Rendering
 * ============================================================================
 *
 * This renderer handles only the cheap markdown cues that make terminal output
 * readable: **bold**, *italic*, inline code, and fenced code blocks.  It is a
 * streaming parser, so it buffers only ambiguous marker bytes long enough to
 * decide whether they are formatting or literal text.
 */

static void agent_tail_capture_append(agent_tail_capture *t,
                                      const char *s, size_t n) {
    if (!t || !n) return;
    if (!t->cap) return;
    if (!t->buf) t->buf = xmalloc(t->cap);
    t->total += n;

    if (n >= t->cap) {
        memcpy(t->buf, s + n - t->cap, t->cap);
        t->start = 0;
        t->len = t->cap;
        return;
    }

    if (t->len < t->cap) {
        size_t free_tail = t->cap - t->len;
        size_t first = n < free_tail ? n : free_tail;
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = first < right ? first : right;
        memcpy(t->buf + pos, s, chunk);
        if (first > chunk) memcpy(t->buf, s + chunk, first - chunk);
        t->len += first;
        s += first;
        n -= first;
    }

    while (n) {
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = n < right ? n : right;
        memcpy(t->buf + pos, s, chunk);
        t->start = (t->start + chunk) % t->cap;
        s += chunk;
        n -= chunk;
    }
}

char *agent_tail_capture_take(agent_tail_capture *t, size_t *len) {
    size_t n = t ? t->len : 0;
    char *out = xmalloc(n + 1);
    if (n) {
        size_t right = t->cap - t->start;
        size_t first = n < right ? n : right;
        memcpy(out, t->buf + t->start, first);
        if (n > first) memcpy(out + first, t->buf, n - first);
    }
    out[n] = '\0';
    if (len) *len = n;
    free(t->buf);
    memset(t, 0, sizeof(*t));
    return out;
}

void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    if (r->capture) agent_tail_capture_append(r->capture, s, n);
    else agent_publish(r->worker, s, n);
}

static void renderer_set_grey(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[38;5;245m", 11);
}

void renderer_reset_color(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

static size_t renderer_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

static bool renderer_has_text_attrs(agent_token_renderer *r) {
    return r->in_think || r->md_bold || r->md_italic ||
           r->md_inline_code || r->md_code_block;
}

static void renderer_set_text_attrs(agent_token_renderer *r) {
    if (!r->use_color) return;
    if (r->in_think) {
        renderer_set_grey(r);
        return;
    }
    if (r->md_code_block) {
        renderer_write(r, "\x1b[38;5;75m", 10);
        return;
    } else if (r->md_inline_code) {
        renderer_write(r, "\x1b[36m", 5);
    }
    if (r->md_bold) renderer_write(r, "\x1b[1m", 4);
    if (r->md_italic) renderer_write(r, "\x1b[3m", 4);
}

void renderer_restore_text_attrs(agent_token_renderer *r) {
    if (!r->use_color || !r->color_open || !renderer_has_text_attrs(r)) return;
    renderer_set_text_attrs(r);
}

static void renderer_write_complete_char_raw(agent_token_renderer *r, const char *s, size_t n) {
    bool styled = r->use_color && renderer_has_text_attrs(r);
    if (styled && !r->color_open) {
        renderer_set_text_attrs(r);
        r->color_open = true;
    } else if (!styled && r->color_open) {
        renderer_reset_color(r);
    }
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    r->last_output_newline = n == 1 && s[0] == '\n';
}

void renderer_flush_utf8(agent_token_renderer *r) {
    if (!r->utf8_pending_len) return;
    renderer_write_complete_char_raw(r, r->utf8_pending, r->utf8_pending_len);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
}

void renderer_write_char_raw(agent_token_renderer *r, char c) {
    unsigned char uc = (unsigned char)c;

    if (r->utf8_pending_len) {
        if ((uc & 0xc0) == 0x80 && r->utf8_pending_len < sizeof(r->utf8_pending)) {
            r->utf8_pending[r->utf8_pending_len++] = c;
            if (r->utf8_pending_len == r->utf8_pending_need) renderer_flush_utf8(r);
            return;
        }
        renderer_flush_utf8(r);
    }

    size_t need = renderer_utf8_need(uc);
    if (need == 1) {
        renderer_write_complete_char_raw(r, &c, 1);
        return;
    }
    r->utf8_pending[0] = c;
    r->utf8_pending_len = 1;
    r->utf8_pending_need = need;
}

void renderer_write_plain_byte(agent_token_renderer *r, char c) {
    bool old_bold = r->md_bold;
    bool old_italic = r->md_italic;
    bool old_inline_code = r->md_inline_code;
    bool old_code_block = r->md_code_block;

    /* Code blocks are streamed immediately in plain text, then repainted with
     * syntax colors when a complete terminal-safe line is available.  Disable
     * markdown attributes only for this byte; renderer_write_char_raw() will
     * reset any tracked manual color once if needed. */
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    renderer_write_char_raw(r, c);
    r->md_bold = old_bold;
    r->md_italic = old_italic;
    r->md_inline_code = old_inline_code;
    r->md_code_block = old_code_block;
}




static const char *agent_kw_generic[] = {
    "if","else","for","while","do","switch","case","default","break",
    "continue","return","try","catch","finally","throw","throws","class",
    "struct","enum","interface","trait","impl","fn","func","function",
    "def","lambda","let","var","const","static","public","private",
    "protected","import","include","from","export","package","module",
    "namespace","new","delete","async","await","yield","match","type",
    "true|","false|","null|","nil|","none|","None|","NULL|","void|",
    "int|","long|","float|","double|","char|","bool|","string|",
    "String|","usize|","isize|","u8|","u16|","u32|","u64|","i8|",
    "i16|","i32|","i64|",NULL
};

static const char *agent_kw_c[] = {
    "auto","break","case","continue","default","do","else","enum",
    "extern","for","goto","if","register","return","sizeof","static",
    "struct","switch","typedef","union","volatile","while",
    "alignas","alignof","and","and_eq","asm","bitand","bitor","class",
    "compl","constexpr","const_cast","decltype","delete","dynamic_cast",
    "explicit","export","false","friend","inline","mutable","namespace",
    "new","noexcept","not","not_eq","nullptr","operator","or","or_eq",
    "private","protected","public","reinterpret_cast","static_assert",
    "static_cast","template","this","thread_local","throw","true","try",
    "typeid","typename","virtual","xor","xor_eq",
    "NULL|","bool|","char|","const|","double|","float|","int|","long|",
    "short|","signed|","size_t|","ssize_t|","uint8_t|","uint16_t|",
    "uint32_t|","uint64_t|","unsigned|","void|",NULL
};

static const char *agent_kw_python[] = {
    "and","as","assert","async","await","break","case","class","continue",
    "def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","match","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
    "False|","None|","True|","bool|","bytes|","dict|","float|","int|",
    "list|","object|","set|","str|","tuple|",NULL
};

static const char *agent_kw_js[] = {
    "async","await","break","case","catch","class","const","continue",
    "debugger","default","delete","do","else","export","extends",
    "finally","for","from","function","get","if","import","in",
    "instanceof","let","new","of","return","set","static","super",
    "switch","this","throw","try","typeof","var","void","while","with",
    "yield","abstract","as","declare","enum","implements","interface",
    "keyof","namespace","private","protected","public","readonly","type",
    "any|","boolean|","false|","never|","null|","number|","string|",
    "symbol|","true|","undefined|","unknown|","void|",NULL
};

static const char *agent_kw_java[] = {
    "abstract","assert","break","case","catch","class","const","continue",
    "default","do","else","enum","extends","final","finally","for","goto",
    "if","implements","import","instanceof","interface","native","new",
    "package","private","protected","public","return","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient",
    "try","volatile","while",
    "boolean|","byte|","char|","double|","false|","float|","int|","long|",
    "null|","short|","true|","void|",NULL
};

static const char *agent_kw_csharp[] = {
    "abstract","as","base","break","case","catch","checked","class","const",
    "continue","default","delegate","do","else","enum","event","explicit",
    "extern","finally","fixed","for","foreach","goto","if","implicit","in",
    "interface","internal","is","lock","namespace","new","operator","out",
    "override","params","private","protected","public","readonly","ref",
    "return","sealed","sizeof","stackalloc","static","struct","switch",
    "this","throw","try","typeof","unchecked","unsafe","using","virtual",
    "volatile","while","async","await","get","init","record","set","var",
    "bool|","byte|","char|","decimal|","double|","false|","float|","int|",
    "long|","null|","object|","sbyte|","short|","string|","true|","uint|",
    "ulong|","ushort|","void|",NULL
};

static const char *agent_kw_go[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface",
    "map","package","range","return","select","struct","switch","type",
    "var","bool|","byte|","complex64|","complex128|","error|","false|",
    "float32|","float64|","int|","int8|","int16|","int32|","int64|",
    "nil|","rune|","string|","true|","uint|","uint8|","uint16|",
    "uint32|","uint64|","uintptr|",NULL
};

static const char *agent_kw_rust[] = {
    "as","async","await","break","const","continue","crate","dyn","else",
    "enum","extern","fn","for","if","impl","in","let","loop","match",
    "mod","move","mut","pub","ref","return","self","Self","static",
    "struct","super","trait","type","unsafe","use","where","while",
    "bool|","char|","false|","f32|","f64|","i8|","i16|","i32|","i64|",
    "i128|","isize|","str|","String|","true|","u8|","u16|","u32|",
    "u64|","u128|","usize|",NULL
};

static const char *agent_kw_shell[] = {
    "case","do","done","elif","else","esac","fi","for","function","if",
    "in","select","then","time","until","while","break","continue",
    "return","export","local","readonly","source","test","true|","false|",
    "echo|","printf|","cd|","pwd|","read|","set|","unset|","shift|",NULL
};

static const char *agent_kw_sql[] = {
    "add","alter","and","as","asc","between","by","case","check","column",
    "constraint","create","delete","desc","distinct","drop","else","end",
    "exists","foreign","from","group","having","in","index","insert",
    "into","is","join","key","left","like","limit","not","null","on",
    "or","order","outer","primary","references","right","select","set",
    "table","then","union","unique","update","values","view","where",
    "bigint|","boolean|","date|","decimal|","false|","int|","integer|",
    "numeric|","real|","text|","timestamp|","true|","varchar|",NULL
};

static const char *agent_kw_ruby[] = {
    "BEGIN","END","alias","and","begin","break","case","class","def",
    "defined?","do","else","elsif","end","ensure","for","if","in",
    "module","next","not","or","redo","rescue","retry","return","self",
    "super","then","undef","unless","until","when","while","yield",
    "false|","nil|","true|",NULL
};

static const char *agent_kw_php[] = {
    "abstract","and","array","as","break","callable","case","catch","class",
    "clone","const","continue","declare","default","die","do","echo","else",
    "elseif","empty","enddeclare","endfor","endforeach","endif","endswitch",
    "endwhile","eval","exit","extends","final","finally","fn","for",
    "foreach","function","global","goto","if","implements","include",
    "include_once","instanceof","insteadof","interface","isset","list",
    "match","namespace","new","or","print","private","protected","public",
    "readonly","require","require_once","return","static","switch","throw",
    "trait","try","unset","use","var","while","xor","bool|","false|",
    "float|","int|","null|","string|","true|","void|",NULL
};

static const char *agent_kw_swift[] = {
    "actor","as","associatedtype","async","await","break","case","catch",
    "class","continue","default","defer","do","else","enum","extension",
    "fallthrough","for","func","guard","if","import","in","init","inout",
    "is","let","nonisolated","operator","private","protocol","public",
    "repeat","return","self","Self","static","struct","subscript","super",
    "switch","throw","throws","try","typealias","var","where","while",
    "Any|","Bool|","Double|","false|","Float|","Int|","nil|","String|",
    "true|","Void|",NULL
};

static const char *agent_kw_kotlin[] = {
    "as","break","class","continue","do","else","false","for","fun","if",
    "in","interface","is","null","object","package","return","super",
    "this","throw","true","try","typealias","typeof","val","var","when",
    "while","actual","annotation","by","catch","companion","const",
    "constructor","crossinline","data","enum","expect","external","final",
    "finally","import","infix","init","inline","inner","internal","lateinit",
    "noinline","open","operator","out","override","private","protected",
    "public","reified","sealed","suspend","tailrec","vararg",
    "Any|","Boolean|","Byte|","Char|","Double|","Float|","Int|","Long|",
    "Short|","String|","Unit|",NULL
};

static const char *agent_kw_zig[] = {
    "addrspace","align","allowzero","and","anyframe","anytype","asm",
    "async","await","break","callconv","catch","comptime","const",
    "continue","defer","else","enum","errdefer","error","export","extern",
    "fn","for","if","inline","linksection","noalias","noinline","nosuspend",
    "opaque","or","orelse","packed","pub","resume","return","struct",
    "suspend","switch","test","threadlocal","try","union","unreachable",
    "usingnamespace","var","volatile","while",
    "bool|","false|","f32|","f64|","i32|","i64|","null|","true|","u8|",
    "u16|","u32|","u64|","usize|","void|",NULL
};

static const char *agent_kw_lua[] = {
    "and","break","do","else","elseif","end","false","for","function",
    "goto","if","in","local","nil","not","or","repeat","return","then",
    "true","until","while",NULL
};

static const char *agent_kw_html[] = {
    "a","body","button","div","doctype","form","h1","h2","h3","head",
    "html","input","label","li","link","main","meta","ol","option","p",
    "script","section","select","span","style","table","tbody","td","th",
    "thead","title","tr","ul","class|","href|","id|","name|","rel|",
    "src|","type|","value|",NULL
};

static const char *agent_kw_css[] = {
    "align-items","background","border","bottom","color","display","flex",
    "font","font-size","gap","grid","height","justify-content","left",
    "margin","max-width","min-width","padding","position","right","top",
    "transform","width","z-index","absolute|","auto|","block|","flex|",
    "grid|","hidden|","inline|","none|","relative|","solid|",NULL
};

const agent_syntax agent_syntaxes[] = {
    {"generic", " text txt", agent_kw_generic, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"c", " c h cpp c++ cc cxx hpp hxx objc objective-c", agent_kw_c, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"python", " py python py3", agent_kw_python, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"javascript", " js jsx javascript typescript ts tsx node mjs cjs", agent_kw_js, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"java", " java", agent_kw_java, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"csharp", " cs c# csharp dotnet", agent_kw_csharp, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"go", " go golang", agent_kw_go, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"rust", " rs rust", agent_kw_rust, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"shell", " sh bash zsh shell fish ksh", agent_kw_shell, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"sql", " sql postgres mysql sqlite", agent_kw_sql, {"--",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_CASE_INSENSITIVE},
    {"ruby", " rb ruby", agent_kw_ruby, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"php", " php", agent_kw_php, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"swift", " swift", agent_kw_swift, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"kotlin", " kt kts kotlin", agent_kw_kotlin, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"zig", " zig", agent_kw_zig, {"//",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"lua", " lua", agent_kw_lua, {"--",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"html", " html htm xml svg", agent_kw_html, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"css", " css scss sass", agent_kw_css, {NULL,NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"json", " json jsonc", NULL, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"yaml", " yaml yml toml ini", NULL, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"markdown", " md markdown", agent_kw_generic, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {NULL, NULL, NULL, {NULL,NULL,NULL}, NULL, NULL, 0}
};
