/* web_internal.h — agent web 抓取模块内部头(重构阶段3, 自 ds4_web.c 机械五分:
 * util=基元/TCP/HTTP, ws=WebSocket/CDP 协议, page=页面等待与滚动,
 * browser=Chrome 启动与标签页, api=JS 注入与公共入口)。命名与拆分前一致。 */
#ifndef DS4_WEB_INTERNAL_H
#define DS4_WEB_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

#include "ds4_web.h"

#define DS4_WEB_CONNECT_TIMEOUT_MS 3000
#define DS4_WEB_CDP_TIMEOUT_MS 20000
#define DS4_WEB_MAX_RESULT_BYTES (1024*1024)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} web_buf;

struct ds4_web {
    char home[PATH_MAX];
    char profile_dir[PATH_MAX];
    int port;
    pid_t chrome_pid;
    bool browser_allowed;
    ds4_web_confirm_fn confirm;
    void *confirm_privdata;
    ds4_web_log_fn log;
    void *log_privdata;
    int next_cdp_id;
};

typedef struct {
    int fd;
    int next_id;
} cdp_ws;

typedef struct {
    char *id;
    char *ws_url;
} web_tab;


char *web_run_page_js(ds4_web *web, const char *url, const char *js,
                             bool dynamic_scroll,
                             char *err, size_t err_len);
char *web_chrome_executable(void);
const char *web_macos_chrome_app_name(void);
bool web_spawn_chrome(ds4_web *web, char *err, size_t err_len);
bool web_ensure_browser(ds4_web *web, char *err, size_t err_len);
void web_tab_free(web_tab *tab);
char *web_browser_ws_url(ds4_web *web, char *err, size_t err_len);
bool web_open_tab(ds4_web *web, const char *url, web_tab *tab,
                         char *err, size_t err_len);
void web_close_tab(ds4_web *web, const web_tab *tab);
char *web_cdp_eval_string(cdp_ws *ws, const char *expr,
                                 char *err, size_t err_len);
bool web_wait_ready(cdp_ws *ws, char *err, size_t err_len);
bool web_cdp_navigate(cdp_ws *ws, const char *url,
                             char *err, size_t err_len);
bool web_page_probe(cdp_ws *ws, char **href_out, char **ready_out,
                           long *text_len_out, char *err, size_t err_len);
bool web_wait_navigated_ready(cdp_ws *ws, const char *url,
                                     char *err, size_t err_len);
bool web_cdp_prepare_page(cdp_ws *ws, char *err, size_t err_len);
void web_scroll_dynamic_page(cdp_ws *ws);
void *web_xmalloc(size_t n);
char *web_xstrdup(const char *s);
void web_buf_append(web_buf *b, const char *s, size_t n);
void web_buf_puts(web_buf *b, const char *s);
char *web_buf_take(web_buf *b);
void web_set_err(char *err, size_t err_len, const char *fmt, ...);
void web_log(ds4_web *web, const char *msg);
bool web_mkdir_p(const char *path);
int web_tcp_connect(const char *host, int port, int timeout_ms,
                           char *err, size_t err_len);
int web_write_all(int fd, const void *buf, size_t len);
ssize_t web_read_some(int fd, char *buf, size_t len, int timeout_ms);
char *web_http_request(const char *method, int port, const char *path,
                              char *err, size_t err_len);
bool web_cdp_alive(ds4_web *web);
char *web_json_get_string(const char *json, const char *key);

char *web_url_encode(const char *s);
void web_random_bytes(unsigned char *buf, size_t len);
char *web_base64(const unsigned char *data, size_t len);
char *web_json_quote(const char *s);
int web_ws_connect(const char *ws_url, cdp_ws *ws,
                          char *err, size_t err_len);
void web_ws_close(cdp_ws *ws);
int web_read_exact(int fd, unsigned char *buf, size_t len, int timeout_ms);
int web_ws_send_text(cdp_ws *ws, const char *text,
                            char *err, size_t err_len);
int web_ws_send_pong(cdp_ws *ws, const unsigned char *payload, size_t len);
char *web_ws_read_message(cdp_ws *ws, char *err, size_t err_len);
bool web_json_id_matches(const char *json, int id);
char *web_cdp_call(cdp_ws *ws, const char *method, const char *params,
                          char *err, size_t err_len);
void web_cdp_call_optional(cdp_ws *ws, const char *method, const char *params);
int web_hex4(const char *p);
void web_utf8_append(web_buf *b, unsigned code);
char *web_json_parse_string_at(const char *q, const char **endp);
char *web_json_get_string(const char *json, const char *key);

#endif /* DS4_WEB_INTERNAL_H */
