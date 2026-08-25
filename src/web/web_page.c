#include "ds4_web.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DS4_WEB_DEFAULT_PORT 9333
#include "web_internal.h"

char *web_cdp_eval_string(cdp_ws *ws, const char *expr,
                                 char *err, size_t err_len) {
    char *qexpr = web_json_quote(expr);
    web_buf params = {0};
    web_buf_puts(&params, "{\"expression\":");
    web_buf_puts(&params, qexpr);
    web_buf_puts(&params, ",\"returnByValue\":true,\"awaitPromise\":true,\"includeCommandLineAPI\":true}");
    free(qexpr);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(ws, "Runtime.evaluate", params_s, err, err_len);
    free(params_s);
    if (!resp) return NULL;
    if (strstr(resp, "\"exceptionDetails\"")) {
        web_set_err(err, err_len, "JavaScript evaluation failed");
        free(resp);
        return NULL;
    }
    char *val = web_json_get_string(resp, "value");
    free(resp);
    if (!val) web_set_err(err, err_len, "Runtime.evaluate did not return a string");
    return val;
}

bool web_wait_ready(cdp_ws *ws, char *err, size_t err_len) {
    const char *expr = "document.readyState";
    for (int i = 0; i < 80; i++) {
        char *state = web_cdp_eval_string(ws, expr, err, err_len);
        if (state && (!strcmp(state, "complete") || !strcmp(state, "interactive"))) {
            free(state);
            usleep(800000);
            return true;
        }
        free(state);
        usleep(250000);
    }
    return true;
}

bool web_cdp_navigate(cdp_ws *ws, const char *url,
                             char *err, size_t err_len) {
    char *qurl = web_json_quote(url);
    web_buf params = {0};
    web_buf_puts(&params, "{\"url\":");
    web_buf_puts(&params, qurl);
    web_buf_puts(&params, "}");
    free(qurl);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(ws, "Page.navigate", params_s, err, err_len);
    free(params_s);
    if (!resp) return false;
    free(resp);
    return true;
}

bool web_page_probe(cdp_ws *ws, char **href_out, char **ready_out,
                           long *text_len_out, char *err, size_t err_len) {
    const char *expr =
        "location.href+'\\n'+document.readyState+'\\n'+"
        "((document.body&&document.body.innerText)||'').length";
    char *probe = web_cdp_eval_string(ws, expr, err, err_len);
    if (!probe) return false;

    char *nl1 = strchr(probe, '\n');
    char *nl2 = nl1 ? strchr(nl1 + 1, '\n') : NULL;
    if (!nl1 || !nl2) {
        free(probe);
        web_set_err(err, err_len, "page readiness probe returned malformed data");
        return false;
    }
    *nl1 = '\0';
    *nl2 = '\0';
    if (href_out) *href_out = web_xstrdup(probe);
    if (ready_out) *ready_out = web_xstrdup(nl1 + 1);
    if (text_len_out) *text_len_out = strtol(nl2 + 1, NULL, 10);
    free(probe);
    return true;
}

bool web_wait_navigated_ready(cdp_ws *ws, const char *url,
                                     char *err, size_t err_len) {
    (void)url;
    long last_len = -1;
    int stable = 0;
    bool saw_real_url = false;

    for (int i = 0; i < 100; i++) {
        char *href = NULL;
        char *ready = NULL;
        long text_len = 0;
        bool ok = web_page_probe(ws, &href, &ready, &text_len, err, err_len);
        if (!ok) {
            free(href);
            free(ready);
            usleep(250000);
            continue;
        }

        bool real_url = href && href[0] &&
                        strcmp(href, "about:blank") &&
                        strncmp(href, "chrome://", 9);
        bool ready_state = ready &&
            (!strcmp(ready, "complete") || !strcmp(ready, "interactive"));
        if (real_url) saw_real_url = true;
        if (text_len > 0 && text_len == last_len) stable++;
        else stable = 0;
        last_len = text_len;

        free(href);
        free(ready);

        if (saw_real_url && ready_state && text_len > 0 && stable >= 2) {
            usleep(500000);
            return true;
        }
        if (saw_real_url && ready_state && i >= 24) return true;
        usleep(250000);
    }
    return true;
}

bool web_cdp_prepare_page(cdp_ws *ws, char *err, size_t err_len) {
    char *resp = web_cdp_call(ws, "Page.enable", "{}", err, err_len);
    if (!resp) return false;
    free(resp);
    resp = web_cdp_call(ws, "Runtime.enable", "{}", err, err_len);
    if (!resp) return false;
    free(resp);
    web_cdp_call_optional(ws, "Emulation.setFocusEmulationEnabled",
                          "{\"enabled\":true}");
    web_cdp_call_optional(ws, "Emulation.setDeviceMetricsOverride",
                          "{\"width\":1365,\"height\":900,\"deviceScaleFactor\":1,\"mobile\":false}");
    return web_wait_ready(ws, err, err_len);
}

void web_scroll_dynamic_page(cdp_ws *ws) {
    const char *expr =
        "(() => new Promise(resolve => {"
        "const root=()=>document.scrollingElement||document.documentElement||document.body;"
        "const blockSel='h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id=\"content-text\"],[class*=\"comment-body\"],[class*=\"comment-content\"],[data-testid*=\"comment-text\"]';"
        "const lazySel='[onscroll],[loading=\"lazy\"],[data-src],[data-lazy],[class*=\"lazy\"],[class*=\"infinite\"],[class*=\"virtual\"],[role=\"feed\"],[id*=\"comment\"],[class*=\"comment\"],[data-testid*=\"comment\"]';"
        "const hookCount=()=>{let n=0;try{if(window.onscroll)n++;if(document.onscroll)n++;if(document.body&&document.body.onscroll)n++;}catch(e){}"
        "try{if(typeof getEventListeners==='function'){for(const o of [window,document,document.body]){if(!o)continue;const ev=getEventListeners(o);if(ev&&ev.scroll)n+=ev.scroll.length;}}}catch(e){}"
        "try{n+=document.querySelectorAll(lazySel).length;}catch(e){}return n;};"
        "const metrics=()=>{const r=root();return {"
        "height:r?r.scrollHeight:0,"
        "view:innerHeight||900,"
        "y:scrollY||(r&&r.scrollTop)||0,"
        "text:((document.body&&document.body.innerText)||'').length,"
        "links:document.links?document.links.length:0,"
        "blocks:document.body?document.body.querySelectorAll(blockSel).length:0,"
        "hooks:hookCount()};};"
        "const sig=m=>[m.height,m.text,m.links,m.blocks].join('|');"
        "const grew=(a,b)=>b.height>a.height+20||b.text>a.text+200||b.links>a.links+2||b.blocks>a.blocks+2;"
        "const scrollOnce=()=>{const r=root();if(!r)return;"
        "const h=Math.max(700,Math.floor((innerHeight||900)*0.85));"
        "window.scrollTo(0,Math.min(r.scrollHeight,(scrollY||r.scrollTop||0)+h));};"
        "let last=metrics(),lastSig=sig(last),same=0,steps=0;"
        "const scrollable=last.height>last.view*1.35;"
        "if(!scrollable||last.hooks===0){resolve('scroll skipped hooks='+last.hooks+' text='+last.text);return;}"
        "const tick=()=>{"
        "if(steps>=28){resolve('scrolled '+steps+' text='+last.text);return;}"
        "const before=last;"
        "scrollOnce();steps++;"
        "setTimeout(()=>{const now=metrics(),nowSig=sig(now);"
        "if(nowSig===lastSig)same++;else same=0;"
        "const loaded=grew(before,now);"
        "last=now;lastSig=nowSig;"
        "if(steps===1&&!loaded){resolve('scroll probe unchanged text='+now.text);return;}"
        "const atBottom=now.y+now.view+20>=now.height;"
        "if(same>=4||(atBottom&&same>=1)){resolve('scrolled '+steps+' text='+now.text);return;}"
        "tick();},900);"
        "};tick();"
        "}))()";
    char err[160] = {0};
    char *res = web_cdp_eval_string(ws, expr, err, sizeof(err));
    free(res);
}

