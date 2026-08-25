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

static const char *web_click_google_consent_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const pats=[/accept all/i,/i agree/i,/agree/i,/accetta tutto/i,/tout accepter/i,/aceptar todo/i,/alle akzeptieren/i];"
"const els=[...document.querySelectorAll('button,[role=button],input[type=submit],a')];"
"for (const el of els){const t=clean(el.innerText||el.value||el.textContent);"
"if(!t)continue; if(pats.some(p=>p.test(t))){el.click(); return 'clicked '+t;}}"
"return '';"
"})()";

static const char *web_extract_search_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const esc=s=>clean(s).replace(/\\\\/g,'\\\\\\\\').replace(/\\[/g,'\\\\[').replace(/\\]/g,'\\\\]').replace(/\\n/g,' ');"
"const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};"
"const bad=h=>(/(^|\\.)google\\./.test(h)||/(^|\\.)gstatic\\./.test(h)||/(^|\\.)googleusercontent\\./.test(h));"
"const lines=['# Google search results','',`URL: ${location.href}`,'','## Visible links'];"
"const seen=new Set();"
"for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;let href=a.href||'';"
"try{const u=new URL(href);if(u.pathname==='/url'&&u.searchParams.get('q'))href=u.searchParams.get('q');}catch{}"
"let u;try{u=new URL(href);}catch{continue;}if(!/^https?:$/.test(u.protocol))continue;if(bad(u.hostname))continue;"
"const text=esc(a.innerText||a.textContent);if(text.length<3)continue;if(seen.has(u.href))continue;seen.add(u.href);"
"lines.push(`- [${text.slice(0,180)}](${u.href})`);if(seen.size>=20)break;}"
"lines.push('','## Text snapshot',clean(document.body.innerText).slice(0,1200));"
"return lines.join('\\n');"
"})()";

static const char *web_extract_page_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const esc=s=>clean(s).replace(/\\\\/g,'\\\\\\\\').replace(/\\[/g,'\\\\[').replace(/\\]/g,'\\\\]').replace(/\\n/g,' ');"
"const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};"
"const inline=n=>{if(!n)return'';if(n.nodeType===3)return n.nodeValue;if(n.nodeType!==1)return'';const el=n;"
"if(el.tagName==='SCRIPT'||el.tagName==='STYLE'||el.tagName==='NOSCRIPT')return'';"
"if(el.tagName==='A'){const t=esc(el.innerText||el.textContent);const h=el.href||'';return t&&h?`[${t}](${h})`:t;}"
"if(el.tagName==='CODE')return '`'+clean(el.innerText||el.textContent).replace(/`/g,'\\\\`')+'`';"
"return [...el.childNodes].map(inline).join('');};"
"const lines=[`# ${clean(document.title)||location.href}`,'',`URL: ${location.href}`,'','## Content'];"
"const blocks=[...document.body.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id=\"content-text\"],[class*=\"comment-body\"],[class*=\"comment-content\"],[data-testid*=\"comment-text\"]')];"
"const seen=new Set();"
"for(const el of blocks){if(!visible(el))continue;let s='';const tag=el.tagName;"
"if(/^H[1-6]$/.test(tag)){s='#'.repeat(Number(tag[1]))+' '+inline(el);}"
"else if(tag==='LI'){s='- '+inline(el);}"
"else if(tag==='PRE'){s='```\\n'+(el.innerText||el.textContent||'').trimEnd()+'\\n```';}"
"else if(tag==='BLOCKQUOTE'){s='> '+clean(el.innerText||el.textContent);}"
"else{s=inline(el);}s=s.trim();if(!s||seen.has(s))continue;seen.add(s);lines.push('',s);"
"if(lines.join('\\n').length>900000){lines.push('','[Content truncated by browser extractor.]');break;}}"
"lines.push('','## Visible links');let n=0;const linkSeen=new Set();"
"for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;const t=esc(a.innerText||a.textContent);if(t.length<3)continue;"
"let u;try{u=new URL(a.href);}catch{continue;}if(!/^https?:$/.test(u.protocol)||linkSeen.has(u.href))continue;linkSeen.add(u.href);"
"lines.push(`- [${t.slice(0,160)}](${u.href})`);if(++n>=80)break;}"
"return lines.join('\\n');"
"})()";

char *web_run_page_js(ds4_web *web, const char *url, const char *js,
                             bool dynamic_scroll,
                             char *err, size_t err_len) {
    if (!web_ensure_browser(web, err, err_len)) return NULL;
    web_tab tab = {0};
    if (!web_open_tab(web, "about:blank", &tab, err, err_len)) return NULL;
    cdp_ws ws = {.fd = -1};
    if (web_ws_connect(tab.ws_url, &ws, err, err_len) != 0) {
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    if (!web_cdp_prepare_page(&ws, err, err_len)) {
        web_ws_close(&ws);
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    if (!web_cdp_navigate(&ws, url, err, err_len) ||
        !web_wait_navigated_ready(&ws, url, err, err_len))
    {
        web_ws_close(&ws);
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    char *clicked = web_cdp_eval_string(&ws, web_click_google_consent_js, err, err_len);
    if (clicked && clicked[0]) {
        web_log(web, clicked);
        usleep(1500000);
        (void)web_wait_navigated_ready(&ws, url, err, err_len);
    }
    free(clicked);
    if (dynamic_scroll) web_scroll_dynamic_page(&ws);
    char *out = web_cdp_eval_string(&ws, js, err, err_len);
    web_ws_close(&ws);
    web_close_tab(web, &tab);
    web_tab_free(&tab);
    return out;
}

ds4_web *ds4_web_create(const ds4_web_config *cfg) {
    ds4_web *web = web_xmalloc(sizeof(*web));
    memset(web, 0, sizeof(*web));
    const char *home = cfg && cfg->home_dir && cfg->home_dir[0] ?
        cfg->home_dir : getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(web->home, sizeof(web->home), "%s", home);
    snprintf(web->profile_dir, sizeof(web->profile_dir), "%s/.ds4/browser", home);
    web->port = cfg && cfg->port > 0 ? cfg->port : DS4_WEB_DEFAULT_PORT;
    web->chrome_pid = 0;
    web->next_cdp_id = 1;
    if (cfg) {
        web->confirm = cfg->confirm;
        web->confirm_privdata = cfg->confirm_privdata;
        web->log = cfg->log;
        web->log_privdata = cfg->log_privdata;
    }
    return web;
}

void ds4_web_free(ds4_web *web) {
    if (!web) return;
    /* Do not kill Chrome.  The browser profile is user-visible state and keeping
     * it alive makes repeated web tool calls cheaper and less suspicious. */
    free(web);
}

char *ds4_web_google_search(ds4_web *web, const char *query,
                            char *err, size_t err_len) {
    if (!web) {
        web_set_err(err, err_len, "web subsystem is not initialized");
        return NULL;
    }
    if (!query || !query[0]) {
        web_set_err(err, err_len, "google_search requires query");
        return NULL;
    }
    char *q = web_url_encode(query);
    web_buf url = {0};
    web_buf_puts(&url, "https://www.google.com/search?q=");
    web_buf_puts(&url, q);
    free(q);
    char *url_s = web_buf_take(&url);
    char *out = web_run_page_js(web, url_s, web_extract_search_js, false, err, err_len);
    free(url_s);
    return out;
}

char *ds4_web_visit_page(ds4_web *web, const char *url,
                         char *err, size_t err_len) {
    if (!web) {
        web_set_err(err, err_len, "web subsystem is not initialized");
        return NULL;
    }
    if (!url || !url[0]) {
        web_set_err(err, err_len, "visit_page requires url");
        return NULL;
    }
    return web_run_page_js(web, url, web_extract_page_js, true, err, err_len);
}
