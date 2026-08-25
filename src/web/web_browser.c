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

char *web_chrome_executable(void) {
    const char *env = getenv("DS4_CHROME");
    if (env && env[0]) return web_xstrdup(env);
#ifdef __APPLE__
    if (access("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", X_OK) == 0)
        return web_xstrdup("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
    if (access("/Applications/Chromium.app/Contents/MacOS/Chromium", X_OK) == 0)
        return web_xstrdup("/Applications/Chromium.app/Contents/MacOS/Chromium");
#endif
    const char *paths[] = {
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/opt/google/chrome/chrome",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], X_OK) == 0) return web_xstrdup(paths[i]);
    }

    const char *names[] = {
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        NULL
    };
    const char *pathenv = getenv("PATH");
    if (pathenv) {
        char *path = web_xstrdup(pathenv);
        char *save = NULL;
        for (char *dir = strtok_r(path, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
            for (int i = 0; names[i]; i++) {
                char candidate[PATH_MAX];
                snprintf(candidate, sizeof(candidate), "%s/%s", dir[0] ? dir : ".", names[i]);
                if (access(candidate, X_OK) == 0) {
                    char *res = web_xstrdup(candidate);
                    free(path);
                    return res;
                }
            }
        }
        free(path);
    }
    return web_xstrdup("google-chrome");
}

#ifdef __APPLE__
const char *web_macos_chrome_app_name(void) {
    if (getenv("DS4_CHROME")) return NULL;
    if (access("/Applications/Google Chrome.app", F_OK) == 0)
        return "Google Chrome";
    if (access("/Applications/Chromium.app", F_OK) == 0)
        return "Chromium";
    return NULL;
}
#endif

bool web_spawn_chrome(ds4_web *web, char *err, size_t err_len) {
    if (!web_mkdir_p(web->profile_dir)) {
        web_set_err(err, err_len, "failed to create Chrome profile dir %s: %s",
                    web->profile_dir, strerror(errno));
        return false;
    }
    char *exe = web_chrome_executable();
#ifdef __APPLE__
    const char *mac_app_name = web_macos_chrome_app_name();
    bool launched_via_open = mac_app_name != NULL && access("/usr/bin/open", X_OK) == 0;
#else
    bool launched_via_open = false;
#endif
    char port_arg[64], profile_arg[PATH_MAX + 64];
    snprintf(port_arg, sizeof(port_arg), "--remote-debugging-port=%d", web->port);
    snprintf(profile_arg, sizeof(profile_arg), "--user-data-dir=%s", web->profile_dir);
    pid_t pid = fork();
    if (pid < 0) {
        web_set_err(err, err_len, "failed to fork Chrome: %s", strerror(errno));
        free(exe);
        return false;
    }
    if (pid == 0) {
        int nullfd = open("/dev/null", O_RDWR);
        if (nullfd >= 0) {
            dup2(nullfd, STDOUT_FILENO);
            dup2(nullfd, STDERR_FILENO);
            if (nullfd > 2) close(nullfd);
        }
#ifdef __APPLE__
        if (launched_via_open) {
            execlp("/usr/bin/open", "open", "-g", "-na", mac_app_name,
                   "--args", port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--use-mock-keychain", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        } else {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--use-mock-keychain", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        }
#else
        if (geteuid() == 0) {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--password-store=basic", "--no-sandbox",
                   "--mute-audio", "about:blank", (char *)NULL);
        } else {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        }
#endif
        _exit(127);
    }
    free(exe);
    web->chrome_pid = pid;
    for (int i = 0; i < 80; i++) {
        if (web_cdp_alive(web)) {
            web_log(web, "Chrome browser session is ready");
            return true;
        }
        int status = 0;
        pid_t rc = waitpid(pid, &status, WNOHANG);
        if (rc == pid) {
            web->chrome_pid = 0;
            if (launched_via_open) continue;
            web_set_err(err, err_len, "Chrome exited before CDP became ready");
            return false;
        }
        usleep(250000);
    }
    web_set_err(err, err_len, "Chrome did not expose CDP on port %d", web->port);
    return false;
}

bool web_ensure_browser(ds4_web *web, char *err, size_t err_len) {
    if (web_cdp_alive(web)) return true;
    if (web->chrome_pid > 0) {
        int status = 0;
        waitpid(web->chrome_pid, &status, WNOHANG);
        web->chrome_pid = 0;
    }
    if (!web->browser_allowed) {
        if (!web->confirm) {
            web_set_err(err, err_len,
                        "starting a visible Chrome browser requires interactive approval");
            return false;
        }
        if (!web->confirm(web->confirm_privdata,
                          "The web tool wants to start a visible Chrome browser. Allow? (y/n) ",
                          err, err_len))
        {
            if (err && !err[0]) web_set_err(err, err_len, "user denied Chrome browser start");
            return false;
        }
        web->browser_allowed = true;
    }
    return web_spawn_chrome(web, err, err_len);
}

void web_tab_free(web_tab *tab) {
    if (!tab) return;
    free(tab->id);
    free(tab->ws_url);
    tab->id = NULL;
    tab->ws_url = NULL;
}

char *web_browser_ws_url(ds4_web *web, char *err, size_t err_len) {
    char *body = web_http_request("GET", web->port, "/json/version", err, err_len);
    if (!body) return NULL;
    char *ws = web_json_get_string(body, "webSocketDebuggerUrl");
    free(body);
    if (!ws) web_set_err(err, err_len, "Chrome did not return a browser WebSocket URL");
    return ws;
}

bool web_open_tab(ds4_web *web, const char *url, web_tab *tab,
                         char *err, size_t err_len) {
    memset(tab, 0, sizeof(*tab));

    char *browser_url = web_browser_ws_url(web, err, err_len);
    if (!browser_url) return false;
    cdp_ws browser = {.fd = -1};
    if (web_ws_connect(browser_url, &browser, err, err_len) != 0) {
        free(browser_url);
        return false;
    }
    free(browser_url);

    char *qurl = web_json_quote(url);
    web_buf params = {0};
    web_buf_puts(&params, "{\"url\":");
    web_buf_puts(&params, qurl);
    web_buf_puts(&params, ",\"background\":true,\"newWindow\":false}");
    free(qurl);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(&browser, "Target.createTarget",
                              params_s, err, err_len);
    free(params_s);
    web_ws_close(&browser);
    if (!resp) return false;

    tab->id = web_json_get_string(resp, "targetId");
    free(resp);
    if (!tab->id) {
        web_tab_free(tab);
        web_set_err(err, err_len, "Chrome did not return a page target id");
        return false;
    }

    char ws_url[PATH_MAX + 128];
    snprintf(ws_url, sizeof(ws_url), "ws://127.0.0.1:%d/devtools/page/%s",
             web->port, tab->id);
    tab->ws_url = web_xstrdup(ws_url);
    return true;
}

void web_close_tab(ds4_web *web, const web_tab *tab) {
    if (!web || !tab || !tab->id || !tab->id[0]) return;
    char *enc = web_url_encode(tab->id);
    web_buf path = {0};
    web_buf_puts(&path, "/json/close/");
    web_buf_puts(&path, enc);
    free(enc);

    char err[160] = {0};
    char *path_s = web_buf_take(&path);
    char *body = web_http_request("GET", web->port, path_s, err, sizeof(err));
    free(path_s);
    if (body) {
        free(body);
    } else if (err[0]) {
        web_log(web, err);
    }
}

