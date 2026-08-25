/* dist_coord_route.c — 机械拆自 ds4_distributed.c: coordinator 注册表与路由规划(Coordinator Worker Registry And Route Planning)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Coordinator Worker Registry And Route Planning
 * =========================================================================
 *
 * A route is a contiguous chain that starts after the coordinator's local
 * slice. The last hop can either return logits directly or return the final
 * hidden state so the coordinator can run its local output head.
 */

void dist_coordinator_add_worker(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port,
        const ds4_dist_hello_fixed *hello,
        const char *model_name) {
    ds4_dist_worker_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory while registering worker\n");
        return;
    }

    entry->fd = fd;
    snprintf(entry->peer_host, sizeof(entry->peer_host), "%s", peer_host);
    snprintf(entry->peer_port, sizeof(entry->peer_port), "%s", peer_port);
    snprintf(entry->model_name, sizeof(entry->model_name), "%s", model_name ? model_name : "unknown");
    entry->model_id = hello->model_id;
    entry->quant_bits = hello->quant_bits;
    entry->layer_start = hello->layer_start;
    entry->layer_end = hello->layer_end;
    entry->has_output = hello->has_output;
    entry->has_hidden = hello->has_hidden;
    entry->ctx_size = hello->ctx_size;
    entry->n_layers = hello->n_layers;
    entry->listen_port = hello->listen_port;

    pthread_mutex_lock(&state->mu);
    if (state->shutting_down) {
        pthread_mutex_unlock(&state->mu);
        free(entry);
        return;
    }
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *old = *link;
        if (strcmp(old->peer_host, peer_host) == 0 &&
            old->model_id == hello->model_id &&
            old->layer_start == hello->layer_start &&
            old->layer_end == hello->layer_end &&
            old->has_output == hello->has_output)
        {
            *link = old->next;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: dropped stale worker %s:%s layers=%u:%u%s\n",
                             old->peer_host,
                             old->peer_port,
                             old->layer_start,
                             old->layer_end,
                             old->has_output ? "+output" : "");
            free(old);
            continue;
        }
        link = &old->next;
    }
    entry->next = state->workers;
    state->workers = entry;
    state->generation++;
    pthread_mutex_unlock(&state->mu);

    char layer_end[32];
    if (entry->has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", entry->layer_end);
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: registered worker %s:%s data_port=%u model_id=%u quant=Q%u layers=%u:%s hidden=%u ctx=%u\n",
                     entry->peer_host,
                     entry->peer_port,
                     entry->listen_port,
                     entry->model_id,
                     entry->quant_bits,
                     entry->layer_start,
                     layer_end,
                     entry->has_hidden,
                     entry->ctx_size);
    if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static int dist_worker_route_cmp(const void *a, const void *b) {
    const ds4_dist_worker_entry *ea = *(const ds4_dist_worker_entry * const *)a;
    const ds4_dist_worker_entry *eb = *(const ds4_dist_worker_entry * const *)b;
    if (ea->layer_start < eb->layer_start) return -1;
    if (ea->layer_start > eb->layer_start) return 1;
    if (ea->has_output != eb->has_output) return ea->has_output ? -1 : 1;
    if (ea->layer_end > eb->layer_end) return -1;
    if (ea->layer_end < eb->layer_end) return 1;
    return 0;
}

static bool dist_worker_route_candidate_ok(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_worker_entry *w,
        uint32_t last) {
    const bool needs_hidden = w->layer_end < last || !w->has_output;
    if (needs_hidden && !w->has_hidden) return false;
    if (w->layer_end >= last && !w->has_output && !state->local_can_output_head) return false;
    return true;
}

static bool dist_route_search_workers(
        const ds4_dist_coordinator_state *state,
        ds4_dist_worker_entry **workers,
        uint32_t n,
        uint32_t next,
        uint32_t last,
        ds4_dist_worker_entry **path,
        uint32_t *path_len,
        uint32_t *missing_layer) {
    bool saw_start = false;
    for (uint32_t i = 0; i < n; i++) {
        ds4_dist_worker_entry *w = workers[i];
        if (w->layer_start < next) continue;
        if (w->layer_start > next) break;
        saw_start = true;
        if (!dist_worker_route_candidate_ok(state, w, last)) continue;

        path[(*path_len)++] = w;
        if (w->layer_end >= last) return true;
        uint32_t child_missing = w->layer_end + 1u;
        if (dist_route_search_workers(state,
                                      workers,
                                      n,
                                      child_missing,
                                      last,
                                      path,
                                      path_len,
                                      &child_missing)) {
            return true;
        }
        if (child_missing > *missing_layer) *missing_layer = child_missing;
        (*path_len)--;
    }
    if (!saw_start && next > *missing_layer) *missing_layer = next;
    return false;
}

void dist_coordinator_report_plan(ds4_dist_coordinator_state *state) {
    if (!dist_coordinator_debug_enabled(state)) return;
    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        fprintf(stderr, "ds4: distributed coordinator: out of memory building route plan\n");
        return;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    bool complete = state->local_start == 0;
    bool has_output = state->local_end == last &&
                      (state->local_has_output || state->local_can_output_head);
    uint32_t next = state->local_end + 1u;
    if (state->local_end >= last) next = state->n_layers;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (complete && !has_output) {
        complete = dist_route_search_workers(state,
                                             workers,
                                             n,
                                             next,
                                             last,
                                             path,
                                             &path_len,
                                             &missing);
        if (complete && path_len != 0) {
            ds4_dist_worker_entry *final = path[path_len - 1u];
            has_output = final->has_output || state->local_can_output_head;
            next = state->n_layers;
        }
    }

    char plan[1024];
    size_t used = 0;
    char local_end[32];
    if (state->local_has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", state->local_end);
    used += (size_t)snprintf(plan + used, used < sizeof(plan) ? sizeof(plan) - used : 0,
                             "local %u:%s",
                             state->local_start,
                             local_end);
    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        if (used < sizeof(plan)) {
            char end[32];
            if (w->has_output) snprintf(end, sizeof(end), "output");
            else snprintf(end, sizeof(end), "%u", w->layer_end);
            used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                     " -> %s:%u Q%u %u:%s",
                                     w->peer_host,
                                     w->listen_port,
                                     w->quant_bits,
                                     w->layer_start,
                                     end);
        }
    }
    if (complete && path_len != 0 &&
        !path[path_len - 1u]->has_output && state->local_can_output_head &&
        used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    if (complete && path_len == 0 &&
        state->local_end == last && !state->local_has_output &&
        state->local_can_output_head && used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    complete = complete && has_output && next == state->n_layers;
    pthread_mutex_unlock(&state->mu);

    if (complete) {
        fprintf(stderr, "ds4: distributed coordinator: complete route ready: %s\n", plan);
    } else {
        fprintf(stderr, "ds4: distributed coordinator: route incomplete; next needed layer %u\n", missing);
    }
    free(path);
    free(workers);
}

void dist_route_plan_free(ds4_dist_route_plan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < plan->count; i++) {
        if (plan->entry[i].fd >= 0) close(plan->entry[i].fd);
    }
    free(plan->entry);
    free(plan->blob);
    memset(plan, 0, sizeof(*plan));
}

static bool dist_route_entry_matches_worker(
        const ds4_dist_route_entry *route,
        const ds4_dist_worker_entry *worker) {
    const bool route_has_output = (route->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    return route->port == worker->listen_port &&
           strcmp(route->host, worker->peer_host) == 0 &&
           route->layer_start == worker->layer_start &&
           route->layer_end == worker->layer_end &&
           route_has_output == (worker->has_output != 0);
}

void dist_coordinator_forget_route_workers(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan) {
    bool removed_any = false;
    pthread_mutex_lock(&state->mu);
    for (uint32_t i = 0; i < plan->count; i++) {
        ds4_dist_worker_entry **link = &state->workers;
        while (*link) {
            ds4_dist_worker_entry *entry = *link;
            if (!dist_route_entry_matches_worker(&plan->entry[i], entry)) {
                link = &entry->next;
                continue;
            }
            *link = entry->next;
            close(entry->fd);
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: forgot failed route worker %s:%u layers=%u:%u%s\n",
                             plan->entry[i].host,
                             plan->entry[i].port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            free(entry);
            removed_any = true;
            break;
        }
    }
    if (removed_any) state->generation++;
    pthread_mutex_unlock(&state->mu);

    if (removed_any && dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static bool dist_route_plan_append_blob(
        ds4_dist_route_plan *plan,
        const ds4_dist_route_entry *entry,
        char *err,
        size_t errlen) {
    const size_t host_len = strlen(entry->host);
    if (host_len == 0 || host_len >= NI_MAXHOST) {
        if (errlen) snprintf(err, errlen, "invalid route host");
        return false;
    }
    const uint64_t add = sizeof(ds4_dist_route_fixed) + host_len;
    if (add > UINT32_MAX || plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    uint32_t old_bytes = plan->blob_bytes;
    uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    uint8_t *p = (uint8_t *)plan->blob + old_bytes;
    ds4_dist_route_fixed fixed = {
        (uint32_t)host_len,
        entry->port,
        entry->layer_start,
        entry->layer_end,
        entry->flags,
    };
    dist_route_to_wire(&fixed);
    memcpy(p, &fixed, sizeof(fixed));
    memcpy(p + sizeof(fixed), entry->host, host_len);
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_route_plan_append_return_upstream(
        ds4_dist_route_plan *plan,
        char *err,
        size_t errlen) {
    const uint64_t add = sizeof(ds4_dist_route_return_fixed);
    if (plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    const uint32_t old_bytes = plan->blob_bytes;
    const uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    ds4_dist_route_return_fixed fixed = {
        DS4_DIST_ROUTE_RETURN_UPSTREAM,
        0,
        0,
    };
    dist_route_return_to_wire(&fixed);
    memcpy((uint8_t *)plan->blob + old_bytes, &fixed, sizeof(fixed));
    plan->blob_bytes = new_bytes;
    return true;
}

bool dist_coordinator_build_route_plan(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    memset(plan, 0, sizeof(*plan));
    if (generation) *generation = 0;

    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        if (errlen) snprintf(err, errlen, "out of memory building route");
        return false;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    if (state->local_start != 0) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "coordinator route does not start at layer 0");
        return false;
    }
    if (state->local_end == last &&
        (state->local_has_output || state->local_can_output_head)) {
        if (generation) *generation = state->generation;
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        return true;
    }

    uint32_t next = state->local_end + 1u;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (!dist_route_search_workers(state,
                                   workers,
                                   n,
                                   next,
                                   last,
                                   path,
                                   &path_len,
                                   &missing)) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "distributed route incomplete: missing layer %u", missing);
        return false;
    }

    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        ds4_dist_route_entry entry;
        memset(&entry, 0, sizeof(entry));
        entry.fd = -1;
        snprintf(entry.host, sizeof(entry.host), "%s", w->peer_host);
        entry.port = w->listen_port;
        entry.layer_start = w->layer_start;
        entry.layer_end = w->layer_end;
        entry.flags = w->has_output ? DS4_DIST_ROUTE_F_OUTPUT_LOGITS : 0u;
        if (state->use_control_for_work && plan->count == 0) {
            entry.fd = dup(w->fd);
            if (entry.fd < 0) {
                pthread_mutex_unlock(&state->mu);
                free(workers);
                free(path);
                dist_route_plan_free(plan);
                if (errlen) snprintf(err, errlen, "failed to duplicate first-hop worker connection: %s", strerror(errno));
                return false;
            }
            dist_set_socket_low_latency(entry.fd);
        }

        ds4_dist_route_entry *new_entries = realloc(plan->entry, (size_t)(plan->count + 1u) * sizeof(plan->entry[0]));
        if (!new_entries) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            if (entry.fd >= 0) close(entry.fd);
            dist_route_plan_free(plan);
            if (errlen) snprintf(err, errlen, "out of memory building route entries");
            return false;
        }
        plan->entry = new_entries;
        plan->entry[plan->count++] = entry;
        if (!dist_route_plan_append_blob(plan, &entry, err, errlen)) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            dist_route_plan_free(plan);
            return false;
        }
    }
    if (generation) *generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    free(workers);
    free(path);
    if (plan->count != 0 && !dist_route_plan_append_return_upstream(plan, err, errlen)) {
        dist_route_plan_free(plan);
        return false;
    }
    return true;
}

int dist_logits_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

bool dist_coordinator_ensure_route(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    return dist_coordinator_build_route_plan(state, plan, generation, err, errlen);
}

uint64_t dist_coordinator_generation(ds4_dist_coordinator_state *state) {
    if (!state) return 0;
    pthread_mutex_lock(&state->mu);
    uint64_t generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    return generation;
}

