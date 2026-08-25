/* =========================================================================
 * ds4_multimodal -- 多模态: the modality-adapter seam, as a self-contained
 * module.
 * =========================================================================
 *
 * DeepSeek V4 Flash is a text model; what the engine actually consumes is a
 * token stream. This module is the ONE place where "content of some modality"
 * becomes "token stream": a registry of named encoders. The text modality is
 * built in (it adapts the engine tokenizer, same callback shape as ds4_mtp);
 * image/audio/anything register an encoder at startup and become available
 * everywhere the registry is consulted -- no engine code changes, matching
 * the project's extension-is-data-not-flags discipline.
 *
 * Honest contract: an unregistered modality fails cleanly with -1
 * ("unsupported"), it is never silently dropped or faked. Encoders that
 * produce embeddings rather than tokens are future work at the graph seam;
 * THIS seam is deliberately token-level so it composes with everything that
 * exists today (prompt rendering, KV cache keys, copy-spec transcripts).
 *
 * Two output forms, one registry:
 *   - TEXT  (ds4_mm_encode_as_text): the canonical form for text-pipeline
 *     consumers -- the server splices it into message content, so rendered
 *     prompts, disk-KV prefix keys and exact-DSML replay stay byte-stable
 *     and identical uploads hit the cache.
 *   - TOKEN (ds4_mm_encode): the same content tokenized, for in-process
 *     consumers that feed the engine directly.
 * Both go through the same encoder entry; an encoder without a text form
 * (token-only) honestly returns -1 from the text call. */
#ifndef DS4_MULTIMODAL_H
#define DS4_MULTIMODAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct ds4_mm ds4_mm;

/* Encoder: turn `len` bytes of modality content into a plain-malloc'd token
 * array (*toks, *n_toks; ownership moves to the caller). Return 0, or -1 on
 * malformed content. */
typedef int (*ds4_mm_encode_fn)(void *ctx, const uint8_t *data, size_t len,
                                int **toks, int *n_toks);

/* Tokenizer callback for the built-in text modality. */
typedef void (*ds4_mm_tokenize_fn)(void *ctx, const char *text,
                                   int **toks, int *n);

/* Create a registry with the built-in "text" modality: its text form is the
 * identity, its token form runs `tokenize`. NULL tokenize => the text
 * modality is text-form-only (token encodes fail until an engine wires a
 * tokenizer; the server's text-level consumers are unaffected). */
ds4_mm *ds4_mm_create(ds4_mm_tokenize_fn tokenize, void *tok_ctx);
void ds4_mm_free(ds4_mm *mm);

/* Register (or replace) the encoder for `modality` (e.g. "image/png",
 * "audio/wav"). Returns 0, -1 on OOM/bad args. */
int ds4_mm_register(ds4_mm *mm, const char *modality,
                    ds4_mm_encode_fn encode, void *ctx);

/* Encode `len` bytes of `modality` content into tokens. Returns 0; -1 with
 * *toks = NULL when the modality has no registered encoder (unsupported) or
 * the encoder rejected the content. Longest-prefix match on the modality
 * name ("image/png" falls back to a registered "image" encoder). */
int ds4_mm_encode(const ds4_mm *mm, const char *modality,
                  const uint8_t *data, size_t len, int **toks, int *n_toks);

/* 1 if `modality` would resolve to an encoder (capability probe for API
 * layers that must reject unsupported attachments up front). */
int ds4_mm_supported(const ds4_mm *mm, const char *modality);

/* Register an EXTERNAL COMMAND encoder for `modality`: content bytes are
 * written to a temp file, `argv0 <tmpfile>` runs, its stdout is the canonical
 * text (and, for token consumers, goes through the registry's tokenizer).
 * One line of shell = one new modality -- the extension contract of this
 * module (production instance: mm-ui, the frontend-domain UI-sketch encoder
 * for the "image" family; geometry/palette/verbatim text a text model can
 * reason about). The command path is copied; the registry owns the copy. */
int ds4_mm_register_command(ds4_mm *mm, const char *modality, const char *argv0);

/* Encode `len` bytes of `modality` content to canonical TEXT (*out_text:
 * malloc'd, NUL-terminated, ownership moves to the caller), then run the
 * modality's ENRICHERS (below) over the encoder output. Returns 0; -1 with
 * *out_text = NULL when unsupported, when the encoder rejected the content,
 * or when the resolved encoder has no text form. */
int ds4_mm_encode_as_text(const ds4_mm *mm, const char *modality,
                          const uint8_t *data, size_t len, char **out_text);

/* Enricher: derives an APPENDED section from the encoder's canonical text
 * (e.g. spatial relations / CSS facts computed from a UI sketch). Returns a
 * malloc'd section (no leading newline) or NULL for "nothing to add".
 * Enrichers only append -- the encoder's own facts are never rewritten -- and
 * every enricher sees the ORIGINAL encoder text, so registration order only
 * orders sections, it never chains parsers. Both output forms run them: the
 * text form appends sections; the token form tokenizes the enriched text. */
typedef char *(*ds4_mm_enrich_fn)(void *ctx, const char *modality,
                                  const char *text);

/* Register an enricher for `modality` (same family-prefix matching as
 * encoders: an "image" enricher runs for "image/png"). Multiple enrichers
 * per modality run in registration order. Returns 0, -1 on OOM/bad args. */
int ds4_mm_register_enricher(ds4_mm *mm, const char *modality,
                             ds4_mm_enrich_fn fn, void *ctx);

/* Strict RFC 4648 base64 decode (the wire form of API image payloads).
 * Whitespace is skipped; anything else non-alphabet, missing padding, or
 * trailing garbage is rejected with -1. *out is malloc'd (NUL-terminated
 * for convenience; *out_len excludes the NUL). */
int ds4_mm_b64_decode(const char *s, size_t slen, uint8_t **out, size_t *out_len);

#endif /* DS4_MULTIMODAL_H */
