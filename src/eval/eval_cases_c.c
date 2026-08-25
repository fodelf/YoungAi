/* ds4-eval embedded question table, part C, plus the split-table accessor
 * (rows moved verbatim from ds4_eval.c; provenance in eval_internal.h). */

#include "eval_internal.h"

const eval_case eval_cases_c[] = {
    {
        .source = "SuperGPQA",
        .id = "591a77df21324272914be82ac6583399",
        .domain = "Science",
        .title = "In $\\Delta G=-177$ K cal for $(1)$ $2",
        .question = "In $\\Delta G=-177$ K cal for\n$(1)$ $2 Fe(s)+\\dfrac{3}{2}O_2(g)\\rightarrow Fe_2O_3(s)$ and $\\Delta G=-19$K cal for\n$(2)$ $4 Fe_2O_3(s)+Fe(s)\\rightarrow 3Fe_3O_4(s)$\nWhat is the Gibbs free energy of formation of $Fe_3O_4$?",
        .choice[0] = "$$-243.3\\dfrac{kcal}{mol}$$",
        .choice[1] = "$$+229.6\\dfrac{kcal}{mol}$$",
        .choice[2] = "$$-230.6\\dfrac{kcal}{mol}$$",
        .choice[3] = "$$-242.3\\dfrac{kcal}{mol}$$",
        .choice[4] = "$$-228.6\\dfrac{kcal}{mol}$$",
        .choice[5] = "$$-727\\dfrac{kcal}{mol}$$",
        .choice[6] = "$$-229.6\\dfrac{kcal}{mol}$$",
        .choice[7] = "$$-245.3\\dfrac{kcal}{mol}$$",
        .choice[8] = "$$-244.3\\dfrac{kcal}{mol}$$",
        .answer = "F",
    },
    {
        .source = "AIME2025",
        .id = "aime2025-15",
        .domain = "Number Theory",
        .title = "Let $N$ denote the numbers of ordered triples of positive",
        .question = "Let $N$ denote the numbers of ordered triples of positive integers $(a, b, c)$ such that $a, b, c \\le 3^6$ and $a^3 + b^3 + c^3$ is a multiple of $3^7$. Find the remainder when $N$ is divided by $1000$.",
        .answer = "735",
    },
    {
        .source = "GPQA Diamond",
        .id = "recMicVBcqy1xM1jq",
        .domain = "Physics",
        .title = "Consider an oscillating charge distribution, which is spheroid",
        .question = "Consider an oscillating charge distribution, which is spheroid in shape (with the symmetry axis along z-axis). Because it oscillates, it would radiate at a wavelength \\lambda. In the radiation zone, the radiated power per unit solid angle will be a function f of \\lambda and the angle \\theta (with z-axis), along which the power is measured. If the maximum power thus radiated is A, the fraction of A that will be radiated at an angle \\theta = 30^0 and a possible form of f are, respectively,",
        .choice[0] = "1/4, \\lambda^(-4)",
        .choice[1] = "3/4, \\lambda^(-6)",
        .choice[2] = "1/2, \\lambda^(-4)",
        .choice[3] = "1/4, \\lambda^(-3)",
        .answer = "B",
    },
    {
        .source = "SuperGPQA",
        .id = "e780f37a5baa4fe094cd9c157486664d",
        .domain = "Sociology",
        .title = "In 1814, 15 British individuals settled on T Island in the",
        .question = "In 1814, 15 British individuals settled on T Island in the middle of the Atlantic Ocean, one of whom was a carrier of the recessive gene for retinitis pigmentosa. By 1960, among the 240 descendants living on this island, there were 4 patients and at least 9 carriers of the recessive gene, a proportion far higher than that in the ancestral British population. What factors caused the change in gene frequency in this population?",
        .choice[0] = "Recessive Selection",
        .choice[1] = "Natural Migration",
        .choice[2] = "Natural Selection",
        .choice[3] = "Gene Mutations",
        .choice[4] = "Migration",
        .choice[5] = "Population Bottleneck",
        .choice[6] = "Genetic Flow",
        .choice[7] = "Genetic Drift",
        .choice[8] = "Artificial Mutation",
        .choice[9] = "Founder Effect",
        .answer = "H",
    },
    {
        .source = "AIME2025",
        .id = "aime2025-30",
        .domain = "Algebra",
        .title = "There are exactly three positive real numbers $k$ such that the",
        .question = "There are exactly three positive real numbers $k$ such that the function\n$$f(x) = \\frac{(x - 18)(x - 72)(x - 98)(x - k)}{x}$$\ndefined over the positive real numbers achieves its minimum value at exactly two positive real numbers $x$. Find the sum of these three values of $k$.",
        .answer = "240",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-076",
        .domain = "FreeBSD / librpcsec_gss",
        .title = "rpcsec_gss_validate() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe function validates a per-request authentication blob before passing it to the GSS layer.\n\nCode:\n```c\n 1: #define AUTH_STACK_MAX 256\n 2: \n 3: static bool check_rpcsec_packet(const uint8_t *wire, size_t wire_left)\n 4: {\n 5:     if (wire_left < 4)\n 6:         return false;\n 7: \n 8:     uint32_t body_len = get_be32(wire);      // attacker-controlled length field\n 9:     const uint8_t *body = wire + 4;\n10:     uint8_t scratch[AUTH_STACK_MAX];\n11: \n12:     /* XDR opaque values are padded to 4-byte alignment on the wire. */\n13:     uint32_t pad = (4 - (body_len & 3)) & 3;\n14:     uint32_t encoded_len = body_len + pad;\n15: \n16:     /* This check only proves the source bytes exist in the packet. */\n17:     if (encoded_len > wire_left - 4)\n18:         return false;\n19: \n20:     memcpy(scratch, body, encoded_len);\n21:     return gss_verify_mic(scratch, body_len);\n22: }\n```",
        .answer = "17-20",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-077",
        .domain = "GNU Inetutils telnetd",
        .title = "add_slc() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThis helper is called once for each Special Line Character option received in a TELNET LINEMODE negotiation.\n\nCode:\n```c\n 1: #define SLCBUF_SIZE 108\n 2: #define SLC_NCHARS 18\n 3: \n 4: struct slc_state {\n 5:     unsigned char buf[SLCBUF_SIZE];\n 6:     size_t used;\n 7: };\n 8: \n 9: static void add_slc(struct slc_state *s,\n10:                     unsigned char func,\n11:                     unsigned char flags,\n12:                     unsigned char value)\n13: {\n14:     if (func <= SLC_NCHARS)\n15:         return;                    // ignore ordinary entries here\n16: \n17:     /* Store one SLC triplet for later reply construction. */\n18:     s->buf[s->used++] = func;\n19:     s->buf[s->used++] = flags;\n20:     s->buf[s->used++] = value;\n21: }\n```",
        .answer = "18-20",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-078",
        .domain = "Botan",
        .title = "sm2_decrypt_and_check_c3() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe function checks the SM2 C3 hash in a decoded ciphertext. Assume parse_sm2_ciphertext() may return a c3 field of any length.\n\nCode:\n```c\n 1: #define SM3_DIGEST_LEN 32\n 2: \n 3: static bool check_sm2_c3(const uint8_t *msg, size_t msg_len,\n 4:                          const struct sm2_ctext *ct)\n 5: {\n 6:     uint8_t expected[SM3_DIGEST_LEN];\n 7: \n 8:     sm3_c3_hash(expected, ct->x2, ct->x2_len, msg, msg_len, ct->y2, ct->y2_len);\n 9: \n10:     /* Constant-time comparison: returns 0 only if all bytes match. */\n11:     if (ct_memcmp(expected, ct->c3.data, SM3_DIGEST_LEN) != 0)\n12:         return false;\n13: \n14:     return true;\n15: }\n```",
        .answer = "11",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-079",
        .domain = "Botan",
        .title = "tls13_server_handle_record() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe server is configured to require TLS 1.3 client authentication. Inspect the reduced record handler.\n\nCode:\n```c\n 1: enum State { NeedClientCert, NeedCertVerify, NeedFinished, Established };\n 2: \n 3: static void handle_tls13_record(struct Conn *c, const Record& r)\n 4: {\n 5:     if (r.type == Alert)\n 6:         return close_conn(c);\n 7: \n 8:     if (r.type == Handshake) {\n 9:         if (r.msg == Certificate && c->state == NeedClientCert)\n10:             c->state = NeedCertVerify;\n11:         else if (r.msg == CertificateVerify && c->state == NeedCertVerify)\n12:             c->state = NeedFinished;\n13:         else if (r.msg == Finished && c->state == NeedFinished)\n14:             c->state = Established;\n15:         return;\n16:     }\n17: \n18:     if (r.type == ApplicationData) {\n19:         decrypt_and_dispatch_application_data(c, r);\n20:         return;\n21:     }\n22: }\n```",
        .answer = "18-19",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-080",
        .domain = "Botan",
        .title = "validate_path_step() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe validator is walking a certificate chain. trust_store.contains_subject(x) returns true if any configured trust anchor has subject DN x.\n\nCode:\n```c\n 1: static ValidationResult validate_path_step(const Certificate& cert,\n 2:                                            const TrustStore& trust_store)\n 3: {\n 4:     /* Optimization: stop once we have reached a known trust anchor subject. */\n 5:     if (trust_store.contains_subject(cert.subject_dn()))\n 6:         return ValidationResult::Valid;\n 7: \n 8:     Certificate issuer = find_issuer_for(cert);\n 9:     if (!cert.verify_signature_with(issuer.public_key()))\n10:         return ValidationResult::BadSignature;\n11: \n12:     return validate_path_step(issuer, trust_store);\n13: }\n```",
        .answer = "5-6",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-081",
        .domain = "uds-c",
        .title = "send_diagnostic_request() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe diagnostic request builder has small protocol-defined fields.\n\nCode:\n```c\n 1: #define MAX_DIAG_PAYLOAD       6\n 2: #define MAX_REQUEST_PAYLOAD    7\n 3: \n 4: static int send_diagnostic_request(uint8_t sid,\n 5:                                    const uint8_t *pid, size_t pid_len,\n 6:                                    const uint8_t *payload, size_t payload_len)\n 7: {\n 8:     uint8_t req[MAX_DIAG_PAYLOAD];\n 9: \n10:     if (pid_len > 2 || payload_len > MAX_REQUEST_PAYLOAD)\n11:         return -1;\n12: \n13:     req[0] = sid;\n14:     memcpy(req + 1, pid, pid_len);\n15:     memcpy(req + 1 + pid_len, payload, payload_len);\n16: \n17:     return transport_send(req, 1 + pid_len + payload_len);\n18: }\n```",
        .answer = "10-15",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-082",
        .domain = "Firebird",
        .title = "xdr_datum() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe decoder writes a cstring value into a slice selected by a descriptor. xdr_bytes_available() is accurate.\n\nCode:\n```c\n 1: static bool xdr_datum(struct Xdr *xdr, const struct SliceDesc *desc, char *base)\n 2: {\n 3:     uint32_t n = xdr_get_u32(xdr);       // declared cstring length\n 4:     char *dst = base + desc->offset;\n 5: \n 6:     if (n > xdr_bytes_available(xdr))\n 7:         return false;\n 8: \n 9:     xdr_read_bytes(xdr, dst, n);\n10:     dst[n] = '\\0';\n11:     return true;\n12: }\n```",
        .answer = "9-10",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-083",
        .domain = "Firebird",
        .title = "decode_specific_data_segments() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe authentication decoder stores numbered CNCT_specific_data segments. Inputs are not guaranteed to arrive in ascending segment order.\n\nCode:\n```c\n 1: static bool read_segments(struct ConnAuth *a, struct Packet *p)\n 2: {\n 3:     int last = -1;\n 4: \n 5:     while (packet_has_segment(p)) {\n 6:         int segno = packet_get_segment_number(p);\n 7:         size_t len = packet_get_segment_length(p);\n 8: \n 9:         int missing = segno - last - 1;\n10:         a->segments.grow(a->segments.size() + missing);\n11:         a->segments[segno] = packet_read_bytes(p, len);\n12: \n13:         last = segno;\n14:     }\n15:     return true;\n16: }\n```",
        .answer = "9-11",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-084",
        .domain = "PHP",
        .title = "php_url_decode() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nOn this target, plain char is signed, and the ctype implementation indexes a table for values other than EOF.\n\nCode:\n```c\n 1: static size_t decode_component(char *s, size_t len)\n 2: {\n 3:     size_t r = 0, w = 0;\n 4: \n 5:     while (r < len) {\n 6:         if (s[r] == '%' && r + 2 < len &&\n 7:             isxdigit(s[r + 1]) && isxdigit(s[r + 2])) {\n 8:             s[w++] = hexpair(s[r + 1], s[r + 2]);\n 9:             r += 3;\n10:         } else {\n11:             s[w++] = s[r++];\n12:         }\n13:     }\n14:     return w;\n15: }\n```",
        .answer = "6-7",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-085",
        .domain = "PHP PDO Firebird",
        .title = "pdo_firebird_quote_token() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe database driver receives a language string token as (ptr,len), where len may include embedded NUL bytes. The function reconstructs a quoted SQL token.\n\nCode:\n```c\n1: static void append_quoted_token(char *sql, size_t sql_cap,\n2:                                 const char *tok, size_t tok_len)\n3: {\n4:     strlcat(sql, \"'\", sql_cap);\n5:     strncat(sql, tok, tok_len);\n6:     strlcat(sql, \"'\", sql_cap);\n7: }\n```",
        .answer = "5",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-086",
        .domain = "libexpat",
        .title = "doContent() tag-buffer reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe parser grows a reusable tag buffer before copying the current element name and closing delimiters. Sizes are attacker-influenced through XML input.\n\nCode:\n```c\n 1: static bool ensure_tag_buffer(struct Parser *p, size_t prefix_len, size_t name_len)\n 2: {\n 3:     unsigned int need = (unsigned int)(prefix_len + name_len + 3);\n 4: \n 5:     if (need > p->tag_cap) {\n 6:         char *q = realloc(p->tag_buf, need);\n 7:         if (q == NULL)\n 8:             return false;\n 9:         p->tag_buf = q;\n10:         p->tag_cap = need;\n11:     }\n12: \n13:     memcpy(p->tag_buf, p->prefix, prefix_len);\n14:     memcpy(p->tag_buf + prefix_len, p->name, name_len);\n15:     memcpy(p->tag_buf + prefix_len + name_len, \"/>\\0\", 3);\n16:     return true;\n17: }\n```",
        .answer = "3,13-15",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-087",
        .domain = "Mbed TLS",
        .title = "x509_inet_pton_ipv6() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThis helper parses an IPv6 address used in X.509 name constraints. It records where :: compression occurred, then shifts already parsed bytes.\n\nCode:\n```c\n 1: static int parse_ipv6_reduced(const char *src, uint8_t dst[16])\n 2: {\n 3:     uint8_t tmp[16];\n 4:     uint8_t *tp = tmp;\n 5:     uint8_t *colonp = NULL;\n 6: \n 7:     if (src[0] == ':' && src[1] == ':') {\n 8:         colonp = --tp;          // remember compression point at the beginning\n 9:         src += 2;\n10:     }\n11: \n12:     while (*src != '\\0') {\n13:         unsigned v = parse_hextet(&src);\n14:         *tp++ = (uint8_t)(v >> 8);\n15:         *tp++ = (uint8_t)v;\n16:         if (*src == ':') src++;\n17:     }\n18: \n19:     if (colonp != NULL) {\n20:         size_t n = (size_t)(tp - colonp);\n21:         memmove(colonp + (16 - (tp - tmp)), colonp, n);\n22:         memset(colonp, 0, 16 - (tp - tmp));\n23:     }\n24: \n25:     memcpy(dst, tmp, 16);\n26:     return 0;\n27: }\n```",
        .answer = "8,20-22",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-088",
        .domain = "Mbed TLS",
        .title = "mbedtls_dhm_export_public() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe function exports a finite-field Diffie-Hellman public value into a fixed-width buffer for the group. The public MPI may be supplied by an application.\n\nCode:\n```c\n 1: static int export_ffdh_public(const struct DhmContext *ctx,\n 2:                               uint8_t *out, size_t out_len)\n 3: {\n 4:     size_t plen = mpi_size(&ctx->P);      // byte length of the group prime\n 5:     size_t ylen = mpi_size(&ctx->Y);      // byte length of public value\n 6: \n 7:     if (out_len < plen)\n 8:         return -1;\n 9: \n10:     memset(out, 0, plen);\n11:     return mpi_write_binary(&ctx->Y, out + (plen - ylen), ylen);\n12: }\n```",
        .answer = "11",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-089",
        .domain = "Mbed TLS",
        .title = "mbedtls_ccm_finish() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe public multipart CCM API lets the caller request a tag length.\n\nCode:\n```c\n 1: static int ccm_finish_reduced(struct CcmCtx *ctx, uint8_t *tag, size_t tag_len)\n 2: {\n 3:     uint8_t full_tag[16];\n 4: \n 5:     ccm_compute_full_tag(ctx, full_tag);\n 6: \n 7:     if (tag == NULL)\n 8:         return -1;\n 9: \n10:     memcpy(tag, full_tag, tag_len);\n11:     return 0;\n12: }\n```",
        .answer = "10",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-090",
        .domain = "Linux kernel",
        .title = "handle_one_ule_extension() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe extension type is the low byte of a 16-bit protocol field. The mandatory-extension table has handlers for values 0 through 254.\n\nCode:\n```c\n 1: typedef int (*ule_handler)(struct skb *, const uint8_t *, size_t);\n 2: static ule_handler mandatory_handlers[255];\n 3: \n 4: static int handle_ule_ext(struct skb *skb, uint16_t type,\n 5:                           const uint8_t *p, size_t len)\n 6: {\n 7:     uint8_t htype = (uint8_t)(type & 0x00ff);\n 8: \n 9:     if (htype < 0x80)\n10:         return handle_optional_ext(skb, htype, p, len);\n11: \n12:     if (mandatory_handlers[htype] != NULL)\n13:         return mandatory_handlers[htype](skb, p, len);\n14: \n15:     return -EINVAL;\n16: }\n```",
        .answer = "12-13",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-091",
        .domain = "Linux kernel USB gadget storage",
        .title = "check_command_size() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe SCSI command expresses transfer length in logical blocks. blkbits is the log2 block size.\n\nCode:\n```c\n 1: static int accept_transfer(struct lun *lun, uint32_t blocks, unsigned blkbits)\n 2: {\n 3:     uint32_t bytes = blocks << blkbits;\n 4: \n 5:     if (bytes > lun->backing_size)\n 6:         return -EINVAL;\n 7: \n 8:     lun->data_size_from_cmnd = bytes;\n 9:     return 0;\n10: }\n```",
        .answer = "3",
    },
    {
        .source = "COMPSEC",
        .id = "compsec-092",
        .domain = "Linux kernel AppArmor",
        .title = "verify_dfa() reduction",
        .question = "Analyze this reduced C/C++ function. It may or may not contain a vulnerability.\nReply with the single best line number where the primary bug is introduced. If the bug is only clear from a small group of adjacent lines, reply with the smallest comma-separated set of exact line numbers.\nReturn 0 if the function is safe under the stated assumptions.\n\nThe verifier checks a serialized DFA. Some states are encoded as differences from another state default table.\n\nCode:\n```c\n 1: static int verify_dfa_reduced(const struct dfa *d)\n 2: {\n 3:     for (uint32_t state = 0; state < d->state_count; state++) {\n 4:         uint32_t j = state;\n 5: \n 6:         if (d->default_table[j] >= d->state_count)\n 7:             return -EINVAL;\n 8: \n 9:         while (d->is_differential[j]) {\n10:             uint32_t parent = d->default_table[j];\n11:             j = parent;                    // parent not rechecked here\n12:         }\n13: \n14:         validate_accept_table(d->accept[j]);\n15:     }\n16:     return 0;\n17: }\n```",
        .answer = "10-14",
    },
};

const size_t eval_cases_c_count = sizeof(eval_cases_c) / sizeof(eval_cases_c[0]);

/* The table is split across three files only because a C array literal
 * cannot span files.  The accessor preserves the exact original order. */
const eval_case *eval_case_at(size_t i) {
    if (i < eval_cases_a_count) return &eval_cases_a[i];
    i -= eval_cases_a_count;
    if (i < eval_cases_b_count) return &eval_cases_b[i];
    return &eval_cases_c[i - eval_cases_b_count];
}

size_t eval_case_count(void) {
    return eval_cases_a_count + eval_cases_b_count + eval_cases_c_count;
}
