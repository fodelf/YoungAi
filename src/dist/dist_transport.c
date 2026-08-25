/* dist_transport.c — 机械拆自 ds4_distributed.c: 激活传输(Activation Transport)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Activation Transport
 * =========================================================================
 *
 * The graph-slice APIs exchange float buffers. Distributed transport can leave
 * those buffers as 32-bit floats or pack them to 16/8 bits on the wire; workers
 * decode back to float before executing the next slice.
 */

uint32_t dist_activation_bits_or_default(uint32_t bits) {
    return bits ? bits : DS4_DIST_ACTIVATION_BITS_DEFAULT;
}

bool dist_activation_bits_valid(uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    return bits == 32u || bits == 16u || bits == 8u;
}

bool dist_activation_wire_bytes(uint32_t bits, uint64_t values, uint32_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint64_t bytes = values * (uint64_t)(bits / 8u);
    if (bytes > UINT32_MAX) return false;
    if (out) *out = (uint32_t)bytes;
    return true;
}

bool dist_activation_values_from_wire_bytes(uint32_t bits, uint32_t bytes, uint64_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint32_t bytes_per_value = bits / 8u;
    if (bytes_per_value == 0 || (bytes % bytes_per_value) != 0) return false;
    if (out) *out = bytes / bytes_per_value;
    return true;
}

bool dist_activation_wire_bytes_from_f32_bytes(uint32_t bits, uint32_t f32_bytes, uint32_t *out) {
    if ((f32_bytes % (uint32_t)sizeof(float)) != 0) return false;
    return dist_activation_wire_bytes(bits, f32_bytes / (uint32_t)sizeof(float), out);
}

static uint16_t dist_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float dist_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint8_t dist_f32_to_f8_e4m3(float f) {
    const uint8_t sign = signbit(f) ? 0x80u : 0u;
    float a = fabsf(f);
    if (a == 0.0f) return sign;
    if (!isfinite(a) || a >= 240.0f) return (uint8_t)(sign | 0x77u);

    if (a < 0.001953125f) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    int exp2 = 0;
    (void)frexpf(a, &exp2);
    int exp = exp2 - 1 + 7;
    if (exp <= 0) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    float base = ldexpf(1.0f, exp2 - 1);
    int mant = (int)floorf(((a / base) - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) {
        mant = 0;
        exp++;
    }
    if (exp >= 15) return (uint8_t)(sign | 0x77u);
    return (uint8_t)(sign | (uint8_t)(exp << 3) | (uint8_t)mant);
}

static float dist_f8_e4m3_to_f32(uint8_t h) {
    const float sign = (h & 0x80u) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 3) & 0x0fu;
    const uint32_t mant = h & 0x07u;
    if (exp == 0) {
        return sign * (float)mant * 0.001953125f;
    }
    if (exp >= 15u) {
        return sign * 240.0f;
    }
    return sign * ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
}

int dist_write_activation_payload(
        int fd,
        const float *src,
        uint64_t values,
        uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) return -1;
    if (values == 0) return 0;
    if (!src) return -1;
    if (bits == 32u) {
        uint32_t bytes = 0;
        if (!dist_activation_wire_bytes(bits, values, &bytes)) return -1;
        return dist_write_full(fd, src, bytes);
    }

    const uint64_t max_values = 1024u * 1024u;
    uint64_t cap = values < max_values ? values : max_values;
    void *buf = malloc((size_t)cap * (size_t)(bits / 8u));
    if (!buf) return -1;
    uint64_t done = 0;
    int rc = 0;
    while (done < values) {
        uint64_t n = values - done;
        if (n > cap) n = cap;
        if (bits == 16u) {
            uint16_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f16(src[done + i]);
        } else {
            uint8_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f8_e4m3(src[done + i]);
        }
        if (dist_write_full(fd, buf, (size_t)n * (size_t)(bits / 8u)) != 0) {
            rc = -1;
            break;
        }
        done += n;
    }
    free(buf);
    return rc;
}

int dist_decode_activation_payload(
        const void *wire,
        uint32_t bits,
        uint32_t wire_bytes,
        float **out,
        uint32_t *out_f32_bytes,
        bool *out_uses_wire,
        char *err,
        size_t errlen) {
    if (out) *out = NULL;
    if (out_f32_bytes) *out_f32_bytes = 0;
    if (out_uses_wire) *out_uses_wire = false;
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation width: %u bits", bits);
        return 1;
    }
    if (wire_bytes != 0 && !wire) {
        if (errlen) snprintf(err, errlen, "missing distributed activation payload");
        return 1;
    }

    uint64_t values = 0;
    if (!dist_activation_values_from_wire_bytes(bits, wire_bytes, &values)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation payload size");
        return 1;
    }
    const uint64_t f32_bytes64 = values * sizeof(float);
    if (f32_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed activation payload is too large");
        return 1;
    }
    const uint32_t f32_bytes = (uint32_t)f32_bytes64;
    if (bits == 32u) {
        if (out) *out = (float *)(void *)wire;
        if (out_f32_bytes) *out_f32_bytes = f32_bytes;
        if (out_uses_wire) *out_uses_wire = true;
        return 0;
    }

    float *dst = f32_bytes ? malloc(f32_bytes) : NULL;
    if (f32_bytes && !dst) {
        if (errlen) snprintf(err, errlen, "out of memory decoding distributed activations");
        return 1;
    }
    if (bits == 16u) {
        const uint16_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f16_to_f32(src[i]);
    } else {
        const uint8_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f8_e4m3_to_f32(src[i]);
    }
    if (out) *out = dst;
    if (out_f32_bytes) *out_f32_bytes = f32_bytes;
    return 0;
}

