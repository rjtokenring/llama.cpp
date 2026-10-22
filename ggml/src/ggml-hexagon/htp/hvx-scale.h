#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } else {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } else {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

// x * s, rounded to f32, for an offset to be added to it.
// Below v79 the f32 add goes through qf32, which takes a -0.0 operand for a value just below
// zero: -0.0 + 16 gives 15.999998, which a truncating cast to i32 turns into 15. A zero
// product therefore gets its sign cleared, so that x * 0 + b and -1 * 0 + b stay exact.
static inline HVX_Vector hvx_scale_offset_mul_f32(HVX_Vector x, HVX_Vector s) {
    HVX_Vector p = hvx_vec_mul_f32_f32(x, s);
#if __HVX_ARCH__ < 79
    const HVX_Vector neg_zero = Q6_V_vsplat_R(0x80000000);
    p = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(p, neg_zero), Q6_V_vzero(), p);
#endif
    return p;
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    do {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        const uint32_t elem_size = sizeof(float);                                         \
        const uint32_t epv = 128 / elem_size;                                             \
        const uint32_t nvec = n / epv;                                                    \
        const uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        for (; i < nvec; ++i) {                                                           \
            vdst[i] = hvx_vec_add_f32_f32(hvx_scale_offset_mul_f32(vsrc[i], vs), vo);     \
        }                                                                                 \
        if (nloe) {                                                                       \
            HVX_Vector p = hvx_scale_offset_mul_f32(vsrc[i], vs);                         \
            vec_store((void *) &vdst[i], nloe * elem_size, hvx_vec_add_f32_f32(p, vo));   \
        }                                                                                 \
    } while(0)

static inline void hvx_scale_offset_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    do {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        const uint32_t nvec = n / VLEN_FP16;                                                                         \
        const uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        for (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = hvx_vec_add_f32_f32(hvx_scale_offset_mul_f32(Q6_V_lo_W(p), vs), vo);                     \
            HVX_Vector r1 = hvx_vec_add_f32_f32(hvx_scale_offset_mul_f32(Q6_V_hi_W(p), vs), vo);                     \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        if (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = hvx_vec_add_f32_f32(hvx_scale_offset_mul_f32(Q6_V_lo_W(p), vs), vo);                     \
            HVX_Vector r1 = hvx_vec_add_f32_f32(hvx_scale_offset_mul_f32(Q6_V_hi_W(p), vs), vo);                     \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } while(0)

static inline void hvx_scale_offset_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f16(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
