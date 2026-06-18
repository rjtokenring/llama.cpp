#ifndef HVX_CLAMP_H
#define HVX_CLAMP_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_clamp_f32_loop_body(dst_type, src_type, vec_store)                       \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vmin = hvx_vec_splat_f32(min);                                    \
        HVX_Vector vmax = hvx_vec_splat_f32(max);                                    \
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
            HVX_Vector v = Q6_Vsf_vmax_VsfVsf(vsrc[i], vmin);                        \
            vdst[i] = Q6_Vsf_vmin_VsfVsf(v, vmax);                                   \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vsf_vmax_VsfVsf(vsrc[i], vmin);                        \
            v = Q6_Vsf_vmin_VsfVsf(v, vmax);                                         \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                       \
        }                                                                            \
    } while(0)

static inline void hvx_clamp_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float min, const float max) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_clamp_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_clamp_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float min, const float max) {
    assert((size_t) dst % 128 == 0);
    hvx_clamp_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_clamp_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float min, const float max) {
    assert((size_t) src % 128 == 0);
    hvx_clamp_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_clamp_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float min, const float max) {
    hvx_clamp_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_clamp_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float min, const float max) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_clamp_f32_aa(dst, src, n, min, max);
        } else {
            hvx_clamp_f32_au(dst, src, n, min, max);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_clamp_f32_ua(dst, src, n, min, max);
        } else {
            hvx_clamp_f32_uu(dst, src, n, min, max);
        }
    }
}

#endif // HVX_CLAMP_H
