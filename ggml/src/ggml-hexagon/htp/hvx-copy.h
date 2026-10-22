#ifndef HVX_COPY_H
#define HVX_COPY_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    do {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        for (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        if (nloe) {                                              \
            vec_store((void *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } while(0)

static inline void hvx_splat_a(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

static inline void hvx_splat_u(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

static inline void hvx_splat_f32_a(void * restrict dst, float v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f32_u(void * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f16_a(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_f16_u(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_u16_a(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u16_u(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u8_a(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

static inline void hvx_splat_u8_u(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    do {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        const uint32_t epv  = 128 / elem_size;                       \
        const uint32_t nvec = n / epv;                               \
        const uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        for (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        if (nloe) {                                                  \
            vec_store((void *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } while(0)

// Generic copy routines
static inline void hvx_copy_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_copy_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_copy_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
static inline void hvx_copy_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
static inline void hvx_copy_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
static inline void hvx_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(float));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
        const uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        if (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } while(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
static inline void hvx_copy_f16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f16_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f16_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        if (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            if (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            if (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * sizeof(float), vd);                \
            }                                                                       \
        }                                                                           \
    } while(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
static inline void hvx_copy_f32_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f32_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f32_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

// Truncating f32 -> i32 like the CPU cast, except that a value within 4 ulps of the next integer
// away from zero becomes that integer. HVX float math can land an ulp or two short of an exact
// integer result (e.g. qf32 adds below v79: 2048 - 37 -> 2010.9998), which truncation would turn
// into the integer below. The distance is measured on the f32 bits, so the check itself is exact.
static inline HVX_Vector hvx_vec_f32_to_i32(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);

    HVX_Vector ax  = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff)); // |x|
    HVX_Vector at  = Q6_Vw_equals_Vsf(ax);                         // trunc(|x|)
    HVX_Vector at1 = Q6_Vw_vadd_VwVw(at, one);

    // |x| is not an integer (all f32 at or above 2^23 are) and lies within 4 ulps of trunc(|x|) + 1
    HVX_VectorPred frac = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(ax, Q6_Vsf_equals_Vw(at)),
                                      Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(0x4b000000), ax));
    HVX_Vector ulps = Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vw(at1), ax);
    HVX_VectorPred snap = Q6_Q_and_QQ(frac, Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(5), ulps));

    HVX_Vector am = Q6_V_vmux_QVV(snap, at1, at);
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(zero, x), Q6_Vw_vsub_VwVw(zero, am), am);
}

#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    do {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        const uint32_t elem_size = sizeof(int32_t);               \
        const uint32_t epv  = 128 / elem_size;                    \
        const uint32_t nvec = n / epv;                            \
        const uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        for (; i < nvec; i++) {                                   \
            vdst[i] = hvx_vec_f32_to_i32(vsrc[i]);                \
        }                                                         \
        if (nloe) {                                               \
            HVX_Vector v = hvx_vec_f32_to_i32(vsrc[i]);           \
            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } while(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
static inline void hvx_copy_i32_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
static inline void hvx_copy_i32_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    do {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        const uint32_t elem_size = sizeof(float);                 \
        const uint32_t epv  = 128 / elem_size;                    \
        const uint32_t nvec = n / epv;                            \
        const uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        for (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        if (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } while(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
static inline void hvx_copy_f32_i32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_i32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#endif // HVX_COPY_H
