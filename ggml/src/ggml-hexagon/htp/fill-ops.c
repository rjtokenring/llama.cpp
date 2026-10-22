#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    const struct htp_tensor * dst = octx->dst; \
                                               \
    const uint32_t ne0 = dst->ne[0];           \
    const uint32_t ne1 = dst->ne[1];           \
    const uint32_t ne2 = dst->ne[2];           \
    const uint32_t ne3 = dst->ne[3];           \
                                               \
    const uint32_t nb1 = dst->nb[1];           \
    const uint32_t nb2 = dst->nb[2];           \
    const uint32_t nb3 = dst->nb[3];           \
                                               \
    const uint32_t nr = ne1 * ne2 * ne3;

struct htp_fill_context {
    struct htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

static void fill_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
    struct htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    if (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        const uint32_t elem_start = ir0 * ne0;
        const uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } else {
        // Non-contiguous path: must respect strides
        for (uint32_t ir = ir0; ir < ir1; ++ir) {
            const uint32_t i1 = ir % ne1;
            const uint32_t i2 = (ir / ne1) % ne2;
            const uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(HIGH, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

int op_fill(struct htp_ops_context * octx) {
    fill_preamble;

    if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = nb1;
        const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    const uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    const bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (int) opt_path);

    float val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], sizeof(float));

    struct htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    switch (dst->type) {
    case HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = sizeof(float);
        break;
    case HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = sizeof(_Float16);
        break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    return HTP_STATUS_OK;
}

// ggml op_params layout for ARANGE:
//   op_params[0], op_params[2] (as float) - start and step: dst[i] = start + step * i

struct htp_arange_context {
    struct htp_ops_context * octx;
    uint32_t n;              // number of elements
    uint32_t n_per_thread;   // multiple of 32
    float    start;
    float    step;
    bool     exact;          // integer start and step: computed in i32 and converted exactly
};

static void arange_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_arange_context * actx = (const struct htp_arange_context *) data;
    struct htp_ops_context          * octx = actx->octx;

    const uint32_t i0 = actx->n_per_thread * ith;
    const uint32_t i1 = MIN(i0 + actx->n_per_thread, actx->n);
    if (i0 >= i1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i0);

    static const int32_t lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    float * dst = (float *) octx->dst->data;

    const HVX_Vector v_32   = Q6_V_vsplat_R(32);
    HVX_Vector       v_idx  = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) lane_idx, Q6_V_vsplat_R((int32_t) i0));

    if (actx->exact) {
        // start + step * i stays below 2^24 here, so the i32 values convert exactly
        const int32_t start = (int32_t) actx->start;
        const int32_t step  = (int32_t) actx->step;
        int32_t first[32] __attribute__((aligned(128)));
        for (int l = 0; l < 32; l++) {
            first[l] = start + step * (int32_t) (i0 + l);
        }
        HVX_Vector       v_val  = *(const HVX_Vector *) first;
        const HVX_Vector v_incr = Q6_V_vsplat_R(32 * step);
        for (uint32_t i = i0; i < i1; i += 32) {
            hvx_vec_store_u(dst + i, MIN(32, i1 - i) * sizeof(float), Q6_Vsf_equals_Vw(v_val));
            v_val = Q6_Vw_vadd_VwVw(v_val, v_incr);
        }
    } else {
        const HVX_Vector v_start = hvx_vec_splat_f32(actx->start);
        const HVX_Vector v_step  = hvx_vec_splat_f32(actx->step);
        for (uint32_t i = i0; i < i1; i += 32) {
            HVX_Vector v = hvx_vec_add_f32_f32(v_start, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(v_idx), v_step));
            hvx_vec_store_u(dst + i, MIN(32, i1 - i) * sizeof(float), v);
            v_idx = Q6_Vw_vadd_VwVw(v_idx, v_32);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i1);
}

int op_arange(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    if (dst->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // a single row: with several sessions the first one writes it, like FILL does
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx != 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n = dst->ne[0];
    if (n == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_arange_context actx = {
        .octx = octx,
        .n    = n,
    };
    memcpy(&actx.start, &octx->op_params[0], sizeof(float));
    memcpy(&actx.step,  &octx->op_params[2], sizeof(float));

    const float last = actx.start + actx.step * (float) (n - 1);
    actx.exact = actx.start == (float) (int32_t) actx.start && actx.step == (float) (int32_t) actx.step &&
                 fabsf(actx.start) < 16777216.0f && fabsf(last) < 16777216.0f && fabsf(actx.step) * 32.0f < 16777216.0f;

    const uint32_t n_threads = MIN(octx->n_threads, (n + 31) / 32);
    actx.n_per_thread = (((n + n_threads - 1) / n_threads) + 31) & ~31u;

    work_queue_run(octx->ctx->work_queue, arange_thread, &actx, n_threads);

    return HTP_STATUS_OK;
}
