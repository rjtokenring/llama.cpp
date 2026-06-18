#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <float.h>
#include <math.h>
#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

// Mirror of enum ggml_op_pool. Kept local to avoid pulling ggml.h on the DSP.
enum htp_pool_op {
    HTP_POOL_MAX = 0,
    HTP_POOL_AVG = 1,
};

struct htp_pool_2d_context {
    const uint8_t * src_data;
    uint8_t       * dst_data;

    uint32_t        ne00;       // src W
    uint32_t        ne01;       // src H
    uint32_t        nb01;       // src row stride (bytes)
    uint32_t        nb02;       // src plane stride (bytes)

    uint32_t        ne0;        // dst W
    uint32_t        ne1;        // dst H
    uint32_t        nb1;        // dst row stride (bytes)
    uint32_t        nb2;        // dst plane stride (bytes)

    int32_t         k0, k1;
    int32_t         s0, s1;
    int32_t         p0, p1;

    uint32_t        n_planes;            // ne[2] * ne[3]
    uint32_t        planes_per_thread;
    uint32_t        op;                  // HTP_POOL_*
};

static void pool_2d_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pool_2d_context * pctx = (const struct htp_pool_2d_context *) data;

    const uint32_t start_plane = pctx->planes_per_thread * ith;
    const uint32_t end_plane   = MIN(start_plane + pctx->planes_per_thread, pctx->n_planes);

    if (start_plane >= end_plane) {
        return;
    }

    const int32_t k0 = pctx->k0;
    const int32_t k1 = pctx->k1;
    const int32_t s0 = pctx->s0;
    const int32_t s1 = pctx->s1;
    const int32_t p0 = pctx->p0;
    const int32_t p1 = pctx->p1;

    const int64_t IW = (int64_t) pctx->ne00;
    const int64_t IH = (int64_t) pctx->ne01;
    const int64_t OW = (int64_t) pctx->ne0;
    const int64_t OH = (int64_t) pctx->ne1;

    const float inv_ka = 1.0f / (float) (k0 * k1);

    for (uint32_t ip = start_plane; ip < end_plane; ip++) {
        const uint8_t * src_plane = pctx->src_data + (size_t) ip * pctx->nb02;
        uint8_t *       dst_plane = pctx->dst_data + (size_t) ip * pctx->nb2;

        for (int64_t oy = 0; oy < OH; oy++) {
            float * dst_row = (float *) (dst_plane + (size_t) oy * pctx->nb1);

            const int64_t iy0 = -p1 + oy * s1;

            for (int64_t ox = 0; ox < OW; ox++) {
                const int64_t ix0 = -p0 + ox * s0;

                float res;
                if (pctx->op == HTP_POOL_AVG) {
                    res = 0.0f;
                } else {
                    res = -FLT_MAX;
                }

                for (int32_t ky = 0; ky < k1; ky++) {
                    const int64_t iy = iy0 + ky;
                    if (iy < 0 || iy >= IH) {
                        continue;
                    }

                    const float * src_row = (const float *) (src_plane + (size_t) iy * pctx->nb01);

                    for (int32_t kx = 0; kx < k0; kx++) {
                        const int64_t ix = ix0 + kx;
                        if (ix < 0 || ix >= IW) {
                            continue;
                        }

                        const float v = src_row[ix];
                        if (pctx->op == HTP_POOL_AVG) {
                            res += v;
                        } else {
                            if (v > res) {
                                res = v;
                            }
                        }
                    }
                }

                if (pctx->op == HTP_POOL_AVG) {
                    res *= inv_ka;
                }

                dst_row[ox] = res;
            }
        }
    }
}

int op_pool_2d(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int32_t * op_params = octx->op_params;
    const uint32_t pool_op = (uint32_t) op_params[0];
    const int32_t  k0 = op_params[1];
    const int32_t  k1 = op_params[2];
    const int32_t  s0 = op_params[3];
    const int32_t  s1 = op_params[4];
    const int32_t  p0 = op_params[5];
    const int32_t  p1 = op_params[6];

    if (pool_op != HTP_POOL_AVG && pool_op != HTP_POOL_MAX) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_planes = src->ne[2] * src->ne[3];
    const uint32_t n_threads = MIN(octx->n_threads, n_planes ? n_planes : 1);
    const uint32_t planes_per_thread = (n_planes + n_threads - 1) / n_threads;

    struct htp_pool_2d_context pctx = {
        .src_data           = (const uint8_t *) src->data,
        .dst_data           = (uint8_t *)       dst->data,
        .ne00               = src->ne[0],
        .ne01               = src->ne[1],
        .nb01               = src->nb[1],
        .nb02               = src->nb[2],
        .ne0                = dst->ne[0],
        .ne1                = dst->ne[1],
        .nb1                = dst->nb[1],
        .nb2                = dst->nb[2],
        .k0                 = k0,
        .k1                 = k1,
        .s0                 = s0,
        .s1                 = s1,
        .p0                 = p0,
        .p1                 = p1,
        .n_planes           = n_planes,
        .planes_per_thread  = planes_per_thread,
        .op                 = pool_op,
    };

    uint64_t t1 = HAP_perf_get_qtimer_count();

    worker_pool_run_func(octx->ctx->worker_pool, pool_2d_thread_f32, &pctx, n_threads);

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "pool_2d-f32 %s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) k=%dx%d s=%dx%d p=%dx%d usec %u\n",
         pool_op == HTP_POOL_AVG ? "avg" : "max",
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         k0, k1, s0, s1, p0, p1,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));

    return HTP_STATUS_OK;
}
