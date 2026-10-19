#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "upscale-ops.h"

struct htp_upscale_context {
    struct htp_ops_context * octx;

    uint32_t nrows;             // dst rows (i1 + ne1 * i3)
    uint32_t nrows_per_thread;

    uint32_t c_start;           // channel range of this device
    uint32_t nc;

    uint32_t pix_stride;        // VTCM bytes per src pixel
    uint32_t row_size;          // VTCM bytes per src row
    uint32_t tile_size;

    float sf0;
    float sf1;
    float pixel_offset;
};

// Source taps of dst index i, same arithmetic as ggml_compute_forward_upscale_f32
static inline void upscale_taps(uint32_t i, float sf, float pixel_offset, uint32_t ne_src,
                                uint32_t * t0, uint32_t * t1, float * d) {
    const float x = ((float) i + pixel_offset) / sf - pixel_offset;

    int32_t x0 = (int32_t) floorf(x);
    int32_t x1 = x0 + 1;

    x0 = MAX(0, MIN(x0, (int32_t) ne_src - 1));
    x1 = MAX(0, MIN(x1, (int32_t) ne_src - 1));

    float dx = x - (float) x0;
    dx = MAX(0.0f, MIN(dx, 1.0f));

    *t0 = (uint32_t) x0;
    *t1 = (uint32_t) x1;
    *d  = dx;
}

static inline void upscale_dma_push(dma_queue * q, dma_data ddata, uint32_t dst_stride, uint32_t src_stride,
                                    uint32_t bytes, uint32_t nrows) {
    if (!dma_queue_push(q, ddata, dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_push(q, ddata, dst_stride, src_stride, bytes, nrows);
    }
}

static void upscale_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_upscale_context * uctx = (struct htp_upscale_context *) data;
    struct htp_ops_context * octx = uctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne0  = dst->ne[0];
    const uint32_t ne1  = dst->ne[1];

    const uint32_t row_start = ith * uctx->nrows_per_thread;
    const uint32_t row_end   = MIN(row_start + uctx->nrows_per_thread, uctx->nrows);
    if (row_start >= row_end) {
        return;
    }

    const uint32_t c_start    = uctx->c_start;
    const uint32_t nc         = uctx->nc;
    const uint32_t pix_stride = uctx->pix_stride;

    dma_queue * q = octx->ctx->dma[ith];

    uint8_t * vtcm    = octx->ctx->vtcm_base + ith * (2 * uctx->row_size + uctx->tile_size);
    uint8_t * slot[2] = { vtcm, vtcm + uctx->row_size };
    uint8_t * tile    = vtcm + 2 * uctx->row_size;
    int32_t   slot_row[2] = { -1, -1 }; // src row (i01 + ne01 * i03) resident in each slot

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    HVX_Vector m[HTP_UPSCALE_TILE_NPIX];
    for (uint32_t j = 0; j < HTP_UPSCALE_TILE_NPIX; j++) {
        m[j] = Q6_V_vzero();
    }

    uint32_t x0[HTP_UPSCALE_TILE_NPIX];
    uint32_t x1[HTP_UPSCALE_TILE_NPIX];
    float    w00[HTP_UPSCALE_TILE_NPIX];
    float    w10[HTP_UPSCALE_TILE_NPIX];
    float    w01[HTP_UPSCALE_TILE_NPIX];
    float    w11[HTP_UPSCALE_TILE_NPIX];

    for (uint32_t row = row_start; row < row_end; row++) {
        const uint32_t i3 = row / ne1;
        const uint32_t i1 = row - i3 * ne1;

        uint32_t y[2];
        float    dy;
        upscale_taps(i1, uctx->sf1, uctx->pixel_offset, ne01, &y[0], &y[1], &dy);

        // bring src rows y0 and y1 into the two VTCM slots, keeping the ones already resident
        const int32_t   need[2] = { (int32_t) (i3 * ne01 + y[0]), (int32_t) (i3 * ne01 + y[1]) };
        const uint8_t * src_row[2];

        dma_queue_flush(q);
        for (uint32_t k = 0; k < 2; k++) {
            int s = (slot_row[0] == need[k]) ? 0 : (slot_row[1] == need[k]) ? 1 : -1;
            if (s < 0) {
                s = (slot_row[0] == need[k ^ 1]) ? 1 : 0;
                const dma_addr_t src_addr = src0->data + (uint64_t) i3 * src0->nb[3] + (uint64_t) y[k] * src0->nb[1] + (uint64_t) c_start * src0->nb[2];
                upscale_dma_push(q, dma_make_data(slot[s], src_addr), pix_stride, src0->nb[0], nc * sizeof(float), ne00);
                slot_row[s] = need[k];
            }
            src_row[k] = slot[s];
        }
        dma_queue_flush(q);

        const float wy0 = 1.0f - dy;
        const float wy1 = dy;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row);

        for (uint32_t p0 = 0; p0 < ne0; p0 += HTP_UPSCALE_TILE_NPIX) {
            const uint32_t np = MIN(HTP_UPSCALE_TILE_NPIX, ne0 - p0);

            for (uint32_t j = 0; j < np; j++) {
                float dx;
                upscale_taps(p0 + j, uctx->sf0, uctx->pixel_offset, ne00, &x0[j], &x1[j], &dx);
                w00[j] = (1.0f - dx) * wy0;
                w10[j] = dx * wy0;
                w01[j] = (1.0f - dx) * wy1;
                w11[j] = dx * wy1;
            }

            dma_queue_flush(q); // previous tile write-out must finish before the tile is reused

            for (uint32_t c = 0; c < nc; c += 32) {
                for (uint32_t j = 0; j < np; j++) {
                    const HVX_Vector a = hvx_vmem(src_row[0] + x0[j] * pix_stride + c * sizeof(float));
                    const HVX_Vector b = hvx_vmem(src_row[0] + x1[j] * pix_stride + c * sizeof(float));
                    const HVX_Vector e = hvx_vmem(src_row[1] + x0[j] * pix_stride + c * sizeof(float));
                    const HVX_Vector f = hvx_vmem(src_row[1] + x1[j] * pix_stride + c * sizeof(float));

                    HVX_Vector v = hvx_vec_mul_f32_f32(a, hvx_vec_splat_f32(w00[j]));
                    v = hvx_vec_add_f32_f32(v, hvx_vec_mul_f32_f32(b, hvx_vec_splat_f32(w10[j])));
                    v = hvx_vec_add_f32_f32(v, hvx_vec_mul_f32_f32(e, hvx_vec_splat_f32(w01[j])));
                    v = hvx_vec_add_f32_f32(v, hvx_vec_mul_f32_f32(f, hvx_vec_splat_f32(w11[j])));
                    m[j] = v;
                }
                hvx_transpose_32x32_w_shuff(tile + c * VLEN, VLEN, m, MIN(32, nc - c), np);
            }

            const dma_addr_t dst_addr = dst->data + (uint64_t) i3 * dst->nb[3] + (uint64_t) i1 * dst->nb[1] + (uint64_t) c_start * dst->nb[2] + (uint64_t) p0 * sizeof(float);
            upscale_dma_push(q, dma_make_data(dst_addr, tile), dst->nb[2], VLEN, np * sizeof(float), nc);
        }

        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row);
    }

    dma_queue_flush(q);

    FARF(HIGH, "upscale %d/%d: (%ux%u -> %ux%u) rows %u:%u channels %u:%u\n",
         ith, nth, ne00, ne01, ne0, ne1, row_start, row_end, c_start, c_start + nc);
}

int op_upscale(struct htp_ops_context * octx) {
    const struct htp_upscale_kernel_params * kparams = (const struct htp_upscale_kernel_params *) octx->kernel_params;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t mode = (uint32_t) octx->op_params[0];

    if (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "upscale: unsupported type %u -> %u\n", src0->type, dst->type);
        return HTP_STATUS_NO_SUPPORT;
    }

    if ((mode & 0xFF) != GGML_SCALE_MODE_BILINEAR || (mode & GGML_SCALE_FLAG_ANTIALIAS)) {
        FARF(ERROR, "upscale: unsupported mode 0x%x\n", mode);
        return HTP_STATUS_NO_SUPPORT;
    }

    if (src0->nb[2] != sizeof(float) || !htp_tensor_is_contiguous(dst, sizeof(float)) ||
        src0->ne[2] != dst->ne[2] || src0->ne[3] != dst->ne[3]) {
        FARF(ERROR, "upscale: unsupported layout\n");
        return HTP_STATUS_NO_SUPPORT;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne0  = dst->ne[0];
    const uint32_t ne1  = dst->ne[1];
    const uint32_t ne2  = dst->ne[2];
    const uint32_t ne3  = dst->ne[3];

    float sf0 = (float) ne0 / ne00;
    float sf1 = (float) ne1 / ne01;
    float pixel_offset = 0.5f;

    if (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float) (ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float) (ne1 - 1) / (ne01 - 1) : sf1;
    }

    // split channel planes between devices in groups of 32: 32 planes are a multiple of 128 bytes for any ne0 x ne1
    uint32_t c_start = 0;
    uint32_t nc      = ne2;
    if (octx->ctx->mdev.count > 1) {
        const uint32_t units_per_chunk = htp_tensor_mdev_data_aligned(dst) ? 32 : 0;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne2, units_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        c_start = range.start;
        nc      = range.count;
    }

    if (nc == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t total_rows = ne1 * ne3;

    struct htp_upscale_context uctx = {
        .octx             = octx,
        .nrows            = total_rows,
        .nrows_per_thread = fastdiv(total_rows + n_threads - 1, &octx->n_threads_div),
        .c_start          = c_start,
        .nc               = nc,
        .pix_stride       = htp_upscale_pix_stride(ne2),
        .row_size         = ne00 * htp_upscale_pix_stride(ne2),
        .tile_size        = htp_upscale_tile_size(ne2),
        .sf0              = sf0,
        .sf1              = sf1,
        .pixel_offset     = pixel_offset,
    };

    work_queue_run(octx->ctx->work_queue, upscale_thread_f32, &uctx, n_threads);

    return HTP_STATUS_OK;
}
