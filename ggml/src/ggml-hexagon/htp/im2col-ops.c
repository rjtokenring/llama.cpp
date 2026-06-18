#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

// Convert a single F32 -> F16 using hardware __fp16 cast.
static inline uint16_t f32_to_f16_bits(float v) {
    union { __fp16 h; uint16_t u; } u;
    u.h = (__fp16) v;
    return u.u;
}

struct htp_im2col_context {
    const uint8_t * src1_data;
    uint8_t       * dst_data;

    // src1 strides (bytes)
    size_t          nb10;
    size_t          nb11;
    size_t          nb12;
    size_t          nb13;

    // src1 data type (HTP_TYPE_F32 or HTP_TYPE_F16)
    uint32_t        src1_type;

    // sizes
    int64_t         IW;
    int64_t         IH;
    int64_t         IC;

    int64_t         KW;
    int64_t         KH;

    int64_t         OW;
    int64_t         OH;
    int64_t         N;

    int32_t         s0, s1;
    int32_t         p0, p1;
    int32_t         d0, d1;

    // total patches = N * OH * OW
    uint32_t        total_patches;
    uint32_t        patches_per_thread;

    // size of a single output column: IC * KH * KW
    uint32_t        col_size;
};

static void im2col_thread_f16(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_im2col_context * ictx = (const struct htp_im2col_context *) data;

    const uint32_t start = ictx->patches_per_thread * ith;
    const uint32_t end   = MIN(start + ictx->patches_per_thread, ictx->total_patches);

    if (start >= end) {
        return;
    }

    const int64_t IW = ictx->IW;
    const int64_t IH = ictx->IH;
    const int64_t IC = ictx->IC;
    const int64_t KW = ictx->KW;
    const int64_t KH = ictx->KH;
    const int64_t OW = ictx->OW;
    const int64_t OH = ictx->OH;

    const int32_t s0 = ictx->s0;
    const int32_t s1 = ictx->s1;
    const int32_t p0 = ictx->p0;
    const int32_t p1 = ictx->p1;
    const int32_t d0 = ictx->d0;
    const int32_t d1 = ictx->d1;

    const size_t nb11 = ictx->nb11;
    const size_t nb12 = ictx->nb12;
    const size_t nb13 = ictx->nb13;

    const bool src_is_f32 = (ictx->src1_type == HTP_TYPE_F32);

    // f32->f16 bulk path applicable when dilation is 1 (KW elements are contiguous in src row)
    // and the kw range is fully in-bounds.
    const bool can_bulk_copy = (d0 == 1);

    for (uint32_t p = start; p < end; p++) {
        const int64_t iow = p % OW;
        const int64_t ioh = (p / OW) % OH;
        const int64_t in  = p / (OW * OH);

        uint16_t * restrict dst_col = (uint16_t *) ictx->dst_data + (size_t) p * ictx->col_size;

        const int64_t ix0 = iow * s0 - p0;
        const int64_t iy0 = ioh * s1 - p1;

        for (int64_t iic = 0; iic < IC; iic++) {
            const uint8_t * src_chan = ictx->src1_data + in * nb13 + iic * nb12;

            for (int64_t ikh = 0; ikh < KH; ikh++) {
                const int64_t iih = iy0 + ikh * d1;
                uint16_t * restrict dst_kh = dst_col + (size_t) iic * (KH * KW) + (size_t) ikh * KW;

                if (iih < 0 || iih >= IH) {
                    memset(dst_kh, 0, (size_t) KW * sizeof(uint16_t));
                    continue;
                }

                const uint8_t * src_row = src_chan + (size_t) iih * nb11;

                if (can_bulk_copy) {
                    const int64_t iiw_first = ix0;
                    const int64_t iiw_last  = ix0 + KW - 1;

                    if (iiw_first >= 0 && iiw_last < IW) {
                        // fully in-bounds, contiguous KW elements
                        if (src_is_f32) {
                            const float * src_f32 = (const float *) src_row + iiw_first;
                            for (int64_t k = 0; k < KW; k++) {
                                dst_kh[k] = f32_to_f16_bits(src_f32[k]);
                            }
                        } else {
                            const uint16_t * src_f16 = (const uint16_t *) src_row + iiw_first;
                            memcpy(dst_kh, src_f16, (size_t) KW * sizeof(uint16_t));
                        }
                        continue;
                    }
                }

                // generic path with per-element bounds check
                for (int64_t ikw = 0; ikw < KW; ikw++) {
                    const int64_t iiw = ix0 + ikw * d0;
                    if (iiw < 0 || iiw >= IW) {
                        dst_kh[ikw] = 0;
                    } else {
                        if (src_is_f32) {
                            const float v = ((const float *) src_row)[iiw];
                            dst_kh[ikw] = f32_to_f16_bits(v);
                        } else {
                            dst_kh[ikw] = ((const uint16_t *) src_row)[iiw];
                        }
                    }
                }
            }
        }
    }
}

int op_im2col(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0]; // kernel  [KW, KH, IC, OC]
    const struct htp_tensor * src1 = octx->src[1]; // image   [IW, IH, IC, N]
    const struct htp_tensor * dst  = octx->dst;   //         [IC*KH*KW, OW, OH, N]

    if (src0->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src1->type != HTP_TYPE_F32 && src1->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int32_t * op_params = octx->op_params;
    const int32_t  s0    = op_params[0];
    const int32_t  s1    = op_params[1];
    const int32_t  p0    = op_params[2];
    const int32_t  p1    = op_params[3];
    const int32_t  d0    = op_params[4];
    const int32_t  d1    = op_params[5];
    const bool     is_2D = op_params[6] == 1;

    if (!is_2D) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // src0 layout: KW=ne00, KH=ne01, IC=ne02, OC=ne03
    // src1 layout: IW=ne10, IH=ne11, IC=ne12, N=ne13   (is_2D)
    // dst  layout: ne[0]=IC*KH*KW, ne[1]=OW, ne[2]=OH, ne[3]=N
    const int64_t KW = src0->ne[0];
    const int64_t KH = src0->ne[1];
    const int64_t IC = src0->ne[2];

    const int64_t IW = src1->ne[0];
    const int64_t IH = src1->ne[1];
    const int64_t N  = src1->ne[3];

    const int64_t OW = dst->ne[1];
    const int64_t OH = dst->ne[2];

    const uint32_t total_patches = (uint32_t) (N * OH * OW);
    if (total_patches == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, total_patches);
    const uint32_t patches_per_thread = (total_patches + n_threads - 1) / n_threads;

    struct htp_im2col_context ictx = {
        .src1_data           = (const uint8_t *) src1->data,
        .dst_data            = (uint8_t *)       dst->data,
        .nb10                = src1->nb[0],
        .nb11                = src1->nb[1],
        .nb12                = src1->nb[2],
        .nb13                = src1->nb[3],
        .src1_type           = src1->type,
        .IW                  = IW,
        .IH                  = IH,
        .IC                  = IC,
        .KW                  = KW,
        .KH                  = KH,
        .OW                  = OW,
        .OH                  = OH,
        .N                   = N,
        .s0                  = s0,
        .s1                  = s1,
        .p0                  = p0,
        .p1                  = p1,
        .d0                  = d0,
        .d1                  = d1,
        .total_patches       = total_patches,
        .patches_per_thread  = patches_per_thread,
        .col_size            = (uint32_t) (IC * KH * KW),
    };

    uint64_t t1 = HAP_perf_get_qtimer_count();

    worker_pool_run_func(octx->ctx->worker_pool, im2col_thread_f16, &ictx, n_threads);

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "im2col-f16: kernel=%dx%dx%d image=%dx%dx%dx%d -> dst=%dx%dx%dx%d s=%dx%d p=%dx%d d=%dx%d usec %u\n",
         (int) KW, (int) KH, (int) IC,
         (int) IW, (int) IH, (int) IC, (int) N,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         s0, s1, p0, p1, d0, d1,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));

    return HTP_STATUS_OK;
}
