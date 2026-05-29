#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "hvx-utils.h"

struct get_rows_context {
    struct htp_ops_context * octx;
    uint32_t tasks_per_thread;
    uint32_t total_tasks;
    uint32_t chunks_per_row;
    uint32_t chunk_size;
    struct fastdiv_values get_rows_div_ne10;
    struct fastdiv_values get_rows_div_ne10_ne11;
    struct fastdiv_values get_rows_div_chunks_per_row;
};

#define get_rows_preamble \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t nr = ne10 * ne11 * ne12;

// Dequantize a single row from Q4_0x4x2 repacked format to f32.
// Q4_0x4x2 row layout: [quants: ne00/2 bytes] [scales: ceil(ne00/256)*16 bytes of fp16]
// Each super-block covers 256 elements (8 Q4_0 blocks of 32 elements each).
// Quant packing: q[j] = (qs[j+128] << 4) | qs[j] where qs is the interleaved unpacked array.
static void dequantize_row_q4x4x2_f32(const uint8_t * restrict src, float * restrict dst, uint32_t ne00) {
    const uint32_t qk         = QK_Q4_0x4x2;      // 256 elements per super-block
    const uint32_t nb         = ne00 / qk;          // full super-blocks
    const uint32_t nloe       = ne00 % qk;          // leftover elements
    const uint32_t nblocks    = nb + (nloe > 0 ? 1 : 0);
    const uint32_t qrow_size  = ne00 / 2;           // total quant bytes
    const uint32_t qblk_size  = qk / 2;             // 128 quant bytes per super-block

    const uint8_t  * restrict q_base = src;
    const __fp16   * restrict d_base = (const __fp16 *)(src + qrow_size);

    for (uint32_t blk = 0; blk < nblocks; blk++) {
        const uint8_t * q = q_base + blk * qblk_size;
        const __fp16  * d = d_base + blk * 8;  // 8 scales per super-block

        const uint32_t elems = (blk < nb) ? qk : nloe;

        // Unpack quants back to the interleaved qs[256] array, then map to output
        // q[j] = (qs[j+128] << 4) | qs[j]  for j=0..127
        // qs[bi*32 + e] = element e of Q4_0 block bi (bi=0..7, e=0..31)
        // Output element index = blk*256 + bi*32 + e
        for (uint32_t j = 0; j < qblk_size && j * 2 < elems; j++) {
            uint8_t qbyte = q[j];
            uint8_t lo = qbyte & 0x0F;       // qs[j]     -> block j/32, element j%32
            uint8_t hi = (qbyte >> 4) & 0x0F; // qs[j+128] -> block (j/32)+4, element j%32

            uint32_t bi_lo  = j / 32;          // block index 0-3
            uint32_t elem   = j % 32;          // element within block
            uint32_t bi_hi  = bi_lo + 4;       // block index 4-7

            float scale_lo = (float) d[bi_lo];
            float scale_hi = (float) d[bi_hi];

            uint32_t idx_lo = blk * qk + bi_lo * 32 + elem;
            uint32_t idx_hi = blk * qk + bi_hi * 32 + elem;

            if (idx_lo < ne00) dst[idx_lo] = ((float)lo - 8.0f) * scale_lo;
            if (idx_hi < ne00) dst[idx_hi] = ((float)hi - 8.0f) * scale_hi;
        }
    }
}

static void get_rows_thread_q4x4x2_f32(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;
        }

        const uint8_t * src0_ptr = (const uint8_t *)(octx->src[0]->data + i01*nb01 + i11*nb02 + i12*nb03);
        float         * dst_ptr  = (float *)(octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3);
        dequantize_row_q4x4x2_f32(src0_ptr, dst_ptr, ne00);
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "get-rows-q4x4x2-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, ir0, ir1, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, (unsigned) qt);
}

static void get_rows_thread_f32_f32_dma(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    dma_queue * dma_queue = octx->ctx->dma[ith];
    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;
        }

        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i11*nb02 + i12*nb03;
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;

        while (!dma_queue_push(dma_queue, dma_make_ptr((void *)dst_ptr, (const void *)src0_ptr), nb1, nb01, ne00 * sizeof(float), 1)) {
            dma_queue_pop(dma_queue);
        }
    }
    dma_queue_flush(dma_queue);

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "get-rows-f32-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, ir0, ir1, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, (unsigned) qt);
}

static void get_rows_thread_f32_f32_hvx(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    const uint32_t chunks_per_row = grctx->chunks_per_row;
    const uint32_t chunk_size     = grctx->chunk_size;
    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t row_idx   = fastdiv(i, &grctx->get_rows_div_chunks_per_row);
        const uint32_t chunk_idx = i - row_idx * chunks_per_row;

        const uint32_t i12 = fastdiv(row_idx, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = row_idx - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;
        }

        const uint32_t offset = chunk_idx * chunk_size;
        if (offset < ne00) {
            const uint32_t copy_size = MIN(chunk_size, ne00 - offset);
            const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i11*nb02 + i12*nb03 + offset * sizeof(float);
            const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3  + offset * sizeof(float);
            hvx_copy_f32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, copy_size);
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "get-rows-f32-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, ir0, ir1, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, (unsigned) qt);
}

int op_get_rows(struct htp_ops_context * octx) {
    get_rows_preamble;

    if (octx->dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // Q4_0 path: dequantize repacked Q4_0x4x2 rows to f32
    if (octx->src[0]->type == HTP_TYPE_Q4_0) {
        struct get_rows_context grctx;
        grctx.octx = octx;
        grctx.get_rows_div_ne10      = init_fastdiv_values(octx->src[1]->ne[0]);
        grctx.get_rows_div_ne10_ne11 = init_fastdiv_values(octx->src[1]->ne[0] * octx->src[1]->ne[1]);
        grctx.chunks_per_row = 1;
        grctx.chunk_size = ne00;
        grctx.total_tasks = nr;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(1);

        const uint32_t n_threads = MIN(nr, octx->n_threads);
        grctx.tasks_per_thread = (nr + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_q4x4x2_f32, &grctx, n_threads);
        return HTP_STATUS_OK;
    }

    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t nb00 = octx->src[0]->nb[0];
    const uint32_t nb0  = octx->dst->nb[0];

    const bool can_use_dma = (nb00 == sizeof(float)) && (nb0 == sizeof(float));
    const bool use_dma = can_use_dma && (ne00 >= 2048);

    struct get_rows_context grctx;
    grctx.octx = octx;
    grctx.get_rows_div_ne10      = init_fastdiv_values(octx->src[1]->ne[0]);
    grctx.get_rows_div_ne10_ne11 = init_fastdiv_values(octx->src[1]->ne[0] * octx->src[1]->ne[1]);

    if (use_dma) {
        grctx.chunks_per_row = 1;
        grctx.chunk_size = ne00;
        grctx.total_tasks = nr;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(1);

        const uint32_t n_threads = MIN(nr, octx->n_threads);
        grctx.tasks_per_thread = (nr + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_f32_f32_dma, &grctx, n_threads);
    } else {
        uint32_t chunks_per_row = 1;
        uint32_t chunk_size = ne00;
        uint32_t total_tasks = nr;

        if (nr < octx->n_threads) {
            const uint32_t min_chunk_size = 1024;
            uint32_t max_chunks = ne00 / min_chunk_size;
            if (max_chunks == 0) {
                max_chunks = 1;
            }
            chunks_per_row = MIN((octx->n_threads + nr - 1) / nr, max_chunks);
            chunk_size = (ne00 + chunks_per_row - 1) / chunks_per_row;
            total_tasks = nr * chunks_per_row;
        }

        grctx.chunks_per_row = chunks_per_row;
        grctx.chunk_size = chunk_size;
        grctx.total_tasks = total_tasks;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(chunks_per_row);

        const uint32_t n_threads = MIN(total_tasks, octx->n_threads);
        grctx.tasks_per_thread = (total_tasks + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_f32_f32_hvx, &grctx, n_threads);
    }
    return HTP_STATUS_OK;
}
