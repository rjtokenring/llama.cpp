#ifndef HTP_UPSCALE_OPS_H
#define HTP_UPSCALE_OPS_H

#include "hex-common.h"
#include <stdint.h>

// Bilinear UPSCALE with src channels innermost (nb[2] == sizeof(float)) and a contiguous dst.
// Per thread VTCM: two src rows (ne00 pixels, channels padded to a 128 byte pixel stride)
// and one dst tile of 32 pixels (one 128 byte row per channel).
struct htp_upscale_kernel_params {
    uint32_t n_threads;
    uint32_t vtcm_size;
};

#define HTP_UPSCALE_TILE_NPIX 32

static inline uint32_t htp_upscale_pix_stride(uint32_t ne02) {
    return hex_round_up(ne02 * (uint32_t) sizeof(float), 128);
}

static inline uint32_t htp_upscale_tile_size(uint32_t ne02) {
    return hex_round_up(ne02, HTP_UPSCALE_TILE_NPIX) * HTP_UPSCALE_TILE_NPIX * (uint32_t) sizeof(float);
}

static inline uint32_t htp_upscale_vtcm_per_thread(uint32_t ne00, uint32_t ne02) {
    return 2 * ne00 * htp_upscale_pix_stride(ne02) + htp_upscale_tile_size(ne02);
}

#endif // HTP_UPSCALE_OPS_H
