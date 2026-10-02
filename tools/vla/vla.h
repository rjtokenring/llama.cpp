#ifndef VLA_H
#define VLA_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

/**
 * libvla: runs vision-language-action robot policies (SmolVLA) from a GGUF made by convert_hf_to_gguf.py.
 * One call to vla_predict() maps camera images, robot state and a task prompt to a chunk of actions.
 */

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define VLA_API __declspec(dllexport)
#        else
#            define VLA_API __declspec(dllimport)
#        endif
#    else
#        define VLA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define VLA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct vla_context;

struct vla_context_params {
    int32_t      n_threads;
    const char * device;     // backend device name, e.g. "HTP0"; NULL picks the first GPU/accelerator, "none" uses the CPU only
    bool         flash_attn;

    ggml_backend_sched_eval_callback cb_eval;
    void *                           cb_eval_user_data;
};

struct vla_info {
    const char * policy_type;
    int32_t n_cameras;
    int32_t image_size;      // images are resized to image_size x image_size
    int32_t state_dim;
    int32_t action_dim;
    int32_t max_action_dim;  // padded action size, the size of one row of noise
    int32_t chunk_size;      // actions per vla_predict() call
    int32_t max_lang_tokens;
    int32_t num_steps;
};

enum vla_image_format {
    VLA_IMAGE_RGB_U8  = 0, // interleaved RGB, uint8
    VLA_IMAGE_RGB_F32 = 1, // interleaved RGB, float in [0, 1]
};

struct vla_image {
    int32_t               nx;
    int32_t               ny;
    enum vla_image_format format;
    const void *          data;
};

struct vla_input {
    const struct vla_image * images;   // n_cameras images, in the order the policy was trained with
    int32_t                  n_images;
    const float *            state;    // state_dim values, not normalized
    int32_t                  n_state;
    const int32_t *          tokens;   // tokenized prompt, see vla_tokenize()
    int32_t                  n_tokens;
    const float *            noise;    // optional chunk_size x max_action_dim initial noise; NULL samples it from seed
    uint32_t                 seed;
};

VLA_API struct vla_context_params vla_context_default_params(void);

VLA_API struct vla_context * vla_init_from_file(const char * path, struct vla_context_params params);
VLA_API void                 vla_free(struct vla_context * ctx);

VLA_API struct vla_info vla_get_info(const struct vla_context * ctx);

// tokenize a task prompt the way the policy was trained; returns the number of tokens, or -n if n_tokens_max is too small
VLA_API int32_t vla_tokenize(const struct vla_context * ctx, const char * prompt, int32_t * tokens, int32_t n_tokens_max);

// writes chunk_size x action_dim actions, not normalized; returns 0 on success
VLA_API int32_t vla_predict(struct vla_context * ctx, const struct vla_input * input, float * actions);

#ifdef __cplusplus
}
#endif

#endif
