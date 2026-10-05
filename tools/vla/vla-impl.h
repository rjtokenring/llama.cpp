#pragma once

#include "vla.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <memory>
#include <string>
#include <vector>

struct llama_model;

struct vla_hparams {
    std::string policy_type;
    std::vector<std::string> camera_names; // images are passed in this order
    std::vector<std::string> state_names;  // optional, empty or state_dim names
    std::vector<std::string> action_names; // optional, empty or action_dim names

    int32_t chunk_size      = 0;
    int32_t num_steps       = 0;
    int32_t state_dim       = 0;
    int32_t action_dim      = 0;
    int32_t max_state_dim   = 0;
    int32_t max_action_dim  = 0;
    int32_t max_lang_tokens = 0;
    int32_t n_cameras       = 0;
    float   time_min_period = 0.0f;
    float   time_max_period = 0.0f;
    float   norm_eps        = 0.0f;

    int32_t v_image_size   = 0;
    int32_t v_patch_size   = 0;
    int32_t v_n_embd       = 0;
    int32_t v_n_ff         = 0;
    int32_t v_n_layer      = 0;
    int32_t v_n_head       = 0;
    int32_t v_scale_factor = 0;
    float   v_eps          = 0.0f;

    int32_t t_n_embd    = 0;
    int32_t t_n_ff      = 0;
    int32_t t_n_layer   = 0;
    int32_t t_n_head    = 0;
    int32_t t_n_head_kv = 0;
    int32_t t_head_dim  = 0;
    float   t_eps       = 0.0f;
    float   t_rope_base = 0.0f;

    int32_t e_n_embd           = 0;
    int32_t e_n_ff             = 0;
    int32_t e_n_layer          = 0;
    int32_t e_self_attn_every_n = 0;

    int32_t n_patches() const { return (v_image_size / v_patch_size) * (v_image_size / v_patch_size); }
    int32_t n_img_tokens() const { return n_patches() / (v_scale_factor * v_scale_factor); }
    bool expert_self_attn(int il) const { return e_self_attn_every_n > 0 && il % e_self_attn_every_n == 0; }

    // images, padded language, state
    int32_t n_prefix() const { return n_img_tokens() * n_cameras + max_lang_tokens + 1; }
};

struct vla_vit_layer {
    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    ggml_tensor * q_w   = nullptr;
    ggml_tensor * q_b   = nullptr;
    ggml_tensor * k_w   = nullptr;
    ggml_tensor * k_b   = nullptr;
    ggml_tensor * v_w   = nullptr;
    ggml_tensor * v_b   = nullptr;
    ggml_tensor * o_w   = nullptr;
    ggml_tensor * o_b   = nullptr;
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    ggml_tensor * up_w   = nullptr;
    ggml_tensor * up_b   = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * down_b = nullptr;
};

struct vla_lm_layer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * q         = nullptr;
    ggml_tensor * k         = nullptr;
    ggml_tensor * v         = nullptr;
    ggml_tensor * o         = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * gate      = nullptr;
    ggml_tensor * up        = nullptr;
    ggml_tensor * down      = nullptr;
};

struct vla_model {
    vla_hparams hparams;

    ggml_tensor * patch_w = nullptr;
    ggml_tensor * patch_b = nullptr;
    ggml_tensor * pos_embd = nullptr;
    std::vector<vla_vit_layer> vit;
    ggml_tensor * post_ln_w = nullptr;
    ggml_tensor * post_ln_b = nullptr;
    ggml_tensor * mm_fc = nullptr;

    int32_t       n_vocab  = 0;
    ggml_tensor * tok_embd = nullptr;
    std::vector<vla_lm_layer> vlm;
    std::vector<vla_lm_layer> exp;
    ggml_tensor * exp_out_norm = nullptr;

    ggml_tensor * state_proj_w = nullptr;
    ggml_tensor * state_proj_b = nullptr;
    ggml_tensor * act_in_w  = nullptr;
    ggml_tensor * act_in_b  = nullptr;
    ggml_tensor * act_out_w = nullptr;
    ggml_tensor * act_out_b = nullptr;
    ggml_tensor * time_in_w  = nullptr;
    ggml_tensor * time_in_b  = nullptr;
    ggml_tensor * time_out_w = nullptr;
    ggml_tensor * time_out_b = nullptr;

    std::vector<float> state_mean;
    std::vector<float> state_std;
    std::vector<float> action_mean;
    std::vector<float> action_std;
};

struct vla_ggml_ctx_deleter    { void operator()(ggml_context * c) { ggml_free(c); } };
struct vla_gguf_ctx_deleter    { void operator()(gguf_context * c) { gguf_free(c); } };
struct vla_buffer_deleter      { void operator()(ggml_backend_buffer_t b) { ggml_backend_buffer_free(b); } };
struct vla_backend_deleter     { void operator()(ggml_backend_t b) { ggml_backend_free(b); } };
struct vla_sched_deleter       { void operator()(ggml_backend_sched_t s) { ggml_backend_sched_free(s); } };

using vla_ggml_ctx_ptr = std::unique_ptr<ggml_context, vla_ggml_ctx_deleter>;
using vla_buffer_ptr   = std::unique_ptr<ggml_backend_buffer, vla_buffer_deleter>;
using vla_backend_ptr  = std::unique_ptr<ggml_backend, vla_backend_deleter>;
using vla_sched_ptr    = std::unique_ptr<ggml_backend_sched, vla_sched_deleter>;

// tensors the graph reads per call; filled by vla_predict() after allocation
struct vla_graph_inputs {
    ggml_tensor * patches = nullptr; // [patch_size * patch_size * 3, n_patches, n_cameras], pixels in [-1, 1]
    ggml_tensor * tokens  = nullptr; // [max_lang_tokens], padded
    ggml_tensor * state   = nullptr; // [max_state_dim], normalized and zero padded
    ggml_tensor * pos     = nullptr; // [n_prefix]
    ggml_tensor * mask    = nullptr; // [n_prefix, n_prefix], 0 or -inf

    ggml_tensor * noise      = nullptr; // [max_action_dim, chunk_size]
    ggml_tensor * time_emb   = nullptr; // [e_n_embd, num_steps]
    ggml_tensor * pos_self   = nullptr; // [chunk_size], after the prefix
    ggml_tensor * pos_cross  = nullptr; // [chunk_size], from 0
    ggml_tensor * mask_self  = nullptr; // [n_prefix + chunk_size, chunk_size]
    ggml_tensor * mask_cross = nullptr; // [n_prefix, chunk_size]

    ggml_tensor * actions = nullptr;    // [max_action_dim, chunk_size], normalized
};

struct vla_context {
    vla_context_params params;
    vla_model          model;

    vla_backend_ptr              backend_cpu;
    vla_backend_ptr              backend_dev;
    std::vector<ggml_backend_t>  backends;
    vla_sched_ptr                sched;

    vla_ggml_ctx_ptr             ctx_w_dev;  // weights on the device (or the CPU without one)
    vla_ggml_ctx_ptr             ctx_w_cpu;  // weights only read by GET_ROWS stay on the host
    std::vector<vla_buffer_ptr>  bufs;

    llama_model * vocab_model = nullptr;

    size_t               n_graph = 0;
    std::vector<uint8_t> compute_meta;
    vla_graph_inputs     inp;
};

ggml_cgraph * vla_build_smolvla(vla_context & ctx, ggml_context * ctx0);
