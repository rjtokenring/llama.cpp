#pragma once

#include "vla-impl.h"

// graph helpers shared by the policies in models/

enum vla_ffn_act {
    VLA_FFN_SILU,      // Llama / SmolLM2
    VLA_FFN_GELU_TANH, // Gemma
};

ggml_tensor * vla_build_norm    (ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps);
ggml_tensor * vla_build_rms_norm(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, float eps);
ggml_tensor * vla_build_linear  (ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * vla_build_rope    (ggml_context * ctx0, ggml_tensor * x, ggml_tensor * pos, float base);
ggml_tensor * vla_build_gelu_tanh(ggml_context * ctx0, ggml_tensor * x);

// q: [d, n_head, n_q, B], k and v: [d, n_head_kv, n_kv, B], mask: [n_kv, n_q] or nullptr -> [d*n_head, n_q, B]
ggml_tensor * vla_build_attn(ggml_context * ctx0, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, float scale, bool flash_attn);

// gated MLP: down(act(gate(x)) * up(x))
ggml_tensor * vla_build_ffn_glu(ggml_context * ctx0, ggml_tensor * x, const vla_lm_layer & l, vla_ffn_act act);

// SigLIP encoder for one camera, up to the post layernorm: [P*P*3, N] -> [v_n_embd, N]
// one camera at a time keeps every activation 2D, which is also what ggml-hexagon flash attention takes
ggml_tensor * vla_build_siglip(vla_context & ctx, ggml_context * ctx0, ggml_tensor * patches, int cam);
