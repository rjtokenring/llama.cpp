#include "vla-graph.h"

#include <cmath>

ggml_tensor * vla_build_norm(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx0, x, eps);
    x = ggml_mul(ctx0, x, w);
    return ggml_add(ctx0, x, b);
}

ggml_tensor * vla_build_rms_norm(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, eps), w);
}

ggml_tensor * vla_build_linear(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(ctx0, w, x);
    return b ? ggml_add(ctx0, x, b) : x;
}

ggml_tensor * vla_build_rope(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * pos, float base) {
    return ggml_rope_ext(ctx0, x, pos, nullptr, x->ne[0], GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

// tanh GELU written out: the CPU ggml_gelu uses an F16 table and some backends map it to GELU_QUICK
ggml_tensor * vla_build_gelu_tanh(ggml_context * ctx0, ggml_tensor * x) {
    ggml_tensor * x3 = ggml_mul(ctx0, ggml_sqr(ctx0, x), x);
    ggml_tensor * t  = ggml_add(ctx0, x, ggml_scale(ctx0, x3, 0.044715f));
    return ggml_mul(ctx0, x, ggml_sigmoid(ctx0, ggml_scale(ctx0, t, 2.0f * sqrtf(2.0f / (float) M_PI))));
}

ggml_tensor * vla_build_attn(ggml_context * ctx0, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, float scale, bool flash_attn) {
    const int64_t d   = q->ne[0];
    const int64_t n_h = q->ne[1];
    const int64_t n_q = q->ne[2];
    const int64_t B   = q->ne[3];

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);

    ggml_tensor * cur;
    if (flash_attn) {
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);
        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        if (mask) {
            mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        }
        cur = ggml_flash_attn_ext(ctx0, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, mask, scale, 0.0f);
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
        cur = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    }
    return ggml_cont_3d(ctx0, cur, d * n_h, n_q, B);
}

ggml_tensor * vla_build_ffn_glu(ggml_context * ctx0, ggml_tensor * x, const vla_lm_layer & l, vla_ffn_act act) {
    ggml_tensor * g = ggml_mul_mat(ctx0, l.gate, x);
    g = act == VLA_FFN_SILU ? ggml_silu(ctx0, g) : vla_build_gelu_tanh(ctx0, g);
    return ggml_mul_mat(ctx0, l.down, ggml_mul(ctx0, g, ggml_mul_mat(ctx0, l.up, x)));
}

ggml_tensor * vla_build_siglip(vla_context & ctx, ggml_context * ctx0, ggml_tensor * patches, int cam) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int64_t C    = hp.v_n_embd;
    const int64_t d    = C / hp.v_n_head;
    const int64_t grid = hp.v_image_size / hp.v_patch_size;
    const int64_t N    = grid * grid;

    ggml_tensor * x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, m.patch_w, patches->ne[0], C), patches);
    x = ggml_add(ctx0, x, m.patch_b);
    x = ggml_add(ctx0, x, m.pos_embd);
    ggml_format_name(x, "v_inp-c%d", cam);

    for (size_t il = 0; il < m.vit.size(); il++) {
        const auto & l = m.vit[il];
        ggml_tensor * h = vla_build_norm(ctx0, x, l.ln1_w, l.ln1_b, hp.v_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx0, vla_build_linear(ctx0, h, l.q_w, l.q_b), d, hp.v_n_head, N);
        ggml_tensor * k = ggml_reshape_3d(ctx0, vla_build_linear(ctx0, h, l.k_w, l.k_b), d, hp.v_n_head, N);
        ggml_tensor * v = ggml_reshape_3d(ctx0, vla_build_linear(ctx0, h, l.v_w, l.v_b), d, hp.v_n_head, N);
        h = vla_build_attn(ctx0, q, k, v, nullptr, 1.0f / sqrtf((float) d), ctx.params.flash_attn);
        x = ggml_add(ctx0, x, vla_build_linear(ctx0, ggml_reshape_2d(ctx0, h, C, N), l.o_w, l.o_b));

        h = vla_build_norm(ctx0, x, l.ln2_w, l.ln2_b, hp.v_eps);
        h = vla_build_gelu_tanh(ctx0, vla_build_linear(ctx0, h, l.up_w, l.up_b));
        x = ggml_add(ctx0, x, vla_build_linear(ctx0, h, l.down_w, l.down_b));
        ggml_format_name(x, "v_out-%zu-c%d", il, cam);
    }
    return vla_build_norm(ctx0, x, m.post_ln_w, m.post_ln_b, hp.v_eps);
}
