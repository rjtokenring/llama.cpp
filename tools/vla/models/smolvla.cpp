#include "../vla-impl.h"

#include <cmath>

static ggml_tensor * build_norm(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx0, x, eps);
    x = ggml_mul(ctx0, x, w);
    return ggml_add(ctx0, x, b);
}

static ggml_tensor * build_linear(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(ctx0, w, x);
    return b ? ggml_add(ctx0, x, b) : x;
}

// q: [d, n_head, n_q, B], k and v: [d, n_head_kv, n_kv, B], mask: [n_kv, n_q] or nullptr -> [d*n_head, n_q, B]
static ggml_tensor * build_attn(ggml_context * ctx0, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
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

// tanh GELU written out: the CPU ggml_gelu uses an F16 table and some backends map it to GELU_QUICK
static ggml_tensor * build_gelu_tanh(ggml_context * ctx0, ggml_tensor * x) {
    ggml_tensor * x3 = ggml_mul(ctx0, ggml_sqr(ctx0, x), x);
    ggml_tensor * t  = ggml_add(ctx0, x, ggml_scale(ctx0, x3, 0.044715f));
    return ggml_mul(ctx0, x, ggml_sigmoid(ctx0, ggml_scale(ctx0, t, 2.0f * sqrtf(2.0f / (float) M_PI))));
}

// SigLIP tower, pixel shuffle and connector of SmolVLM: [P*P*3, N, B] -> [n_embd_text, n_img_tokens, B]
static ggml_tensor * build_vision(vla_context & ctx, ggml_context * ctx0, ggml_tensor * patches) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int64_t C    = hp.v_n_embd;
    const int64_t d    = C / hp.v_n_head;
    const int64_t grid = hp.v_image_size / hp.v_patch_size;
    const int64_t N    = grid * grid;
    const int64_t B    = patches->ne[2];
    const int64_t s    = hp.v_scale_factor;

    ggml_tensor * x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, m.patch_w, patches->ne[0], C), patches);
    x = ggml_add(ctx0, x, m.patch_b);
    x = ggml_add(ctx0, x, m.pos_embd);
    ggml_set_name(x, "v_inp");

    for (size_t il = 0; il < m.vit.size(); il++) {
        const auto & l = m.vit[il];
        ggml_tensor * h = build_norm(ctx0, x, l.ln1_w, l.ln1_b, hp.v_eps);
        ggml_tensor * q = ggml_reshape_4d(ctx0, build_linear(ctx0, h, l.q_w, l.q_b), d, hp.v_n_head, N, B);
        ggml_tensor * k = ggml_reshape_4d(ctx0, build_linear(ctx0, h, l.k_w, l.k_b), d, hp.v_n_head, N, B);
        ggml_tensor * v = ggml_reshape_4d(ctx0, build_linear(ctx0, h, l.v_w, l.v_b), d, hp.v_n_head, N, B);
        h = build_attn(ctx0, q, k, v, nullptr, 1.0f / sqrtf((float) d), ctx.params.flash_attn);
        x = ggml_add(ctx0, x, build_linear(ctx0, h, l.o_w, l.o_b));

        h = build_norm(ctx0, x, l.ln2_w, l.ln2_b, hp.v_eps);
        h = build_gelu_tanh(ctx0, build_linear(ctx0, h, l.up_w, l.up_b));
        x = ggml_add(ctx0, x, build_linear(ctx0, h, l.down_w, l.down_b));
        ggml_format_name(x, "v_out-%zu", il);
    }
    x = build_norm(ctx0, x, m.post_ln_w, m.post_ln_b, hp.v_eps);
    ggml_set_name(x, "v_post_ln");

    // pixel shuffle, as clip_graph::build_patch_merge_permute with a batch dim
    x = ggml_reshape_4d(ctx0, x, C * s, grid / s, grid, B);
    x = ggml_permute(ctx0, x, 0, 2, 1, 3);
    x = ggml_cont_4d(ctx0, x, C * s * s, grid / s, grid / s, B);
    x = ggml_permute(ctx0, x, 0, 2, 1, 3);
    x = ggml_cont_3d(ctx0, x, C * s * s, N / (s * s), B);

    return ggml_mul_mat(ctx0, m.mm_fc, x);
}

ggml_cgraph * vla_build_smolvla(vla_context & ctx, ggml_context * ctx0) {
    const vla_hparams & hp = ctx.model.hparams;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, ctx.n_graph, false);

    const int64_t P = hp.v_patch_size;
    ctx.inp.patches = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, P * P * 3, hp.n_patches(), hp.n_cameras);
    ggml_set_name(ctx.inp.patches, "patches");
    ggml_set_input(ctx.inp.patches);

    ggml_tensor * img = build_vision(ctx, ctx0, ctx.inp.patches);
    ggml_set_name(img, "img_tokens");
    ggml_set_output(img);

    ggml_build_forward_expand(gf, img);
    return gf;
}
