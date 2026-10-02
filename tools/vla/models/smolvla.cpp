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

static ggml_tensor * build_rms_norm(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, eps), w);
}

static ggml_tensor * build_rope(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * pos, float base) {
    return ggml_rope_ext(ctx0, x, pos, nullptr, x->ne[0], GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

static ggml_tensor * build_ffn_silu(ggml_context * ctx0, ggml_tensor * x, const vla_lm_layer & l) {
    ggml_tensor * g = ggml_silu(ctx0, ggml_mul_mat(ctx0, l.gate, x));
    return ggml_mul_mat(ctx0, l.down, ggml_mul(ctx0, g, ggml_mul_mat(ctx0, l.up, x)));
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

// runs the VLM over the prefix; the expert only reads the per-layer K (after RoPE) and V
static void build_vlm(vla_context & ctx, ggml_context * ctx0, ggml_tensor * x,
        std::vector<ggml_tensor *> & k_cache, std::vector<ggml_tensor *> & v_cache) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int64_t hd = hp.t_head_dim;
    const int64_t n  = x->ne[1];

    for (int il = 0; il < hp.t_n_layer; il++) {
        const auto & l = m.vlm[il];
        ggml_tensor * h = build_rms_norm(ctx0, x, l.attn_norm, hp.t_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.q, h), hd, hp.t_n_head, n);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, h), hd, hp.t_n_head_kv, n);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.v, h), hd, hp.t_n_head_kv, n);
        q = build_rope(ctx0, q, ctx.inp.pos, hp.t_rope_base);
        k = build_rope(ctx0, k, ctx.inp.pos, hp.t_rope_base);
        ggml_format_name(k, "Kcur-%d", il);
        ggml_format_name(v, "Vcur-%d", il);
        k_cache[il] = k;
        v_cache[il] = v;

        // the last layer output is never read, ggml drops these nodes
        h = build_attn(ctx0, q, k, v, ctx.inp.mask, 1.0f / sqrtf((float) hd), ctx.params.flash_attn);
        x = ggml_add(ctx0, x, ggml_mul_mat(ctx0, l.o, h));
        x = ggml_add(ctx0, x, build_ffn_silu(ctx0, build_rms_norm(ctx0, x, l.ffn_norm, hp.t_eps), l));
    }
}

// flow matching: from noise at t = 1, num_steps Euler steps of the expert velocity down to t = 0
static ggml_tensor * build_expert(vla_context & ctx, ggml_context * ctx0,
        const std::vector<ggml_tensor *> & k_cache, const std::vector<ggml_tensor *> & v_cache) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int64_t hd    = hp.t_head_dim;
    const int64_t n_kv  = k_cache[0]->ne[2];
    const int64_t chunk = hp.chunk_size;
    const float   kq_scale = 1.0f / sqrtf((float) hd);
    const float   dt = -1.0f / hp.num_steps;

    // cross-attention layers read the prefix K/V through their own projections; same for every step
    std::vector<ggml_tensor *> k_cross(hp.e_n_layer), v_cross(hp.e_n_layer);
    for (int il = 0; il < hp.e_n_layer; il++) {
        if (hp.expert_self_attn(il)) {
            continue;
        }
        const auto & l = m.exp[il];
        ggml_tensor * k = ggml_reshape_2d(ctx0, k_cache[il], hd * hp.t_n_head_kv, n_kv);
        ggml_tensor * v = ggml_reshape_2d(ctx0, v_cache[il], hd * hp.t_n_head_kv, n_kv);
        k_cross[il] = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, k), hd, hp.t_n_head_kv, n_kv);
        v_cross[il] = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.v, v), hd, hp.t_n_head_kv, n_kv);
    }

    ggml_tensor * x_t = ctx.inp.noise;
    for (int step = 0; step < hp.num_steps; step++) {
        ggml_tensor * act  = build_linear(ctx0, x_t, m.act_in_w, m.act_in_b);
        ggml_tensor * temb = ggml_view_2d(ctx0, ctx.inp.time_emb, hp.e_n_embd, 1, ctx.inp.time_emb->nb[1],
                                          step * ctx.inp.time_emb->nb[1]);
        ggml_tensor * h = ggml_concat(ctx0, act, ggml_repeat(ctx0, temb, act), 0);
        h = ggml_silu(ctx0, build_linear(ctx0, h, m.time_in_w, m.time_in_b));
        h = build_linear(ctx0, h, m.time_out_w, m.time_out_b);

        for (int il = 0; il < hp.e_n_layer; il++) {
            const auto & l = m.exp[il];
            ggml_tensor * n = build_rms_norm(ctx0, h, l.attn_norm, hp.t_eps);
            ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.q, n), hd, hp.t_n_head, chunk);
            ggml_tensor * k;
            ggml_tensor * v;
            ggml_tensor * mask;
            if (hp.expert_self_attn(il)) {
                k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, n), hd, hp.t_n_head_kv, chunk);
                v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.v, n), hd, hp.t_n_head_kv, chunk);
                q = build_rope(ctx0, q, ctx.inp.pos_self, hp.t_rope_base);
                k = build_rope(ctx0, k, ctx.inp.pos_self, hp.t_rope_base);
                k = ggml_concat(ctx0, k_cache[il], k, 2);
                v = ggml_concat(ctx0, v_cache[il], v, 2);
                mask = ctx.inp.mask_self;
            } else {
                // the prefix K is already rotated; only the suffix queries get RoPE, from position 0
                q = build_rope(ctx0, q, ctx.inp.pos_cross, hp.t_rope_base);
                k = k_cross[il];
                v = v_cross[il];
                mask = ctx.inp.mask_cross;
            }
            ggml_tensor * a = build_attn(ctx0, q, k, v, mask, kq_scale, ctx.params.flash_attn);
            h = ggml_add(ctx0, h, ggml_mul_mat(ctx0, l.o, a));
            h = ggml_add(ctx0, h, build_ffn_silu(ctx0, build_rms_norm(ctx0, h, l.ffn_norm, hp.t_eps), l));
        }

        h = build_rms_norm(ctx0, h, m.exp_out_norm, hp.t_eps);
        ggml_tensor * v_t = build_linear(ctx0, h, m.act_out_w, m.act_out_b);
        ggml_format_name(v_t, "v_t-%d", step);
        x_t = ggml_add(ctx0, x_t, ggml_scale(ctx0, v_t, dt));
    }
    return x_t;
}

ggml_cgraph * vla_build_smolvla(vla_context & ctx, ggml_context * ctx0) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, ctx.n_graph, false);

    const int64_t P = hp.v_patch_size;
    const int64_t n_prefix = hp.n_prefix();
    ctx.inp.patches = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, P * P * 3, hp.n_patches(), hp.n_cameras);
    ctx.inp.tokens  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hp.max_lang_tokens);
    ctx.inp.state   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hp.max_state_dim);
    ctx.inp.pos     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prefix);
    ctx.inp.mask    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix, n_prefix);
    ctx.inp.noise      = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.max_action_dim, hp.chunk_size);
    ctx.inp.time_emb   = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.e_n_embd, hp.num_steps);
    ctx.inp.pos_self   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hp.chunk_size);
    ctx.inp.pos_cross  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hp.chunk_size);
    ctx.inp.mask_self  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix + hp.chunk_size, hp.chunk_size);
    ctx.inp.mask_cross = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix, hp.chunk_size);
    for (ggml_tensor * t : { ctx.inp.patches, ctx.inp.tokens, ctx.inp.state, ctx.inp.pos, ctx.inp.mask, ctx.inp.noise,
                             ctx.inp.time_emb, ctx.inp.pos_self, ctx.inp.pos_cross, ctx.inp.mask_self, ctx.inp.mask_cross }) {
        ggml_set_input(t);
    }

    ggml_tensor * img = build_vision(ctx, ctx0, ctx.inp.patches);
    ggml_set_name(img, "img_tokens");

    // SmolVLA scales image and language embeddings by sqrt(n_embd), not the state
    const float scale = sqrtf((float) hp.t_n_embd);
    img = ggml_scale(ctx0, ggml_reshape_2d(ctx0, img, hp.t_n_embd, img->ne[1] * img->ne[2]), scale);
    ggml_tensor * lang  = ggml_scale(ctx0, ggml_get_rows(ctx0, m.tok_embd, ctx.inp.tokens), scale);
    ggml_tensor * state = build_linear(ctx0, ggml_reshape_2d(ctx0, ctx.inp.state, hp.max_state_dim, 1), m.state_proj_w, m.state_proj_b);
    ggml_tensor * prefix = ggml_concat(ctx0, ggml_concat(ctx0, img, lang, 1), state, 1);
    ggml_set_name(prefix, "prefix_emb");

    std::vector<ggml_tensor *> k_cache(hp.t_n_layer), v_cache(hp.t_n_layer);
    build_vlm(ctx, ctx0, prefix, k_cache, v_cache);

    ctx.inp.actions = build_expert(ctx, ctx0, k_cache, v_cache);
    ggml_set_name(ctx.inp.actions, "actions_norm");
    ggml_set_output(ctx.inp.actions);
    ggml_build_forward_expand(gf, ctx.inp.actions);
    return gf;
}
