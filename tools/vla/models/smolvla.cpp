#include "../vla-graph.h"

#include <cmath>

// SigLIP, pixel shuffle and connector of SmolVLM for one camera: [P*P*3, N] -> [n_embd_text, n_img_tokens]
static ggml_tensor * build_vision(vla_context & ctx, ggml_context * ctx0, ggml_tensor * patches, int cam) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int64_t C    = hp.v_n_embd;
    const int64_t grid = hp.v_image_size / hp.v_patch_size;
    const int64_t N    = grid * grid;
    const int64_t s    = hp.v_scale_factor;

    ggml_tensor * x = vla_build_siglip(ctx, ctx0, patches, cam);

    // pixel shuffle, as clip_graph::build_patch_merge_permute
    x = ggml_reshape_3d(ctx0, x, C * s, grid / s, grid);
    x = ggml_permute(ctx0, x, 0, 2, 1, 3);
    x = ggml_cont_3d(ctx0, x, C * s * s, grid / s, grid / s);
    x = ggml_permute(ctx0, x, 0, 2, 1, 3);
    x = ggml_cont_2d(ctx0, x, C * s * s, N / (s * s));

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
        ggml_tensor * h = vla_build_rms_norm(ctx0, x, l.attn_norm, hp.t_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.q, h), hd, hp.t_n_head, n);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, h), hd, hp.t_n_head_kv, n);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.v, h), hd, hp.t_n_head_kv, n);
        q = vla_build_rope(ctx0, q, ctx.inp.pos, hp.t_rope_base);
        k = vla_build_rope(ctx0, k, ctx.inp.pos, hp.t_rope_base);
        ggml_format_name(k, "Kcur-%d", il);
        ggml_format_name(v, "Vcur-%d", il);
        k_cache[il] = k;
        v_cache[il] = v;

        // the last layer output is never read, ggml drops these nodes
        h = vla_build_attn(ctx0, q, k, v, ctx.inp.mask, 1.0f / sqrtf((float) hd), ctx.params.flash_attn);
        x = ggml_add(ctx0, x, ggml_mul_mat(ctx0, l.o, h));
        x = ggml_add(ctx0, x, vla_build_ffn_glu(ctx0, vla_build_rms_norm(ctx0, x, l.ffn_norm, hp.t_eps), l, VLA_FFN_SILU));
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
        ggml_tensor * act  = vla_build_linear(ctx0, x_t, m.act_in_w, m.act_in_b);
        ggml_tensor * temb = ggml_view_2d(ctx0, ctx.inp.time_emb, hp.e_n_embd, 1, ctx.inp.time_emb->nb[1],
                                          step * ctx.inp.time_emb->nb[1]);
        ggml_tensor * h = ggml_concat(ctx0, act, ggml_repeat(ctx0, temb, act), 0);
        h = ggml_silu(ctx0, vla_build_linear(ctx0, h, m.time_in_w, m.time_in_b));
        h = vla_build_linear(ctx0, h, m.time_out_w, m.time_out_b);

        for (int il = 0; il < hp.e_n_layer; il++) {
            const auto & l = m.exp[il];
            ggml_tensor * n = vla_build_rms_norm(ctx0, h, l.attn_norm, hp.t_eps);
            ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.q, n), hd, hp.t_n_head, chunk);
            ggml_tensor * k;
            ggml_tensor * v;
            ggml_tensor * mask;
            if (hp.expert_self_attn(il)) {
                k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, n), hd, hp.t_n_head_kv, chunk);
                v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.v, n), hd, hp.t_n_head_kv, chunk);
                q = vla_build_rope(ctx0, q, ctx.inp.pos_self, hp.t_rope_base);
                k = vla_build_rope(ctx0, k, ctx.inp.pos_self, hp.t_rope_base);
                k = ggml_concat(ctx0, k_cache[il], k, 2);
                v = ggml_concat(ctx0, v_cache[il], v, 2);
                mask = ctx.inp.mask_self;
            } else {
                // the prefix K is already rotated; only the suffix queries get RoPE, from position 0
                q = vla_build_rope(ctx0, q, ctx.inp.pos_cross, hp.t_rope_base);
                k = k_cross[il];
                v = v_cross[il];
                mask = ctx.inp.mask_cross;
            }
            ggml_tensor * a = vla_build_attn(ctx0, q, k, v, mask, kq_scale, ctx.params.flash_attn);
            h = ggml_add(ctx0, h, ggml_mul_mat(ctx0, l.o, a));
            h = ggml_add(ctx0, h, vla_build_ffn_glu(ctx0, vla_build_rms_norm(ctx0, h, l.ffn_norm, hp.t_eps), l, VLA_FFN_SILU));
        }

        h = vla_build_rms_norm(ctx0, h, m.exp_out_norm, hp.t_eps);
        ggml_tensor * v_t = vla_build_linear(ctx0, h, m.act_out_w, m.act_out_b);
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

    ggml_tensor * img = nullptr;
    for (int c = 0; c < hp.n_cameras; c++) {
        ggml_tensor * patches = ggml_view_2d(ctx0, ctx.inp.patches, P * P * 3, hp.n_patches(), ctx.inp.patches->nb[1],
                                             c * ctx.inp.patches->nb[2]);
        ggml_tensor * cur = build_vision(ctx, ctx0, patches, c);
        img = img ? ggml_concat(ctx0, img, cur, 1) : cur;
    }
    ggml_set_name(img, "img_tokens");

    // SmolVLA scales image and language embeddings by sqrt(n_embd), not the state
    const float scale = sqrtf((float) hp.t_n_embd);
    img = ggml_scale(ctx0, img, scale);
    ggml_tensor * lang  = ggml_scale(ctx0, ggml_get_rows(ctx0, m.tok_embd, ctx.inp.tokens), scale);
    ggml_tensor * state = vla_build_linear(ctx0, ggml_reshape_2d(ctx0, ctx.inp.state, hp.max_state_dim, 1), m.state_proj_w, m.state_proj_b);
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
