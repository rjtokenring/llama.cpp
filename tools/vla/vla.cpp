#include "vla-impl.h"

#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>

#define VLA_LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define VLA_LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

template <typename... Args>
static std::string format(const char * fmt, Args... args) {
    char buf[512];
    snprintf(buf, sizeof(buf), fmt, args...);
    return buf;
}

//
// loader
//

struct vla_loader {
    gguf_context * gguf;
    ggml_context * meta;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> to_load; // (meta, dst)

    int64_t key(const char * name) const {
        const int64_t i = gguf_find_key(gguf, name);
        if (i < 0) {
            throw std::runtime_error(format("missing key %s", name));
        }
        return i;
    }

    int32_t get_i32(const char * name, int32_t max_val = 1 << 20) const {
        const int64_t i = key(name);
        if (gguf_get_kv_type(gguf, i) != GGUF_TYPE_UINT32) {
            throw std::runtime_error(format("key %s is not a uint32", name));
        }
        const uint32_t v = gguf_get_val_u32(gguf, i);
        if (v == 0 || v > (uint32_t) max_val) {
            throw std::runtime_error(format("key %s = %u out of range", name, v));
        }
        return (int32_t) v;
    }

    float get_f32(const char * name) const {
        const int64_t i = key(name);
        if (gguf_get_kv_type(gguf, i) != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(format("key %s is not a float32", name));
        }
        const float v = gguf_get_val_f32(gguf, i);
        if (!std::isfinite(v)) {
            throw std::runtime_error(format("key %s is not finite", name));
        }
        return v;
    }

    // returns an empty vector for a missing optional key
    std::vector<std::string> get_str_arr(const char * name, bool required, size_t max_n) const {
        const int64_t i = gguf_find_key(gguf, name);
        if (i < 0) {
            if (required) {
                throw std::runtime_error(format("missing key %s", name));
            }
            return {};
        }
        if (gguf_get_kv_type(gguf, i) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, i) != GGUF_TYPE_STRING) {
            throw std::runtime_error(format("key %s is not a string array", name));
        }
        const size_t n = gguf_get_arr_n(gguf, i);
        if (n == 0 || n > max_n) {
            throw std::runtime_error(format("key %s has %zu values", name, n));
        }
        std::vector<std::string> v;
        for (size_t j = 0; j < n; j++) {
            const char * s = gguf_get_arr_str(gguf, i, j);
            if (strlen(s) > 256) {
                throw std::runtime_error(format("key %s: name too long", name));
            }
            v.push_back(s);
        }
        return v;
    }

    std::string get_str(const char * name) const {
        const int64_t i = key(name);
        if (gguf_get_kv_type(gguf, i) != GGUF_TYPE_STRING) {
            throw std::runtime_error(format("key %s is not a string", name));
        }
        return gguf_get_val_str(gguf, i);
    }

    ggml_tensor * get(ggml_context * dst_ctx, const std::string & name, std::initializer_list<int64_t> ne) {
        ggml_tensor * cur = ggml_get_tensor(meta, name.c_str());
        if (!cur) {
            throw std::runtime_error("missing tensor " + name);
        }
        int i = 0;
        for (int64_t n : ne) {
            if (cur->ne[i] != n) {
                throw std::runtime_error(format("tensor %s: ne[%d] = %lld, expected %lld", name.c_str(), i, (long long) cur->ne[i], (long long) n));
            }
            i++;
        }
        for (; i < GGML_MAX_DIMS; i++) {
            if (cur->ne[i] != 1) {
                throw std::runtime_error(format("tensor %s has too many dims", name.c_str()));
            }
        }
        ggml_tensor * dst = ggml_dup_tensor(dst_ctx, cur);
        ggml_set_name(dst, cur->name);
        to_load.push_back({cur, dst});
        return dst;
    }

    std::vector<float> get_f32_vec(const std::string & name, int64_t n, std::ifstream & fin) const {
        ggml_tensor * cur = ggml_get_tensor(meta, name.c_str());
        if (!cur || cur->type != GGML_TYPE_F32 || ggml_nelements(cur) != n || cur->ne[0] != n) {
            throw std::runtime_error("missing or malformed tensor " + name);
        }
        std::vector<float> v(n);
        fin.seekg(offset(name), std::ios::beg);
        fin.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
        if (!fin) {
            throw std::runtime_error("failed to read " + name);
        }
        return v;
    }

    size_t offset(const std::string & name) const {
        const int64_t i = gguf_find_tensor(gguf, name.c_str());
        GGML_ASSERT(i >= 0);
        return gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
    }
};

static void vla_load_hparams(vla_loader & ml, vla_hparams & hp) {
    hp.policy_type = ml.get_str("vla.policy_type");
    if (hp.policy_type != "smolvla") {
        throw std::runtime_error("unsupported policy type " + hp.policy_type);
    }

    hp.chunk_size      = ml.get_i32("vla.chunk_size", 4096);
    hp.num_steps       = ml.get_i32("vla.num_steps", 1000);
    hp.state_dim       = ml.get_i32("vla.state_dim", 4096);
    hp.action_dim      = ml.get_i32("vla.action_dim", 4096);
    hp.max_state_dim   = ml.get_i32("vla.max_state_dim", 4096);
    hp.max_action_dim  = ml.get_i32("vla.max_action_dim", 4096);
    hp.max_lang_tokens = ml.get_i32("vla.max_lang_tokens", 4096);
    hp.camera_names    = ml.get_str_arr("vla.camera_names", true, 16);
    hp.n_cameras       = (int32_t) hp.camera_names.size();
    hp.time_min_period = ml.get_f32("vla.time_min_period");
    hp.time_max_period = ml.get_f32("vla.time_max_period");
    hp.norm_eps        = ml.get_f32("vla.norm_eps");

    hp.v_image_size   = ml.get_i32("vla.vision.image_size", 4096);
    hp.v_patch_size   = ml.get_i32("vla.vision.patch_size", 256);
    hp.v_n_embd       = ml.get_i32("vla.vision.embedding_length");
    hp.v_n_ff         = ml.get_i32("vla.vision.feed_forward_length");
    hp.v_n_layer      = ml.get_i32("vla.vision.block_count", 256);
    hp.v_n_head       = ml.get_i32("vla.vision.head_count", 256);
    hp.v_scale_factor = ml.get_i32("vla.vision.scale_factor", 64);
    hp.v_eps          = ml.get_f32("vla.vision.layer_norm_epsilon");

    hp.t_n_embd    = ml.get_i32("vla.text.embedding_length");
    hp.t_n_ff      = ml.get_i32("vla.text.feed_forward_length");
    hp.t_n_layer   = ml.get_i32("vla.text.block_count", 256);
    hp.t_n_head    = ml.get_i32("vla.text.head_count", 256);
    hp.t_n_head_kv = ml.get_i32("vla.text.head_count_kv", 256);
    hp.t_head_dim  = ml.get_i32("vla.text.key_length", 1024);
    hp.t_eps       = ml.get_f32("vla.text.layer_norm_rms_epsilon");
    hp.t_rope_base = ml.get_f32("vla.text.rope.freq_base");

    hp.e_n_embd            = ml.get_i32("vla.expert.embedding_length");
    hp.e_n_ff              = ml.get_i32("vla.expert.feed_forward_length");
    hp.e_n_layer           = ml.get_i32("vla.expert.block_count", 256);
    hp.e_self_attn_every_n = ml.get_i32("vla.expert.self_attn_every_n_layers", 256);

    hp.state_names  = ml.get_str_arr("vla.state_names", false, 4096);
    hp.action_names = ml.get_str_arr("vla.action_names", false, 4096);
    if ((!hp.state_names.empty() && (int32_t) hp.state_names.size() != hp.state_dim)
            || (!hp.action_names.empty() && (int32_t) hp.action_names.size() != hp.action_dim)) {
        throw std::runtime_error("state or action names do not match their size");
    }

    const int32_t grid = hp.v_image_size / hp.v_patch_size;
    if (hp.v_image_size % hp.v_patch_size != 0 || grid % hp.v_scale_factor != 0 || hp.v_n_embd % hp.v_n_head != 0) {
        throw std::runtime_error("inconsistent vision hparams");
    }
    if (hp.t_n_head % hp.t_n_head_kv != 0 || hp.t_head_dim % 2 != 0 || hp.e_n_embd % 2 != 0 || hp.e_n_layer != hp.t_n_layer || hp.state_dim > hp.max_state_dim
            || hp.action_dim > hp.max_action_dim) {
        throw std::runtime_error("inconsistent policy hparams");
    }
}

static void vla_load_tensors(vla_loader & ml, vla_context & ctx) {
    vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    ggml_context * cd = ctx.ctx_w_dev.get();

    const int64_t vd = hp.v_n_embd;
    const int64_t P  = hp.v_patch_size;
    m.patch_w   = ml.get(cd, "v.patch_embd.weight", {P, P, 3, vd});
    m.patch_b   = ml.get(cd, "v.patch_embd.bias", {vd});
    m.pos_embd  = ml.get(cd, "v.position_embd.weight", {vd, hp.n_patches()});
    m.post_ln_w = ml.get(cd, "v.post_ln.weight", {vd});
    m.post_ln_b = ml.get(cd, "v.post_ln.bias", {vd});
    m.vit.resize(hp.v_n_layer);
    for (int il = 0; il < hp.v_n_layer; il++) {
        auto & l = m.vit[il];
        auto name = [&](const char * t) { return format("v.blk.%d.%s", il, t); };
        l.ln1_w  = ml.get(cd, name("ln1.weight"), {vd});
        l.ln1_b  = ml.get(cd, name("ln1.bias"), {vd});
        l.q_w    = ml.get(cd, name("attn_q.weight"), {vd, vd});
        l.q_b    = ml.get(cd, name("attn_q.bias"), {vd});
        l.k_w    = ml.get(cd, name("attn_k.weight"), {vd, vd});
        l.k_b    = ml.get(cd, name("attn_k.bias"), {vd});
        l.v_w    = ml.get(cd, name("attn_v.weight"), {vd, vd});
        l.v_b    = ml.get(cd, name("attn_v.bias"), {vd});
        l.o_w    = ml.get(cd, name("attn_out.weight"), {vd, vd});
        l.o_b    = ml.get(cd, name("attn_out.bias"), {vd});
        l.ln2_w  = ml.get(cd, name("ln2.weight"), {vd});
        l.ln2_b  = ml.get(cd, name("ln2.bias"), {vd});
        l.up_w   = ml.get(cd, name("ffn_up.weight"), {vd, hp.v_n_ff});
        l.up_b   = ml.get(cd, name("ffn_up.bias"), {hp.v_n_ff});
        l.down_w = ml.get(cd, name("ffn_down.weight"), {hp.v_n_ff, vd});
        l.down_b = ml.get(cd, name("ffn_down.bias"), {vd});
    }
    m.mm_fc = ml.get(cd, "mm.model.fc.weight", {vd * hp.v_scale_factor * hp.v_scale_factor, hp.t_n_embd});

    const int64_t td = hp.t_n_embd;
    const int64_t hd = hp.t_head_dim;
    ggml_tensor * tok_meta = ggml_get_tensor(ml.meta, "token_embd.weight");
    if (!tok_meta || tok_meta->ne[1] < 1 || tok_meta->ne[1] > (1 << 24)) {
        throw std::runtime_error("missing or malformed token_embd.weight");
    }
    m.n_vocab  = (int32_t) tok_meta->ne[1];
    // only read by GET_ROWS, keep it on the host
    m.tok_embd = ml.get(ctx.ctx_w_cpu.get(), "token_embd.weight", {td, m.n_vocab});
    m.state_proj_w = ml.get(cd, "state_proj.weight", {hp.max_state_dim, td});
    m.state_proj_b = ml.get(cd, "state_proj.bias", {td});
    m.vlm.resize(hp.t_n_layer);
    for (int il = 0; il < hp.t_n_layer; il++) {
        auto & l = m.vlm[il];
        auto name = [&](const char * t) { return format("blk.%d.%s", il, t); };
        l.attn_norm = ml.get(cd, name("attn_norm.weight"), {td});
        l.q         = ml.get(cd, name("attn_q.weight"), {td, hp.t_n_head * hd});
        l.k         = ml.get(cd, name("attn_k.weight"), {td, hp.t_n_head_kv * hd});
        l.v         = ml.get(cd, name("attn_v.weight"), {td, hp.t_n_head_kv * hd});
        l.o         = ml.get(cd, name("attn_output.weight"), {hp.t_n_head * hd, td});
        l.ffn_norm  = ml.get(cd, name("ffn_norm.weight"), {td});
        l.gate      = ml.get(cd, name("ffn_gate.weight"), {td, hp.t_n_ff});
        l.up        = ml.get(cd, name("ffn_up.weight"), {td, hp.t_n_ff});
        l.down      = ml.get(cd, name("ffn_down.weight"), {hp.t_n_ff, td});
    }

    const int64_t ed  = hp.e_n_embd;
    const int64_t kvd = hp.t_n_head_kv * hd;
    m.exp.resize(hp.e_n_layer);
    for (int il = 0; il < hp.e_n_layer; il++) {
        auto & l = m.exp[il];
        auto name = [&](const char * t) { return format("exp.blk.%d.%s", il, t); };
        // cross-attention layers project the VLM K/V, not the expert hidden state
        const int64_t kv_in = hp.expert_self_attn(il) ? ed : kvd;
        l.attn_norm = ml.get(cd, name("attn_norm.weight"), {ed});
        l.q         = ml.get(cd, name("attn_q.weight"), {ed, hp.t_n_head * hd});
        l.k         = ml.get(cd, name("attn_k.weight"), {kv_in, kvd});
        l.v         = ml.get(cd, name("attn_v.weight"), {kv_in, kvd});
        l.o         = ml.get(cd, name("attn_output.weight"), {hp.t_n_head * hd, ed});
        l.ffn_norm  = ml.get(cd, name("ffn_norm.weight"), {ed});
        l.gate      = ml.get(cd, name("ffn_gate.weight"), {ed, hp.e_n_ff});
        l.up        = ml.get(cd, name("ffn_up.weight"), {ed, hp.e_n_ff});
        l.down      = ml.get(cd, name("ffn_down.weight"), {hp.e_n_ff, ed});
    }
    m.exp_out_norm = ml.get(cd, "exp.output_norm.weight", {ed});

    m.act_in_w   = ml.get(cd, "act.in_proj.weight", {hp.max_action_dim, ed});
    m.act_in_b   = ml.get(cd, "act.in_proj.bias", {ed});
    m.act_out_w  = ml.get(cd, "act.out_proj.weight", {ed, hp.max_action_dim});
    m.act_out_b  = ml.get(cd, "act.out_proj.bias", {hp.max_action_dim});
    m.time_in_w  = ml.get(cd, "act.time_mlp_in.weight", {2 * ed, ed});
    m.time_in_b  = ml.get(cd, "act.time_mlp_in.bias", {ed});
    m.time_out_w = ml.get(cd, "act.time_mlp_out.weight", {ed, ed});
    m.time_out_b = ml.get(cd, "act.time_mlp_out.bias", {ed});
}

//
// API
//

vla_context_params vla_context_default_params() {
    vla_context_params p = {};
    p.n_threads  = 4;
    p.device     = nullptr;
    p.flash_attn = false;
    return p;
}

static void vla_init_backends(vla_context & ctx) {
    ctx.backend_cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!ctx.backend_cpu) {
        throw std::runtime_error("failed to init the CPU backend");
    }
    ggml_backend_cpu_set_n_threads(ctx.backend_cpu.get(), ctx.params.n_threads);

    const char * dev_name = ctx.params.device;
    if (!dev_name || strcmp(dev_name, "none") != 0) {
        ggml_backend_dev_t dev = nullptr;
        if (dev_name) {
            dev = ggml_backend_dev_by_name(dev_name);
            if (!dev) {
                throw std::runtime_error(format("unknown device %s", dev_name));
            }
        } else {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (dev) {
            ctx.backend_dev.reset(ggml_backend_dev_init(dev, nullptr));
            if (!ctx.backend_dev) {
                throw std::runtime_error(format("failed to init device %s", ggml_backend_dev_name(dev)));
            }
        }
    }

    if (ctx.backend_dev) {
        ctx.backends.push_back(ctx.backend_dev.get());
    }
    ctx.backends.push_back(ctx.backend_cpu.get());
    VLA_LOG_INF("%s: using %s backend\n", __func__, ggml_backend_name(ctx.backends[0]));
}

static size_t vla_graph_size(const vla_hparams & hp) {
    return 64 * (size_t) (hp.v_n_layer + hp.t_n_layer + hp.num_steps * hp.e_n_layer) + 1024;
}

static void vla_load(vla_context & ctx, const char * path) {
    ggml_context * meta = nullptr;
    gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    std::unique_ptr<gguf_context, vla_gguf_ctx_deleter> gguf(gguf_init_from_file(path, gparams));
    if (!gguf) {
        throw std::runtime_error(format("failed to open %s", path));
    }
    vla_ggml_ctx_ptr meta_ptr(meta);

    const int64_t gi = gguf_find_key(gguf.get(), "general.architecture");
    if (gi < 0 || gguf_get_kv_type(gguf.get(), gi) != GGUF_TYPE_STRING || strcmp(gguf_get_val_str(gguf.get(), gi), "vla") != 0) {
        throw std::runtime_error(format("%s is not a vla GGUF", path));
    }

    vla_loader ml = { gguf.get(), meta, {} };
    vla_load_hparams(ml, ctx.model.hparams);

    const size_t n_tensors = gguf_get_n_tensors(gguf.get());
    ggml_init_params params = { (n_tensors + 1) * ggml_tensor_overhead(), nullptr, true };
    ctx.ctx_w_dev.reset(ggml_init(params));
    ctx.ctx_w_cpu.reset(ggml_init(params));

    vla_load_tensors(ml, ctx);

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error(format("failed to open %s", path));
    }

    ggml_backend_buffer_type_t buft_dev = ggml_backend_get_default_buffer_type(ctx.backends[0]);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_get_default_buffer_type(ctx.backend_cpu.get());
    for (auto [c, buft] : { std::make_pair(ctx.ctx_w_dev.get(), buft_dev), std::make_pair(ctx.ctx_w_cpu.get(), buft_cpu) }) {
        if (ggml_get_first_tensor(c) == nullptr) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate the weights");
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ctx.bufs.emplace_back(buf);
    }

    std::vector<uint8_t> read_buf;
    for (auto & [src, dst] : ml.to_load) {
        const size_t n = ggml_nbytes(dst);
        fin.seekg(ml.offset(src->name), std::ios::beg);
        read_buf.resize(n);
        fin.read(reinterpret_cast<char *>(read_buf.data()), n);
        if (!fin) {
            throw std::runtime_error(format("failed to read tensor %s", src->name));
        }
        ggml_backend_tensor_set(dst, read_buf.data(), 0, n);
    }

    vla_model & m = ctx.model;
    m.state_mean  = ml.get_f32_vec("norm.state_mean", m.hparams.state_dim, fin);
    m.state_std   = ml.get_f32_vec("norm.state_std", m.hparams.state_dim, fin);
    m.action_mean = ml.get_f32_vec("norm.action_mean", m.hparams.action_dim, fin);
    m.action_std  = ml.get_f32_vec("norm.action_std", m.hparams.action_dim, fin);

    ctx.n_graph = vla_graph_size(ctx.model.hparams);
    ctx.compute_meta.resize(ggml_tensor_overhead() * ctx.n_graph + ggml_graph_overhead_custom(ctx.n_graph, false));
    ctx.sched.reset(ggml_backend_sched_new(ctx.backends.data(), nullptr, ctx.backends.size(), ctx.n_graph, false, true));
    if (ctx.params.cb_eval) {
        ggml_backend_sched_set_eval_callback(ctx.sched.get(), ctx.params.cb_eval, ctx.params.cb_eval_user_data);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    ctx.vocab_model = llama_model_load_from_file(path, mparams);
    if (!ctx.vocab_model) {
        throw std::runtime_error("failed to load the vocab");
    }

    VLA_LOG_INF("%s: %s, %zu tensors, %d cameras %dx%d, chunk %d x %d\n", __func__, ctx.model.hparams.policy_type.c_str(),
        ml.to_load.size(), ctx.model.hparams.n_cameras, ctx.model.hparams.v_image_size, ctx.model.hparams.v_image_size,
        ctx.model.hparams.chunk_size, ctx.model.hparams.action_dim);
}

vla_context * vla_init_from_file(const char * path, vla_context_params params) {
    auto ctx = std::make_unique<vla_context>();
    ctx->params = params;
    try {
        vla_init_backends(*ctx);
        vla_load(*ctx, path);
    } catch (const std::exception & e) {
        VLA_LOG_ERR("%s: %s\n", __func__, e.what());
        vla_free(ctx.release());
        return nullptr;
    }
    return ctx.release();
}

void vla_free(vla_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->vocab_model) {
        llama_model_free(ctx->vocab_model);
    }
    delete ctx;
}

vla_info vla_get_info(const vla_context * ctx) {
    const vla_hparams & hp = ctx->model.hparams;
    vla_info info = {};
    info.policy_type     = hp.policy_type.c_str();
    info.n_cameras       = hp.n_cameras;
    info.image_size      = hp.v_image_size;
    info.state_dim       = hp.state_dim;
    info.action_dim      = hp.action_dim;
    info.max_action_dim  = hp.max_action_dim;
    info.chunk_size      = hp.chunk_size;
    info.max_lang_tokens = hp.max_lang_tokens;
    info.num_steps       = hp.num_steps;
    return info;
}

static const char * vla_name(const std::vector<std::string> & names, int32_t i) {
    return i >= 0 && i < (int32_t) names.size() ? names[i].c_str() : nullptr;
}

const char * vla_camera_name(const vla_context * ctx, int32_t i) {
    return vla_name(ctx->model.hparams.camera_names, i);
}

const char * vla_state_name(const vla_context * ctx, int32_t i) {
    return vla_name(ctx->model.hparams.state_names, i);
}

const char * vla_action_name(const vla_context * ctx, int32_t i) {
    return vla_name(ctx->model.hparams.action_names, i);
}

int32_t vla_tokenize(const vla_context * ctx, const char * prompt, int32_t * tokens, int32_t n_tokens_max) {
    // LeRobot appends a newline to the SmolVLA task, then truncates to max_lang_tokens
    const std::string text = std::string(prompt) + "\n";
    const llama_vocab * vocab = llama_model_get_vocab(ctx->vocab_model);
    std::vector<llama_token> tmp(text.size() + 8);
    const int32_t n = llama_tokenize(vocab, text.c_str(), text.size(), tmp.data(), tmp.size(), true, false);
    if (n < 0) {
        return n;
    }
    const int32_t n_keep = std::min(n, ctx->model.hparams.max_lang_tokens);
    if (n_keep > n_tokens_max) {
        return -n_keep;
    }
    memcpy(tokens, tmp.data(), n_keep * sizeof(int32_t));
    return n_keep;
}

// the patch embedding conv has stride == kernel, so cut the patches here and let the graph do a plain matmul
static float vla_pixel(const vla_image & img, int64_t x, int64_t y, int ch) {
    const int64_t i = (y * img.nx + x) * 3 + ch;
    return img.format == VLA_IMAGE_RGB_U8 ? ((const uint8_t *) img.data)[i] / 255.0f : ((const float *) img.data)[i];
}

// LeRobot resize_with_pad for SmolVLA: keep the aspect ratio, bilinear with align_corners=False and no
// antialias, pad on the left and top with 0; returns S x S interleaved RGB in [0, 1]
static std::vector<float> vla_resize_with_pad(const vla_image & img, int64_t S) {
    std::vector<float> out(S * S * 3, 0.0f);
    if (img.nx == S && img.ny == S) {
        for (int64_t y = 0; y < S; y++) {
            for (int64_t x = 0; x < S; x++) {
                for (int ch = 0; ch < 3; ch++) {
                    out[(y * S + x) * 3 + ch] = vla_pixel(img, x, y, ch);
                }
            }
        }
        return out;
    }
    const double  ratio = std::max((double) img.nx / S, (double) img.ny / S);
    const int64_t rw    = (int64_t) (img.nx / ratio);
    const int64_t rh    = (int64_t) (img.ny / ratio);
    auto src = [](int64_t i, int64_t n_in, int64_t n_out, int64_t & i0, int64_t & i1, float & f) {
        const float s = std::max(((float) i + 0.5f) * n_in / n_out - 0.5f, 0.0f);
        i0 = std::min((int64_t) s, n_in - 1);
        i1 = std::min(i0 + 1, n_in - 1);
        f  = s - i0;
    };
    for (int64_t y = 0; y < rh; y++) {
        int64_t y0, y1;
        float fy;
        src(y, img.ny, rh, y0, y1, fy);
        for (int64_t x = 0; x < rw; x++) {
            int64_t x0, x1;
            float fx;
            src(x, img.nx, rw, x0, x1, fx);
            for (int ch = 0; ch < 3; ch++) {
                const float top = vla_pixel(img, x0, y0, ch) * (1 - fx) + vla_pixel(img, x1, y0, ch) * fx;
                const float bot = vla_pixel(img, x0, y1, ch) * (1 - fx) + vla_pixel(img, x1, y1, ch) * fx;
                out[((S - rh + y) * S + S - rw + x) * 3 + ch] = top * (1 - fy) + bot * fy;
            }
        }
    }
    return out;
}

static void vla_set_patches(const vla_context & ctx, const vla_input & in) {
    const vla_hparams & hp = ctx.model.hparams;
    const int64_t S = hp.v_image_size;
    const int64_t P = hp.v_patch_size;
    const int64_t G = S / P;
    std::vector<float> patches(S * S * 3 * hp.n_cameras);
    for (int c = 0; c < hp.n_cameras; c++) {
        const std::vector<float> img = vla_resize_with_pad(in.images[c], S);
        for (int64_t y = 0; y < S; y++) {
            for (int64_t x = 0; x < S; x++) {
                const int64_t patch = (y / P) * G + x / P;
                for (int ch = 0; ch < 3; ch++) {
                    // same element order as the conv kernel [P, P, 3, n_embd]; SigLIP takes pixels in [-1, 1]
                    patches[((c * G * G + patch) * 3 + ch) * P * P + (y % P) * P + x % P] = img[(y * S + x) * 3 + ch] * 2.0f - 1.0f;
                }
            }
        }
    }
    ggml_backend_tensor_set(ctx.inp.patches, patches.data(), 0, patches.size() * sizeof(float));
}

// prefix = [images, language padded to max_lang_tokens, state]
// LeRobot attention rule: images and language see each other, the state token sees everything, padding is never a key
static void vla_set_prefix(const vla_context & ctx, const vla_input & in) {
    const vla_model & m = ctx.model;
    const vla_hparams & hp = m.hparams;
    const int32_t n_img    = hp.n_img_tokens() * hp.n_cameras;
    const int32_t n_prefix = hp.n_prefix();

    std::vector<int32_t> tokens(hp.max_lang_tokens, 0);
    memcpy(tokens.data(), in.tokens, in.n_tokens * sizeof(int32_t));

    std::vector<float> state(hp.max_state_dim, 0.0f);
    for (int i = 0; i < hp.state_dim; i++) {
        state[i] = (in.state[i] - m.state_mean[i]) / (m.state_std[i] + hp.norm_eps);
    }

    std::vector<uint8_t> valid(n_prefix, 1);
    std::vector<int32_t> block(n_prefix, 0);
    for (int i = n_img + in.n_tokens; i < n_img + hp.max_lang_tokens; i++) {
        valid[i] = 0;
    }
    block[n_prefix - 1] = 1;

    std::vector<int32_t> pos(n_prefix);
    for (int i = 0, p = -1; i < n_prefix; i++) {
        p += valid[i];
        pos[i] = std::max(p, 0);
    }

    // padded queries are never read; let them see the valid keys so no row is all -inf
    std::vector<float> mask((size_t) n_prefix * n_prefix);
    for (int q = 0; q < n_prefix; q++) {
        for (int k = 0; k < n_prefix; k++) {
            mask[(size_t) q * n_prefix + k] = valid[k] && block[k] <= block[q] ? 0.0f : -INFINITY;
        }
    }

    ggml_backend_tensor_set(ctx.inp.tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ctx.inp.state, state.data(), 0, state.size() * sizeof(float));
    ggml_backend_tensor_set(ctx.inp.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ctx.inp.mask, mask.data(), 0, mask.size() * sizeof(float));
}

// suffix = chunk_size action tokens: causal among themselves, they see the valid prefix
static void vla_set_suffix(const vla_context & ctx, const vla_input & in) {
    const vla_hparams & hp = ctx.model.hparams;
    const int32_t n_img    = hp.n_img_tokens() * hp.n_cameras;
    const int32_t n_prefix = hp.n_prefix();
    const int32_t chunk    = hp.chunk_size;
    const int32_t n_valid  = n_img + in.n_tokens + 1;

    std::vector<float> noise((size_t) hp.max_action_dim * chunk);
    if (in.noise) {
        memcpy(noise.data(), in.noise, noise.size() * sizeof(float));
    } else {
        std::mt19937 rng(in.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float & v : noise) {
            v = dist(rng);
        }
    }

    // openpi sinusoidal embedding: float32 time, float64 sin/cos over log-spaced periods
    const int32_t half = hp.e_n_embd / 2;
    std::vector<float> time_emb((size_t) hp.e_n_embd * hp.num_steps);
    for (int s = 0; s < hp.num_steps; s++) {
        const double dt = -1.0 / hp.num_steps;
        const double t  = (float) (1.0 + s * dt);
        for (int i = 0; i < half; i++) {
            const double frac   = half > 1 ? (double) i / (half - 1) : 0.0;
            const double period = hp.time_min_period * std::pow((double) hp.time_max_period / hp.time_min_period, frac);
            const double x      = t * 2.0 * M_PI / period;
            time_emb[(size_t) s * hp.e_n_embd + i]        = (float) std::sin(x);
            time_emb[(size_t) s * hp.e_n_embd + half + i] = (float) std::cos(x);
        }
    }

    std::vector<int32_t> pos_self(chunk), pos_cross(chunk);
    for (int i = 0; i < chunk; i++) {
        pos_self[i]  = n_valid + i;
        pos_cross[i] = i;
    }

    std::vector<float> mask_self((size_t) (n_prefix + chunk) * chunk);
    std::vector<float> mask_cross((size_t) n_prefix * chunk);
    for (int q = 0; q < chunk; q++) {
        for (int k = 0; k < n_prefix; k++) {
            const bool valid = k < n_img + in.n_tokens || k == n_prefix - 1;
            mask_self [(size_t) q * (n_prefix + chunk) + k] = valid ? 0.0f : -INFINITY;
            mask_cross[(size_t) q * n_prefix + k]           = valid ? 0.0f : -INFINITY;
        }
        for (int k = 0; k < chunk; k++) {
            mask_self[(size_t) q * (n_prefix + chunk) + n_prefix + k] = k <= q ? 0.0f : -INFINITY;
        }
    }

    ggml_backend_tensor_set(ctx.inp.noise, noise.data(), 0, noise.size() * sizeof(float));
    ggml_backend_tensor_set(ctx.inp.time_emb, time_emb.data(), 0, time_emb.size() * sizeof(float));
    ggml_backend_tensor_set(ctx.inp.pos_self, pos_self.data(), 0, pos_self.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ctx.inp.pos_cross, pos_cross.data(), 0, pos_cross.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ctx.inp.mask_self, mask_self.data(), 0, mask_self.size() * sizeof(float));
    ggml_backend_tensor_set(ctx.inp.mask_cross, mask_cross.data(), 0, mask_cross.size() * sizeof(float));
}

int32_t vla_predict(vla_context * ctx, const vla_input * in, float * actions) {
    const vla_hparams & hp = ctx->model.hparams;
    if (in->n_images != hp.n_cameras || in->n_state != hp.state_dim || in->n_tokens < 1 || in->n_tokens > hp.max_lang_tokens) {
        VLA_LOG_ERR("%s: expected %d images, %d state values and 1..%d tokens\n", __func__, hp.n_cameras, hp.state_dim, hp.max_lang_tokens);
        return 1;
    }
    for (int i = 0; i < in->n_tokens; i++) {
        if (in->tokens[i] < 0 || in->tokens[i] >= ctx->model.n_vocab) {
            VLA_LOG_ERR("%s: token %d out of range\n", __func__, in->tokens[i]);
            return 1;
        }
    }
    for (int i = 0; i < in->n_images; i++) {
        const vla_image & img = in->images[i];
        if (!img.data || img.nx < 1 || img.ny < 1 || img.nx > 8192 || img.ny > 8192
                || (img.format != VLA_IMAGE_RGB_U8 && img.format != VLA_IMAGE_RGB_F32)) {
            VLA_LOG_ERR("%s: image %d: bad size or format\n", __func__, i);
            return 1;
        }
    }

    ggml_init_params params = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    vla_ggml_ctx_ptr ctx0(ggml_init(params));
    ggml_cgraph * gf = vla_build_smolvla(*ctx, ctx0.get());

    ggml_backend_sched_reset(ctx->sched.get());
    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {
        VLA_LOG_ERR("%s: failed to allocate the graph\n", __func__);
        return 2;
    }
    vla_set_patches(*ctx, *in);
    vla_set_prefix(*ctx, *in);
    vla_set_suffix(*ctx, *in);

    if (ggml_backend_sched_graph_compute(ctx->sched.get(), gf) != GGML_STATUS_SUCCESS) {
        VLA_LOG_ERR("%s: graph compute failed\n", __func__);
        return 2;
    }

    const vla_model & m = ctx->model;
    std::vector<float> out((size_t) hp.max_action_dim * hp.chunk_size);
    ggml_backend_tensor_get(ctx->inp.actions, out.data(), 0, out.size() * sizeof(float));
    for (int i = 0; i < hp.chunk_size; i++) {
        for (int j = 0; j < hp.action_dim; j++) {
            actions[i * hp.action_dim + j] = out[(size_t) i * hp.max_action_dim + j] * (m.action_std[j] + hp.norm_eps) + m.action_mean[j];
        }
    }
    return 0;
}
