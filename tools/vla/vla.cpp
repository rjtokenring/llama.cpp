#include "vla-impl.h"

#include "ggml-cpu.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
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
    hp.n_cameras       = ml.get_i32("vla.n_cameras", 16);
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

    const int32_t grid = hp.v_image_size / hp.v_patch_size;
    if (hp.v_image_size % hp.v_patch_size != 0 || grid % hp.v_scale_factor != 0 || hp.v_n_embd % hp.v_n_head != 0) {
        throw std::runtime_error("inconsistent vision hparams");
    }
    if (hp.t_n_head % hp.t_n_head_kv != 0 || hp.e_n_layer != hp.t_n_layer || hp.state_dim > hp.max_state_dim
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
static void vla_set_patches(const vla_context & ctx, const vla_input & in) {
    const vla_hparams & hp = ctx.model.hparams;
    const int64_t S = hp.v_image_size;
    const int64_t P = hp.v_patch_size;
    const int64_t G = S / P;
    std::vector<float> patches(S * S * 3 * hp.n_cameras);
    for (int c = 0; c < hp.n_cameras; c++) {
        const vla_image & img = in.images[c];
        for (int64_t y = 0; y < S; y++) {
            for (int64_t x = 0; x < S; x++) {
                const int64_t patch = (y / P) * G + x / P;
                for (int ch = 0; ch < 3; ch++) {
                    const int64_t i = (y * S + x) * 3 + ch;
                    const float v = img.format == VLA_IMAGE_RGB_U8 ? ((const uint8_t *) img.data)[i] / 255.0f
                                                                   : ((const float *) img.data)[i];
                    // same element order as the conv kernel [P, P, 3, n_embd]; SigLIP takes pixels in [-1, 1]
                    patches[((c * G * G + patch) * 3 + ch) * P * P + (y % P) * P + x % P] = v * 2.0f - 1.0f;
                }
            }
        }
    }
    ggml_backend_tensor_set(ctx.inp.patches, patches.data(), 0, patches.size() * sizeof(float));
}

int32_t vla_predict(vla_context * ctx, const vla_input * in, float * actions) {
    const vla_hparams & hp = ctx->model.hparams;
    if (in->n_images != hp.n_cameras || in->n_state != hp.state_dim || in->n_tokens < 1 || in->n_tokens > hp.max_lang_tokens) {
        VLA_LOG_ERR("%s: expected %d images, %d state values and 1..%d tokens\n", __func__, hp.n_cameras, hp.state_dim, hp.max_lang_tokens);
        return 1;
    }
    for (int i = 0; i < in->n_images; i++) {
        if (in->images[i].nx != hp.v_image_size || in->images[i].ny != hp.v_image_size) {
            VLA_LOG_ERR("%s: images must be %dx%d\n", __func__, hp.v_image_size, hp.v_image_size);
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

    if (ggml_backend_sched_graph_compute(ctx->sched.get(), gf) != GGML_STATUS_SUCCESS) {
        VLA_LOG_ERR("%s: graph compute failed\n", __func__);
        return 2;
    }
    GGML_UNUSED(actions);
    return 0;
}
