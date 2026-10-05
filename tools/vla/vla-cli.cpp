#include "vla.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// predicts one action chunk from camera images, robot state and a task prompt, printed as a JSON array of steps:
//   llama-vla-cli -m policy.gguf --image cam0.png [--image cam1.png ...] --state 0.1,0.2,... -p "pick up the bowl" [--seed N]
// the cameras, their order and the state/action sizes come from the GGUF; --image name=path picks a camera by name
// or runs a policy over recorded samples:
//   llama-vla-cli -m policy.gguf --samples file.bin [--noise noise.f32] [--out actions.bin] [--dump dir] [-t N] [--device NAME] [-fa]
//
// actions file: i32 n_samples, i32 chunk_size * action_dim, then the f32 actions of each sample
//
// samples file, little-endian: "VLAR" i32 version=1, n_samples, n_images, width, height, state_dim, pixel_format
// then per sample: i32 n_tok, i32 tok[n_tok], f32 state[state_dim], images as u8 (pixel_format 0) or f32 (1) RGB

struct dump_state {
    std::string           dir;
    int                   sample = 0;
    std::set<std::string> names = { "img_tokens", "prefix_emb", "Kcur-0", "Vcur-0", "Kcur-last", "Vcur-last", "v_t-0", "actions_norm" };
};

static void write_npy(const std::string & path, const std::vector<float> & data, const ggml_tensor * t) {
    std::string shape;
    for (int i = ggml_n_dims(t) - 1; i >= 0; i--) {
        shape += std::to_string(t->ne[i]) + ", ";
    }
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shape + "), }";
    header.append(63 - (10 + header.size()) % 64, ' ');
    header += '\n';
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    const uint16_t hlen = (uint16_t) header.size();
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(header.data(), 1, header.size(), f);
    fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
}

static bool dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (ask) {
        return st->names.count(t->name) > 0;
    }
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        fprintf(stderr, "skipping %s: not a contiguous F32/F16 tensor\n", t->name);
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> data(ggml_nelements(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), data.data(), data.size());
    } else {
        memcpy(data.data(), raw.data(), raw.size());
    }
    write_npy(st->dir + "/" + std::to_string(st->sample) + "-" + t->name + ".npy", data, t);
    return true;
}

// the vocab load in libvla is chatty; show only warnings and errors unless -v
static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    const bool verbose = *(const bool *) user_data;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (verbose || last >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m policy.gguf (--image [name=]img ... --state v,v,... -p prompt [--seed N] | --samples file.bin)\n"
        "       [--noise noise.f32] [--out actions.bin] [--dump dir [--dump-names a,b]] [-t N] [--device NAME] [-fa] [-v]\n", argv0);
    exit(1);
}

static std::vector<float> parse_floats(const std::string & list) {
    std::vector<float> v;
    for (size_t p = 0, q; p < list.size(); p = q + 1) {
        q = list.find(',', p);
        q = q == std::string::npos ? list.size() : q;
        const std::string item = list.substr(p, q - p);
        char * end = nullptr;
        errno = 0;
        const float f = strtof(item.c_str(), &end);
        if (item.empty() || *end != '\0' || errno != 0) {
            fprintf(stderr, "bad number '%s'\n", item.c_str());
            exit(1);
        }
        v.push_back(f);
    }
    return v;
}

static bool read_noise(const std::string & path, const vla_info & info, std::vector<float> & noise) {
    if (path.empty()) {
        return true;
    }
    noise.resize((size_t) info.chunk_size * info.max_action_dim);
    std::ifstream fn(path, std::ios::binary);
    fn.read((char *) noise.data(), noise.size() * sizeof(float));
    if (!fn) {
        fprintf(stderr, "%s: expected %zu floats\n", path.c_str(), noise.size());
        return false;
    }
    return true;
}

static FILE * open_actions(const std::string & path, int32_t n_samples, size_t n_actions) {
    if (path.empty()) {
        return nullptr;
    }
    FILE * f = fopen(path.c_str(), "wb");
    const int32_t hdr[2] = { n_samples, (int32_t) n_actions };
    if (!f || fwrite(hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    return f;
}

// the input/output contract comes from the GGUF, the same for every policy
static void print_contract(const vla_context * ctx, const vla_info & info) {
    fprintf(stderr, "cameras:");
    for (int c = 0; c < info.n_cameras; c++) {
        fprintf(stderr, " %s", vla_camera_name(ctx, c));
    }
    fprintf(stderr, "\nstate:   %d values", info.state_dim);
    for (int i = 0; vla_state_name(ctx, i); i++) {
        fprintf(stderr, "%s%s", i ? ", " : " (", vla_state_name(ctx, i));
    }
    fprintf(stderr, "%s\nactions: %d steps x %d values", vla_state_name(ctx, 0) ? ")" : "", info.chunk_size, info.action_dim);
    for (int i = 0; vla_action_name(ctx, i); i++) {
        fprintf(stderr, "%s%s", i ? ", " : " (", vla_action_name(ctx, i));
    }
    fprintf(stderr, "%s\n", vla_action_name(ctx, 0) ? ")" : "");
}

// --image takes the cameras in order, or as name=path
static bool order_images(const vla_context * ctx, const vla_info & info, const std::vector<std::string> & args,
        std::vector<std::string> & paths) {
    paths.assign(info.n_cameras, "");
    for (size_t i = 0; i < args.size(); i++) {
        const size_t eq = args[i].find('=');
        int cam = -1;
        for (int c = 0; eq != std::string::npos && c < info.n_cameras; c++) {
            if (args[i].compare(0, eq, vla_camera_name(ctx, c)) == 0) {
                cam = c;
            }
        }
        if (cam >= 0) {
            paths[cam] = args[i].substr(eq + 1);
        } else if (i < paths.size()) {
            paths[i] = args[i];
        }
    }
    for (const auto & p : paths) {
        if (p.empty() || args.size() != paths.size()) {
            return false;
        }
    }
    return true;
}

static int run_single(vla_context * ctx, const vla_info & info, const std::vector<std::string> & image_args,
        const std::vector<float> & state, const std::string & prompt, const std::vector<float> & noise, uint32_t seed,
        const std::string & out_path) {
    print_contract(ctx, info);
    std::vector<std::string> image_paths;
    if (!order_images(ctx, info, image_args, image_paths) || (int) state.size() != info.state_dim) {
        fprintf(stderr, "error: expected %d --image (in the order above, or name=path) and %d --state values\n", info.n_cameras, info.state_dim);
        return 1;
    }
    std::vector<vla_image> images(info.n_cameras);
    std::vector<stbi_uc *> pixels(info.n_cameras, nullptr);
    for (int c = 0; c < info.n_cameras; c++) {
        int nx = 0, ny = 0, nc = 0;
        pixels[c] = stbi_load(image_paths[c].c_str(), &nx, &ny, &nc, 3);
        if (!pixels[c]) {
            fprintf(stderr, "%s: %s\n", image_paths[c].c_str(), stbi_failure_reason());
            return 1;
        }
        images[c] = { nx, ny, VLA_IMAGE_RGB_U8, pixels[c] };
    }

    std::vector<int32_t> tokens(info.max_lang_tokens);
    const int32_t n_tok = vla_tokenize(ctx, prompt.c_str(), tokens.data(), tokens.size());
    if (n_tok < 1) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        return 1;
    }

    vla_input in = {};
    in.images   = images.data();
    in.n_images = info.n_cameras;
    in.state    = state.data();
    in.n_state  = info.state_dim;
    in.tokens   = tokens.data();
    in.n_tokens = n_tok;
    in.noise    = noise.empty() ? nullptr : noise.data();
    in.seed     = seed;

    std::vector<float> actions((size_t) info.chunk_size * info.action_dim);
    const int32_t ret = vla_predict(ctx, &in, actions.data());
    for (stbi_uc * p : pixels) {
        stbi_image_free(p);
    }
    if (ret != 0) {
        fprintf(stderr, "predict failed\n");
        return 1;
    }
    // one array per step, not normalized
    printf("[\n");
    for (int i = 0; i < info.chunk_size; i++) {
        printf("  [");
        for (int j = 0; j < info.action_dim; j++) {
            printf("%s%.6f", j ? ", " : "", actions[(size_t) i * info.action_dim + j]);
        }
        printf("]%s\n", i + 1 < info.chunk_size ? "," : "");
    }
    printf("]\n");
    if (FILE * f = open_actions(out_path, 1, actions.size())) {
        fwrite(actions.data(), sizeof(float), actions.size(), f);
        fclose(f);
    }
    return 0;
}

static int run_samples(vla_context * ctx, const vla_info & info, const std::string & samples_path,
        const std::vector<float> & noise, const std::string & out_path, dump_state & dump) {
    std::ifstream fin(samples_path, std::ios::binary);
    char magic[4];
    int32_t hdr[7];
    fin.read(magic, 4);
    fin.read((char *) hdr, sizeof(hdr));
    if (!fin || memcmp(magic, "VLAR", 4) != 0 || hdr[0] != 1 || hdr[2] != info.n_cameras || hdr[5] != info.state_dim
            || hdr[3] <= 0 || hdr[4] <= 0 || hdr[3] > 8192 || hdr[4] > 8192 || (hdr[6] != 0 && hdr[6] != 1)) {
        fprintf(stderr, "%s: not a replay file for this policy\n", samples_path.c_str());
        return 1;
    }
    const int32_t n_samples = hdr[1], nx = hdr[3], ny = hdr[4];
    const size_t px_bytes = (size_t) nx * ny * 3 * (hdr[6] ? sizeof(float) : 1);

    std::vector<float> actions((size_t) info.chunk_size * info.action_dim);
    FILE * fout = open_actions(out_path, n_samples, actions.size());
    for (int32_t s = 0; s < n_samples; s++) {
        int32_t n_tok = 0;
        fin.read((char *) &n_tok, sizeof(n_tok));
        if (!fin || n_tok < 1 || n_tok > info.max_lang_tokens) {
            fprintf(stderr, "sample %d: bad token count\n", s);
            return 1;
        }
        std::vector<int32_t> tokens(n_tok);
        std::vector<float> state(info.state_dim);
        std::vector<std::vector<uint8_t>> pixels(info.n_cameras, std::vector<uint8_t>(px_bytes));
        fin.read((char *) tokens.data(), n_tok * sizeof(int32_t));
        fin.read((char *) state.data(), state.size() * sizeof(float));
        std::vector<vla_image> images(info.n_cameras);
        for (int c = 0; c < info.n_cameras; c++) {
            fin.read((char *) pixels[c].data(), px_bytes);
            images[c] = { nx, ny, hdr[6] ? VLA_IMAGE_RGB_F32 : VLA_IMAGE_RGB_U8, pixels[c].data() };
        }
        if (!fin) {
            fprintf(stderr, "sample %d: truncated\n", s);
            return 1;
        }

        vla_input in = {};
        in.images   = images.data();
        in.n_images = info.n_cameras;
        in.state    = state.data();
        in.n_state  = info.state_dim;
        in.tokens   = tokens.data();
        in.n_tokens = n_tok;
        in.noise    = noise.empty() ? nullptr : noise.data();
        dump.sample = s;
        if (vla_predict(ctx, &in, actions.data()) != 0) {
            fprintf(stderr, "sample %d: predict failed\n", s);
            return 1;
        }
        printf("sample %d: actions[0] =", s);
        for (int j = 0; j < info.action_dim; j++) {
            printf(" %.4f", actions[j]);
        }
        printf("\n");
        if (fout) {
            fwrite(actions.data(), sizeof(float), actions.size(), fout);
        }
    }
    if (fout) {
        fclose(fout);
    }
    return 0;
}

int main(int argc, char ** argv) {
    std::string model_path, samples_path, noise_path, out_path, prompt;
    std::vector<std::string> image_paths;
    std::vector<float> state;
    uint32_t seed = 0;
    bool verbose = false;
    dump_state dump;
    vla_context_params params = vla_context_default_params();

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = next();
        } else if (a == "--image") {
            image_paths.push_back(next());
        } else if (a == "--state") {
            state = parse_floats(next());
        } else if (a == "-p") {
            prompt = next();
        } else if (a == "--seed") {
            seed = (uint32_t) strtoul(next(), nullptr, 10);
        } else if (a == "--samples") {
            samples_path = next();
        } else if (a == "--noise") {
            noise_path = next();
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "--dump") {
            dump.dir = next();
        } else if (a == "--dump-names") {
            dump.names.clear();
            std::string list = next();
            for (size_t p = 0, q; p <= list.size(); p = q + 1) {
                q = list.find(',', p);
                q = q == std::string::npos ? list.size() : q;
                dump.names.insert(list.substr(p, q - p));
            }
        } else if (a == "-t") {
            params.n_threads = atoi(next());
        } else if (a == "--device") {
            params.device = next();
        } else if (a == "-fa") {
            params.flash_attn = true;
        } else if (a == "-v") {
            verbose = true;
        } else {
            usage(argv[0]);
        }
    }
    if (model_path.empty() || samples_path.empty() == image_paths.empty()) {
        usage(argv[0]);
    }
    if (!dump.dir.empty()) {
        params.cb_eval           = dump_cb;
        params.cb_eval_user_data = &dump;
    }

    llama_log_set(log_cb, &verbose);
    ggml_backend_load_all();
    vla_context * ctx = vla_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        return 1;
    }
    const vla_info info = vla_get_info(ctx);

    std::vector<float> noise;
    int ret = 1;
    if (read_noise(noise_path, info, noise)) {
        ret = samples_path.empty() ? run_single(ctx, info, image_paths, state, prompt, noise, seed, out_path)
                                   : run_samples(ctx, info, samples_path, noise, out_path, dump);
    }
    vla_free(ctx);
    return ret;
}
