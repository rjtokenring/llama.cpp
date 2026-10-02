#include "vla.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// runs a policy over recorded samples:
//   llama-vla-cli -m policy.gguf --samples file.bin [--noise noise.f32] [--dump dir] [-t N] [--device NAME] [-fa]
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

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m policy.gguf --samples file.bin [--noise noise.f32] [--dump dir [--dump-names a,b]] [-t N] [--device NAME] [-fa]\n", argv0);
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, samples_path, noise_path;
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
        } else if (a == "--samples") {
            samples_path = next();
        } else if (a == "--noise") {
            noise_path = next();
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
        } else {
            usage(argv[0]);
        }
    }
    if (model_path.empty() || samples_path.empty()) {
        usage(argv[0]);
    }
    if (!dump.dir.empty()) {
        params.cb_eval           = dump_cb;
        params.cb_eval_user_data = &dump;
    }

    ggml_backend_load_all();
    vla_context * ctx = vla_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        return 1;
    }
    const vla_info info = vla_get_info(ctx);

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

    std::vector<float> noise;
    if (!noise_path.empty()) {
        noise.resize((size_t) info.chunk_size * info.max_action_dim);
        std::ifstream fn(noise_path, std::ios::binary);
        fn.read((char *) noise.data(), noise.size() * sizeof(float));
        if (!fn) {
            fprintf(stderr, "%s: expected %zu floats\n", noise_path.c_str(), noise.size());
            return 1;
        }
    }

    std::vector<float> actions((size_t) info.chunk_size * info.action_dim);
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
        printf("sample %d: ok\n", s);
    }

    vla_free(ctx);
    return 0;
}
