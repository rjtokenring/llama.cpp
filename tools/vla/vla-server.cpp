#include "vla.h"

#include "base64.hpp"
#include "json.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// HTTP server for a vision-language-action policy:
//   llama-vla-server -m policy.gguf [--host 127.0.0.1] [--port 8080] [--api-key KEY] [-t N] [--device NAME] [-fa]
//
//   GET  /health
//   GET  /v1/info    the policy contract: cameras, state and action sizes
//   POST /v1/act     {"images": {camera: image} or [image, ...], "state": [...], "prompt": "..." or "tokens": [...],
//                     "seed": N, "noise": [...]} -> {"actions": [[...], ...], ...}
//   an image is a base64 PNG/JPEG (a data: URI prefix is accepted) or
//   {"width": W, "height": H, "format": "rgb_u8" | "rgb_f32", "data": base64 of the interleaved RGB pixels}

struct bad_request : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct server_image {
    std::vector<uint8_t> pixels;
    vla_image            view = {};
};

static std::string error_json(int code, const std::string & msg) {
    common_json err = common_json::object();
    err["code"]    = code;
    err["message"] = msg;
    err["type"]    = code == 401 ? "authentication_error" : code < 500 ? "invalid_request_error" : "server_error";
    common_json res = common_json::object();
    res["error"] = err;
    return res.dump_safe();
}

static std::string b64_decode(const std::string & s) {
    const size_t comma = s.rfind(',');
    const bool   is_uri = s.compare(0, 5, "data:") == 0 && comma != std::string::npos;
    try {
        return base64::decode(is_uri ? s.substr(comma + 1) : s);
    } catch (const std::exception &) {
        throw bad_request("invalid base64");
    }
}

static void decode_image(const common_json & j, server_image & img) {
    if (j.is_string()) {
        const std::string bytes = b64_decode(j.get<std::string>());
        int nx = 0, ny = 0, nc = 0;
        stbi_uc * data = stbi_load_from_memory((const stbi_uc *) bytes.data(), (int) bytes.size(), &nx, &ny, &nc, 3);
        if (!data) {
            throw bad_request(std::string("cannot decode image: ") + stbi_failure_reason());
        }
        img.pixels.assign(data, data + (size_t) nx * ny * 3);
        stbi_image_free(data);
        img.view = { nx, ny, VLA_IMAGE_RGB_U8, nullptr };
    } else if (j.is_object()) {
        if (!j.contains("width") || !j.contains("height") || !j.at("width").is_number_integer() || !j.at("height").is_number_integer()) {
            throw bad_request("raw image needs integer width and height");
        }
        const int64_t nx = j.at("width").get<int64_t>();
        const int64_t ny = j.at("height").get<int64_t>();
        const std::string format = j.value("format", std::string("rgb_u8"));
        if (nx < 1 || ny < 1 || nx > 8192 || ny > 8192 || (format != "rgb_u8" && format != "rgb_f32")) {
            throw bad_request("raw image: width/height must be 1..8192 and format rgb_u8 or rgb_f32");
        }
        const size_t elt = format == "rgb_f32" ? sizeof(float) : 1;
        const std::string bytes = b64_decode(j.value("data", std::string()));
        if (bytes.size() != (size_t) nx * ny * 3 * elt) {
            throw bad_request("raw image: data size does not match width x height x 3");
        }
        img.pixels.assign(bytes.begin(), bytes.end());
        img.view = { (int32_t) nx, (int32_t) ny, elt == 1 ? VLA_IMAGE_RGB_U8 : VLA_IMAGE_RGB_F32, nullptr };
    } else {
        throw bad_request("an image is a base64 string or a raw image object");
    }
    img.view.data = img.pixels.data();
}

static std::vector<float> get_floats(const common_json & j, const char * name, size_t n) {
    if (!j.is_array() || j.size() != n) {
        throw bad_request(std::string(name) + " must be an array of " + std::to_string(n) + " numbers");
    }
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        if (!j.at(i).is_number()) {
            throw bad_request(std::string(name) + " must hold numbers");
        }
        v[i] = (float) j.at(i).get<double>();
        if (!std::isfinite(v[i])) {
            throw bad_request(std::string(name) + " must be finite");
        }
    }
    return v;
}

static common_json names(const char * (*get)(const vla_context *, int32_t), const vla_context * ctx, int32_t n) {
    common_json arr = common_json::array();
    for (int32_t i = 0; i < n && get(ctx, i); i++) {
        arr.push_back(get(ctx, i));
    }
    return arr;
}

static common_json info_json(const vla_context * ctx) {
    const vla_info info = vla_get_info(ctx);
    common_json res = common_json::object();
    res["policy_type"]     = info.policy_type;
    res["cameras"]         = names(vla_camera_name, ctx, info.n_cameras);
    res["image_size"]      = info.image_size;
    res["state_dim"]       = info.state_dim;
    res["action_dim"]      = info.action_dim;
    res["chunk_size"]      = info.chunk_size;
    res["num_steps"]       = info.num_steps;
    res["max_lang_tokens"] = info.max_lang_tokens;
    if (vla_state_name(ctx, 0)) {
        res["state_names"] = names(vla_state_name, ctx, info.state_dim);
    }
    if (vla_action_name(ctx, 0)) {
        res["action_names"] = names(vla_action_name, ctx, info.action_dim);
    }
    return res;
}

static std::string act(vla_context * ctx, std::mutex & mtx, const std::string & body) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const vla_info info = vla_get_info(ctx);

    common_json req = common_json::parse_no_throw(body);
    if (req.is_discarded()) {
        throw bad_request("invalid JSON (NaN and Infinity are not JSON)");
    }
    if (!req.is_object()) {
        throw bad_request("body must be a JSON object");
    }

    // images by camera name or in camera order
    if (!req.contains("images")) {
        throw bad_request("missing images");
    }
    const common_json & jimgs = req.at("images");
    std::vector<server_image> imgs(info.n_cameras);
    for (int c = 0; c < info.n_cameras; c++) {
        const char * cam = vla_camera_name(ctx, c);
        if (jimgs.is_object() && jimgs.contains(cam)) {
            decode_image(jimgs.at(cam), imgs[c]);
        } else if (jimgs.is_array() && jimgs.size() == (size_t) info.n_cameras) {
            decode_image(jimgs.at(c), imgs[c]);
        } else {
            throw bad_request(std::string("missing image for camera ") + cam);
        }
    }
    if (jimgs.size() != (size_t) info.n_cameras) {
        throw bad_request("expected " + std::to_string(info.n_cameras) + " images");
    }

    if (!req.contains("state")) {
        throw bad_request("missing state");
    }
    const std::vector<float> state = get_floats(req.at("state"), "state", info.state_dim);

    std::vector<int32_t> tokens(info.max_lang_tokens);
    int32_t n_tok = 0;
    if (req.contains("prompt") && req.at("prompt").is_string()) {
        const std::string prompt = req.at("prompt").get<std::string>();
        if (prompt.size() > 4096) {
            throw bad_request("prompt too long");
        }
        n_tok = vla_tokenize(ctx, prompt.c_str(), tokens.data(), tokens.size());
    } else if (req.contains("tokens") && req.at("tokens").is_array()) {
        const common_json & jt = req.at("tokens");
        if (jt.size() < 1 || jt.size() > tokens.size()) {
            throw bad_request("tokens: 1.." + std::to_string(tokens.size()) + " values");
        }
        for (size_t i = 0; i < jt.size(); i++) {
            if (!jt.at(i).is_number_integer()) {
                throw bad_request("tokens must be integers");
            }
            const int64_t t = jt.at(i).get<int64_t>();
            if (t < 0 || t > INT32_MAX) {
                throw bad_request("token out of range");
            }
            tokens[i] = (int32_t) t;
        }
        n_tok = (int32_t) jt.size();
    } else {
        throw bad_request("missing prompt (string) or tokens (array)");
    }
    if (n_tok < 1) {
        throw bad_request("empty prompt");
    }

    std::vector<float> noise;
    if (req.contains("noise")) {
        noise = get_floats(req.at("noise"), "noise", (size_t) info.chunk_size * info.max_action_dim);
    }
    uint32_t seed = 0;
    if (req.contains("seed")) {
        if (!req.at("seed").is_number_integer() || req.at("seed").get<int64_t>() < 0 || req.at("seed").get<int64_t>() > UINT32_MAX) {
            throw bad_request("seed must be an integer in 0..2^32-1");
        }
        seed = (uint32_t) req.at("seed").get<int64_t>();
    }

    std::vector<vla_image> views(info.n_cameras);
    for (int c = 0; c < info.n_cameras; c++) {
        views[c] = imgs[c].view;
    }
    vla_input in = {};
    in.images   = views.data();
    in.n_images = info.n_cameras;
    in.state    = state.data();
    in.n_state  = info.state_dim;
    in.tokens   = tokens.data();
    in.n_tokens = n_tok;
    in.noise    = noise.empty() ? nullptr : noise.data();
    in.seed     = seed;

    std::vector<float> actions((size_t) info.chunk_size * info.action_dim);
    const auto t1 = clock::now();
    int32_t ret;
    {
        std::lock_guard<std::mutex> lock(mtx);
        ret = vla_predict(ctx, &in, actions.data());
    }
    const auto t2 = clock::now();
    if (ret == 1) {
        throw bad_request("input rejected by the policy, see the server log");
    }
    if (ret != 0) {
        throw std::runtime_error("predict failed");
    }

    common_json rows = common_json::array();
    for (int i = 0; i < info.chunk_size; i++) {
        rows.push_back(std::vector<float>(actions.begin() + (size_t) i * info.action_dim, actions.begin() + (size_t) (i + 1) * info.action_dim));
    }
    auto ms = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    common_json timings = common_json::object();
    timings["decode_ms"]  = ms(t0, t1);
    timings["predict_ms"] = ms(t1, t2);
    timings["total_ms"]   = ms(t0, clock::now());

    common_json res = common_json::object();
    res["actions"]    = rows;
    res["chunk_size"] = info.chunk_size;
    res["action_dim"] = info.action_dim;
    res["timings"]    = timings;
    return res.dump_safe();
}

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m policy.gguf [--host 127.0.0.1] [--port 8080] [--api-key KEY] [-t N] [--device NAME] [-fa] [-v]\n", argv0);
    exit(1);
}

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

int main(int argc, char ** argv) {
    std::string model_path, host = "127.0.0.1", api_key;
    int port = 8080;
    bool verbose = false;
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
        } else if (a == "--host") {
            host = next();
        } else if (a == "--port") {
            port = atoi(next());
        } else if (a == "--api-key") {
            api_key = next();
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
    if (model_path.empty() || port < 1 || port > 65535) {
        usage(argv[0]);
    }

    llama_log_set(log_cb, &verbose);
    ggml_backend_load_all();
    vla_context * ctx = vla_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        return 1;
    }

    std::mutex mtx;
    httplib::Server svr;
    // two cameras of raw float images at 512x512 are about 8 MiB of base64
    svr.set_payload_max_length(64 * 1024 * 1024);
    svr.new_task_queue = [] { return new httplib::ThreadPool(4); };

    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (api_key.empty() || req.path == "/health") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        std::string key = req.get_header_value("Authorization");
        if (key.compare(0, 7, "Bearer ") == 0) {
            key = key.substr(7);
        }
        if (key == api_key) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        res.status = 401;
        res.set_content(error_json(401, "invalid API key"), "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/info", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(info_json(ctx).dump_safe(), "application/json");
    });
    svr.Post("/v1/act", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            res.set_content(act(ctx, mtx, req.body), "application/json");
        } catch (const bad_request & e) {
            res.status = 400;
            res.set_content(error_json(400, e.what()), "application/json");
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(error_json(500, e.what()), "application/json");
        }
    });

    fprintf(stderr, "%s: listening on http://%s:%d\n", __func__, host.c_str(), port);
    if (!svr.listen(host, port)) {
        fprintf(stderr, "%s: cannot listen on %s:%d\n", __func__, host.c_str(), port);
        vla_free(ctx);
        return 1;
    }
    vla_free(ctx);
    return 0;
}
