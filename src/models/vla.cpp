#include "models.h"

// Stub so libllama can load the vocab of a VLA policy GGUF and llama-quantize can open it

[[noreturn]]
void llama_model_vla::load_arch_hparams(llama_model_loader &) {
    GGML_ABORT("VLA is a vocab/quant-only stub; load_arch_hparams should not be called");
}

[[noreturn]]
void llama_model_vla::load_arch_tensors(llama_model_loader &) {
    GGML_ABORT("VLA is a vocab/quant-only stub; load_arch_tensors should not be called");
}

[[noreturn]]
std::unique_ptr<llm_graph_context> llama_model_vla::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("VLA has no inference graph via llama_model dispatch; runtime lives in tools/vla");
}
