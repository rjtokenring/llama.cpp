from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import torch

from .base import ModelBase, TextModel, gguf, logger

# LeRobot vision-language-action policies, run by tools/vla.
# A LeRobot checkpoint has a "type" in config.json instead of "architectures", names its base VLM only
# by vlm_model_name, and keeps the state/action statistics in policy_{pre,post}processor safetensors.
# The GGUF holds all of it, plus the VLM tokenizer.

_LEROBOT_POLICIES = {"smolvla": "SmolVLAPolicy"}


def _lerobot_config(dir_model: Path) -> dict[str, Any] | None:
    path = dir_model / "config.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "architectures" in config or config.get("type") not in _LEROBOT_POLICIES:
        return None
    return config


def _count_layers(names: Iterable[str], pattern: str) -> int:
    ids = {int(m.group(1)) for name in names if (m := re.search(pattern, name))}
    return max(ids) + 1 if ids else 0


@ModelBase.register_hparams_loader(lambda dir_model: _lerobot_config(dir_model) is not None)
def _load_lerobot_hparams(dir_model: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    config = _lerobot_config(dir_model)
    assert config is not None
    # only config and tokenizer, the VLM weights are in the policy checkpoint
    vlm_dir = Path(snapshot_download(config["vlm_model_name"], allow_patterns=[
        "config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]))
    logger.info(f"gguf: LeRobot {config['type']} policy, tokenizer from {vlm_dir}")
    with open(vlm_dir / "config.json", "r", encoding="utf-8") as f:
        vlm = json.load(f)

    names: list[str] = []
    for part in ModelBase.get_model_part_names(dir_model, "model", ".safetensors"):
        with gguf.utility.SafetensorsLocal(dir_model / part) as f:
            names.extend(f.keys())

    return {
        **config,
        "architectures": [_LEROBOT_POLICIES[config["type"]]],
        "vlm_config": vlm,
        "vlm_dir": str(vlm_dir),
        "vocab_size": vlm["text_config"]["vocab_size"],
        # num_vlm_layers can truncate the VLM, count the layers the checkpoint has
        "num_hidden_layers": _count_layers(names, r"text_model\.layers\.(\d+)\."),
    }


@ModelBase.register("SmolVLAPolicy")
# [TAG_HF_EXAMPLE_MISSING] LeRobot checkpoint, not a transformers model
class SmolVLAModel(TextModel):
    model_arch = gguf.MODEL_ARCH.VLA

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # keep the vision tower, TextModel drops it
        return ModelBase.filter_tensors(item)

    def set_vocab(self):
        dir_model = self.dir_model
        self.dir_model = Path(self.hparams["vlm_dir"])
        try:
            self._set_vocab_gpt2()
        finally:
            self.dir_model = dir_model

    def set_gguf_parameters(self):
        hp = self.hparams
        text = hp["vlm_config"]["text_config"]
        # SmolVLM leaves the SigLIP sizes to the transformers defaults
        vision = {"patch_size": 16, "intermediate_size": 3072, "num_attention_heads": 12, "layer_norm_eps": 1e-6,
                  **hp["vlm_config"].get("vision_config", {})}
        n_ff_exp, n_embd_exp = self.model_tensors["model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight"]().shape
        n_layer_exp = _count_layers(self.model_tensors, r"lm_expert\.layers\.(\d+)\.")
        n_layer_vit = _count_layers(self.model_tensors, r"vision_model\.encoder\.layers\.(\d+)\.")

        if hp["attention_mode"] != "cross_attn" or n_layer_exp != self.block_count or hp["prefix_length"] > 0 \
                or hp["add_image_special_tokens"] or any(n not in ("IDENTITY", "MEAN_STD") for n in hp["normalization_mapping"].values()):
            raise NotImplementedError("SmolVLA variant not supported")

        K = gguf.Keys.VLA
        w = self.gguf_writer
        w.add_file_type(self.ftype)
        w.add_block_count(self.block_count)
        w.add_string(K.POLICY_TYPE, hp["type"])
        w.add_uint32(K.CHUNK_SIZE, hp["chunk_size"])
        w.add_uint32(K.NUM_STEPS, hp["num_steps"])
        w.add_uint32(K.STATE_DIM, hp["input_features"]["observation.state"]["shape"][0])
        w.add_uint32(K.ACTION_DIM, hp["output_features"]["action"]["shape"][0])
        w.add_uint32(K.MAX_STATE_DIM, hp["max_state_dim"])
        w.add_uint32(K.MAX_ACTION_DIM, hp["max_action_dim"])
        w.add_uint32(K.MAX_LANG_TOKENS, hp["tokenizer_max_length"])
        # images are passed in this order; the names are the LeRobot observation.images.* keys
        w.add_array(K.CAMERA_NAMES, [k.removeprefix("observation.images.") for k, f in hp["input_features"].items() if f["type"] == "VISUAL"])
        for key, feature, n in ((K.STATE_NAMES, "observation.state", hp["input_features"]["observation.state"]["shape"][0]),
                                (K.ACTION_NAMES, "action", hp["output_features"]["action"]["shape"][0])):
            names = self._dataset_names(feature)
            if names is not None and len(names) == n:
                w.add_array(key, names)
        w.add_float32(K.TIME_MIN_PERIOD, hp["min_period"])
        w.add_float32(K.TIME_MAX_PERIOD, hp["max_period"])
        w.add_float32(K.NORM_EPS, self._processor_step("policy_postprocessor.json", "unnormalizer_processor")["config"]["eps"])

        w.add_uint32(K.Vision.IMAGE_SIZE, hp["resize_imgs_with_padding"][0])
        w.add_uint32(K.Vision.PATCH_SIZE, vision["patch_size"])
        w.add_uint32(K.Vision.EMBEDDING_LENGTH, vision["hidden_size"])
        w.add_uint32(K.Vision.FEED_FORWARD_LENGTH, vision["intermediate_size"])
        w.add_uint32(K.Vision.BLOCK_COUNT, n_layer_vit)
        w.add_uint32(K.Vision.HEAD_COUNT, vision["num_attention_heads"])
        w.add_float32(K.Vision.LAYERNORM_EPS, vision["layer_norm_eps"])
        w.add_uint32(K.Vision.SCALE_FACTOR, hp["vlm_config"]["scale_factor"])

        w.add_uint32(K.Text.EMBEDDING_LENGTH, text["hidden_size"])
        w.add_uint32(K.Text.FEED_FORWARD_LENGTH, text["intermediate_size"])
        w.add_uint32(K.Text.BLOCK_COUNT, self.block_count)
        w.add_uint32(K.Text.HEAD_COUNT, text["num_attention_heads"])
        w.add_uint32(K.Text.HEAD_COUNT_KV, text["num_key_value_heads"])
        w.add_uint32(K.Text.KEY_LENGTH, text["head_dim"])
        w.add_float32(K.Text.RMS_EPS, text["rms_norm_eps"])
        # SmolVLA has its own RoPE with base 10000, it ignores the VLM rope_theta
        w.add_float32(K.Text.ROPE_FREQ_BASE, 10000.0)

        w.add_uint32(K.Expert.EMBEDDING_LENGTH, n_embd_exp)
        w.add_uint32(K.Expert.FEED_FORWARD_LENGTH, n_ff_exp)
        w.add_uint32(K.Expert.BLOCK_COUNT, n_layer_exp)
        w.add_uint32(K.Expert.SELF_ATTN_EVERY_N, hp["self_attn_every_n_layers"])

    def _dataset_names(self, feature: str) -> list[str] | None:
        # only checkpoints with a train_config.json name their dataset, whose meta/info.json may name each value
        path = self.dir_model / "train_config.json"
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            repo_id = json.load(f).get("dataset", {}).get("repo_id")
        if not repo_id:
            return None
        try:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset"), "r", encoding="utf-8") as f:
                names = json.load(f)["features"][feature].get("names")
        except Exception as e:
            logger.warning(f"no {feature} names from dataset {repo_id}: {e}")
            return None
        return names if isinstance(names, list) and all(isinstance(n, str) for n in names) else None

    def _processor_step(self, file: str, step: str) -> dict[str, Any]:
        with open(self.dir_model / file, "r", encoding="utf-8") as f:
            return next(s for s in json.load(f)["steps"] if s["registry_name"] == step)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        from safetensors.torch import load_file

        pre = load_file(self.dir_model / self._processor_step("policy_preprocessor.json", "normalizer_processor")["state_file"])
        post = load_file(self.dir_model / self._processor_step("policy_postprocessor.json", "unnormalizer_processor")["state_file"])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.VLA_NORM_STATE_MEAN, suffix=""), pre["observation.state.mean"].float()
        yield self.format_tensor_name(gguf.MODEL_TENSOR.VLA_NORM_STATE_STD, suffix=""), pre["observation.state.std"].float()
        yield self.format_tensor_name(gguf.MODEL_TENSOR.VLA_NORM_ACTION_MEAN, suffix=""), post["action.mean"].float()
        yield self.format_tensor_name(gguf.MODEL_TENSOR.VLA_NORM_ACTION_STD, suffix=""), post["action.std"].float()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # LeRobot keeps some small heads and projections in F32, keep them so
        if name in self.model_tensors and self.model_tensors[name]().dtype == torch.float32:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("norm."):  # from generate_extra_tensors
            yield name, data_torch
            return

        name = name.removeprefix("model.vlm_with_expert.").removeprefix("vlm.")
        name = name.replace("model.text_model.", "model.")
        if name in ("lm_head.weight", "model.norm.weight"):
            return  # the policy never reads the VLM logits

        yield self.map_tensor_name(name), data_torch
