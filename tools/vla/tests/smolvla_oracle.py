#!/usr/bin/env python3
"""Reference SmolVLA run with LeRobot, dumping the intermediate tensors llama-vla compares against.

    smolvla_oracle.py --ckpt HuggingFaceVLA/smolvla_libero (or a local dir) \
        --samples libero3.samples.bin --noise noise.f32 --out oracle/

For each sample it writes oracle/<i>.npz with the stages llama-vla-cli --dump names the same way:
    img_tokens   [n_views, 64, 960]   connector output per view, before the sqrt(hidden) scale
    prefix_emb   [n_prefix, 960]      image + language + state embeddings fed to the VLM
    k_0, v_0     [n_prefix, 5, 64]    layer-0 K (after RoPE) and V of the prefix cache
    k_last, v_last                    the same for the last layer
    v_t_0        [50, 32]             velocity of the first Euler step
    actions_norm [50, 32]             integrated actions, normalized
    actions      [50, n_action]       unnormalized with the checkpoint's postprocessor stats

The samples file is the one tests/libero_replay.py builds (images already resized to the policy's
input as float32 RGB in [0, 1]); the noise file is chunk_size * max_action_dim float32 values.
Everything runs in float32 on CPU so the comparison measures llama-vla, not bf16 rounding.
"""

import argparse
import os
import struct

import numpy as np
import torch


def read_samples(path):
    with open(path, "rb") as f:
        if f.read(4) != b"VLAR":
            raise SystemExit(f"{path}: not a replay file")
        _, n, n_img, w, h, sdim, pix = struct.unpack("<7i", f.read(28))
        if pix != 1:
            raise SystemExit(f"{path}: expected float32 pixels")
        for _ in range(n):
            nt, = struct.unpack("<i", f.read(4))
            tokens = np.frombuffer(f.read(4 * nt), "<i4").copy()
            state = np.frombuffer(f.read(4 * sdim), "<f4").copy()
            imgs = [np.frombuffer(f.read(4 * 3 * w * h), "<f4").reshape(h, w, 3).copy() for _ in range(n_img)]
            yield tokens, state, imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.numpy import load_file

    policy = SmolVLAPolicy.from_pretrained(a.ckpt, device="cpu").to("cpu", torch.float32).eval()
    model = policy.model
    cfg = policy.config
    stats_dir = a.ckpt if os.path.isdir(a.ckpt) else None
    if stats_dir is None:
        from huggingface_hub import snapshot_download
        stats_dir = snapshot_download(a.ckpt, allow_patterns=["*.json", "*.safetensors"])
    pre = load_file(os.path.join(stats_dir, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post = load_file(os.path.join(stats_dir, "policy_postprocessor_step_1_unnormalizer_processor.safetensors"))
    eps = 1e-8

    noise = torch.from_numpy(np.fromfile(a.noise, "<f4").reshape(1, cfg.chunk_size, cfg.max_action_dim))
    os.makedirs(a.out, exist_ok=True)

    probes = {}
    vlm = model.vlm_with_expert
    embed_image = vlm.embed_image
    vlm.embed_image = lambda img: probes.setdefault("img", []).append(embed_image(img)) or probes["img"][-1]
    embed_prefix = model.embed_prefix

    def embed_prefix_hook(*args, **kw):
        out = embed_prefix(*args, **kw)
        probes["prefix_emb"] = out[0][0]
        return out
    model.embed_prefix = embed_prefix_hook

    vlm_forward = vlm.forward

    def forward_hook(*args, **kw):
        out = vlm_forward(*args, **kw)
        if kw.get("inputs_embeds", [None, None])[1] is None and "k_0" not in probes:
            cache = out[1]
            # DynamicCache stores [batch, heads, seq, head_dim]; dump [seq, heads, head_dim]
            for name, layer in (("0", cache.layers[0]), ("last", cache.layers[-1])):
                probes["k_" + name] = layer.keys[0].permute(1, 0, 2).clone()
                probes["v_" + name] = layer.values[0].permute(1, 0, 2).clone()
        return out
    vlm.forward = forward_hook

    denoise_step = model.denoise_step

    def denoise_hook(*args, **kw):
        v = denoise_step(*args, **kw)
        probes.setdefault("v_t", []).append(v[0].clone())
        return v
    model.denoise_step = denoise_hook

    for i, (tokens, state, imgs) in enumerate(read_samples(a.samples)):
        probes.clear()
        images = [torch.from_numpy(im).permute(2, 0, 1)[None] * 2.0 - 1.0 for im in imgs]
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in imgs]
        lang = torch.from_numpy(tokens).long()[None]
        lang_mask = torch.ones_like(lang, dtype=torch.bool)
        s = (state - pre["observation.state.mean"]) / (pre["observation.state.std"] + eps)
        s_pad = torch.zeros(1, cfg.max_state_dim)
        s_pad[0, : s.shape[0]] = torch.from_numpy(s)
        with torch.no_grad():
            x = model.sample_actions(images, img_masks, lang, lang_mask, s_pad, noise=noise.clone())[0]
        n_act = post["action.mean"].shape[0]
        act = x[:, :n_act].numpy() * (post["action.std"] + eps) + post["action.mean"]
        np.savez(os.path.join(a.out, f"{i}.npz"),
                 img_tokens=torch.stack([t[0] for t in probes["img"]]).numpy(),
                 prefix_emb=probes["prefix_emb"].numpy(),
                 k_0=probes["k_0"].numpy(), v_0=probes["v_0"].numpy(),
                 k_last=probes["k_last"].numpy(), v_last=probes["v_last"].numpy(),
                 v_t_0=probes["v_t"][0].numpy(), actions_norm=x.numpy(), actions=act.astype(np.float32))
        print(f"sample {i}: {len(tokens)} tokens, prefix {probes['prefix_emb'].shape[0]}, "
              f"actions[0] = {np.array2string(act[0], precision=4)}")


if __name__ == "__main__":
    main()
