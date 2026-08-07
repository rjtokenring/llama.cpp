#!/usr/bin/env python3
"""Predict whether a GGUF model fits on the Hexagon NPU, and with which settings.

Reads a GGUF file and replicates llama.cpp's allocation decisions to answer:
  - will this model run on the NPU at all (are its weight types supported)?
  - how many sessions (GGML_HEXAGON_NDEV) does it need?
  - what is the largest usable context?

Each Hexagon session is a separate DSP process domain with a hard VA-mapping budget
(3200 MiB by default, GGML_HEXAGON_VMEM). Weights, the KV cache of the layers assigned
to that session, and compute buffers all share that budget - which is why a model can
need both more sessions and a smaller context.

Self-contained: standard library only, no gguf-py, no numpy. Copy it anywhere.

Output is JSON by default; --human prints a table. Exit status is 0 if every model
fits, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

MiB = 1024 * 1024

# ---------------------------------------------------------------------------
# ggml types
# ---------------------------------------------------------------------------

# name -> id, mirroring GGMLQuantizationType in gguf-py/gguf/constants.py
GGML_TYPES = {
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
    "Q8_1": 9, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
    "Q8_K": 15, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19,
    "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "I8": 24, "I16": 25,
    "I32": 26, "I64": 27, "F64": 28, "IQ1_M": 29, "BF16": 30, "TQ1_0": 34,
    "TQ2_0": 35, "MXFP4": 39, "NVFP4": 40, "Q1_0": 41, "Q2_0": 42,
}
TYPE_NAME = {v: k for k, v in GGML_TYPES.items()}

QK_K = 256
# id -> (block size, type size), mirroring GGML_QUANT_SIZES
QUANT_SIZES = {
    GGML_TYPES["F32"]:     (1, 4),
    GGML_TYPES["F16"]:     (1, 2),
    GGML_TYPES["Q4_0"]:    (32, 2 + 16),
    GGML_TYPES["Q4_1"]:    (32, 2 + 2 + 16),
    GGML_TYPES["Q5_0"]:    (32, 2 + 4 + 16),
    GGML_TYPES["Q5_1"]:    (32, 2 + 2 + 4 + 16),
    GGML_TYPES["Q8_0"]:    (32, 2 + 32),
    GGML_TYPES["Q8_1"]:    (32, 4 + 4 + 32),
    GGML_TYPES["Q2_K"]:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGML_TYPES["Q3_K"]:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGML_TYPES["Q4_K"]:    (256, 2 + 2 + QK_K // 2 + 12),
    GGML_TYPES["Q5_K"]:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGML_TYPES["Q6_K"]:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGML_TYPES["Q8_K"]:    (256, 4 + QK_K + QK_K // 8),
    GGML_TYPES["IQ2_XXS"]: (256, 2 + QK_K // 4),
    GGML_TYPES["IQ2_XS"]:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGML_TYPES["IQ3_XXS"]: (256, 2 + QK_K // 4 + QK_K // 8),
    GGML_TYPES["IQ1_S"]:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGML_TYPES["IQ4_NL"]:  (32, 2 + 16),
    GGML_TYPES["IQ3_S"]:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGML_TYPES["IQ2_S"]:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGML_TYPES["IQ4_XS"]:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGML_TYPES["I8"]:      (1, 1),
    GGML_TYPES["I16"]:     (1, 2),
    GGML_TYPES["I32"]:     (1, 4),
    GGML_TYPES["I64"]:     (1, 8),
    GGML_TYPES["F64"]:     (1, 8),
    GGML_TYPES["IQ1_M"]:   (256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    GGML_TYPES["BF16"]:    (1, 2),
    GGML_TYPES["TQ1_0"]:   (256, 2 + 4 * 13),
    GGML_TYPES["TQ2_0"]:   (256, 2 + 64),
    GGML_TYPES["MXFP4"]:   (32, 1 + 16),
    GGML_TYPES["NVFP4"]:   (64, 4 + 32),
    GGML_TYPES["Q1_0"]:    (128, 2 + 16),
    GGML_TYPES["Q2_0"]:    (64, 2 + 16),
}


def pad_up(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def row_size(t: int, ne: int) -> int:
    blck, tsize = QUANT_SIZES[t]
    return tsize * ne // blck


def nbytes(t: int, ne) -> int:
    """ggml_nbytes for a contiguous tensor."""
    return row_size(t, ne[0]) * ne[1] * ne[2] * ne[3]


# ---------------------------------------------------------------------------
# GGUF reader (metadata + tensor directory only, no tensor data)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747

# GGUFValueType -> (struct format, size); STRING and ARRAY handled separately
_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}
_T_STRING = 8
_T_ARRAY = 9

# arrays longer than this are not materialized (tokenizer vocabs etc.)
_MAX_ARRAY = 8192


class GgufTensor:
    __slots__ = ("name", "type", "ne")

    def __init__(self, name: str, type_: int, ne):
        self.name = name
        self.type = type_
        self.ne = ne  # always 4 elements, ggml order (ne0 = row length)

    @property
    def type_name(self) -> str:
        return TYPE_NAME.get(self.type, "type%d" % self.type)


class Gguf:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            blob = f.read(64 * MiB)  # header + metadata + tensor directory
        self.buf = blob
        self.kv = {}
        self.tensors = []
        self._parse()
        del self.buf

    # -- primitives ---------------------------------------------------------
    def _u32(self, o):
        return struct.unpack_from("<I", self.buf, o)[0], o + 4

    def _u64(self, o):
        return struct.unpack_from("<Q", self.buf, o)[0], o + 8

    def _str(self, o):
        n, o = self._u64(o)
        s = self.buf[o:o + n].decode("utf-8", "replace")
        return s, o + n

    def _value(self, o, vtype):
        if vtype == _T_STRING:
            return self._str(o)
        if vtype == _T_ARRAY:
            itype, o = self._u32(o)
            count, o = self._u64(o)
            if itype == _T_STRING:
                # skip the payload; only the count is ever interesting
                for _ in range(count):
                    n, o = self._u64(o)
                    o += n
                return {"_array_len": count}, o
            if itype == _T_ARRAY:
                raise ValueError("nested arrays are not supported")
            fmt, size = _SCALARS[itype]
            if count > _MAX_ARRAY:
                return {"_array_len": count}, o + size * count
            vals = list(struct.unpack_from("<%d%s" % (count, fmt[1]), self.buf, o))
            return vals, o + size * count
        fmt, size = _SCALARS[vtype]
        return struct.unpack_from(fmt, self.buf, o)[0], o + size

    def _parse(self):
        o = 0
        magic, o = self._u32(o)
        if magic != GGUF_MAGIC:
            raise ValueError("%s: not a GGUF file" % self.path)
        version, o = self._u32(o)
        if version not in (2, 3):
            raise ValueError("%s: unsupported GGUF version %d" % (self.path, version))
        n_tensors, o = self._u64(o)
        n_kv, o = self._u64(o)

        for _ in range(n_kv):
            key, o = self._str(o)
            vtype, o = self._u32(o)
            val, o = self._value(o, vtype)
            self.kv[key] = val

        for _ in range(n_tensors):
            name, o = self._str(o)
            n_dims, o = self._u32(o)
            ne = [1, 1, 1, 1]
            for i in range(n_dims):
                ne[i], o = self._u64(o)
            ttype, o = self._u32(o)
            _offset, o = self._u64(o)
            self.tensors.append(GgufTensor(name, ttype, ne))

    # -- typed access -------------------------------------------------------
    def get(self, key, default=None):
        return self.kv.get(key, default)

    def get_arr(self, key, n, default=None):
        """Read a key that may be a scalar or a per-layer array; returns a list of n."""
        v = self.kv.get(key)
        if v is None:
            return None if default is None else [default] * n
        if isinstance(v, dict):  # skipped oversized array
            return None if default is None else [default] * n
        if isinstance(v, list):
            if len(v) >= n:
                return [int(x) for x in v[:n]]
            return [int(x) for x in v] + [int(v[-1])] * (n - len(v))
        return [int(v)] * n


# ---------------------------------------------------------------------------
# model hyper-parameters
# ---------------------------------------------------------------------------

class Hparams:
    """The subset of llama_hparams that drives memory use."""

    def __init__(self, g: Gguf, warnings: list):
        arch = g.get("general.architecture")
        if not arch:
            raise ValueError("%s: missing general.architecture" % g.path)
        self.arch = arch
        k = lambda name: "%s.%s" % (arch, name)

        n_layer = g.get(k("block_count"))
        if n_layer is None:
            raise ValueError("%s: missing %s" % (g.path, k("block_count")))
        self.n_layer_all = int(n_layer)
        self.n_embd = int(g.get(k("embedding_length"), 0))

        self.n_head_arr = g.get_arr(k("attention.head_count"), self.n_layer_all, 0)
        self.n_head_kv_arr = g.get_arr(k("attention.head_count_kv"), self.n_layer_all)
        if self.n_head_kv_arr is None:
            self.n_head_kv_arr = list(self.n_head_arr)
        self.n_ff_arr = g.get_arr(k("feed_forward_length"), self.n_layer_all, 0)

        n_head = self.n_head_arr[0] or 1
        dflt_head_dim = self.n_embd // n_head if n_head else 0
        self.n_embd_head_k = int(g.get(k("attention.key_length"), dflt_head_dim))
        self.n_embd_head_v = int(g.get(k("attention.value_length"), dflt_head_dim))
        self.n_embd_head_k_swa = int(g.get(k("attention.key_length_swa"), self.n_embd_head_k))
        self.n_embd_head_v_swa = int(g.get(k("attention.value_length_swa"), self.n_embd_head_v))

        self.n_ctx_train = int(g.get(k("context_length"), 0))
        self.n_expert = int(g.get(k("expert_count"), 0))
        self.n_expert_used = int(g.get(k("expert_used_count"), 0))

        # vocab size: prefer the token_embd shape, it is always right
        self.n_vocab = 0
        for t in g.tensors:
            if t.name == "token_embd.weight":
                self.n_vocab = t.ne[1]
                break
        if not self.n_vocab:
            self.n_vocab = int(g.get(k("vocab_size"), 0)) or len(
                g.get("tokenizer.ggml.tokens", {}).get("_array_len", 0) and [] or [])

        # sliding-window attention
        self.n_swa = int(g.get(k("attention.sliding_window"), 0))
        self.is_swa = [False] * self.n_layer_all
        pattern = g.get(k("attention.sliding_window_pattern"))
        if self.n_swa > 0:
            if isinstance(pattern, list) and len(pattern) >= self.n_layer_all:
                # gemma4 style: one flag per layer
                self.is_swa = [bool(x) for x in pattern[:self.n_layer_all]]
            elif isinstance(pattern, int) and pattern > 0:
                # llama_hparams::set_swa_pattern, dense_first = false
                self.is_swa = [(il % pattern) < (pattern - 1) for il in range(self.n_layer_all)]
            else:
                warnings.append(
                    "model declares sliding_window=%d but no usable "
                    "%s.attention.sliding_window_pattern; assuming every layer keeps a "
                    "full-size KV cache (pessimistic)" % (self.n_swa, arch))

        # layers past n_layer_kv_from_start reuse an earlier layer's KV (gemma3n/gemma4)
        self.n_layer_kv_from_start = -1
        shared_kv = g.get(k("attention.shared_kv_layers"))
        if shared_kv:
            self.n_layer_kv_from_start = self.n_layer_all - int(shared_kv)
        elif arch == "gemma3n":
            self.n_layer_kv_from_start = 20

    # -- per-layer accessors, mirroring llama-hparams.cpp -------------------
    def n_head_kv(self, il):
        return self.n_head_kv_arr[il]

    def n_embd_k_gqa(self, il):
        head = self.n_embd_head_k_swa if self.is_swa[il] else self.n_embd_head_k
        return head * self.n_head_kv(il)

    def n_embd_v_gqa(self, il):
        head = self.n_embd_head_v_swa if self.is_swa[il] else self.n_embd_head_v
        return head * self.n_head_kv(il)

    def n_embd_v_gqa_max(self):
        return max(self.n_embd_v_gqa(il) for il in range(self.n_layer_all))

    def has_kv(self, il):
        if self.n_layer_kv_from_start >= 0:
            return il < self.n_layer_kv_from_start
        return True

    @property
    def n_ff(self):
        return max(self.n_ff_arr) if self.n_ff_arr else 0

    @property
    def n_head(self):
        return max(self.n_head_arr) if self.n_head_arr else 0


# ---------------------------------------------------------------------------
# Hexagon backend rules (ggml/src/ggml-hexagon/ggml-hexagon.cpp)
# ---------------------------------------------------------------------------

# ggml_hexagon_is_repack_type
REPACK_TYPES = {GGML_TYPES[n] for n in ("Q4_0", "Q4_1", "Q8_0", "IQ4_NL", "MXFP4")}
# types accepted by the plain (non-repack) buffer for MUL_MAT
PLAIN_MM_TYPES = {GGML_TYPES[n] for n in ("F16", "F32")}
Q6_K = GGML_TYPES["Q6_K"]
BF16 = GGML_TYPES["BF16"]
F16 = GGML_TYPES["F16"]
F32 = GGML_TYPES["F32"]

HTP_ALIGN = 128           # ggml_backend_hexagon_buffer_type_get_alignment
CPU_ALIGN = 32            # TENSOR_ALIGNMENT
HTP_MAX_MMAPS = 16        # htp/htp-ctx.h
VMEM_V75 = 3200 * MiB     # HTP_OP_MAX_VMEM_DEFAULT
VMEM_OLD = 3000 * MiB     # arch < v75
MBUF_DEFAULT = 1024 * MiB # GGML_HEXAGON_MBUF
MAX_SESSIONS = 16         # GGML_HEXAGON_MAX_SESSIONS

# ggml_hexagon_supported_mul_mat's bound on src0->ne[1]. Binaries built before commit
# f5a2c0df0 ("add q6_k repack") used 32768, which refuses any real lm-head and keeps it
# - plus the logits buffer - on the CPU. Use --max-out-rows to model such a build.
MAX_OUT_ROWS = 1 << 21
MAX_OUT_ROWS_OLD = 32768

# compute-buffer model: not derivable from the GGUF, llama.cpp gets these from a real
# ggml-alloc reserve pass. Calibrate with --verify and adjust here.
COMPUTE_MODEL = {
    # f32 activations of shape [n_embd, n_tokens] live simultaneously in a session's
    # compute buffer. 12 reproduces the 16.06 MiB seen for gpt-oss-20b at ubatch 128
    # (docs/backend/snapdragon/developer.md).
    "K_ACT": 12,
    # extra bytes per session, absorbs graph slop and cross-device copies
    "HEADROOM_MIB": 256,
}


def repack_storage_type(t: int) -> int:
    """ggml_hexagon_repack_storage_type: type as actually stored in a repack buffer."""
    if t == BF16:
        return F16
    if t == Q6_K:
        return GGML_TYPES["Q8_0"]
    return t


def repack_alloc_size(t: GgufTensor) -> int:
    """ggml_backend_hexagon_buffer_type_get_alloc_size."""
    stype = repack_storage_type(t.type)
    if stype in REPACK_TYPES:
        ne0 = pad_up(t.ne[0], 32)
        ne1 = pad_up(t.ne[1], 32)
        return row_size(stype, ne0) * ne1 * t.ne[2] * t.ne[3]
    return nbytes(t.type, t.ne)


# ---------------------------------------------------------------------------
# tensor classification
# ---------------------------------------------------------------------------

_BLK_RE = re.compile(r"^blk\.(\d+)\.")

# LLM_TENSOR_LAYER_INPUT, src/llama-arch.cpp - always on the CPU
INPUT_TENSORS = {
    "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
    "position_embd.weight", "token_types.weight", "per_layer_token_embd.weight",
    "conv1d.weight", "conv1d.bias",
}
# LLM_TENSOR_LAYER_OUTPUT - follows the output pseudo-layer
OUTPUT_PREFIXES = ("output.", "output_norm.", "cls.", "cls_out.", "dense_2_out.",
                   "dense_3_out.")

PLACE_CPU_INPUT = "cpu_input"        # never offloaded by llama.cpp
PLACE_CPU_UNSUP = "cpu_unsupported"  # the NPU has no kernel for this weight type
PLACE_CPU_LIMIT = "cpu_limit"        # supported type, rejected on shape/limit grounds
PLACE_REPACK = "repack"
PLACE_PLAIN = "plain"


def is_moe_expert(t: GgufTensor) -> bool:
    """MUL_MAT_ID weight: stacked experts."""
    return "_exps." in t.name


def classify(t: GgufTensor, max_out_rows: int = MAX_OUT_ROWS):
    """Return (placement, reason) assuming the tensor's layer is offloaded.

    Mirrors ggml_hexagon_supported_mul_mat / _mul_mat_id / _binary via the
    weight_buft_supported probe: the plain buffer type is tried first and rejects
    quantized weights, so those land in the repack buffer.
    """
    name = t.name
    is_2d_weight = name.endswith(".weight") and t.ne[1] > 1
    is_norm = "_norm" in name or name.endswith("norm.weight")
    if not is_2d_weight or is_norm:
        # norms, biases, rope freqs: F32 binary ops on the plain buffer
        if t.type in (F32, F16):
            return PLACE_PLAIN, None
        return PLACE_CPU_UNSUP, "%s not supported for elementwise ops" % t.type_name

    if t.type in REPACK_TYPES or t.type == Q6_K:
        if t.ne[0] % 32:
            return PLACE_CPU_LIMIT, "ne0=%d is not a multiple of 32" % t.ne[0]
        if t.ne[1] > max_out_rows:
            return PLACE_CPU_LIMIT, ("ne1=%d exceeds the %d output-row limit"
                                     % (t.ne[1], max_out_rows))
        if t.type == Q6_K and is_moe_expert(t):
            return PLACE_CPU_LIMIT, "Q6_K is not supported for MoE expert matmul"
        return PLACE_REPACK, None
    if t.type == BF16:
        return PLACE_REPACK, None  # converted to F16 on set_tensor
    if t.type in PLAIN_MM_TYPES:
        return PLACE_PLAIN, None
    return PLACE_CPU_UNSUP, "%s matmul is not implemented on the NPU" % t.type_name


# ---------------------------------------------------------------------------
# layer -> session assignment (src/llama-model.cpp:1315-1340)
# ---------------------------------------------------------------------------

def layer_devices(n_layer_all: int, ngl: int, ndev: int):
    """Return a list of length n_layer_all+1 (last entry = output pseudo-layer).

    Each entry is a session index, or None for the CPU. Hexagon reports 0/0 free
    memory so the default tensor_split degenerates to an equal split.
    """
    n_slots = n_layer_all + 1
    if ndev <= 0:
        return [None] * n_slots
    act = min(ngl, n_slots)
    i_gpu_start = max(n_slots - ngl, 0)
    out = []
    for il in range(n_slots):
        if il < i_gpu_start or (il - i_gpu_start) >= act:
            out.append(None)
            continue
        frac = float(il - i_gpu_start) / act
        # std::upper_bound over the normalized prefix sums (k+1)/ndev
        dev = ndev - 1
        for k in range(ndev):
            if (k + 1) / float(ndev) > frac:
                dev = k
                break
        out.append(dev)
    return out


# ---------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, args, hp: Hparams):
        self.ngl = args.ngl if args.ngl is not None else hp.n_layer_all + 1
        self.n_ctx = args.ctx if args.ctx else hp.n_ctx_train
        self.n_batch = args.batch
        self.n_ubatch = min(args.ubatch, args.batch)
        self.n_seq_max = args.seq_max
        self.kv_unified = args.kv_unified
        self.type_k = GGML_TYPES[args.ctk.upper()]
        self.type_v = GGML_TYPES[args.ctv.upper()]
        self.flash_attn = args.flash_attn != "off"
        self.swa_full = args.swa_full
        self.kv_offload = not args.no_kv_offload
        self.vmem = args.vmem * MiB if args.vmem else (
            VMEM_V75 if args.arch >= 75 else VMEM_OLD)
        self.mbuf = args.mbuf * MiB
        self.headroom = args.headroom * MiB
        self.arch = args.arch
        self.max_out_rows = args.max_out_rows
        self.ram = (args.ram * MiB) if args.ram else host_ram()


class Buffers:
    """Accumulates padded tensor sizes per (owner, buffer type) and chunks them."""

    def __init__(self):
        self.sizes = {}   # key -> list of padded tensor sizes

    def add(self, key, size, align):
        self.sizes.setdefault(key, []).append(pad_up(size, align))

    def total(self, key):
        return sum(self.sizes.get(key, ()))

    def chunks(self, key, mbuf):
        """Replicate ggml_backend_alloc_ctx_tensors_from_buft_impl's greedy split."""
        n, cur = 0, 0
        for s in self.sizes.get(key, ()):
            if cur > 0 and cur + s > mbuf:
                n += 1
                cur = s
            else:
                cur += s
        return n + 1 if cur > 0 else 0


def kv_cache_bytes(hp: Hparams, cfg: Config, devs, n_ctx: int, warnings=None):
    """Per-session (and host) KV bytes; mirrors llama-kv-cache.cpp + iswa."""
    n_ctx = pad_up(n_ctx, 256)
    if cfg.kv_unified:
        n_stream, kv_size = 1, n_ctx
    else:
        n_stream = cfg.n_seq_max
        kv_size = pad_up(n_ctx // cfg.n_seq_max, 256)

    size_base = kv_size
    if any(hp.is_swa) and not cfg.swa_full:
        window = hp.n_swa * (cfg.n_seq_max if cfg.kv_unified else 1) + cfg.n_ubatch
        size_swa = pad_up(min(size_base, window), 256)
    else:
        size_swa = size_base

    per_owner = {}
    v_dim_all = hp.n_embd_v_gqa_max()
    for il in range(hp.n_layer_all):
        if not hp.has_kv(il):
            continue
        owner = devs[il] if cfg.kv_offload else None
        size = size_swa if hp.is_swa[il] else size_base
        k = row_size(cfg.type_k, hp.n_embd_k_gqa(il)) * size * n_stream
        v_dim = hp.n_embd_v_gqa(il) if cfg.flash_attn else v_dim_all
        v = row_size(cfg.type_v, v_dim) * size * n_stream
        align = CPU_ALIGN if owner is None else HTP_ALIGN
        per_owner[owner] = per_owner.get(owner, 0) + pad_up(k, align) + pad_up(v, align)
    return per_owner


def estimate(g: Gguf, args) -> dict:
    warnings = []
    hp = Hparams(g, warnings)
    cfg = Config(args, hp)

    ndev_fixed = args.ndev
    result = {
        "model": os.path.abspath(g.path),
        "arch": hp.arch,
        "n_layer": hp.n_layer_all,
        "n_ctx_train": hp.n_ctx_train,
        "n_vocab": hp.n_vocab,
        "file_size_mib": os.path.getsize(g.path) // MiB,
    }

    if not any(_BLK_RE.match(t.name) for t in g.tensors):
        raise ValueError("no layer weight tensors found - this looks like a "
                         "vocab-only or split GGUF, not a full model")

    # --- NPU weight-type support, independent of any config ----------------
    unsupported = {}
    unsup_bytes = 0
    limit_bytes = 0
    for t in g.tensors:
        if t.name in INPUT_TENSORS:
            continue
        place, reason = classify(t, cfg.max_out_rows)
        if place == PLACE_CPU_UNSUP:
            unsup_bytes += nbytes(t.type, t.ne)
            unsupported.setdefault(reason, 0)
            unsupported[reason] += 1
        elif place == PLACE_CPU_LIMIT:
            limit_bytes += nbytes(t.type, t.ne)
    weight_bytes = sum(nbytes(t.type, t.ne) for t in g.tensors)
    unsup_frac = unsup_bytes / weight_bytes if weight_bytes else 0.0
    result["npu_limit_rejected_mib"] = limit_bytes // MiB
    result["npu_unsupported_mib"] = unsup_bytes // MiB
    result["npu_unsupported_fraction"] = round(unsup_frac, 3)
    result["npu_supported"] = unsup_frac < 0.10
    result["npu_unsupported_reason"] = None
    if unsupported:
        worst = max(unsupported.items(), key=lambda kv: kv[1])
        result["npu_unsupported_reason"] = "%s (%d tensors)" % (worst[0], worst[1])
    if not result["npu_supported"]:
        warnings.append(
            "%.0f%% of the weights cannot run on the NPU and stay on the CPU - "
            "re-quantize to Q4_0 for NPU support" % (100 * unsup_frac))

    # --- try the requested / solved configuration --------------------------
    ndev_list = [ndev_fixed] if ndev_fixed else list(range(1, MAX_SESSIONS + 1))
    chosen = None
    for ndev in ndev_list:
        plan = plan_config(g, hp, cfg, ndev, cfg.n_ctx)
        if chosen is None:
            chosen = plan
        if plan["fits"]:
            chosen = plan
            break

    result["requested"] = {
        "ndev": chosen["ndev"],
        "ngl": cfg.ngl,
        "n_ctx": cfg.n_ctx,
        "n_batch": cfg.n_batch,
        "n_ubatch": cfg.n_ubatch,
        "n_seq_max": cfg.n_seq_max,
        "type_k": TYPE_NAME[cfg.type_k],
        "type_v": TYPE_NAME[cfg.type_v],
        "flash_attn": cfg.flash_attn,
        "hexagon_arch": cfg.arch,
        "vmem_mib": cfg.vmem // MiB,
        "solved_ndev": ndev_fixed is None,
    }
    result["fits"] = chosen["fits"]
    result["reason"] = chosen["reason"]
    result["sessions"] = chosen["sessions"]
    result["host"] = chosen["host"]
    result["compute_estimated"] = True

    # --- host RAM ----------------------------------------------------------
    result["total_mib"] = chosen["total_mib"]
    result["ram_available_mib"] = cfg.ram // MiB if cfg.ram else None

    # --- largest usable context per session count --------------------------
    result["max_ctx_by_ndev"] = {}
    for ndev in range(1, max(ndev_fixed or 0, 8) + 1):
        result["max_ctx_by_ndev"][str(ndev)] = max_ctx(g, hp, cfg, ndev)

    # --- recommendation ----------------------------------------------------
    result["recommended"] = recommend(g, hp, cfg, result)
    if not result["npu_supported"] and result["recommended"].get("args"):
        result["recommended"]["note"] = (
            "this config fits in memory but %.0f%% of the weights run on the CPU"
            % (100 * unsup_frac))

    # --- hints -------------------------------------------------------------
    if not result["fits"] and cfg.n_ubatch > 128:
        small = Config(args, hp)
        small.n_ubatch = 128
        alt = plan_config(g, hp, small, chosen["ndev"], cfg.n_ctx)
        if alt["fits"]:
            warnings.append("would fit with -ub 128 (smaller logits buffer)")
    if cfg.arch < 75 and cfg.mbuf > 2048 * MiB:
        warnings.append("on Hexagon v%d a single buffer above 2 GiB makes the DSP "
                        "abort() (HAP_mmap limit); keep GGML_HEXAGON_MBUF <= 2048"
                        % cfg.arch)
    if any(s["chunks"] > HTP_MAX_MMAPS for s in chosen["sessions"]):
        warnings.append("a session needs more than %d buffer mappings; raise "
                        "GGML_HEXAGON_MBUF" % HTP_MAX_MMAPS)
    if cfg.flash_attn and hp.n_embd_v_gqa_max() != hp.n_embd_v_gqa(0):
        warnings.append("estimate assumes flash attention is active; if llama.cpp "
                        "disables it the V cache grows to the largest layer's size")
    if cfg.type_k != F16 or cfg.type_v != F16:
        warnings.append("the Hexagon flash-attention kernel only accepts F16 K and V; "
                        "a quantized KV cache pushes attention back onto the CPU")
    lm_head = next((t for t in g.tensors if t.name == "output.weight"), None)
    if lm_head is not None and lm_head.ne[1] > MAX_OUT_ROWS_OLD:
        on_npu = classify(lm_head, cfg.max_out_rows)[0] not in (
            PLACE_CPU_UNSUP, PLACE_CPU_LIMIT)
        warnings.append(
            "the lm-head (%d MiB, %s) is assumed to be %s; binaries built before "
            "commit f5a2c0df0 refuse output rows > %d and keep it on the CPU "
            "(--max-out-rows %d models those)"
            % (repack_alloc_size(lm_head) // MiB, lm_head.type_name,
               "offloaded" if on_npu else "kept on the CPU",
               MAX_OUT_ROWS_OLD, MAX_OUT_ROWS_OLD))
    result["assumptions"] = [
        "compute buffers are estimated (K_ACT=%d, headroom=%d MiB), not computed"
        % (COMPUTE_MODEL["K_ACT"], cfg.headroom // MiB),
        "matmul output-row limit: %d" % cfg.max_out_rows,
    ]
    result["warnings"] = warnings
    return result


def plan_config(g: Gguf, hp: Hparams, cfg: Config, ndev: int, n_ctx: int) -> dict:
    """Full per-session accounting for one (ndev, n_ctx) pair."""
    devs = layer_devices(hp.n_layer_all, cfg.ngl, ndev)
    out_dev = devs[hp.n_layer_all]

    bufs = Buffers()
    have_output_weight = any(t.name == "output.weight" for t in g.tensors)

    for t in g.tensors:
        if t.name in INPUT_TENSORS:
            bufs.add((None, "cpu"), nbytes(t.type, t.ne), CPU_ALIGN)
            if t.name == "token_embd.weight" and not have_output_weight:
                # tied embeddings: a second, full-size copy is created as `output`
                place_tensor(bufs, t, out_dev, cfg)
            continue
        m = _BLK_RE.match(t.name)
        if m:
            dev = devs[int(m.group(1))]
        elif t.name.startswith(OUTPUT_PREFIXES):
            dev = out_dev
        else:
            dev = None  # unknown tensor: assume the CPU keeps it
        place_tensor(bufs, t, dev, cfg)

    kv = kv_cache_bytes(hp, cfg, devs, n_ctx)

    # the logits buffer lives wherever the lm-head matmul runs
    logits_dev = out_dev
    lm_head = next((t for t in g.tensors if t.name == "output.weight"), None)
    if lm_head is None:
        lm_head = next((t for t in g.tensors if t.name == "token_embd.weight"), None)
    if lm_head is not None and classify(lm_head, cfg.max_out_rows)[0] in (
            PLACE_CPU_UNSUP, PLACE_CPU_LIMIT):
        logits_dev = None

    # compute buffers
    n_tok = min(pad_up(n_ctx, 256), cfg.n_ubatch)
    act = 4 * n_tok * COMPUTE_MODEL["K_ACT"] * hp.n_embd
    logits = 4 * hp.n_vocab * min(n_tok, cfg.n_batch)

    sessions = []
    reason = None
    fits = True
    for d in range(ndev):
        model_b = bufs.total((d, "plain")) + bufs.total((d, "repack"))
        kv_b = kv.get(d, 0)
        comp_b = act + cfg.headroom + (logits if d == logits_dev else 0)
        total_b = model_b + kv_b + comp_b
        chunks = bufs.chunks((d, "plain"), cfg.mbuf) + bufs.chunks((d, "repack"), cfg.mbuf)
        ok = total_b <= cfg.vmem
        if not ok and fits:
            fits = False
            reason = ("session HTP%d needs %d MiB but the per-session budget is %d MiB"
                      % (d, total_b // MiB, cfg.vmem // MiB))
        mine = [il for il in range(hp.n_layer_all) if devs[il] == d]
        sessions.append({
            "name": "HTP%d" % d,
            "layers": [mine[0], mine[-1]] if mine else [],
            "n_layers": len(mine),
            "model_mib": model_b // MiB,
            "repack_mib": bufs.total((d, "repack")) // MiB,
            "kv_mib": kv_b // MiB,
            "compute_mib": comp_b // MiB,
            "total_mib": total_b // MiB,
            "budget_mib": cfg.vmem // MiB,
            "chunks": chunks,
            "fits": ok,
        })

    host_model = bufs.total((None, "cpu"))
    host_kv = kv.get(None, 0)
    host_comp = cfg.headroom + (logits if logits_dev is None else 0)
    # llama.cpp's own output buffer (logits + sampling scratch), always on the host
    host_out = 4 * hp.n_vocab * max(cfg.n_seq_max, 1) * 3
    host = {
        "model_mib": host_model // MiB,
        "kv_mib": host_kv // MiB,
        "compute_mib": (host_comp + host_out) // MiB,
        "total_mib": (host_model + host_kv + host_comp + host_out) // MiB,
    }
    if fits and not sessions:
        fits = False
        reason = "no layers assigned to the NPU (ngl too low)"

    total_mib = sum(s["total_mib"] for s in sessions) + host["total_mib"]
    if fits and cfg.ram and total_mib * MiB > cfg.ram:
        fits = False
        reason = ("needs %d MiB of DDR in total but only %d MiB is available"
                  % (total_mib, cfg.ram // MiB))
    return {"ndev": ndev, "fits": fits, "reason": reason, "total_mib": total_mib,
            "sessions": sessions, "host": host}


def place_tensor(bufs: Buffers, t: GgufTensor, dev, cfg: Config):
    if dev is None:
        bufs.add((None, "cpu"), nbytes(t.type, t.ne), CPU_ALIGN)
        return
    place, _ = classify(t, cfg.max_out_rows)
    if place == PLACE_REPACK:
        bufs.add((dev, "repack"), repack_alloc_size(t), HTP_ALIGN)
    elif place == PLACE_PLAIN:
        bufs.add((dev, "plain"), nbytes(t.type, t.ne), HTP_ALIGN)
    else:  # PLACE_CPU_UNSUP or PLACE_CPU_LIMIT
        bufs.add((None, "cpu"), nbytes(t.type, t.ne), CPU_ALIGN)


def max_ctx(g: Gguf, hp: Hparams, cfg: Config, ndev: int) -> int:
    """Largest multiple of 256 that still fits, 0 if even the smallest does not."""
    lo, hi = 0, max(pad_up(hp.n_ctx_train, 256), 256)
    if not plan_config(g, hp, cfg, ndev, 256)["fits"]:
        return 0
    lo = 256
    while lo < hi:
        mid = pad_up((lo + hi + 256) // 2, 256)
        if mid > hi:
            break
        if plan_config(g, hp, cfg, ndev, mid)["fits"]:
            lo = mid
        else:
            hi = mid - 256
    return lo


def recommend(g: Gguf, hp: Hparams, cfg: Config, result: dict) -> dict:
    """Smallest session count that fits, with the largest context it allows.

    Always searches every session count, including when --ndev pinned a different one:
    the point is to say what to change. Prefers the fewest sessions that still reach the
    requested context; if nothing does, picks whatever reaches the largest context -
    more sessions is worth it when it buys context.
    """
    caps = [(ndev, max_ctx(g, hp, cfg, ndev)) for ndev in range(1, MAX_SESSIONS + 1)]
    return _pick(caps, cfg)


def _pick(caps, cfg) -> dict:
    """caps: [(ndev, max_ctx)] in increasing ndev order."""
    target = pad_up(cfg.n_ctx, 256) if cfg.n_ctx else 0
    best = None
    if target:
        best = next(((n, target) for n, c in caps if c >= target), None)
    if best is None:
        reachable = [(n, c) for n, c in caps if c > 0]
        if reachable:
            top = max(c for _, c in reachable)
            best = next((n, c) for n, c in reachable if c == top)
    if best is None:
        return {"ndev": None, "n_ctx": 0, "env": {}, "args": None,
                "note": "no configuration of up to %d sessions fits" % MAX_SESSIONS}
    ndev, c = best
    return {
        "ndev": ndev,
        "ngl": cfg.ngl,
        "n_ctx": c,
        "env": {"GGML_HEXAGON_NDEV": str(ndev)} if ndev > 1 else {},
        "args": "-ngl %d -c %d" % (cfg.ngl, c),
    }


# ---------------------------------------------------------------------------
# several models resident at once (llama-server LLAMA_ARG_MODELS_MAX > 1)
# ---------------------------------------------------------------------------

def combined_plan(entries, ndev: int, n_ctx: int) -> dict:
    """Sum the per-session and host footprints of models that are resident together.

    Every resident model maps its own weights, KV cache and compute buffers into the
    same sessions, so they share one VA budget per session.
    """
    plans = [(g, plan_config(g, hp, cfg, ndev, n_ctx)) for g, hp, cfg in entries]
    cfg0 = entries[0][2]
    sessions = []
    fits, reason = True, None
    for d in range(ndev):
        agg = {"name": "HTP%d" % d, "model_mib": 0, "kv_mib": 0, "compute_mib": 0,
               "total_mib": 0, "budget_mib": cfg0.vmem // MiB, "chunks": 0}
        for _, plan in plans:
            s_d = plan["sessions"][d]
            for key in ("model_mib", "kv_mib", "compute_mib", "total_mib", "chunks"):
                agg[key] += s_d[key]
        agg["fits"] = agg["total_mib"] * MiB <= cfg0.vmem
        if not agg["fits"] and fits:
            fits = False
            reason = ("session HTP%d needs %d MiB for all resident models but the "
                      "per-session budget is %d MiB"
                      % (d, agg["total_mib"], cfg0.vmem // MiB))
        sessions.append(agg)

    host = {"model_mib": 0, "kv_mib": 0, "compute_mib": 0, "total_mib": 0}
    for _, plan in plans:
        for key in host:
            host[key] += plan["host"][key]

    total_mib = sum(s["total_mib"] for s in sessions) + host["total_mib"]
    if fits and cfg0.ram and total_mib * MiB > cfg0.ram:
        fits = False
        reason = ("all resident models need %d MiB of DDR but only %d MiB is available"
                  % (total_mib, cfg0.ram // MiB))
    return {"ndev": ndev, "fits": fits, "reason": reason, "total_mib": total_mib,
            "sessions": sessions, "host": host,
            "per_model": [{"model": os.path.basename(g.path),
                           "total_mib": sum(x["total_mib"] for x in p["sessions"])}
                          for g, p in plans]}


def combined_max_ctx(entries, ndev: int) -> int:
    hi = max(pad_up(hp.n_ctx_train, 256) for _, hp, _ in entries)
    if not combined_plan(entries, ndev, 256)["fits"]:
        return 0
    lo = 256
    while lo < hi:
        mid = pad_up((lo + hi + 256) // 2, 256)
        if mid > hi:
            break
        if combined_plan(entries, ndev, mid)["fits"]:
            lo = mid
        else:
            hi = mid - 256
    return lo


def estimate_coexist(paths, args) -> dict:
    entries = []
    for path in paths:
        g = Gguf(path)
        warns = []
        hp = Hparams(g, warns)
        entries.append((g, hp, Config(args, hp)))
    cfg0 = entries[0][2]
    n_ctx = cfg0.n_ctx or max(hp.n_ctx_train for _, hp, _ in entries)

    ndev_list = [args.ndev] if args.ndev else list(range(1, MAX_SESSIONS + 1))
    chosen = None
    for ndev in ndev_list:
        plan = combined_plan(entries, ndev, n_ctx)
        if chosen is None:
            chosen = plan
        if plan["fits"]:
            chosen = plan
            break

    max_by = {}
    caps = []
    for ndev in range(1, max(args.ndev or 0, 8) + 1):
        max_by[str(ndev)] = combined_max_ctx(entries, ndev)
        caps.append((ndev, max_by[str(ndev)]))
    rec = _pick(caps, cfg0)
    if rec.get("args") is None:
        rec["note"] = ("no session count up to %d holds all these models"
                       % len(caps))
    return {
        "coexist": [os.path.abspath(p) for p in paths],
        "arch": "+".join(hp.arch for _, hp, _ in entries),
        "n_layer": sum(hp.n_layer_all for _, hp, _ in entries),
        "n_vocab": 0,
        "file_size_mib": sum(os.path.getsize(p) // MiB for p in paths),
        "npu_supported": True,
        "npu_unsupported_reason": None,
        "npu_unsupported_mib": 0,
        "npu_unsupported_fraction": 0.0,
        "requested": {"ndev": chosen["ndev"], "ngl": cfg0.ngl, "n_ctx": n_ctx,
                      "n_batch": cfg0.n_batch, "n_ubatch": cfg0.n_ubatch,
                      "n_seq_max": cfg0.n_seq_max, "type_k": TYPE_NAME[cfg0.type_k],
                      "type_v": TYPE_NAME[cfg0.type_v], "flash_attn": cfg0.flash_attn,
                      "vmem_mib": cfg0.vmem // MiB, "solved_ndev": args.ndev is None},
        "fits": chosen["fits"],
        "reason": chosen["reason"],
        "sessions": chosen["sessions"],
        "host": chosen["host"],
        "per_model": chosen["per_model"],
        "compute_estimated": True,
        "total_mib": chosen["total_mib"],
        "ram_available_mib": cfg0.ram // MiB if cfg0.ram else None,
        "max_ctx_by_ndev": max_by,
        "recommended": rec,
        "assumptions": ["all listed models are resident at the same time"],
        "warnings": [],
    }


def host_ram():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# verification against a real llama.cpp log
# ---------------------------------------------------------------------------

_LOG_PATTERNS = [
    (re.compile(r"load_tensors:\s+(\S+) model buffer size\s*=\s*([\d.]+) MiB"), "model"),
    (re.compile(r"llama_kv_cache:\s+(\S+) KV buffer size\s*=\s*([\d.]+) MiB"), "kv"),
    (re.compile(r"llama_context:\s+(\S+) compute buffer size\s*=\s*([\d.]+) MiB"), "compute"),
]


def parse_log(path: str):
    actual = {}
    with open(path) as f:
        for line in f:
            for rx, kind in _LOG_PATTERNS:
                m = rx.search(line)
                if m:
                    dev, val = m.group(1), float(m.group(2))
                    actual.setdefault(dev, {}).setdefault(kind, 0.0)
                    actual[dev][kind] += val
    return actual


def print_verify(result: dict, actual: dict):
    print("verification against the log (MiB)")
    print("  %-16s %10s %10s %10s" % ("buffer", "predicted", "actual", "delta"))
    rows = []
    for s in result["sessions"]:
        name = s["name"]
        rows.append((name + "-REPACK", s["repack_mib"], actual.get(name + "-REPACK", {}).get("model")))
        rows.append((name + " model", s["model_mib"] - s["repack_mib"], actual.get(name, {}).get("model")))
        rows.append((name + " KV", s["kv_mib"], actual.get(name, {}).get("kv")))
        rows.append((name + " compute", s["compute_mib"], actual.get(name, {}).get("compute")))
    rows.append(("CPU model", result["host"]["model_mib"], actual.get("CPU", {}).get("model")))
    for name, pred, act in rows:
        if act is None:
            print("  %-16s %10d %10s %10s" % (name, pred, "-", "-"))
        else:
            print("  %-16s %10d %10.1f %+10.1f" % (name, pred, act, pred - act))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def print_human(r: dict):
    print("%s" % r["model"])
    print("  arch %s, %d layers, vocab %d, file %d MiB"
          % (r["arch"], r["n_layer"], r["n_vocab"], r["file_size_mib"]))
    if not r["npu_supported"]:
        print("  NPU: UNSUPPORTED - %s" % r["npu_unsupported_reason"])
        print("       %d MiB (%.0f%%) of weights would stay on the CPU"
              % (r["npu_unsupported_mib"], 100 * r["npu_unsupported_fraction"]))
    q = r["requested"]
    print("  config: ndev %d, ngl %d, ctx %d, ubatch %d, kv %s/%s, budget %d MiB/session"
          % (q["ndev"], q["ngl"], q["n_ctx"], q["n_ubatch"], q["type_k"], q["type_v"],
             q["vmem_mib"]))
    print("  %-8s %9s %9s %9s %9s %7s" % ("", "model", "KV", "compute", "total", "budget"))
    for s in r["sessions"]:
        print("  %-8s %9d %9d %9d %9d %7d  %s"
              % (s["name"], s["model_mib"], s["kv_mib"], s["compute_mib"],
                 s["total_mib"], s["budget_mib"], "ok" if s["fits"] else "OVER"))
    h = r["host"]
    print("  %-8s %9d %9d %9d %9d" % ("Host", h["model_mib"], h["kv_mib"],
                                      h["compute_mib"], h["total_mib"]))
    print("  total %d MiB of DDR%s" % (r["total_mib"],
          "" if r["ram_available_mib"] is None else " (available: %d MiB)" % r["ram_available_mib"]))
    print("  verdict: %s%s" % ("FITS" if r["fits"] else "DOES NOT FIT",
                               "" if r["fits"] else " - " + (r["reason"] or "")))
    rec = r["recommended"]
    if rec.get("args"):
        env = " ".join("%s=%s" % kv for kv in sorted(rec["env"].items()))
        print("  run with: %s %s" % (env, rec["args"]) if env else "  run with: %s" % rec["args"])
        if rec.get("note"):
            print("            (%s)" % rec["note"])
    else:
        print("  no usable configuration: %s" % rec.get("note", ""))
    print("  max context: %s" % ", ".join(
        "ndev %s -> %d" % (k, v) for k, v in sorted(r["max_ctx_by_ndev"].items(),
                                                    key=lambda kv: int(kv[0]))))
    for w in r["warnings"]:
        print("  warning: %s" % w)


def print_human_coexist(r: dict):
    print("resident together:")
    for m, per in zip(r["coexist"], r["per_model"]):
        print("  %-50s %6d MiB on the NPU" % (os.path.basename(m), per["total_mib"]))
    q = r["requested"]
    print("  config: ndev %d, ngl %d, ctx %d, ubatch %d, budget %d MiB/session"
          % (q["ndev"], q["ngl"], q["n_ctx"], q["n_ubatch"], q["vmem_mib"]))
    print("  %-8s %9s %9s %9s %9s %7s" % ("", "model", "KV", "compute", "total", "budget"))
    for s in r["sessions"]:
        print("  %-8s %9d %9d %9d %9d %7d  %s"
              % (s["name"], s["model_mib"], s["kv_mib"], s["compute_mib"],
                 s["total_mib"], s["budget_mib"], "ok" if s["fits"] else "OVER"))
    h = r["host"]
    print("  %-8s %9d %9d %9d %9d" % ("Host", h["model_mib"], h["kv_mib"],
                                      h["compute_mib"], h["total_mib"]))
    print("  total %d MiB of DDR%s" % (r["total_mib"],
          "" if r["ram_available_mib"] is None else
          " (available: %d MiB)" % r["ram_available_mib"]))
    print("  verdict: %s%s" % ("FITS" if r["fits"] else "DOES NOT FIT",
                               "" if r["fits"] else " - " + (r["reason"] or "")))
    rec = r["recommended"]
    if rec.get("args"):
        env = " ".join("%s=%s" % kv for kv in sorted(rec["env"].items()))
        print("  run with: %s %s" % (env, rec["args"]) if env else
              "  run with: %s" % rec["args"])
    else:
        print("  no usable configuration: %s" % rec.get("note", ""))
    print("  max shared context: %s" % ", ".join(
        "ndev %s -> %d" % (k, v) for k, v in sorted(r["max_ctx_by_ndev"].items(),
                                                    key=lambda kv: int(kv[0]))))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Predict whether a GGUF model fits on the Hexagon NPU.",
        epilog="examples:\n"
               "  %(prog)s --human model.gguf\n"
               "  %(prog)s --ndev 2 -c 4096 --human model.gguf\n"
               "  %(prog)s /models/*.gguf            # JSON array for a model picker\n"
               "  %(prog)s --ndev 2 -c 4096 --verify server.log --human model.gguf\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", metavar="MODEL.gguf")
    p.add_argument("--ndev", type=int, default=None,
                   help="sessions (GGML_HEXAGON_NDEV); default: solve for the smallest")
    p.add_argument("--ngl", type=int, default=None,
                   help="layers to offload; default: all")
    p.add_argument("-c", "--ctx", type=int, default=0,
                   help="context size; default 0 = the model's training context")
    p.add_argument("-b", "--batch", type=int, default=2048)
    p.add_argument("-ub", "--ubatch", type=int, default=512)
    p.add_argument("--seq-max", type=int, default=1, help="parallel sequences (-np)")
    p.add_argument("--kv-unified", action="store_true")
    p.add_argument("-ctk", "--ctk", default="f16", help="K cache type (f16, q8_0, ...)")
    p.add_argument("-ctv", "--ctv", default="f16", help="V cache type")
    p.add_argument("--flash-attn", choices=["on", "off", "auto"], default="auto")
    p.add_argument("--swa-full", action="store_true")
    p.add_argument("--no-kv-offload", action="store_true")
    p.add_argument("--arch", type=int, default=75,
                   help="Hexagon arch version; picks the default VA budget")
    p.add_argument("--vmem", type=int, default=0,
                   help="per-session VA budget in MiB (GGML_HEXAGON_VMEM)")
    p.add_argument("--mbuf", type=int, default=MBUF_DEFAULT // MiB,
                   help="max buffer size in MiB (GGML_HEXAGON_MBUF)")
    p.add_argument("--ram", type=int, default=0,
                   help="available DDR in MiB; default: /proc/meminfo MemAvailable")
    p.add_argument("--headroom", type=int, default=COMPUTE_MODEL["HEADROOM_MIB"],
                   help="per-session slack for compute buffers, in MiB")
    p.add_argument("--max-out-rows", type=int, default=MAX_OUT_ROWS,
                   help="matmul output-row limit; pass %d to model a binary built "
                        "before commit f5a2c0df0, which never offloads the lm-head"
                        % MAX_OUT_ROWS_OLD)
    p.add_argument("--coexist", action="store_true",
                   help="treat every listed model as resident at the same time "
                        "(llama-server with LLAMA_ARG_MODELS_MAX > 1)")
    p.add_argument("--verify", metavar="LOGFILE",
                   help="compare the prediction against a real llama.cpp startup log")
    p.add_argument("--human", action="store_true", help="human-readable output")
    p.add_argument("--json", action="store_true", help="force JSON output (default)")
    args = p.parse_args(argv)

    if args.coexist:
        try:
            results = [estimate_coexist(args.models, args)]
        except Exception as exc:
            results = [{"coexist": args.models, "error": str(exc), "fits": False}]
        failed = not results[0].get("fits")
        if args.human:
            if "error" in results[0]:
                print("error: %s" % results[0]["error"])
            else:
                print_human_coexist(results[0])
        else:
            json.dump(results[0], sys.stdout, indent=2)
            sys.stdout.write("\n")
        return 1 if failed else 0

    results, failed = [], False
    for path in args.models:
        try:
            r = estimate(Gguf(path), args)
        except Exception as exc:  # keep going over a directory of models
            r = {"model": os.path.abspath(path), "error": str(exc), "fits": False}
        results.append(r)
        if not r.get("fits"):
            failed = True

    if args.human:
        for r in results:
            if "error" in r:
                print("%s\n  error: %s" % (r["model"], r["error"]))
            else:
                print_human(r)
            print()
    else:
        json.dump(results[0] if len(results) == 1 else results, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if args.verify:
        for r in results:
            if "error" not in r:
                print_verify(r, parse_log(args.verify))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
