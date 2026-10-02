#!/usr/bin/env python3
"""Compare llama-vla-cli --dump tensors with smolvla_oracle.py outputs.

    compare.py --oracle oracle/ --dump dump/ [--samples 3]
"""

import argparse
import glob
import os
import re

import numpy as np

# dump name -> oracle name; "last" is the highest layer index found in the dump
PROBES = {
    "img_tokens": "img_tokens",
    "prefix_emb": "prefix_emb",
    "Kcur-0":     "k_0",
    "Vcur-0":     "v_0",
    "Kcur-last":  "k_last",
    "Vcur-last":  "v_last",
}


def unpad(got, want):
    # llama-vla pads the language to max_lang_tokens before the state token, the oracle does not
    if got.shape[0] == want.shape[0]:
        return got
    return np.concatenate([got[: want.shape[0] - 1], got[-1:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()

    print(f"{'sample':>6} {'probe':<14} {'max|d|':>10} {'rel rms':>10} {'max|ref|':>10}")
    for i in range(a.samples):
        ref = np.load(os.path.join(a.oracle, f"{i}.npz"))
        layers = [int(m.group(1)) for f in glob.glob(os.path.join(a.dump, f"{i}-Kcur-*.npy"))
                  if (m := re.search(r"Kcur-(\d+)\.npy$", f))]
        for dname, oname in PROBES.items():
            if "last" in dname and layers:
                dname = dname.replace("last", str(max(layers)))
            path = os.path.join(a.dump, f"{i}-{dname}.npy")
            if not os.path.exists(path):
                continue
            want = ref[oname]
            got = np.load(path)
            got = unpad(got, want) if got.shape[1:] == want.shape[1:] else got
            got = got.reshape(want.shape)
            d = got - want
            rel = np.sqrt((d ** 2).mean() / (want ** 2).mean())
            print(f"{i:>6} {dname:<14} {np.abs(d).max():>10.3e} {rel:>10.3e} {np.abs(want).max():>10.3e}")


if __name__ == "__main__":
    main()
