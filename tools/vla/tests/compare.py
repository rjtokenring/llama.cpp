#!/usr/bin/env python3
"""Compare llama-vla-cli --dump tensors with smolvla_oracle.py outputs.

    compare.py --oracle oracle/ --dump dump/ [--samples 3]
"""

import argparse
import os

import numpy as np

# dump name -> oracle name
PROBES = {
    "img_tokens": "img_tokens",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()

    print(f"{'sample':>6} {'probe':<14} {'max|d|':>10} {'rel rms':>10} {'max|ref|':>10}")
    for i in range(a.samples):
        ref = np.load(os.path.join(a.oracle, f"{i}.npz"))
        for dname, oname in PROBES.items():
            path = os.path.join(a.dump, f"{i}-{dname}.npy")
            if not os.path.exists(path):
                continue
            got = np.load(path).reshape(ref[oname].shape)
            want = ref[oname]
            d = got - want
            rel = np.sqrt((d ** 2).mean() / (want ** 2).mean())
            print(f"{i:>6} {dname:<14} {np.abs(d).max():>10.3e} {rel:>10.3e} {np.abs(want).max():>10.3e}")


if __name__ == "__main__":
    main()
