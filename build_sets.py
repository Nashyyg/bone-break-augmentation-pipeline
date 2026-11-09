

#decides which images make up the training set of each condition.


# A "condition" is (pipeline, synthetic proportion p, seed). This module turns that triple into a
# manifest: a table of image paths + labels + an is_synthetic flag. train.py trains on the manifest.
# Manifests are also saved to work/manifests/ as an audit trail ("which images did run X use?").



import numpy as np
import pandas as pd

import config


def condition_name(pipeline, p, seed):
    """e.g. ('bonebreak', 0.25, 1) -> 'bonebreak_p025_s1' (used for result/manifests filenames)."""
    return f"{pipeline}_p{int(round(p * 100)):03d}_s{seed}"


def build_manifest(pipeline, p, seed, mode=None, save=True):
    mode = mode or config.MIX_MODE
    split = pd.read_csv(config.SPLIT_CSV)
    train = split[split.split == "train"].reset_index(drop=True)
    n = len(train)                                  # N = number of REAL training images
    rng = np.random.default_rng(seed)
    real_perm = rng.permutation(n)                  # fixed shuffled order of real images for this seed

    if pipeline == "baseline" or p == 0:
        # Experiment 1: no synthetic data at all.
        man = (train.iloc[real_perm][["path", "label"]].assign(is_synthetic=False)
               .reset_index(drop=True))
        pipeline = "baseline"
    else:
        pool = pd.read_csv(config.SYN_DIR / f"{pipeline}_pool.csv")
        pool = pool[pool.ok].reset_index(drop=True)
        syn_perm = rng.permutation(len(pool))       # fixed shuffled order of synthetic images
        if mode == "fixed_size":
            n_syn = int(round(p * n))               # p of the N slots are synthetic...
            n_real = n - n_syn                      # ...the rest are real. Total stays N.
        elif mode == "additive":
            n_syn, n_real = int(round(p * n)), n    # keep all N real, ADD p*N synthetic
        else:
            raise ValueError(f"unknown MIX_MODE {mode!r}")
        if n_syn > len(pool):
            raise ValueError(f"Need {n_syn} synthetic images but the pool has {len(pool)}; "
                             f"increase POOL_MULT in config.py and re-run generate_synthetic.py")
        real = train.iloc[real_perm[:n_real]][["path", "label"]].assign(is_synthetic=False)
        syn = pool.iloc[syn_perm[:n_syn]][["path", "label"]].assign(is_synthetic=True)
        man = pd.concat([real, syn], ignore_index=True)

    if save:
        config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        man.to_csv(config.MANIFEST_DIR / f"{condition_name(pipeline, p, seed)}.csv", index=False)
    return man


if __name__ == "__main__":
    # Running this file directly prints the size of every condition - a quick sanity check.
    for pl in config.PIPELINES:
        for p in config.PROPORTIONS:
            m = build_manifest(pl, p, seed=0)
            print(f"{pl:10s} p={p:4.2f}  total={len(m):5d}  synthetic={int(m.is_synthetic.sum()):5d}")
