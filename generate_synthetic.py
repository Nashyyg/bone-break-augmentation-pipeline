

#create the synthetic images offline

# INPUT   work/split.csv (train rows only are augmented)
# OUTPUT  work/synthetic/<pipeline>/<class>/*.png     the synthetic images
#         work/synthetic/<pipeline>_pool.csv          one row per synthetic image (source, type, % removed)
#         work/test_broken/... + work/test_broken.csv secondary 'fragmented' test set


# ONLY TRAIN IMAGES
#     If validation/test images were augmented and used for training, the model would have seen
#     (altered copies of) its own exam questions -> inflated, invalid results.

# THE POOL
#     Make one synthetic variant of every training image (POOL_MULT=1). Later (build_sets.py)
#     a condition simply picks p*N images from this pool. Generating the pool once and sub-sampling
#     it guarantees both pipelines are built from exactly the same source images.

# THE FRAGMENTED TEST SET
#     The original test images are intact bones, but the real-world problem (and research question
#     3) is about fragmented specimens. So we also write a copy of the test set with bone-breaks
#     applied (fixed seed, identical for every experiment)

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from augment import PIPELINE_FUNCS, bonebreak_augment
from common import read_bgr, rel


def main():
    split = pd.read_csv(config.SPLIT_CSV)
    train = split[split.split == "train"].reset_index(drop=True)

    for pi, pipeline in enumerate(config.PIPELINES):
        fn = PIPELINE_FUNCS[pipeline]
        rows = []
        for v in range(config.POOL_MULT):
            for i, r in train.iterrows():
                img = read_bgr(r.path)
                # Seed = (global seed, pipeline index, variant, image index). Every synthetic image
                # therefore has its own fixed random stream: re-running gives identical files, and
                # adding images does not change the others.
                rng = np.random.default_rng([config.SYN_SEED, pi, v, i])
                out, info = fn(img, rng)
                dst = config.SYN_DIR / pipeline / r.label / f"{Path(r.path).stem}_v{v}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), out)
                rows.append(dict(src_path=r.path, path=rel(dst), label=r.label, **info))
        pool = pd.DataFrame(rows)
        pool.to_csv(config.SYN_DIR / f"{pipeline}_pool.csv", index=False)

        n_fail = int((~pool.ok).sum())
        print(f"[{pipeline}] {len(pool)} synthetic images; failed/unchanged: {n_fail} "
              f"({100 * n_fail / len(pool):.1f}%)")
        if pipeline == "bonebreak":
            ok = pool[pool.ok]
            print("   break types:", ok.aug.value_counts().to_dict(),
                  f"| mean area removed: {ok.frac.mean():.2f}")
            print("   failure reasons:", pool[~pool.ok].reason.value_counts().to_dict())
            if n_fail / len(pool) > 0.05:
                # A high failure rate usually means the foreground mask is wrong for the photos.
                print("   WARNING: >5% of images could not be broken. Run preview_augmentations.py "
                      "and check the foreground mask / background assumptions.")

    test = split[split.split == "test"].reset_index(drop=True)
    rows = []
    for i, r in test.iterrows():
        rng = np.random.default_rng([config.BROKEN_TEST_SEED, i])
        out, info = bonebreak_augment(read_bgr(r.path), rng)
        dst = config.BROKEN_TEST_DIR / r.label / f"{Path(r.path).stem}_broken.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        rows.append(dict(path=rel(dst), label=r.label, **info))
    tb = pd.DataFrame(rows)
    tb.to_csv(config.WORK / "test_broken.csv", index=False)
    # Images whose mask/break failed are NOT broken. train.py evaluates the fragmented test set
    print(f"[test_broken] {int(tb.ok.sum())}/{len(tb)} test images genuinely fragmented "
          f"(the other {int((~tb.ok).sum())} are excluded from the fragmented test set)")


if __name__ == "__main__":
    main()
