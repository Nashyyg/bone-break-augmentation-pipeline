"""


preprocess every image and create the train/val/test split.


INPUT   data/raw/<class_name>/<image files>      (folder name = class label)
OUTPUT  work/processed/<class>/<name>.png        every image resized to 224x224
        work/split.csv                           path, label, group(specimen), split
        work/classes.json                        ordered class names


  1. Resize each image to 224x224 (letterboxed by default). The network needs a fixed input
     size, and 224 matches EfficientNet-B0's pre-training. Doing this ONCE here (instead of
     every epoch) makes training faster and guarantees every experiment sees identical pixels.
  2. Split into train / validation / test:
       * train      - the model learns from this (and ONLY this gets synthetic augmentation)
       * validation - used during training to choose the best epoch
       * test       - used once at the end to report results; never used for any decision
     The split is made once with a fixed seed and saved to split.csv, so every experiment
     uses the same test set 
  3. The split is STRATIFIED (each class appears in every split in roughly the same proportion)
     and, because the specimens have several photos, GROUPED BY SPECIMEN (all photos of
     one bone go to the same split). Without this, near-identical photos of the same bone
     would sit in both train and test and inflate accuracy (data leakage).
  4. A leak check verifies afterwards that no specimen appears in two splits.

"""
import json
import re
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import config
from common import border_color, rel

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def letterbox(img, size, bg):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = bg
    y0, x0 = (size - nh) // 2, (size - nw) // 2     # centre the image on the canvas
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def group_split(df, seed):
    k_test = round(1 / config.SPLIT[2])
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    tv_idx, te_idx = next(sgkf.split(df, df.label.values, df.group.values))
    tv, test = df.iloc[tv_idx], df.iloc[te_idx]

    k_val = round((config.SPLIT[0] + config.SPLIT[1]) / config.SPLIT[1])
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf2.split(tv, tv.label.values, tv.group.values))
    return tv.iloc[tr_idx], tv.iloc[va_idx], test


def stratified_split(df, seed):
    tv, test = train_test_split(df, test_size=config.SPLIT[2], stratify=df.label, random_state=seed)
    val_frac = config.SPLIT[1] / (config.SPLIT[0] + config.SPLIT[1])   # val as fraction of the remainder
    train, val = train_test_split(tv, test_size=val_frac, stratify=tv.label, random_state=seed)
    return train, val, test


def main():
    if not config.RAW_DIR.exists():
        sys.exit(f"{config.RAW_DIR} not found. Put images in data/raw/<class_name>/ "
                 f"(or run make_dummy_data.py to try the pipeline).")
    classes = sorted(d.name for d in config.RAW_DIR.iterdir() if d.is_dir())  
    if len(classes) < 2:
        sys.exit("Need at least two class sub-folders in data/raw/")

    # preprocess every image
    rows = []
    for label in classes:
        cdir = config.RAW_DIR / label
        for f in sorted(p for p in cdir.rglob("*") if p.suffix.lower() in EXTS):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)   
            if img is None:
                print(f"  ! unreadable, skipped: {f}")
                continue
            bg = border_color(img)
            if config.RESIZE_MODE == "letterbox":
                out = letterbox(img, config.IMG_SIZE, bg)
            else:
                out = cv2.resize(img, (config.IMG_SIZE,) * 2, interpolation=cv2.INTER_AREA)
            
            name = "__".join(f.relative_to(cdir).with_suffix("").parts) + ".png"
            dst = config.PROCESSED_DIR / label / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), out)

            group = rel(dst)                               
            if config.SPECIMEN_REGEX:
                m = re.search(config.SPECIMEN_REGEX, f.stem)
                if not m:
                    print(f"  ! no specimen id in filename (treated as its own specimen): {f.name}")
                group = f"{label}/{m.group(1) if m else f.stem}"
            rows.append(dict(path=rel(dst), label=label, group=group))

    df = pd.DataFrame(rows)
    counts = df.label.value_counts().sort_index()
    print("Images per class:\n", counts.to_string(), sep="")
    print(f"Specimens (groups): {df.group.nunique()}")
    if counts.min() < 10:
        print("WARNING: a class has <10 images; splitting may fail or be unreliable.")
    if counts.max() / counts.min() > 3:
        print("NOTE: class imbalance >3x. Report macro-averaged metrics and per-class results.")

    # split 
    split_fn = group_split if config.SPECIMEN_REGEX else stratified_split
    train, val, test = split_fn(df, config.SPLIT_SEED)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    out = out.sort_values("path").reset_index(drop=True)

    # leak check: a specimen must live in exactly ONE split 
    leaks = out.groupby("group").split.nunique()
    if (leaks > 1).any():
        sys.exit(f"DATA LEAK: specimens in more than one split: {list(leaks[leaks > 1].index)[:10]}")
    print("\nLeak check passed: no specimen appears in more than one split.")

    config.WORK.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.SPLIT_CSV, index=False)
    with open(config.CLASSES_JSON, "w") as f:
        json.dump(classes, f, indent=2)

    print("\nSplit sizes (images):", out.split.value_counts().to_dict())
    print(pd.crosstab(out.label, out.split).to_string())
    spec = out.groupby(["label", "split"]).group.nunique().unstack(fill_value=0)
    print("\nSpecimens per class per split:\n", spec.to_string(), sep="")
    low = [c for c in classes if (out[(out.label == c) & (out.split == "test")].shape[0] < 3)]
    if low:
        print(f"\nWARNING: fewer than 3 TEST images for {low}. Per-class test metrics for these "
              f"classes will be extremely unreliable (consider cross-validation).")
    print(f"\nWrote {config.SPLIT_CSV}")


if __name__ == "__main__":
    main()
