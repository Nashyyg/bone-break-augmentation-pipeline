
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from augment import bonebreak_augment, segment_bone
from common import read_bgr


def overlay(img, mask):
    out = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)       # red outline (BGR)
    return out


def grid(imgs, titles, colors, fname, ncols=8):
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t, c in zip(np.atleast_1d(axes).ravel(), imgs, titles, colors):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=6, color=c)
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(config.WORK / fname, dpi=110)
    plt.close(fig)
    print("saved", config.WORK / fname)


def main(n=48, seed=0):
    split = pd.read_csv(config.SPLIT_CSV)
    rows = []
    for _, r in split.iterrows():
        _, info = segment_bone(read_bgr(r.path))
        rows.append(dict(path=r.path, label=r.label, split=r.split, **info))
    rep = pd.DataFrame(rows)
    rep.to_csv(config.WORK / "mask_report.csv", index=False)
    print(f"mask pass rate: {rep.ok.mean():.1%} of {len(rep)} images")
    print(rep.reason.value_counts().to_string())
    print("\npass rate per class:")
    print(rep.groupby("label").ok.mean().round(2).to_string())

    pick = split.sample(min(n, len(split)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    a_imgs, a_t, a_c, b_imgs, b_t, b_c = [], [], [], [], [], []
    for _, r in pick.iterrows():
        img = read_bgr(r.path)
        m, info = segment_bone(img)
        col = "green" if info["ok"] else "red"
        a_imgs.append(overlay(img, m))
        a_t.append(f"{r.label} {info['reason']} a={info['area']:.2f}"); a_c.append(col)
        out, bi = bonebreak_augment(img, rng)
        b_imgs.append(out)
        b_t.append(f"{bi['aug']} {bi['frac']:.2f}"); b_c.append("green" if bi["ok"] else "red")
    grid(a_imgs, a_t, a_c, "mask_grid.png")
    grid(b_imgs, b_t, b_c, "break_grid.png")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
