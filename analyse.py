

# OUTPUTS (in results/)
#     all_runs.csv                one row per run (every metric)
#     summary_by_condition.csv    mean and std over seeds for each (pipeline, proportion)
#     pipeline_comparison.csv     paired difference bonebreak - standard at each proportion
#     curve_*.png                 metric vs % synthetic, one line per pipeline (error bars = std over seeds)
#     confusion_matrices.png      FIXED conditions (config.CONFUSION_CONDITIONS), chosen before seeing results

import json

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

import sys
WEIGHTED = "--weighted" in sys.argv
RUNS = config.RESULTS_DIR / ("runs_weighted" if WEIGHTED else "runs")
OUT = config.RESULTS_DIR / ("weighted" if WEIGHTED else "")
COLORS = {"standard": "tab:blue", "bonebreak": "tab:red"}


def macro_f1_excluding(cm, classes, excl):
    cm = np.array(cm, float)
    keep = [i for i, c in enumerate(classes) if c != excl]
    cm = cm[keep, :]                                 
    f1s = []
    for r, i in enumerate(keep):
        tp = cm[r, i]
        fn = cm[r].sum() - tp
        fp = cm[:, i].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def load():
    recs, raw = [], {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.load(open(f))
        raw[r["name"]] = r
        row = dict(name=r["name"], pipeline=r["pipeline"], p=r["p"], seed=r["seed"],
                   n_train=r["n_train"], n_synthetic=r["n_synthetic"], best_epoch=r["best_epoch"],
                   gap=r["generalisation_gap"], train_acc=r["train_acc_at_best"],
                   val_macro_f1=r["val"]["macro_f1"])
        for split in ("test", "test_broken"):
            if split in r:
                row[f"{split}_macro_f1_no_{config.EXCLUDE_CLASS}"] = macro_f1_excluding(
                    r[split]["confusion"], r["classes"], config.EXCLUDE_CLASS)
                for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
                    row[f"{split}_{k}"] = r[split][k]
        recs.append(row)
    return pd.DataFrame(recs), raw


def with_baseline_on_both(df):
    base = df[df.pipeline == "baseline"]
    parts = [df[df.pipeline != "baseline"]]
    for pl in config.PIPELINES:
        parts.append(base.assign(pipeline=pl))
    return pd.concat(parts, ignore_index=True)


def curve_plot(agg, metric, title, fname):
    #One figure: metric (mean +- std over seeds) against % synthetic data, per pipeline.
    fig, ax = plt.subplots(figsize=(6, 4))
    for pl, g in agg.groupby("pipeline"):
        g = g.sort_values("p")
        ax.errorbar(g.p * 100, g[(metric, "mean")], yerr=g[(metric, "std")].fillna(0),
                    marker="o", capsize=3, label=pl, color=COLORS.get(pl))
    ax.set_xlabel("Synthetic data in training set (%)")
    ax.set_ylabel(title)
    ax.set_title(f"{title} vs synthetic proportion")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def confusion_plot(raw, names, labels, classes, fname):
    #Row-normalised confusion matrices (summed over seeds) side by side.
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, nm_list, lab in zip(axes, names, labels):
        cm = sum(np.array(raw[n]["test"]["confusion"]) for n in nm_list)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)       # each row sums to 1
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def main():
    df, raw = load()
    if df.empty:
        raise SystemExit("No results in results/runs/. Run run_experiments.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    # Used only seeds for which EVERY condition finished, so each condition averages over the same seeds
    # (otherwise a half-finished seed would bias some conditions and not others).
    need = {("baseline", 0.0)} | {(pl, p) for pl in config.PIPELINES for p in config.PROPORTIONS if p > 0}
    if WEIGHTED:   # the extra experiment only covers 0% and 100%
        need = {("baseline", 0.0), ("standard", 1.0), ("bonebreak", 1.0)}
    have = df.groupby("seed").apply(lambda g: need <= set(zip(g.pipeline, g.p)), include_groups=False)
    complete = sorted(have[have].index)
    print(f"seeds with all {len(need)} conditions finished: {complete} ({len(complete)}); "
          f"incomplete seeds ignored: {sorted(set(df.seed) - set(complete))}")
    df = df[df.seed.isin(complete)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No seed has all conditions finished yet.")
    df.to_csv(OUT / "all_runs.csv", index=False)
    full = with_baseline_on_both(df)

    # mean / std over seeds for each condition 
    metrics = [c for c in full.columns if c.startswith(("test_", "gap"))]
    agg = full.groupby(["pipeline", "p"])[metrics].agg(["mean", "std"]).reset_index()
    flat = agg.copy()
    flat.columns = ["_".join(c).strip("_") for c in flat.columns]
    flat.to_csv(OUT / "summary_by_condition.csv", index=False)
    print(flat[["pipeline", "p", "test_accuracy_mean", "test_accuracy_std",
                "test_macro_f1_mean", "test_macro_f1_std"]].round(4).to_string(index=False))

    for m, t in [("test_macro_f1", "Macro F1"), ("test_accuracy", "Accuracy"),
                 ("test_macro_precision", "Macro precision"), ("test_macro_recall", "Macro recall"),
                 ("gap", "Generalisation gap (train acc - test acc)")]:
        curve_plot(agg, m, t, f"curve_{m}.png")
    if "test_broken_macro_f1" in full.columns:
        curve_plot(agg, "test_broken_macro_f1", "Macro F1 on fragmented test set", "curve_broken_macro_f1.png")

    # effect sizes with 95% bootstrap CIs (over seeds); p-values are secondary 
    # With few test images the test has little power, so leaning on effect sizes and plots.
    rng = np.random.default_rng(0)

    def ci(d):
        d = np.asarray(d, float)
        if len(d) < 2:
            return np.nan, np.nan
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        return tuple(np.percentile(bs, [2.5, 97.5]))

    rows = []
    for p in sorted(x for x in df.p.unique() if x > 0):
        std_ = df[(df.pipeline == "standard") & (df.p == p)].set_index("seed")
        bb_ = df[(df.pipeline == "bonebreak") & (df.p == p)].set_index("seed")
        base_ = df[df.pipeline == "baseline"].set_index("seed")
        for metric in [c for c in df.columns if c.startswith(("test_macro_f1", "test_broken_macro_f1", "test_accuracy"))]:
            for lab, x, y in (("bonebreak - standard", bb_, std_), ("standard - baseline", std_, base_),
                              ("bonebreak - baseline", bb_, base_)):
                common = x.index.intersection(y.index)
                if len(common) == 0 or metric not in x.columns:
                    continue
                d = (x.loc[common, metric] - y.loc[common, metric]).values
                lo, hi = ci(d)
                pval = np.nan
                if len(d) >= 6 and np.any(d != 0):
                    try:
                        from scipy.stats import wilcoxon
                        pval = wilcoxon(d).pvalue
                    except ImportError:      
                        pval = np.nan
                rows.append(dict(p=p, metric=metric, comparison=lab, n_seeds=len(d), mean_diff=d.mean(),
                                 ci95_low=lo, ci95_high=hi, std_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                                 wilcoxon_p=pval))
    if rows:
        comp = pd.DataFrame(rows)
        comp.to_csv(OUT / "effect_sizes.csv", index=False)
        print("\nEffect sizes (mean paired difference, 95% bootstrap CI over seeds), macro-F1 on the test set:")
        print(comp[comp.metric == "test_macro_f1"].round(4).to_string(index=False))
        print("(CI that includes 0 = no clear effect. p-values are in effect_sizes.csv but have little power here.)")
        # forest-style plots: augmentation question (intact test set) and robustness question (fragmented test set),
        # each with and without digesto
        ex_ = config.EXCLUDE_CLASS
        panels = [("test_macro_f1", "effect_sizes.png", "A. Augmentation: intact test set"),
                  (f"test_macro_f1_no_{ex_}", f"effect_sizes_no_{ex_}.png", f"A. Augmentation: intact test set, no {ex_}"),
                  ("test_broken_macro_f1", "effect_sizes_fragmented.png", "B. Robustness: fragmented test set"),
                  (f"test_broken_macro_f1_no_{ex_}", f"effect_sizes_fragmented_no_{ex_}.png",
                   f"B. Robustness: fragmented test set, no {ex_}")]
        styles = [("bonebreak - standard", "tab:red"), ("bonebreak - baseline", "tab:purple"),
                  ("standard - baseline", "tab:blue")]

        def forest(ax, metric, title):
            k = comp[comp.metric == metric]
            for j, (lab, c) in enumerate(styles):
                g = k[k.comparison == lab].sort_values("p")
                if g.empty:
                    continue
                ax.errorbar(g.p * 100 + (j - 1) * 2, g.mean_diff,
                            yerr=[g.mean_diff - g.ci95_low, g.ci95_high - g.mean_diff],
                            marker="o", capsize=3, color=c, label=lab)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks([25, 50, 100]); ax.set_xlabel("Synthetic data (%)")
            ax.set_ylabel("Paired difference in macro-F1"); ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        for metric, fname, title in panels:
            if metric in set(comp.metric):
                fig, ax = plt.subplots(figsize=(6, 3.5))
                forest(ax, metric, title)
                fig.tight_layout(); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
                print(f"\n{title}  (mean paired difference [95% CI], {metric})")
                t = comp[comp.metric == metric]
                print(t[["p", "comparison", "mean_diff", "ci95_low", "ci95_high"]].round(4).to_string(index=False))
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for ax, (metric, fname, title) in zip(axes.ravel(), panels):
            forest(ax, metric, title)
        fig.tight_layout(); fig.savefig(OUT / "effect_sizes_2x2.png", dpi=130); plt.close(fig)

    # every seed as a dot (shows the spread instead of hiding it in a mean) 
    fig, ax = plt.subplots(figsize=(6, 4))
    for pl, g in full.groupby("pipeline"):
        ax.scatter(g.p * 100 + (-1.2 if pl == "standard" else 1.2), g.test_macro_f1, s=14, alpha=0.5,
                   color=COLORS.get(pl), label=pl)
        m = g.groupby("p").test_macro_f1.mean()
        ax.plot(m.index * 100 + (-1.2 if pl == "standard" else 1.2), m.values, color=COLORS.get(pl))
    ax.set_xlabel("Synthetic data (%)"); ax.set_ylabel("Test macro-F1 (one dot per seed)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "seed_scatter.png", dpi=130); plt.close(fig)

    # confusion matrices
    classes = next(iter(raw.values()))["classes"]
    names, labels = [], []
    for pl, p in config.CONFUSION_CONDITIONS:
        sel = df[(df.pipeline == pl) & (df.p == p)]
        if len(sel):
            names.append(list(sel.name)); labels.append(f"{pl} ({int(p * 100)}%)")
    confusion_plot(raw, names, labels, classes, "confusion_matrices.png")
    # same, without the digesto class (not a true bone type)
    ex = [i for i, c in enumerate(classes) if c != config.EXCLUDE_CLASS]
    raw_ex = {n: dict(test=dict(confusion=np.array(r["test"]["confusion"])[np.ix_(ex, ex)].tolist()))
              for n, r in raw.items()}
    confusion_plot(raw_ex, names, labels, [classes[i] for i in ex], f"confusion_matrices_no_{config.EXCLUDE_CLASS}.png")
    print(f"\nWrote tables and figures to {OUT}")


if __name__ == "__main__":
    main()
