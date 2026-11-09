

# Run every condition x seed 


# CONDITIONS 
#     Experiment 1  baseline   0% synthetic (no augmentation)
#     Experiment 2  standard   x p in {25,50,100}%
#     Experiment 3  bonebreak  x p in {25,50,100}%
# 7 conditions; each is repeated for every seed in config.SEEDS.


import argparse

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", default=config.PIPELINES)
    ap.add_argument("--proportions", nargs="+", type=float, default=config.PROPORTIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS (smoke tests)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke test)")
    ap.add_argument("--force", action="store_true", help="re-run finished conditions")
    ap.add_argument("--class-weighted", action="store_true",
                    help="EXTRA experiment: class-weighted loss; results go to results/runs_weighted/")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    conditions = []
    if 0.0 in a.proportions:
        conditions.append(("baseline", 0.0))
    conditions += [(pl, p) for pl in a.pipelines for p in a.proportions if p > 0]
    print(f"{len(conditions)} conditions x {len(a.seeds)} seeds = {len(conditions) * len(a.seeds)} runs")
    if a.dry_run:
        for c in conditions:
            print("  ", c)
        return
    from train import run   # imported here so --dry-run works on a machine without PyTorch
    for seed in a.seeds:
        for pl, p in conditions:
            run(pl, p, seed, epochs=a.epochs, pretrained=False if a.no_pretrained else None, force=a.force, class_weighted=a.class_weighted)


if __name__ == "__main__":
    main()
