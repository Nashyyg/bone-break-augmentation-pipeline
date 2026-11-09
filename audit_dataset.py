
    #Reads data/raw/<class>/<file> and checks the filename convention
    #   <SpecimenID>_<Bone>_<View>.<ext>      e.g.  COH0042_Femur_V2.png
    #then prints a report about problems that would silently damage the results.


   # python audit_dataset.py            full report (reads every image for its size)

import argparse
import difflib
import re
from collections import Counter, defaultdict

import cv2
import pandas as pd

import config

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# <specimen>_<bone>[_<view...>] ; specimen = letters+digits, bone = one token without underscores.
# Everything after the bone token (V1, V2, anterior, left_lateral, ...) is the "view".
NAME_RE = re.compile(r"^(?P<spec>[A-Za-z]+\d+)_(?P<bone>[^_]+)(?:_(?P<view>.+))?$")


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def same_bone(folder, name_token):
    a, b = norm(folder), norm(name_token)
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dims", action="store_true", help="skip reading images (filenames only)")
    args = ap.parse_args()

    if not config.RAW_DIR.exists():
        raise SystemExit(f"{config.RAW_DIR} not found")

    rows = []
    for cdir in sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir()):
        for f in sorted(p for p in cdir.rglob("*") if p.suffix.lower() in EXTS):
            m = NAME_RE.match(f.stem)
            row = dict(label=cdir.name, file=f.name,
                       specimen=m["spec"] if m else None,
                       bone_in_name=m["bone"] if m else None,
                       view=(m["view"] if m and m["view"] else None),
                       name_ok=bool(m), width=None, height=None)
            if not args.no_dims:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is not None:
                    row["height"], row["width"] = img.shape[:2]
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No images found.")

    problems = 0
    print(f"{len(df)} images, {df.label.nunique()} classes, {df.specimen.nunique()} specimens\n")

    # 1. filename does not follow the convention
    bad = df[~df.name_ok]
    if len(bad):
        problems += len(bad)
        print("FILENAMES NOT MATCHING <ID>_<Bone>_<View>:")
        print(bad[["label", "file"]].to_string(index=False), "\n")

    # 2. label vs bone name inside the filename
    ok = df[df.name_ok].copy()
    ok["label_matches_name"] = [same_bone(l, b) for l, b in zip(ok.label, ok.bone_in_name)]
    mism = ok[~ok.label_matches_name]
    if len(mism):
        problems += len(mism)
        print("POSSIBLE MISLABELS (folder vs bone name in filename disagree):")
        print(mism[["label", "file"]].to_string(index=False))
        print("  -> check these by eye; fix the folder or the filename.\n")
    df = df.merge(ok[["label", "file", "label_matches_name"]], on=["label", "file"], how="left")

    # 3. unusual specimen-id formats (typos)
    digits = Counter(len(re.sub(r"\D", "", s)) for s in ok.specimen.unique())
    common_len = digits.most_common(1)[0][0]
    odd = sorted(s for s in ok.specimen.unique() if len(re.sub(r"\D", "", s)) != common_len)
    if odd:
        problems += len(odd)
        print(f"SPECIMEN IDS WITH UNUSUAL LENGTH (most IDs have {common_len} digits): {odd}")
        print("  -> possible typos; a typo makes one bone look like two specimens in the split.\n")

    # 4. a specimen appearing under more than one class
    spec_classes = ok.groupby("specimen").label.nunique()
    multi = spec_classes[spec_classes > 1]
    if len(multi):
        problems += len(multi)
        print("SPECIMENS UNDER MORE THAN ONE CLASS:")
        print(ok[ok.specimen.isin(multi.index)][["specimen", "label", "file"]].to_string(index=False), "\n")

    # 5. specimens/images per class -> how reliable can per-class metrics be?
    per = ok.groupby("label").agg(images=("file", "size"), specimens=("specimen", "nunique"))
    per["imgs_per_specimen"] = (per.images / per.specimens).round(1)
    # expected specimens in test ~ 15% of specimens
    per["~test_specimens"] = (per.specimens * config.SPLIT[2]).round(1)
    print("PER-CLASS SIZE (the split is by specimen, so 'specimens' is what matters):")
    print(per.to_string(), "\n")
    if (per.specimens < 8).any():
        print("  NOTE: classes with <8 specimens will have only ~1 specimen in the test set;")
        print("        per-class precision/recall will be very noisy. Consider cross-validation.\n")

    # 6. image size and what survives resizing to 224
    if not args.no_dims and df.width.notna().any():
        d = df.dropna(subset=["width"]).copy()
        d["aspect"] = d[["width", "height"]].max(axis=1) / d[["width", "height"]].min(axis=1)
        # With letterboxing the LONG side becomes IMG_SIZE, so the SHORT side becomes IMG_SIZE/aspect.
        d["short_side_at_224"] = (config.IMG_SIZE / d.aspect).round(0)
        print("IMAGE GEOMETRY (aspect = long side / short side):")
        print(d.groupby("label").agg(min_w=("width", "min"), max_w=("width", "max"),
                                     med_aspect=("aspect", "median"), max_aspect=("aspect", "max"),
                                     med_short_px=("short_side_at_224", "median"),
                                     min_short_px=("short_side_at_224", "min")).round(1).to_string())
        thin = int((d.short_side_at_224 < 40).sum())
        print(f"\n{thin} images would have a short side < 40 px after letterboxing to "
              f"{config.IMG_SIZE}px. Fine detail will be lost; see README 'Resolution'.")
        df = df.merge(d[["label", "file", "aspect", "short_side_at_224"]], on=["label", "file"], how="left")

    config.WORK.mkdir(parents=True, exist_ok=True)
    out = config.WORK / "audit_report.csv"
    df.to_csv(out, index=False)
    print(f"\n{problems} potential problem(s) flagged. Per-image table: {out}")


if __name__ == "__main__":
    main()
