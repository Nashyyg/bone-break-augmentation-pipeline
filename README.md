# Bone-break augmentation for microfauna classification (EfficientNet-B0)


## Run order
pip install -r requirements.txt

# 1. put images in data/raw/<class_name>/   (folder name = label; filenames <SpecimenID>_<Bone>_<View>.png)
python audit_dataset.py          # Step 0: find mislabels, ID typos, tiny classes, extreme aspect ratios
python prepare_data.py           # Step 1: resize to 224x224, specimen-grouped stratified split
python generate_synthetic.py     # Step 2: offline synthetic pools (train only) + fragmented test set
python check_masks.py            #         LOOK at work/mask_grid.png and work/break_grid.png (48 images) BEFORE continuing.
python run_experiments.py --dry-run
python run_experiments.py --epochs 1 --seeds 0 --no-pretrained   # smoke test
python run_experiments.py        # Step 3: full run (resumable)
python analyse.py                # Step 4: tables + figures in results/


## File guide
| `config.py` | every setting, with the reason for each |
| `common.py` | shared helpers (background colour, paths, image reading) |
| `audit_dataset.py` | dataset sanity checks (Step 0) |
| `prepare_data.py` | preprocessing + split + leak check |
| `augment.py` | the two augmentation methods (standard, bone-break) |
| `generate_synthetic.py` | writes synthetic pools and the fragmented test set |
| `build_sets.py` | which images form each condition's training set |
| `train.py` | EfficientNet-B0 training/evaluation for one condition |
| `run_experiments.py` | loops over all conditions and seeds |
| `analyse.py` | aggregation, plots, paired comparison |
| `preview_augmentations.py` | visual check of masks and augmentations |



