

#train and evaluate ONE condition with EfficientNet-B0 (transfer learning).


# A "run" = one (pipeline, synthetic proportion p, seed). run() does:
#     1. build the training manifest (build_sets.py)
#     2. load a pretrained EfficientNet-B0 and swap its final layer for one with our class count
#     3. train for EPOCHS epochs with Adam; after each epoch evaluate on the REAL validation set
#     4. keep the weights of the epoch with the best validation macro-F1
#     5. evaluate those weights ONCE on the test set (and the fragmented test set)
#     6. save everything to results/runs/<condition>.json


import json
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import config
from build_sets import build_manifest, condition_name
from common import load_classes

# Pretrained EfficientNet-B0 expects inputs normalised with the ImageNet channel mean/std.
# Using other numbers would shift the input distribution away from what the weights were trained on.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ImageDataset(Dataset):

    def __init__(self, df, class_to_idx):
        self.paths = [str(config.ROOT / p) for p in df.path]
        self.labels = [class_to_idx[l] for l in df.label]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV is BGR; the model expects RGB
        if img.shape[:2] != (config.IMG_SIZE, config.IMG_SIZE):
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD   # scale to 0-1, then standardise per channel
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.labels[i]   # HWC -> CHW


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False   # benchmark mode picks algorithms non-deterministically


def get_device():
    #Use an NVIDIA GPU if present, else Apple-silicon GPU, else CPU (CPU works but is slow)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_classes, pretrained):
 
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


@torch.no_grad()   
def predict(model, loader, device, criterion):
    """Run the model over a loader. Returns (true labels, predicted labels, mean loss)."""
    model.eval()  
    ys, ps, total_loss, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)   # weight by batch size -> true mean
        n += len(yb)
        ys.append(yb.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())               # predicted class = highest score
    return np.concatenate(ys), np.concatenate(ps), total_loss / max(n, 1)


def metrics(y, p, loss, n_classes, class_names):
    labels = list(range(n_classes))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y, p)), macro_precision=float(pr), macro_recall=float(rc),
        macro_f1=float(f1), weighted_f1=float(wf1), loss=float(loss),
        confusion=confusion_matrix(y, p, labels=labels).tolist(),
        per_class=classification_report(y, p, labels=labels, target_names=class_names,
                                        output_dict=True, zero_division=0),
    )


def run(pipeline, p, seed, epochs=None, pretrained=None, force=False, class_weighted=None):
    """Train + evaluate one condition. Returns (and saves) the result dict."""
    epochs = epochs or config.EPOCHS
    pretrained = config.PRETRAINED if pretrained is None else pretrained
    name = condition_name("baseline" if p == 0 else pipeline, p, seed)
    # class_weighted=True is the EXTRA experiment (loss weighted by inverse class frequency). Its results go in
    # a separate folder so they can never be mixed up with, or overwrite, the main unweighted results.
    class_weighted = config.USE_CLASS_WEIGHTS if class_weighted is None else class_weighted
    out_path = config.RESULTS_DIR / ("runs_weighted" if class_weighted else "runs") / f"{name}.json"
    if out_path.exists() and not force:       # makes a long experiment sweep resumable after a crash
        print(f"[skip] {name} (already done)")
        return json.load(open(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = load_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    split = pd.read_csv(config.SPLIT_CSV)
    manifest = build_manifest(pipeline, p, seed)
    sets = {"train": manifest,                                  # real + synthetic mix for this condition
            "val": split[split.split == "val"],                 # real only
            "test": split[split.split == "test"]}               # real only
    broken_csv = config.WORK / "test_broken.csv"
    if broken_csv.exists():
        tb = pd.read_csv(broken_csv)
        # keep ONLY images that were really broken (mask valid + break applied)
        sets["test_broken"] = tb[tb.ok].reset_index(drop=True) if "ok" in tb.columns else tb

    set_seed(seed)
    device = get_device()
    g = torch.Generator().manual_seed(seed)                     # seeds the shuffling order of batches
    train_ds = ImageDataset(sets["train"], c2i)
    bs = config.BATCH_SIZE
    loaders = {"train": DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g,
                                   drop_last=(len(train_ds) % bs == 1),   # a final batch of 1 can upset BatchNorm
                                   num_workers=config.NUM_WORKERS, pin_memory=device.type == "cuda")}
    for key in ("val", "test", "test_broken"):
        if key in sets:
            loaders[key] = DataLoader(ImageDataset(sets[key], c2i), batch_size=64, shuffle=False,
                                      num_workers=config.NUM_WORKERS)

    model = build_model(k, pretrained).to(device)
    weight = None
    if class_weighted:
        # Off in the main experiment so all conditions use the same loss.
        counts = np.bincount(train_ds.labels, minlength=k).astype(np.float32)
        weight = torch.tensor(counts.sum() / (k * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)   # standard loss for multi-class classification
    eval_criterion = nn.CrossEntropyLoss()           # unweighted, so losses are comparable across runs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    history, best = [], dict(f1=-1.0, loss=1e9, state=None, epoch=-1, train_acc=None)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # one pass over the training data 
        model.train()                               # train mode: dropout/batch-norm behave for training
        run_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                   # clear gradients from the previous step
            logits = model(xb)                      # forward pass
            loss = criterion(logits, yb)
            loss.backward()                         # backpropagation: compute gradients
            optimizer.step()                        # Adam update of the weights
            run_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)

        # evaluate on the real validation set 
        yv, pv, vloss = predict(model, loaders["val"], device, eval_criterion)
        vm = metrics(yv, pv, vloss, k, classes)
        rec = dict(epoch=epoch, train_loss=run_loss / seen, train_acc=correct / seen,
                   val_loss=vloss, val_acc=vm["accuracy"], val_macro_f1=vm["macro_f1"])
        history.append(rec)                         # kept for learning-curve plots
        print(f"  {name} ep{epoch:02d} train_loss={rec['train_loss']:.3f} train_acc={rec['train_acc']:.3f} "
              f"val_loss={vloss:.3f} val_f1={vm['macro_f1']:.3f}")

        # Keep the best epoch (validation macro-F1; ties broken by lower validation loss). 
        if vm["macro_f1"] > best["f1"] or (vm["macro_f1"] == best["f1"] and vloss < best["loss"]):
            best.update(f1=vm["macro_f1"], loss=vloss, epoch=epoch, train_acc=rec["train_acc"],
                        state={kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()})

    # final evaluation with the BEST epoch's weights 
    model.load_state_dict(best["state"])
    result = dict(name=name, pipeline="baseline" if p == 0 else pipeline, p=p, seed=seed,
                  mix_mode=config.MIX_MODE, epochs=epochs, pretrained=pretrained,
                  n_train=len(manifest), n_synthetic=int(manifest.is_synthetic.sum()),
                  best_epoch=best["epoch"], train_acc_at_best=best["train_acc"],
                  history=history, classes=classes, seconds=time.time() - t0)
    yv, pv, vl = predict(model, loaders["val"], device, eval_criterion)
    result["val"] = metrics(yv, pv, vl, k, classes)
    for key in ("test", "test_broken"):
        if key in loaders:
            y, pr, l = predict(model, loaders[key], device, eval_criterion)
            result[key] = metrics(y, pr, l, k, classes)
  
    result["generalisation_gap"] = best["train_acc"] - result["test"]["accuracy"]

    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[done] {name}: test acc={result['test']['accuracy']:.3f} "
          f"macro-F1={result['test']['macro_f1']:.3f} ({result['seconds']:.0f}s)")
    return result
