
import cv2
import numpy as np

import config
from common import border_color

# Experiment 2: Standard augmentation 

def standard_augment(img, rng):
    # Random rotation, flips, crop+scale, brightness/contrast, Gaussian blur.

    # Each operation is applied INDEPENDENTLY with some probability, so different synthetic
    # images get different combinations (more diversity than always applying everything).
   
    h, w = img.shape[:2]
    bg = tuple(int(c) for c in border_color(img)) 
    out = img.copy()
    applied = []

    if rng.random() < 0.8:
        # Rotation +-30 degrees. Corners that rotate into view are filled with the BACKGROUND
        # colour (not black) so they look like more empty stage, not a new black object.
        ang = rng.uniform(-config.ROT_DEG, config.ROT_DEG)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
        applied.append("rot")
    if rng.random() < 0.5:                 # horizontal flip: bones are roughly mirror-symmetric
        out = cv2.flip(out, 1)             
        applied.append("hflip")
    if rng.random() < 0.5:                 # vertical flip
        out = cv2.flip(out, 0)
        applied.append("vflip")
    if rng.random() < 0.5:
        # Random crop then resize back: simulates different framing/zoom. A crop keeps
        # 75-100% of each side (config) - small enough to vary, large enough to keep the bone.
        s = rng.uniform(*config.CROP_SIDE_RANGE)
        ch, cw = max(8, int(h * s)), max(8, int(w * s))
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        applied.append("crop")
    if rng.random() < 0.6:
        # new = alpha * pixel + beta  (alpha = contrast, beta = brightness), clipped to 0-255.
        alpha = rng.uniform(*config.CONTRAST_RANGE)
        beta = rng.uniform(*config.BRIGHTNESS_RANGE)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        applied.append("bc")
    if rng.random() < 0.4:              
        k = int(rng.choice(config.BLUR_KERNELS))
        out = cv2.GaussianBlur(out, (k, k), 0)
        applied.append("blur")

    if not applied:
        # Guarantee the "synthetic" image differs from the original; otherwise a copy of the real
        # image would be counted as synthetic data and weaken the experiment.
        out = cv2.flip(out, 1)
        applied.append("hflip")
    return out, dict(aug="+".join(applied), frac=np.nan, ok=True, reason="ok")


# Experiment 3: Domain specific bone-break augmentation 

def local_background(img, lab=None):
 
    if lab is None:
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)
    return cv2.medianBlur(lab, config.MASK_BG_KERNEL).astype(np.float32), lab


def segment_bone(img):
    """Find the bone on the uniform green stage. Returns (mask_0_255, info).

    Method:
      1. Lab colour space (L = lightness, a/b = colour). Bone is grey/white/brown, the stage green.
      2. Local stage model = large median filter (see local_background).
      3. distance = colour difference from the local stage (a,b dominate; lightness counts half).
      4. Hysteresis threshold: pixels > MASK_MIN_DIST are candidates, but a component is kept only
         if it contains a clearly different pixel (> MASK_SEED_DIST). Catches the faint ends of thin
         bones without admitting noise.
      5. Morphological open/close; fill small holes; keep the largest component (even if a long bone reaches the
         image edge) plus other pieces >= MASK_KEEP_REL of it that do not touch the image edge
         (small pieces at the edge are stage-edge artefacts); fragmented bones keep their pieces.
      6. VALIDITY CHECKS (reject rather than guess): area outside [MASK_MIN_AREA, MASK_MAX_AREA]
         or more than MASK_MAX_BORDER of the border ring covered -> rejected with a reason.
    info = dict(ok, reason, area, border).
    """
    h, w = img.shape[:2]
    bg, lab = local_background(img)
    d = lab.astype(np.float32) - bg
    dist = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2) + config.MASK_L_WEIGHT * np.abs(d[..., 0])
    step = cv2.morphologyEx(bg[..., 0], cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > config.MASK_EDGE_STEP
    step = cv2.dilate(step.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dist[step] = 0
    cand = (dist > config.MASK_MIN_DIST).astype(np.uint8)
    n, lbl = cv2.connectedComponents(cand, connectivity=8)
    strong = np.unique(lbl[dist > config.MASK_SEED_DIST])
    m = (np.isin(lbl, strong[strong > 0]) * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inv = (m == 0).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, ar = st[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and ar < config.MASK_HOLE_FRAC * h * w:
            m[lbl == i] = 255
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= config.MASK_MIN_PIXELS]
    if not ids:
        return np.zeros((h, w), np.uint8), dict(ok=False, reason="empty", area=0.0, border=0.0)
    areas = {i: st[i, cv2.CC_STAT_AREA] for i in ids}
    biggest = max(ids, key=areas.get)

    def at_edge(i):
        return st[i, 0] == 0 or st[i, 1] == 0 or st[i, 0] + st[i, 2] == w or st[i, 1] + st[i, 3] == h
    # Keep the largest piece (even if a long bone reaches the image edge) plus other pieces that are
    # big enough relative to it. Smaller pieces touching the image edge are stage-edge artefacts.
    keep = [i for i in ids if i == biggest or
            (areas[i] >= config.MASK_KEEP_REL * areas[biggest] and not at_edge(i))]
    m = (np.isin(lbl, keep) * 255).astype(np.uint8)

    rw = max(2, int(round(config.MASK_RING_FRAC * min(h, w))))
    ring = np.zeros((h, w), bool)
    ring[:rw] = ring[-rw:] = True
    ring[:, :rw] = ring[:, -rw:] = True
    area = float((m > 0).mean())
    border = float((m[ring] > 0).mean())
    if area < config.MASK_MIN_AREA:
        reason = "too_small"
    elif area > config.MASK_MAX_AREA:
        reason = "too_big"
    elif border > config.MASK_MAX_BORDER:
        reason = "touches_border"
    else:
        reason = "ok"
    return m, dict(ok=reason == "ok", reason=reason, area=area, border=border)


def foreground_mask(img):
    #Mask only (0/255), kept for callers that do not need the validity info.
    return segment_bone(img)[0]


def _jagged_halfplane(center, n_dir, rng, jitter, extent):
    #Polygon covering everything on one side of a JAGGED line (used for 'end breaks').

    d = np.array([-n_dir[1], n_dir[0]])          # direction ALONG the cut (perpendicular to n_dir)
    far = 4 * extent
    ts = np.linspace(-extent, extent, int(rng.integers(10, 16)))
    line = [center + d * t + n_dir * rng.normal(0, jitter) for t in ts]
    poly = ([center - d * far] + line +
            [center + d * far, center + d * far + n_dir * far, center - d * far + n_dir * far])
    return np.round(np.array(poly)).astype(np.int32)


def _chip_polygon(center, radius, rng):
    k = int(rng.integers(6, 11))
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))          # sorted angles -> a simple (non-self-crossing) polygon
    rad = radius * rng.uniform(0.5, 1.0, k)               # varying radius -> irregular outline
    pts = center + np.stack([np.cos(ang), np.sin(ang)], axis=1) * rad[:, None]
    return np.round(pts).astype(np.int32)


def bonebreak_augment(img, rng):
    """Simulate breakage by DELETING part of the bone (replacing it with background).

    Two break types, chosen at random:
      * 'end_break' - a ragged, roughly transverse or oblique (+-35 degrees) cut removes one END
                      of the bone. Long bones typically break across their length, leaving a
                      missing end. The principal axis of the bone is found with PCA (eigenvector
                      of the pixel-coordinate covariance), so the cut is placed relative to the
                      bone's own orientation, whatever way it lies in the picture.
      * 'chip'      - an irregular polygon centred on a point of the bone's outline removes a
                      piece from the edge (edge chipping / partial loss, as in the proposal).

    ACCEPTANCE RULE: a candidate break is kept only if it removes between BREAK_MIN_FRAC and
    BREAK_MAX_FRAC of the bone area. Too little = no visible change (wasted synthetic image).
    Too much = the bone may no longer be identifiable, so the label would be wrong. If no valid
    break is found in BREAK_MAX_TRIES attempts the image is returned unchanged and flagged
    ok=False; build_sets.py then excludes it from the training mix.

    """
    h, w = img.shape[:2]
    bg = border_color(img)
    fg, minfo = segment_bone(img)
    area = int((fg > 0).sum())
   
    if not minfo["ok"]:
        return img.copy(), dict(aug="skipped_" + minfo["reason"], frac=0.0, ok=False,
                                reason=minfo["reason"])

    # Geometry of the bone, computed once and reused by every attempt.
    ys, xs = np.nonzero(fg)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)                                   # bone centroid
    _, evecs = np.linalg.eigh(np.cov((pts - mean).T))         # PCA; eigh returns ascending eigenvalues
    axis = evecs[:, 1]                                        # eigenvector of the LARGEST variance = long axis
    proj = (pts - mean) @ axis                                # each pixel's position along the long axis
    lo, hi = proj.min(), proj.max()                           # the two tips of the bone
    x, y, bw, bh = cv2.boundingRect(fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = np.concatenate([c.reshape(-1, 2) for c in cnts], axis=0)   # all outline points
    base = np.arctan2(axis[1], axis[0])                       # angle of the long axis
    size = max(h, w)

    for attempt in range(config.BREAK_MAX_TRIES):
        # After 15 failed attempts only try chips: they are smaller, so more likely to fit the size limits.
        kind = "end_break" if (attempt < 15 and rng.random() < config.BREAK_P_END) else "chip"
        if kind == "end_break":
            s = 1 if rng.random() < 0.5 else -1               # which tip breaks off
            t = rng.uniform(0.4, 0.95)                        # how far from the centroid toward that tip the cut falls
            center = mean + axis * (t * (hi if s > 0 else lo))
            ang = base + np.deg2rad(rng.uniform(-35, 35))     # +-35 deg => transverse/oblique breaks
            n_dir = np.array([np.cos(ang), np.sin(ang)]) * s  # points toward the part that is removed
            poly = _jagged_halfplane(center, n_dir, rng, jitter=0.025 * size, extent=0.75 * size)
        else:
            c = contour[int(rng.integers(len(contour)))].astype(np.float64)  
            poly = _chip_polygon(c, rng.uniform(0.08, 0.25) * max(bw, bh), rng)

        pm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pm, [poly], 255)                        # rasterise the polygon
        removed = (pm > 0) & (fg > 0)                        # ONLY bone pixels are erased, background untouched
        frac = removed.sum() / area
        if config.BREAK_MIN_FRAC <= frac <= config.BREAK_MAX_FRAC:
            out = img.copy()
            # Fill with the local stage model rather than one flat colour, so the
            # break is seamless on a vignetted stage and carries no obvious flat-patch cue.
            # (Telea inpainting smeared bone colour into the stage on large removals, so the fill comes
            # from the local stage model instead, with a 2-3 px feathered edge.)
            bg_lab, _ = local_background(img)
            bg_bgr = cv2.cvtColor(np.clip(bg_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            alpha = cv2.GaussianBlur(cv2.dilate(removed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32),
                                     (5, 5), 0)[..., None]
            out = np.clip(img * (1 - alpha) + bg_bgr * alpha, 0, 255).astype(np.uint8)
            return out, dict(aug=kind, frac=float(frac), ok=True, reason="ok")

    return img.copy(), dict(aug="no_valid_break", frac=0.0, ok=False, reason="no_valid_break")


# Lookup used by generate_synthetic.py so pipelines are selected by name from config.PIPELINES.
PIPELINE_FUNCS = {"standard": standard_augment, "bonebreak": bonebreak_augment}
