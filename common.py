

#helper methods 

import json
from pathlib import Path

import cv2
import numpy as np

import config


def border_color(img):
    #Estimate the background colour of a photo as the MEDIAN colour of its 1-pixel border.

    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
    return np.median(border, axis=0).astype(np.uint8)


def rel(path):
    #Path , to enable the use of CSVs

    return Path(path).resolve().relative_to(config.ROOT).as_posix()


def load_classes():
    #Ordered list of class names.
    with open(config.CLASSES_JSON) as f:
        return json.load(f)


def read_bgr(rel_path):
    #Read an image (path relative to ROOT) as a BGR uint8 array; fail loudly if missing.

    img = cv2.imread(str(config.ROOT / rel_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rel_path)
    return img
