#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==== HRF pairing: images ↔ manual1 (GT), optional FoV mask ====
import os, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
# ---- change only ROOT if your path differs
ROOT = Path(r"C:\Users\SC\Documents\Data and Code\AV DRIVE\AV_DRIVE\training")
IMAGES_DIR  = ROOT / "images"
MANUAL1_DIR = ROOT / "av"
FOV_DIR     = ROOT / "masks"       # optional

assert IMAGES_DIR.exists(),  f"Missing: {IMAGES_DIR}"
assert MANUAL1_DIR.exists(), f"Missing: {MANUAL1_DIR}"
if not FOV_DIR.exists():
    print("[WARN] FoV folder not found; continuing without FoV masks.")

SEED = 42
random.seed(SEED)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
def is_img(p: Path): return p.is_file() and p.suffix.lower() in IMG_EXTS

# HRF manual suffixes we may encounter
MANUAL_SUFFIXES = ["_manual1", "-manual1", "_vessel", "_vessels", "-vessel", "-vessels"]
MASK_SUFFIXES   = ["_mask", "-mask", "_fov", "-fov"]

def strip_any_suffix(stem: str, suffixes):
    s = stem.lower()
    for suf in suffixes:
        if s.endswith(suf):
            return stem[:len(stem)-len(suf)]
    return stem

# collect files
imgs  = sorted([p for p in IMAGES_DIR.rglob("*")  if is_img(p)])
mans  = sorted([p for p in MANUAL1_DIR.rglob("*") if is_img(p)])
fovs  = sorted([p for p in FOV_DIR.rglob("*")     if is_img(p)]) if FOV_DIR.exists() else []

# index manual1 and fov by several keys
def index_by_stem(paths, extra_suffixes):
    idx = defaultdict(list)
    for p in paths:
        st = p.stem
        idx[st].append(p)                # exact
        idx[st.lower()].append(p)        # lower
        base = strip_any_suffix(st, extra_suffixes)
        idx[base].append(p)
        idx[base.lower()].append(p)
    return idx

man_idx = index_by_stem(mans, MANUAL_SUFFIXES)
fov_idx = index_by_stem(fovs, MASK_SUFFIXES) if fovs else {}

pairs, missing_gt = [], []
for ip in imgs:
    key1 = ip.stem; key2 = ip.stem.lower()
    cand = list({*man_idx.get(key1, []), *man_idx.get(key2, [])})
    if not cand:
        # last try: attach known suffixes and re-check
        for suf in MANUAL_SUFFIXES:
            cand += man_idx.get(ip.stem + suf, [])
    cand = list(set(cand))
    if not cand:
        missing_gt.append(ip.name)
        continue
    # choose lexicographically stable (usually one)
    gt_path = sorted(cand)[0]
    # fov optional
    fov_path = None
    if fov_idx:
        c2 = list({*fov_idx.get(key1, []), *fov_idx.get(key2, [])})
        if not c2:
            for suf in MASK_SUFFIXES:
                c2 += fov_idx.get(ip.stem + suf, [])
        c2 = list(set(c2))
        if c2:
            fov_path = sorted(c2)[0]
    pairs.append({"img": ip, "gt": gt_path, "fov": fov_path})

print(f"Found {len(imgs)} images | Paired {len(pairs)} | Unpaired (no GT) {len(missing_gt)}")
if missing_gt: print("Examples:", missing_gt[:5])

# 70/15/15 split
random.Random(SEED).shuffle(pairs)
n = len(pairs)
n_tr = int(round(n*0.70))
n_va = int(round(n*0.15))
n_te = n - n_tr - n_va

train_recs = pairs[:n_tr]
val_recs   = pairs[n_tr:n_tr+n_va]
test_recs  = pairs[n_tr+n_va:]

print(f"SPLIT -> train {len(train_recs)} | val {len(val_recs)} | test {len(test_recs)}")

# map for true-original visualization later
ID2PATH_TEST = {r["img"].stem: str(r["img"]) for r in test_recs}




