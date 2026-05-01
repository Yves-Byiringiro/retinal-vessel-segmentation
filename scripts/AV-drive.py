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


# ==== sanity check: 3 samples with overlay ====
import matplotlib.pyplot as plt

def _overlay_mask(img, msk, color=(255,0,0), alpha=0.45):
    v = img.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    idx = msk.astype(bool)
    v[idx] = (1-alpha)*v[idx] + alpha*col
    return v.clip(0,255).astype(np.uint8)

def show_samples(records, n=3, seed=42, title="train"):
    picks = random.Random(seed).sample(records, k=min(n, len(records)))
    for r in picks:
        img = np.array(Image.open(r["img"]).convert("RGB"))
        gt  = np.array(Image.open(r["gt"]).convert("L"))
        gt  = (gt > 0).astype(np.uint8)
        ov  = _overlay_mask(img, gt)
        fig, axs = plt.subplots(1,3, figsize=(11,3.5))
        axs[0].imshow(img); axs[0].set_title(f"{title} | {r['img'].stem}"); axs[0].axis("off")
        axs[1].imshow(gt,cmap="gray"); axs[1].set_title("GT (manual1)"); axs[1].axis("off")
        axs[2].imshow(ov); axs[2].set_title("Overlay"); axs[2].axis("off")
        plt.tight_layout(); plt.show()

show_samples(train_recs, n=3, seed=42, title="train")


# ==== Dataset & DataLoaders ====
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE   = 512
BATCH_SIZE = 2

class HRFDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = np.array(Image.open(rec["img"]).convert("RGB"))
        gt  = np.array(Image.open(rec["gt"]).convert("L"))
        gt  = (gt > 0).astype(np.uint8)
        fov = None
        if rec.get("fov"):
            fov = np.array(Image.open(rec["fov"]).convert("L"))
            fov = (fov > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=img, mask=gt)
            img, gt = aug["image"], aug["mask"]
            if fov is not None:
                fov = A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_NEAREST)(image=fov)["image"]

        img = img.astype(np.float32) / 255.0
        gt  = gt.astype(np.float32)
        if fov is not None: fov = fov.astype(np.float32)

        out = {
            "image": torch.from_numpy(img.transpose(2,0,1)),
            "mask":  torch.from_numpy(gt).unsqueeze(0),
            "id":    rec["img"].stem
        }
        if fov is not None:
            out["fov"] = torch.from_numpy(fov).unsqueeze(0).float()
        return out

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1,0.1,p=0.5),
])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR)])

pin = torch.cuda.is_available()
train_loader = DataLoader(HRFDataset(train_recs, transform=train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
val_loader   = DataLoader(HRFDataset(val_recs,   transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
test_loader  = DataLoader(HRFDataset(test_recs,  transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

b = next(iter(train_loader))
print("Batch shapes:", tuple(b["image"].shape), tuple(b["mask"].shape), "FOV" if "fov" in b else "(no FOV)")

