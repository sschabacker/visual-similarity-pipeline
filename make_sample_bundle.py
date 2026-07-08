"""
make_sample_bundle.py — run ONCE on the full data to create sample_data/
=========================================================================
Builds a small, shareable demo slice so 01_similarity_pipeline.ipynb runs in
DEMO_MODE without the full dataset.

How it picks images: the similarity pairs form many separate groups (each
arrangement task compares a fixed set of images only among themselves), so
the pair graph is a set of disconnected cliques rather than one web. Picking
random images, or snowballing a single group, leaves the test side of the
image split empty and the notebook errors. Instead this keeps WHOLE groups
(connected components) until it reaches the target image count, so every kept
image sits in a fully-paired group. That guarantees pairs on both sides of
the split and repeat judgments for the noise ceiling.

Konkle stimuli/ratings are publicly available data, so this small slice is
fine to commit to the repo.

Run in Colab (Drive mounted) or locally with the full CSVs on hand.
"""
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

BASE = "/content/drive/MyDrive/visual-similarity-data/Features/"
OUT = "sample_data/"
N_IMAGES = 400         # target images to keep (whole groups, a few MB of features)
SEED = 42
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

clip = pd.read_csv(BASE + "konkle_clip_features.csv")
dino = pd.read_csv(BASE + "konkle_dino_features.csv")
pairs = pd.read_csv(BASE + "konkle_similarity_pairs.csv")
pairs["image1"] = pairs["image1"].astype(str)
pairs["image2"] = pairs["image2"].astype(str)

# only pairs whose BOTH images have features
have_feats = set(clip["basename"].astype(str)) & set(dino["basename"].astype(str))
pairs = pairs[pairs["image1"].isin(have_feats) & pairs["image2"].isin(have_feats)]

# build adjacency over the unique pairs
adj = defaultdict(set)
for a, b in set(map(tuple, pairs[["image1", "image2"]].values)):
    adj[a].add(b)
    adj[b].add(a)

# find connected components (each is one fully-paired image group)
seen = set()
components = []
for node in adj:
    if node in seen:
        continue
    comp = {node}
    q = deque([node])
    while q:
        for nb in adj[q.popleft()]:
            if nb not in comp:
                comp.add(nb)
                q.append(nb)
    seen |= comp
    components.append(comp)

# add whole groups (shuffled for variety) until we reach the target size
components = [components[i] for i in rng.permutation(len(components))]
included = set()
for comp in components:
    included |= comp
    if len(included) >= N_IMAGES:
        break

# keep every judgment among the included images (repeats preserved)
keep = pairs[pairs["image1"].isin(included) & pairs["image2"].isin(included)]
imgs = set(keep["image1"]) | set(keep["image2"])

clip[clip["basename"].astype(str).isin(imgs)].to_csv(OUT + "clip_sample.csv", index=False)
dino[dino["basename"].astype(str).isin(imgs)].to_csv(OUT + "dino_sample.csv", index=False)
keep.to_csv(OUT + "pairs_sample.csv", index=False)

n_unique = keep.groupby(["image1", "image2"]).ngroups
print(f"Sample bundle written to {OUT}")
print(f"  images: {len(imgs)}   unique pairs: {n_unique}   judgments: {len(keep):,}")
print("  clip_sample.csv / dino_sample.csv / pairs_sample.csv")
print("Commit sample_data/ to the repo; the notebook runs on it in DEMO_MODE.")
