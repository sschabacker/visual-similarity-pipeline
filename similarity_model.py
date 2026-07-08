"""
similarity_model.py
===================
Reusable core for the visual-similarity pipeline: load CLIP + DINO features,
build a leak-proof (image-disjoint) train/test split, engineer pairwise
features, race candidate models, compute a split-half noise ceiling, and
evaluate predictions against consensus human similarity.

The canonical notebook (notebooks/01_similarity_pipeline.ipynb) walks the
same steps inline for readability. This module exposes them as importable
functions so the pipeline can be scripted or reused on another dataset.

Expected input format:
  * CLIP CSV: a "basename" column plus feature columns prefixed "clip_".
  * DINO CSV: a "basename" column plus feature columns prefixed "dino_".
  * Pairs CSV: columns "image1", "image2", "similarity" (lower = more
    similar in the source arrangement; it is inverted internally so higher
    means more similar). Pairs may repeat, one row per human judgment; the
    repeat structure is what makes the noise ceiling computable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler


def load_features(clip_csv, dino_csv):
    """Load per-image CLIP and DINO features and merge on image basename.

    Returns (feats, idx, merged): feats is an (n_images, n_dims) float32
    array, idx maps each basename to its row in feats, and merged is the
    joined dataframe.
    """
    clip_df = pd.read_csv(clip_csv)
    dino_df = pd.read_csv(dino_csv)
    cc = [c for c in clip_df.columns if c.startswith("clip_")]
    dc = [c for c in dino_df.columns if c.startswith("dino_")]
    merged = clip_df[["basename"] + cc].merge(dino_df[["basename"] + dc], on="basename")
    feats = merged[cc + dc].to_numpy(dtype=np.float32)
    idx = {b: i for i, b in enumerate(merged["basename"].astype(str))}
    return feats, idx, merged


def leakproof_split(pairs, idx, test_frac=0.2, seed=42):
    """Image-disjoint split.

    Partition the images test_frac / (1 - test_frac), then keep a pair only
    if both of its images fall on the same side. This guarantees that no
    image the model trained on can reappear in the test set. The target is
    inverted (higher = more similar) with min/max fit on train only, so no
    normalization information leaks from test into train.

    Returns (train_pairs, test_pairs, n_overlap); n_overlap is the number of
    images shared across the two sides and should be 0.
    """
    rng = np.random.default_rng(seed)
    pairs = pairs.copy()
    pairs["image1"] = pairs["image1"].astype(str)
    pairs["image2"] = pairs["image2"].astype(str)

    feat_imgs = set(idx)
    pairs = pairs[pairs["image1"].isin(feat_imgs) & pairs["image2"].isin(feat_imgs)].reset_index(drop=True)

    imgs = sorted(set(pairs["image1"]) | set(pairs["image2"]))
    perm = rng.permutation(len(imgs))
    n_test = int(round(test_frac * len(imgs)))
    test_imgs = {imgs[i] for i in perm[:n_test]}
    train_imgs = {imgs[i] for i in perm[n_test:]}

    tr = pairs["image1"].isin(train_imgs) & pairs["image2"].isin(train_imgs)
    te = pairs["image1"].isin(test_imgs) & pairs["image2"].isin(test_imgs)
    train_pairs, test_pairs = pairs[tr].copy(), pairs[te].copy()

    overlap = (set(train_pairs.image1) | set(train_pairs.image2)) & \
              (set(test_pairs.image1) | set(test_pairs.image2))

    ytr = train_pairs["similarity"].to_numpy(float)
    yte = test_pairs["similarity"].to_numpy(float)
    lo = ytr.min()
    span = (ytr.max() - ytr.min()) or 1.0
    train_pairs["y"] = 1 - (ytr - lo) / span
    test_pairs["y"] = 1 - (yte - lo) / span

    return train_pairs, test_pairs, len(overlap)


def engineer_pairs(df, feats, idx):
    """Build the pair representation from two images' feature vectors.

    Each pair becomes abs_diff | product | cosine, concatenated. With
    896-d image features that is a 2,688-d pair vector. Returns (X, y).
    """
    a = df["image1"].map(idx).to_numpy()
    b = df["image2"].map(idx).to_numpy()
    f1, f2 = feats[a], feats[b]
    ad, pr = np.abs(f1 - f2), f1 * f2
    den = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-10
    cos = np.repeat((np.sum(f1 * f2, axis=1) / den).astype(np.float32)[:, None], f1.shape[1], 1)
    X = np.concatenate([ad, pr, cos], 1).astype(np.float32)
    y = df["y"].to_numpy(float)
    return X, y


def race_models(train_pairs, test_pairs, feats, idx, subsample=40_000, n_strata=10, seed=42):
    """Fit five candidate models on a stratified subsample of train and score
    each by Pearson r against individual test judgments. Linear models get a
    StandardScaler fit on train; trees run on raw features. Returns a
    dataframe sorted best-first. This is a model-selection step, not the
    final evaluation.
    """
    rng = np.random.default_rng(seed)
    y_all = train_pairs["y"].to_numpy()
    bins = np.clip((y_all * n_strata).astype(int), 0, n_strata - 1)
    frac = min(subsample, len(train_pairs)) / len(train_pairs)
    sel = np.concatenate([
        rng.choice(np.where(bins == k)[0], int(round((bins == k).sum() * frac)), replace=False)
        for k in range(n_strata) if (bins == k).sum() > 0
    ])
    sub = train_pairs.iloc[sel]

    Xs, ys = engineer_pairs(sub, feats, idx)
    Xt, yt = engineer_pairs(test_pairs, feats, idx)
    sc = StandardScaler().fit(Xs)
    Xs_s, Xt_s = sc.transform(Xs), sc.transform(Xt)

    racers = [
        ("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=20, random_state=seed, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed), False),
        ("Ridge", RidgeCV(alphas=np.logspace(-2, 4, 13)), True),
        ("Lasso", LassoCV(alphas=np.logspace(-4, 1, 10), random_state=seed, n_jobs=-1, max_iter=5000), True),
        ("Linear", LinearRegression(), True),
    ]
    rows = []
    for name, m, scaled in racers:
        Xtr, Xte = (Xs_s, Xt_s) if scaled else (Xs, Xt)
        m.fit(Xtr, ys)
        rows.append((name, float(pearsonr(yt, m.predict(Xte))[0])))
    return pd.DataFrame(rows, columns=["model", "honest_r"]).sort_values(
        "honest_r", ascending=False).reset_index(drop=True)


def noise_ceiling(test_pairs, n_splits=50, seed=42):
    """Split-half, Spearman-Brown-corrected noise ceiling.

    Each pair is judged several times. Repeatedly split those judgments in
    half, correlate the two half-means across pairs, and Spearman-Brown
    correct. The mean over splits is the highest Pearson r any model
    predicting the consensus could reach, given how much raters disagree.
    Returns (ceiling, n_pairs_with_repeats).
    """
    rng = np.random.default_rng(seed)
    g = test_pairs.groupby(["image1", "image2"])["y"].apply(list)
    multi = [np.array(v) for v in g if len(v) >= 2]
    rs = []
    for _ in range(n_splits):
        A, B = [], []
        for v in multi:
            vv = v.copy()
            rng.shuffle(vv)
            h = len(vv) // 2
            if h:
                A.append(vv[:h].mean())
                B.append(vv[h:2 * h].mean())
        rh = pearsonr(A, B)[0]
        rs.append(2 * rh / (1 + rh))
    return float(np.mean(rs)), len(multi)


def evaluate_consensus(train_pairs, test_pairs, feats, idx, fit_cap=80_000, seed=42, n_boot=2000):
    """Fit Ridge on a capped train sample and score against the consensus
    (per-pair mean) judgment rather than individual noisy judgments, which is
    the correct target for predicting human-like similarity. Returns a dict
    with r_individual, r_consensus, a bootstrap 95% CI, n_pairs, and the
    fitted model.
    """
    rng = np.random.default_rng(seed)
    fit_df = train_pairs.sample(min(fit_cap, len(train_pairs)), random_state=seed)
    Xf, yf = engineer_pairs(fit_df, feats, idx)
    Xte, yte = engineer_pairs(test_pairs, feats, idx)
    scf = StandardScaler().fit(Xf)
    model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(scf.transform(Xf), yf)
    pred = model.predict(scf.transform(Xte))

    r_ind = float(pearsonr(yte, pred)[0])
    te = test_pairs.copy()
    te["pred"] = pred
    agg = te.groupby(["image1", "image2"]).agg(y=("y", "mean"), p=("pred", "mean")).reset_index()
    r_con = float(pearsonr(agg.y, agg.p)[0])
    boot = [pearsonr(agg.y.values[s], agg.p.values[s])[0]
            for s in (rng.integers(0, len(agg), len(agg)) for _ in range(n_boot))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"r_individual": r_ind, "r_consensus": r_con,
            "ci95": [float(lo), float(hi)], "n_pairs": int(len(agg)), "model": model}
