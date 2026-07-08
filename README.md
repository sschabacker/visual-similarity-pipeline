# Visual Similarity ML Pipeline

Predicting **human similarity judgments** on images from deep vision-model features (CLIP + DINO), and measuring how much of human similarity is actually recoverable from perception alone.

## Headline result

On a **leak-free, image-disjoint** train/test split, the model predicts **consensus human similarity at r ≈ 0.38** (95% CI [0.31, 0.44]) against a **split-half noise ceiling of 0.75**, recovering roughly **half of the reliably predictable structure**. A naive pair-level split had inflated this to r ≈ 0.34 through test-set leakage; correcting the split revealed the honest number and, with the noise ceiling, put it in context.

> The interesting part isn't the 0.38, it's the gap. Human raters agree strongly (ceiling 0.75), and frozen perceptual features capture about half of that agreement. The other half, not recoverable from CLIP/DINO, is where categorical and semantic structure likely lives.

## What it does

1. Extracts CLIP (ViT-B/32, 512d) and DINO (dino_vits8, 384d) features per image.
2. Builds pairwise features (abs-diff | product | cosine = 2,688d).
3. Splits by **image** (not pair), so no image appears in both train and test.
4. Races five models (Random Forest, Gradient Boosting, Ridge, Lasso, Linear) on the leak-free split.
5. Evaluates against **consensus** (mean) human similarity, not individual noisy judgments.
6. Computes a **noise ceiling** (split-half, Spearman-Brown) so performance is judged against what's achievable.

## Method notes (why the numbers are trustworthy)

- **Image-disjoint split.** A pair is kept only if both its images are on the same side of the split; straddlers are dropped (~32%). Verified zero image overlap between train and test.
- **Target normalization fit on train only** (no pooled leakage).
- **Consensus target.** Each pair was judged ~19 times; scoring against the pair mean (the consensus) rather than individual arrangements is the correct target for predicting human-like similarity.
- **Noise ceiling as denominator.** All five models converge to a similar honest r, evidence the signal is largely linearly accessible and that model capacity was not the bottleneck.

## Repo structure

```
visual-similarity-pipeline/
├── README.md
├── LICENSE                     # AGPL-3.0
├── CITATION.cff
├── requirements.txt
├── similarity_model.py         # reusable core
├── notebooks/
│   └── 01_similarity_pipeline.ipynb   # canonical trunk (runs the whole story)
├── sample_data/                # tiny slice so the notebook runs in demo mode
└── results/                    # small: race table, ceiling, headline plot
```

## Running it

**Demo mode (default):** the canonical notebook runs end to end on the small bundle in `sample_data/`, so you can see the full pipeline execute without the large dataset. Demo numbers are illustrative, not the headline result.

**Full data:** set `DEMO_MODE = False` and point the paths at the full feature and pair files (hosted separately, see below). This reproduces the headline result.

## Data & models

The full feature files, similarity matrices, and trained models are large (up to 263 MB) and are hosted on [OSF/Zenodo — DOI link TBD], not in this repo. A runnable sample lives in `sample_data/`.

## Status

The core pipeline is complete and validated. Feature interpretation, cross-dataset transfer, and application work are in progress.

## License & citation

Licensed under **AGPL-3.0**: free to use, study, and build on, with attribution required and derivatives kept open. See `LICENSE`. If you use this work, please cite it (see `CITATION.cff`).

---

*Built by Sydney Schabacker.*
