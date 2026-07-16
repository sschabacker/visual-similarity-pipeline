# Visual Similarity ML Pipeline

Predicting **human similarity judgments** on images from deep vision-model
features (CLIP + DINO), and measuring how much of human similarity is
actually recoverable from perception alone.

![Headline result](results/headline_plot.png)

## Headline result

On a **leak-free, image-disjoint** train/test split, the model predicts
**consensus human similarity at r ≈ 0.38** (95% CI [0.31, 0.44]) against a
**split-half noise ceiling of 0.75**; a ceiling-normalized score of about
**0.51**, or roughly half of the achievable ceiling.

An earlier version of this pipeline split by **pair** rather than by image,
letting the same image appear on both sides of the train/test boundary, and
reported r ≈ 0.34 against individual judgments. Fixing the split cut
same-target performance roughly in half (all five models converge at
r ≈ 0.14 to 0.18 against individual judgments), showing the leaky number
was mostly memorization. The second problem was the evaluation itself:
individual judgments are noisy, and raters only agree with each other at
0.75, so no model scored against single raters can approach 1.0. Scoring
the clean model against the rater **consensus** and reporting it against
the **noise ceiling** gives a number that is both correct and
interpretable: 0.38 of an achievable 0.75.

> The interesting part isn't the 0.38, it's the gap. Human raters agree
> strongly (ceiling 0.75), and frozen perceptual features capture about half
> of that agreement. The other half, not recoverable from CLIP/DINO, is where
> categorical and semantic structure likely lives.

## Why this question

This pipeline grew out of experimental work on how category labels shape
visual memory (MA thesis plus doctoral research on categorical interference).
A recurring problem in that literature is separating *perceptual* similarity
from *categorical* similarity: when two objects are confused in memory, is
that because they look alike or because they share a category? Quantifying
how much of human similarity judgment is predictable from perceptual
features alone puts a number on that boundary, and the residual is where
category structure can be studied directly. The human similarity data come
from the spatial-arrangement (SpAM) paradigm; the images extend the Konkle
lab Mass Memory Database.

## What it does

1. Extracts CLIP (ViT-B/32, 512d) and DINO (dino_vits8, 384d) features per image.
2. Builds pairwise features (abs-diff | product | cosine = 2,688d).
3. Splits by **image** (not pair), so no image appears in both train and test.
4. Races five models (Random Forest, Gradient Boosting, Ridge, Lasso, Linear) on the leak-free split.
5. Evaluates against **consensus** (mean) human similarity, not individual noisy judgments.
6. Computes a **noise ceiling** (split-half, Spearman-Brown) so performance is judged against what's achievable.

## Method notes (why the numbers are trustworthy)

- **Image-disjoint split.** A pair is kept only if both its images are on the
  same side of the split; straddlers are dropped (~32%). Verified zero image
  overlap between train and test.
- **Target normalization fit on train only** (no pooled leakage).
- **Consensus target.** Each pair was judged ~19 times; scoring against the
  pair mean (the consensus) rather than individual arrangements is the
  correct target for predicting human-like similarity.
- **Noise ceiling as denominator.** All five models converge to a similar
  honest r, evidence the signal is largely linearly accessible and that
  model capacity was not the bottleneck.

## Repo structure

```
visual-similarity-pipeline/
├── README.md
├── LICENSE                     # AGPL-3.0
├── CITATION.cff
├── requirements.txt
├── similarity_model.py         # reusable core
├── make_sample_bundle.py       # builds the demo slice from the full data
├── notebooks/
│   └── 01_similarity_pipeline.ipynb   # canonical trunk (runs the whole story)
├── sample_data/                # tiny slice so the notebook runs in demo mode
└── results/                    # small: race table, ceiling, headline plot
```

## Running it

Install dependencies first:

```
pip install -r requirements.txt
```

**Demo mode (default):** the canonical notebook runs end to end on the small
bundle in `sample_data/`, so you can see the full pipeline execute without
the large dataset. Demo numbers are illustrative, not the headline result.

**Full data:** set `DEMO_MODE = False` and point the paths at the full
feature and pair files (hosted separately; see below). This reproduces the
headline result.

## Data & models

The full feature files, similarity matrices, and trained models are large
(up to 263 MB) and are hosted separately rather than in this repo; they are
available on request while a permanent archive (OSF/Zenodo DOI) is set up.
A runnable sample lives in `sample_data/`.

## Status & roadmap

The core pipeline is complete and validated. Planned work is tracked in
[Issues](../../issues), including:

- **Feature interpretation:** which feature dimensions carry the predictive signal.
- **Cross-dataset transfer:** does a model trained on one image set predict similarity on another?
- **Application work:** using the residual (the non-perceptual half) to study categorical structure directly.

## Related work

This is one of several research tools I maintain; see also
[trial-data-extractor](https://github.com/sschabacker/trial-data-extractor),
a full-stack extraction and QC system for research trial PDFs.

## License & citation

Licensed under **AGPL-3.0**: free to use, study, and build on, with
attribution required and derivatives kept open. See `LICENSE`. If you use
this work, please cite it (see `CITATION.cff`).

---

*Built by Sydney Schabacker.*
