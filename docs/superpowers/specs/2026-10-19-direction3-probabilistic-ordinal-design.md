# Direction 3: Probabilistic (vMF) Grade Embeddings — Design

Branch `direction3-probabilistic-ordinal` (from `direction2-ordinal-text-lora`). Source: Direction 3
of `docs/lit_review_text_embedding.md`.

## Idea

DR grades are ambiguous at their borders (graders disagree most between grades 1–2 and 2–3). A single
prototype per grade cannot express that. Here each grade is a **distribution** on CLIP's unit sphere,
a von Mises–Fisher vMF(μ_k, κ_k): μ_k is where the grade lies, κ_k how sharply it is defined. The
classifier is the vMF-mixture posterior, so a broad grade spreads probability onto its neighbours.
That is the ordinal smoothing CLIP-DR adds post hoc with SMS, here built into the model.

## Model (`src/models/prompt_learner/vmf_grades.py`)

- **Means** μ_k: learned directly on the sphere (as in Direction 1), initialised from the zero-shot
  ICDR description ensemble of Direction 2 (frozen encoder), renormalised after each step.
- **Concentrations** κ_k = exp(log κ_k), one per grade, initialised at κ⁰ = 100 (CLIP's logit scale).
- **Likelihood scoring** (default):
  `logit_k = κ_k μ_kᵀz + log C_D(κ_k) − log C_D(κ⁰)`,
  `log C_D(κ) = ν log κ − (ν+1) log 2π − log I_ν(κ)`, ν = D/2 − 1 = 511 for RN50.
  The constant `log C_D(κ⁰)` cancels in the softmax; subtracting it in float64 keeps the logits at
  CLIP's scale (log C_D is about +2000 at κ = 100). With equal κ the posterior is exactly CLIP's softmax
  (tested), so zero-shot equals the description-ensemble zero-shot.
- **Expected-similarity scoring** (ablation): `logit_k = s · A_D(κ_k)/A_D(κ⁰) · μ_kᵀz`, the expected
  cosine to a prototype drawn from the grade, with A_D = I_{ν+1}/I_ν.
- **Bessel function** (`src/models/vmf.py`): uniform asymptotic (Debye) expansion of log I_ν with the
  first correction term; matches mpmath to about 1e-11 relative for κ = 1 … 1000 at ν = 511 (tested),
  differentiable.

## Losses

CE + CLIP-DR ranking (on the vMF logits) + L2RCLIP pairwise (on the means), Direction 1's ladder and
anchor on the means, and a log-normal prior `w_kappa · Σ_k (log κ_k − log κ⁰)²` that keeps the
concentrations from collapsing or exploding with few shots. SMS is off by default.

## Metrics

Besides QWK/MAE/F1/AUC/ODC/ECE, every run (all branches built from this one) now reports **NLL** and
the multi-class **Brier score**, proper scoring rules for the predicted grade distribution.
`geometry.json` adds the per-grade concentrations κ_k per epoch.

## Ablations (`configs/direction3/`)

shared κ (a learned temperature), fixed κ (deterministic prototypes), expected-similarity scoring,
with SMS, no ladder, no anchor, no κ prior, one clinical sentence for the init, κ⁰ = 50.

## Notebook

`notebooks/DeepDRiD_Direction3.ipynb`: same flow as the Direction 1/2 notebooks, results in
`results_d3/`; adds NLL/Brier/ECE curves and a κ-per-grade plot.
