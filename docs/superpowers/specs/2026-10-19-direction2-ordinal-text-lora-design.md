# Direction 2: Ordinal Text-Encoder LoRA — Design

Branch `direction2-ordinal-text-lora` (from `direction1-ordinal-geometry`). Source: Direction 2 of
`docs/lit_review_text_embedding.md`.

## Idea

Keep the grade prompts as clinical language and adapt the **text encoder itself** (LoRA), instead of
learning prompt tokens (OrdinalCLIP, CLIP-DR, RankFormer) or free prototypes (Direction 1). Ordinal
structure is enforced on the encoder's **outputs**, so the adapted encoder, not only five vectors,
learns that grades are ordered.

## Model

- **Descriptions.** Five ICDR-based descriptions per grade (`PROMPT_SETS["icdr_ensemble"]` in
  `src/models/prompt_learner/text_lora_prototypes.py`): the Direction 1 clinical sentence, the main
  fundus findings, and the ICDR level.
- **Text-LoRA** (`src/models/text_lora.py`). Each attention block of CLIP's 12-layer text transformer
  is wrapped so that `W'x = Wx + (α/r)·B A dropout(x)` on q, k, v (optionally o), with W frozen and
  B = 0 at the start. Defaults follow CLIP-LoRA: r = 2, α = 1, dropout 0.25. With B = 0 the block
  reproduces CLIP exactly (tested).
- **Prototypes.** `e_{k,m} = f_θ(s_{k,m})/‖·‖`, `t_k = mean_m e_{k,m}/‖·‖` (prompt ensembling as in
  CuPL). The zero-shot prototypes (B = 0) are the anchors. At k = 0 the model is the zero-shot
  description ensemble.
- **Image side.** Unchanged: RN50 with LoRA, CE + CLIP-DR ranking + L2RCLIP pairwise + SMS, as in
  Direction 1, so the two directions differ only in how the grade embeddings are produced.

## Ordinal regularisers on the text outputs

| Term | Weight | Definition |
|---|---|---|
| ladder | `w_ladder` | Direction 1's ladder loss on the prototypes: `d(i, i+Δ) + (Δ'−Δ)m < d(i, i+Δ')` |
| description ordering | `w_desc` | each description, scored against all prototypes with the CLIP logit scale, gets CE + CLIP-DR ranking loss towards its own grade (`src/models/text_losses.py`) |
| anchor | `w_anchor` | Direction 1's anchor loss: `Σ_k 2λ_k(1 − ⟨t_k, t⁰_k⟩)` |

## Baselines and ablations

- **CLIP-LoRA** (`configs/baselines/cliplora.yaml`): LoRA on both encoders, template
  "a photo of a {grade}.", CE only.
- **CLIP-LoRA + ordinal losses** (`cliplora_ordinal.yaml`): plus CLIP-DR ranking, L2RCLIP pairwise
  and SMS — the fair baseline for "is the text-side ordinal structure needed?".
- **Ablations** (`configs/direction2/`): no ladder / description ordering / anchor / any text
  regulariser; frozen text encoder (fixed zero-shot ensemble); one clinical sentence; plain grade
  sentences; text-LoRA rank 8; text-LoRA on q, k, v, o.

## Diagnostics

`geometry.json` per run: prototype distance matrix, step angles, ladder satisfaction, drift from the
zero-shot prototypes, and the spread of each grade's descriptions around its prototype.

## Notebook

`notebooks/DeepDRiD_Direction2.ipynb`: same flow as the Direction 1 notebook (per-epoch best/last
checkpoints, resume, tests on the best checkpoint, run / epoch / batch progress bars), results in
`results_d2/`.
