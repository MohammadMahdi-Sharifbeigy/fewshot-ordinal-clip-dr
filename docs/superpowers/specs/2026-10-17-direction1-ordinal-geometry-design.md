# Direct Ordinal-Geometry Text Embeddings (Direction 1) — Design

*Date:* 2026-10-17 · *Status:* approved by the user 2026-10-17 · *Branch:* `direction1-ordinal-geometry` (worktree `.claude/worktrees/direction1`), based on `0eb0c8b`

## 1. Intent and why

The project is a few-shot ordinal-aware CLIP model for 5-grade diabetic retinopathy (DR) grading on DeepDRiD (CFP), built on RN50 with LoRA on the image encoder. The text side today is `RankFormerPromptLearner`: 16 context tokens and 5 rank tokens are learned from random initialisation and passed through the frozen CLIP text encoder.

**The problem, seen in our own results** (`results/ablation_results.csv`, produced with pre-`7d74268` code). The learned prompt side hurts in the low-shot regime:

| QWK | 1-shot | 4-shot | 16-shot | full |
|---|---|---|---|---|
| Ours (Full) | 0.135 | 0.297 | 0.696 | 0.870 |
| just LoRA | 0.441 | 0.518 | 0.699 | 0.851 |

Zero-shot QWK is ≈ 0 because the prompts are placeholders.

**The idea.** Replace the prompt learner with five grade embeddings t₀…t₄ that live directly in CLIP's joint space:
- They are initialised from clinical text through the frozen encoder, which is called only once.
- They are then optimised end-to-end together with LoRA.
- Training applies explicit ordinal-geometry constraints: a distance ladder, a smooth path, and a per-grade anchor to the initial text.

The literature review (`docs/lit_review_text_embedding.md` in the main checkout) found no prior work that combines direct, text-initialised prototype learning with ordinal geometry for few-shot ordinal medical grading. The closest prior work:
- **CLAP:** learns prototypes directly, but has no notion of grade order.
- **OrdinalCLIP, L2RCLIP, CLIP-DR:** model grade order, but learn it through prompt tokens.

**Success (the user's definition).**
- Beat **both CLIP-DR and L2RCLIP** on DeepDRiD, with **QWK as the headline metric**.
- The confusion matrix must be **ordinal-shaped**: error mass decreases steadily as you move away from the diagonal.
- A proposed metric that measures this shape (ODC, §6) is part of the contribution.

## 2. Goals / non-goals / anti-goals

**Goals**
- A new text module, `OrdinalGeometryEmbeddings` (direct trainable t₀…t₄), as a drop-in alternative to the prompt learners.
- Three geometry losses (ladder, path, anchor) and modular loss weights, so every ablation is a config change.
- Three initialisation modes and two anchor-weighting modes, plus a `clap` mode used only by the CLAP baseline.
- Rebuilt baselines (CLIP-DR, L2RCLIP, CLAP) in the same codebase, backbone and protocol.
- The ODC metric family, a saved confusion matrix, and per-epoch geometry diagnostics.
- Unit tests for every new loss, the metric and the module.

**Non-goals**
- Directions 2 (text-encoder LoRA) and 3 (distributional embeddings).
- Changing the data pipeline, splits or backbone.
- Running the full sweep. Implementation ends with unit tests and smoke runs.

**Anti-goals** (failure even if the code runs)
- Gains that come from changing the embedding norm or the logit scale instead of the angular geometry. This is why L2 normalisation is mandatory.
- Baselines that are unfaithful, or not run under the identical protocol.
- Breaking existing configs or results. `RankFormerPromptLearner`, `OrdinalCLIPHead` and every existing config must behave exactly as before.
- Touching the main checkout.

## 3. Constraints

- **Git:**
  - Work only in this worktree and on this branch.
  - No push without explicit permission.
  - No `Co-Authored-By: Claude` lines.
  - Never commit `CLIP-DR/` or `repos/`.
- **Data:** `data/` and `results/` are git-ignored, so the worktree has no data of its own.
  - Runs read the main checkout's `data/DeepDRiD/CFP_manifest.csv` and `data/DeepDRiD/splits/` (read-only) through absolute-path CLI overrides.
  - The test set is the manifest's `split == "test"` rows (400 images).
- **Configs:** all Direction-1 configs build on `configs/deepdrid_cfp.yaml`.
- **Protocol:**
  - RN50, LoRA r = 16 on conv1/2/3.
  - k ∈ {0, 1, 2, 4, 8, 16, 32, −1}, 3 seeds.
  - Early stopping on `val_qwk` over the query set.
  - Test on the official test split.

## 4. Method (exact formulas, decided by the user)

### 4.1 Module and normalisation
- `t ∈ ℝ^{K×D}` is an `nn.Parameter` (K = 5, D = 1024 for RN50).
- **Every forward:** t̂_k = t_k / (‖t_k‖₂ + ε). This happens in-graph, before the logits and before any geometry loss.
- The anchors t̂⁰ are stored L2-normalised as a buffer.
- Logits: `sim = logit_scale · f̂ T̂ᵀ`, where `logit_scale` is CLIP's (clamped at 100).
- The text encoder is used only at init, and never in forward.
- Cosine distance: d(t̂_i, t̂_j) = 1 − ⟨t̂_i, t̂_j⟩.

### 4.2 Distance ladder
L_ladder = Σ_i Σ_{1 ≤ Δ < Δ' ≤ K−1−i} max(0, d(t̂_i, t̂_{i+Δ}) − d(t̂_i, t̂_{i+Δ'}) + (Δ' − Δ)·m), with m = 0.05 by default.

### 4.3 Path (`path_mode` ∈ {smooth, align, both}; default `both`)
- L_smooth = Σ_{k=1}^{K−2} ‖(t̂_{k+1} − t̂_k) − (t̂_k − t̂_{k−1})‖²
- With v_k = t̂_{k+1} − t̂_k: L_align = Σ_{k=1}^{K−2} (1 − ⟨v_k, v_{k−1}⟩ / (‖v_k‖‖v_{k−1}‖ + ε))
- `both` = L_smooth + L_align.
- L_smooth and L_align are always logged separately (`train_l_smooth`, `train_l_align`).

### 4.4 Anchor (`anchor_weighting`)
L_anchor = Σ_k λ_k ‖t̂_k − t̂⁰_k‖² = Σ_k 2λ_k (1 − ⟨t̂_k, t̂⁰_k⟩)
- `uniform`: λ_k = 1.
- `inverse_sqrt_freq`: λ_k ∝ 1/√N_k^pop, normalised so that mean_k λ_k = 1. N_k^pop is **always the grade count of the full DeepDRiD training split** (manifest rows with `split == "train"`), never the support set.
- `clap` (CLAP baseline only): λ_k = the mean zero-shot softmax probability of grade k over that grade's support images. It is computed once at train start and then held fixed (CLAP's single outer ALM step).

### 4.5 Total loss
L = w_ce·CE + w_clipdr·L_rank(CLIP-DR) + w_l2r·L_pairwise(L2RCLIP) + w_ladder·L_ladder + w_path·L_path + w_anchor·L_anchor
- **SMS** remains `model.sms.enabled`. `loss.w_sms` ∈ {0, 1} is an alias: 0 turns SMS off.
- **Back-compatibility:**
  - `loss.lambda_rank` and `loss.mu_pairwise` map to `w_clipdr` and `w_l2r` when the new keys are absent.
  - `w_ce` defaults to 1.
  - The geometry weights default to 0, so existing configs reproduce exactly.
- **Direction-1 config defaults:** all weights = 1, m = 0.05, `path_mode = both`.

### 4.6 Initialisation (`init_mode`; default `icdr_clinical`)
**icdr_clinical**
0. "A fundus photograph showing no diabetic retinopathy, with normal retinal vasculature and no microaneurysms or hemorrhages."
1. "A fundus photograph showing mild non-proliferative diabetic retinopathy, characterized by the presence of microaneurysms only."
2. "A fundus photograph showing moderate non-proliferative diabetic retinopathy, with multiple microaneurysms, dot and blot hemorrhages, hard exudates, or cotton wool spots, less than severe."
3. "A fundus photograph showing severe non-proliferative diabetic retinopathy, with extensive intraretinal hemorrhages in four quadrants, venous beading, or intraretinal microvascular abnormalities."
4. "A fundus photograph showing proliferative diabetic retinopathy, characterized by neovascularization on the optic disc or elsewhere, vitreous hemorrhage, or preretinal hemorrhage."

**plain**
0. "A fundus photograph with no diabetic retinopathy."
1. "A fundus photograph with mild non-proliferative diabetic retinopathy."
2. "A fundus photograph with moderate non-proliferative diabetic retinopathy."
3. "A fundus photograph with severe non-proliferative diabetic retinopathy."
4. "A fundus photograph with proliferative diabetic retinopathy."

Sentences must fit in CLIP's 77-token context; raise an error if one would be truncated.

**random_ortho:** QR of a seeded Gaussian (seed = `data.seed`). No text encoder is used, and the anchors are the same vectors.

### 4.7 Training details
- The embeddings get their own param group: lr = `train.text_lr` (default `train.lr`), **weight_decay = 0**.
- **Post-step renormalisation:** `with torch.no_grad(): t.div_(t.norm(dim=-1, keepdim=True))`, controlled by `renorm_after_step: true`. The in-graph normalisation in forward stays as well.
- **k = 0:** every method is evaluated in its initial state, untrained.

### 4.8 Diagnostics
At the end of every train epoch, append the following to `<out_dir>/geometry.json`:
- the K×K cosine-distance matrix;
- adjacent step angles in degrees;
- anchor drift d(t̂_k, t̂⁰_k);
- the fraction of ladder constraints satisfied without margin.

## 5. Baselines and ablations

**Main table: matched baselines.** All use RN50, LoRA r = 16, AdamW and 100 epochs (shared recipe).
- **CLIP-DR:** OrdinalCLIP head + ranking loss + SMS, with LoRA.
- **L2RCLIP:** RankFormer + pairwise loss, with LoRA.
- **CLAP:** direct embeddings, `plain` init, CE + anchor (`clap` λ), no ordinal losses, with LoRA.

**Secondary rows: native recipes.**
- Stock CLIP-DR and L2RCLIP with the encoder frozen (existing `clipdr_on_ordinalclip.yaml` and `l2rclip_stock.yaml`).
- Native CLAP: frozen encoder, SGD momentum 0.9, lr 0.1 with cosine decay, 300 epochs.

**Ablations** (`configs/direction1/`)
1. Current RankFormer ("Ours (Full)").
2. Direct embeddings, no geometry.
3. Direct embeddings + full geometry: "CE + Geometry" and "CE + CLIP-DR + L2RCLIP + SMS + Geometry" (the main method).
4. Leave one out: − ladder, − path, − anchor.
5. Init: `icdr_clinical` / `plain` / `random_ortho`.
6. Anchor weighting: `uniform` / `inverse_sqrt_freq`.

Sweeps use `scripts/sweep.py --kshots 0,1,2,4,8,16,32,-1 --seeds 1,2,3`.

## 6. Metrics

**Kept:** QWK (headline), MAE, macro-F1, accuracy, AUC, LOS.

**Added:**
- **Confusion matrix** (raw counts) saved into `test_stats.json`.
- **ODC (locked by the user).**
  - **Definitions.** C is the K×K confusion matrix (rows = true grade), and P_{i,j} = C_{i,j} / Σ_c C_{i,c}. Rows with zero support are skipped.
  - **`ODC_offdiag`** measures off-diagonal error decay only:
    V_i^left = Σ_{d=1}^{i−1} max(0, P_{i,i−d−1} − P_{i,i−d}), V_i^right = Σ_{d=1}^{K−2−i} max(0, P_{i,i+d+1} − P_{i,i+d})
  - **`ODC_full`** is the same walk started at d = 0, so it also includes max(0, P_{i,i−1} − P_{i,i}) and max(0, P_{i,i+1} − P_{i,i}).
  - **Score:** V_i = V_i^left + V_i^right ∈ [0, 1], and ODC = 1 − mean_{present i} V_i. Ties incur 0 violation.
  - **`monotone_row_frac`:** the fraction of present rows with V_i = 0, computed for both variants.
  - **`mean_mass_by_dist[Δ]`**, Δ = 0…K−1: the average over present rows of Σ_{j : |i−j| = Δ} P_{i,j}.
  - **Sanity check:** a classifier that always predicts grade 2 gives row 0 = (0, 0, 1, 0, 0), so V_0 = 1 in both variants.
- **ECE** (15 bins), added because Direction 3 will need it.
- **Related metrics to cite and differentiate:** OCI / UOC (Cardoso & Sousa 2011) and per-sample %unimodal (Cardoso et al. 2023). Re-check novelty against OCI before claiming ODC is new.

## 7. Left to the builder
- File placement and naming.
- A runner branch that skips the text encoder for direct-prototype modules.
- Test design (CPU, with a fake text encoder where needed).
- Config layout.

## 8. Decisions (resolved with the user, 2026-10-17)
1. **Base commit:** `0eb0c8b`.
2. **Baselines:** matched (LoRA + shared recipe) in the main table; stock and native CLAP as secondary rows.
3. **ODC:** as in §6, with both variants.
4. **`inverse_sqrt_freq`:** always the population prior from the DeepDRiD training split.
5. **Approved builder's calls:**
   - `path_mode = both`, m = 0.05, weights = 1.
   - No-decay param group, post-step renormalisation plus in-graph normalisation.
   - Untrained evaluation at k = 0.
   - `w_sms` ∈ {0, 1}.
   - Back-compatible loss keys.
   - The 400-image test split.
