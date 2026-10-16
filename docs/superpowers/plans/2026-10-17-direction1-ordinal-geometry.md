# Direction 1: Direct Ordinal-Geometry Text Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a text module whose five grade embeddings are learned directly on the unit sphere. They are initialised from clinical text and trained with ladder, path and anchor losses. Alongside it: modular loss weights, the ODC metrics, matched and native baselines, and ablation configs.

**Architecture:**
- `OrdinalGeometryEmbeddings` returns normalised prototypes `[K, D]` directly. The runner skips the text encoder when a module sets `direct_prototypes = True`.
- Geometry losses live in pure functions in one file, and the metrics in another.
- The runner resolves loss weights through a helper that maps old keys to new ones.
- `train.py` gains recursive config defaults, a k = 0 no-fit path, and JSON-safe stats.

**Tech Stack:** Python 3.13, PyTorch 2.8, PyTorch Lightning 2.x, OpenAI `clip` (RN50 cached at `~/.cache/clip/RN50.pt`), OmegaConf, numpy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-direction1-ordinal-geometry-design.md`

## Global Constraints

- Work only in worktree `.claude/worktrees/direction1`, branch `direction1-ordinal-geometry`. Never edit the main checkout. No push. Commit messages carry **no** `Co-Authored-By` line.
- Python interpreter: `E:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\Workspace\env\Scripts\python.exe` (call it `PY`). Run it from PowerShell with `& $PY ...`. Install pytest into it once (Task 1).
- **Import isolation:** the env may have the main checkout installed in editable mode, so `import src` could silently load the main checkout's code. Always run tests as `& $PY -m pytest` from the worktree root, which puts cwd first on `sys.path`. Run scripts with `$env:PYTHONPATH = (Get-Location).Path` set. Task 1 Step 1 also verifies that `src.__file__` points into the worktree.
- Existing configs and modules must behave exactly as before: all geometry weights default to 0, `w_ce` to 1, and `lambda_rank`/`mu_pairwise` keep working.
- K = `model.prompt_learner.num_ranks` (5). D = CLIP text projection width (1024 for RN50). ε = 1e-8.
- Defaults: margin `m = 0.05`, `path_mode = "both"`, geometry weights in Direction-1 configs = 1.0, `renorm_after_step = true`, text param group weight decay 0.0, `text_lr` = `train.lr` unless set.
- Data for runs is passed as CLI overrides: `data.manifest="E:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\Workspace\data\DeepDRiD\CFP_manifest.csv"` and `data.splits_dir="E:\Research\Dr.Mohammad(LAB)\ordinal classify\MMRDR\Workspace\data\DeepDRiD\splits"`. Manifest image paths are absolute.
- Prompt sentences are copied verbatim from spec §4.6.

## Review Focus

1. **Gradient flow through normalisation.** If forward used the raw `t` (or normalised under `no_grad`), the geometry losses would push the norm instead of the angle. Test: `test_forward_is_unit_norm_and_differentiable` (Task 2).
2. **Back-compat of old configs.** A config with only `lambda_rank`/`mu_pairwise` must produce the same total loss as before. Test: `test_resolve_loss_weights_backcompat` (Task 4).
3. **Grades missing from the test set.** A row with no support must be skipped, not counted as a division by zero or NaN. Test: `test_odc_skips_empty_rows` (Task 3).
4. **Post-step renorm must not break the optimiser.** It must leave `requires_grad` intact and keep ‖t_k‖ = 1 after a real optimiser step. Test: `test_renorm_after_optimizer_step` (Task 4).
5. **Nested config defaults.** A config under `configs/direction1/` that defaults to `../deepdrid_cfp` (which itself defaults to `default`) must still carry every key from `default.yaml`. Test: `test_load_config_recursive_defaults` (Task 5).

---

### Task 1: Geometry losses

**Files:**
- Create: `src/models/geometry_losses.py`
- Test: `tests/test_geometry_losses.py`

**Interfaces:**
- Produces:
  - `cosine_distance_matrix(t_hat: Tensor[K,D]) -> Tensor[K,K]` (1 − t̂t̂ᵀ)
  - `ladder_loss(t_hat, margin: float = 0.05) -> Tensor[]`
  - `ladder_satisfied_frac(t_hat) -> float` (fraction of ladder pairs with d_near < d_far, no margin)
  - `path_losses(t_hat, eps=1e-8) -> tuple[Tensor, Tensor]` returning `(l_smooth, l_align)`
  - `path_loss(t_hat, mode: str) -> Tensor` with `mode ∈ {"smooth","align","both"}`; raises `ValueError` otherwise
  - `anchor_loss(t_hat, anchors_hat, lambdas: Tensor[K]) -> Tensor` = Σ 2λ_k(1 − ⟨t̂_k, t̂⁰_k⟩)
  - `inverse_sqrt_freq_lambdas(counts: Sequence[int]) -> Tensor[K]` (∝ 1/√N, mean 1; raises `ValueError` if any count is 0)
  - `step_angles_deg(t_hat) -> list[float]` (K−2 angles between v_k and v_{k−1})

- [ ] **Step 1: Install pytest and check import isolation**
  - Run `& $PY -m pip install pytest`. Expected: `Successfully installed pytest-...` (or "already satisfied").
  - Run `& $PY -c "import src; print(src.__file__)"` from the worktree root. Expected: a path inside `.claude\worktrees\direction1`.
- [ ] **Step 2: Write the failing tests**

```python
import math, torch, torch.nn.functional as F
from src.models.geometry_losses import (cosine_distance_matrix, ladder_loss, ladder_satisfied_frac,
    path_losses, path_loss, anchor_loss, inverse_sqrt_freq_lambdas, step_angles_deg)

def arc(K=5, D=8, span=math.pi/2):
    # K points on a great circle, equally spaced: perfectly ordinal & smooth
    th = torch.linspace(0, span, K)
    t = torch.zeros(K, D); t[:, 0] = th.cos(); t[:, 1] = th.sin(); return t

def test_ladder_zero_on_wide_arc():
    assert ladder_loss(arc(span=math.pi*0.9), margin=0.0).item() == 0.0
    assert ladder_satisfied_frac(arc()) == 1.0

def test_ladder_positive_when_order_swapped():
    t = arc()[[0, 2, 1, 3, 4]]
    assert ladder_loss(t, margin=0.05).item() > 0
    assert ladder_satisfied_frac(t) < 1.0

def test_ladder_margin_term():
    # equal distances → each pair violated by exactly (Δ'-Δ)m
    t = F.normalize(torch.eye(3), dim=-1)  # K=3, all pairwise d = 1
    # pairs: i=0: (Δ=1,Δ'=2) → margin*1 ; i=1: none ; total = 0.05
    assert math.isclose(ladder_loss(t, margin=0.05).item(), 0.05, rel_tol=1e-5)

def test_path_closed_form_on_equally_spaced_arc():
    # great-circle step δ: second difference = 2(cosδ-1)·t_k, adjacent steps differ by angle δ
    d = math.pi / 2 / 4; s, a = path_losses(arc())
    assert math.isclose(s.item(), 3 * (2 * (1 - math.cos(d))) ** 2, rel_tol=1e-4)
    assert math.isclose(a.item(), 3 * (1 - math.cos(d)), rel_tol=1e-4)

def test_path_detects_zigzag():
    t = arc(); t[2] = F.normalize(t[2] + torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0]), dim=0)
    s, a = path_losses(t); s0, a0 = path_losses(arc())
    assert s > s0 and a > a0
    assert torch.isclose(path_loss(t, "both"), s + a)

def test_path_mode_invalid():
    import pytest
    with pytest.raises(ValueError): path_loss(arc(), "bogus")

def test_anchor_loss_values():
    t = arc(); assert anchor_loss(t, t, torch.ones(5)).item() < 1e-6
    t2 = arc(); t2[0] = -t2[0]          # cos = -1 → 2*λ*2 = 4
    assert math.isclose(anchor_loss(t2, arc(), torch.ones(5)).item(), 4.0, rel_tol=1e-5)

def test_inverse_sqrt_freq_lambdas():
    lam = inverse_sqrt_freq_lambdas([100, 25, 25, 4, 4])
    assert math.isclose(lam.mean().item(), 1.0, rel_tol=1e-6)
    assert lam[3] > lam[1] > lam[0] and math.isclose((lam[3]/lam[0]).item(), 5.0, rel_tol=1e-5)

def test_losses_are_differentiable():
    t = arc().requires_grad_(True)
    (ladder_loss(t) + path_loss(t, "both") + anchor_loss(t, arc(), torch.ones(5))).backward()
    assert t.grad is not None

def test_step_angles_and_distance_matrix():
    assert len(step_angles_deg(arc())) == 3
    D = cosine_distance_matrix(arc())
    assert D.shape == (5, 5) and torch.allclose(D.diag(), torch.zeros(5), atol=1e-6)
```

- [ ] **Step 3: Run to verify failure** — `& $PY -m pytest tests/test_geometry_losses.py -q`. Expected: ImportError / collection error.
- [ ] **Step 4: Implement `src/models/geometry_losses.py`** using the exact formulas of spec §4.2–4.4. Loop over (i, Δ, Δ') explicitly; K is tiny.
- [ ] **Step 5: Run to verify pass** — the same command. Expected: `10 passed`.
- [ ] **Step 6: Commit** — `git add src/models/geometry_losses.py tests/test_geometry_losses.py` then `git commit -m "feat: add ordinal geometry losses (ladder, path, anchor)"`.

---

### Task 2: `OrdinalGeometryEmbeddings` module

**Files:**
- Create: `src/models/prompt_learner/ordinal_geometry.py`
- Test: `tests/test_ordinal_geometry.py`

**Interfaces:**
- Consumes: `step_angles_deg`, `cosine_distance_matrix`, `ladder_satisfied_frac` from Task 1.
- Produces:
  - `GRADE_PROMPTS: dict[str, list[str]]` with keys `"icdr_clinical"` and `"plain"` (spec §4.6, verbatim).
  - `build_init_embeddings(clip_model, init_mode: str, num_ranks: int, seed: int) -> Tensor[K,D]` (L2-normalised, float32).
    - For text modes: `clip.tokenize(sentences, truncate=False)` (raises on overflow), then `clip_model.encode_text` under `no_grad`.
    - For `random_ortho`: `torch.linalg.qr(torch.randn(D, K, generator=g))[0].T` with `g = torch.Generator().manual_seed(seed)`; D is `clip_model.text_projection.shape[1]`.
    - Raises `ValueError` for an unknown mode, or for a text mode when `num_ranks != 5`.
  - `class OrdinalGeometryEmbeddings(nn.Module)`:
    - `__init__(self, init_embeddings: Tensor[K,D])`. Attributes:
      - `direct_prototypes = True` (class attribute)
      - `self.t = nn.Parameter(init.clone())`
      - buffer `anchors` (normalised init)
      - buffer `anchor_lambdas = ones(K)`
    - `forward() -> Tensor[K,D]`: returns `F.normalize(self.t, dim=-1, eps=1e-8)` in-graph.
    - `set_anchor_lambdas(lambdas: Tensor[K]) -> None`
    - `renormalize_() -> None`: in place under `no_grad`.
    - `diagnostics() -> dict` with keys:
      - `"distance_matrix"`: K×K list
      - `"step_angles_deg"`: list
      - `"anchor_drift"`: list, d(t̂_k, t̂⁰_k)
      - `"ladder_satisfied_frac"`: float

- [ ] **Step 1: Write the failing tests** (fake CLIP: no download)

```python
import pytest, torch, torch.nn.functional as F
from src.models.prompt_learner.ordinal_geometry import (GRADE_PROMPTS, build_init_embeddings,
    OrdinalGeometryEmbeddings)

class FakeClip(torch.nn.Module):
    def __init__(self, D=16):
        super().__init__(); self.text_projection = torch.nn.Parameter(torch.zeros(8, D))
    def encode_text(self, tokens):  # deterministic per sentence
        g = torch.Generator().manual_seed(int(tokens.sum()))
        return torch.randn(tokens.shape[0], self.text_projection.shape[1], generator=g) * 3

def test_prompts_verbatim():
    assert GRADE_PROMPTS["plain"][0] == "A fundus photograph with no diabetic retinopathy."
    assert GRADE_PROMPTS["icdr_clinical"][1].endswith("characterized by the presence of microaneurysms only.")
    assert all(len(v) == 5 for v in GRADE_PROMPTS.values())

@pytest.mark.parametrize("mode", ["icdr_clinical", "plain", "random_ortho"])
def test_init_is_unit_norm(mode):
    t = build_init_embeddings(FakeClip(), mode, 5, seed=1)
    assert t.shape == (5, 16) and torch.allclose(t.norm(dim=-1), torch.ones(5), atol=1e-5)

def test_random_ortho_orthonormal_and_seeded():
    a = build_init_embeddings(FakeClip(), "random_ortho", 5, seed=3)
    assert torch.allclose(a @ a.T, torch.eye(5), atol=1e-5)
    assert torch.equal(a, build_init_embeddings(FakeClip(), "random_ortho", 5, seed=3))

def test_unknown_mode_raises():
    with pytest.raises(ValueError): build_init_embeddings(FakeClip(), "nope", 5, seed=1)

def test_forward_is_unit_norm_and_differentiable():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16) * 4)
    out = m(); assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
    out[:, 0].sum().backward()
    g = m.t.grad; assert g is not None and g.abs().sum() > 0
    # gradient of a normalised output is orthogonal to t (angular only)
    assert torch.allclose((g * m.t.detach()).sum(-1), torch.zeros(5), atol=1e-4)

def test_anchors_normalised_and_frozen():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16) * 4)
    assert torch.allclose(m.anchors.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert "anchors" not in dict(m.named_parameters())

def test_renormalize_and_diagnostics():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16))
    with torch.no_grad(): m.t.mul_(7)
    m.renormalize_(); assert torch.allclose(m.t.norm(dim=-1), torch.ones(5), atol=1e-5)
    d = m.diagnostics()
    assert set(d) == {"distance_matrix", "step_angles_deg", "anchor_drift", "ladder_satisfied_frac"}
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest tests/test_ordinal_geometry.py -q`. Expected: ImportError.
- [ ] **Step 3: Implement** `src/models/prompt_learner/ordinal_geometry.py` per the Interfaces block. Tokens go to the device of `clip_model.text_projection`. The result is cast to float32 and moved to CPU.
- [ ] **Step 4: Run to verify pass** — the same command. Expected: `9 passed`.
- [ ] **Step 5: Real-CLIP smoke check** — run `& $PY -c "import clip,torch; from src.models.prompt_learner.ordinal_geometry import build_init_embeddings as b; m,_=clip.load('RN50',device='cpu'); t=b(m.float(),'icdr_clinical',5,1); print(t.shape, (t@t.T).diag())"`. Expected: `torch.Size([5, 1024])` and ones. No truncation error.
- [ ] **Step 6: Commit** — `git add src/models/prompt_learner/ordinal_geometry.py tests/test_ordinal_geometry.py` then `git commit -m "feat: add direct ordinal-geometry grade embeddings"`.

---

### Task 3: Ordinal metrics (confusion matrix, ODC, ECE)

**Files:**
- Create: `src/utils/ordinal_metrics.py`
- Test: `tests/test_ordinal_metrics.py`

**Interfaces:**
- Produces:
  - `confusion_matrix(targets, preds, K) -> np.ndarray[K,K]` (int counts; rows = true grade)
  - `odc(cm, include_diagonal_step: bool) -> tuple[float, float]` → `(odc, monotone_row_frac)`. `True` = `ODC_full`, `False` = `ODC_offdiag`. Returns `(nan, nan)` if no row has support.
  - `mean_mass_by_dist(cm) -> list[float]` (length K)
  - `expected_calibration_error(probs, targets, n_bins=15) -> float` (top-1 confidence, equal-width bins)
  - `ordinal_shape_metrics(targets, preds, probs, K) -> dict` with keys `confusion_matrix` (list of lists), `odc_full`, `odc_offdiag`, `monotone_row_frac_full`, `monotone_row_frac_offdiag`, `mean_mass_by_dist`, `ece`.

ODC follows spec §6. Row i is walked outward on each side. `ODC_full` starts the walk at the diagonal. `ODC_offdiag` starts at d = 1, so its left sum runs d = 1…i−1 and its right sum d = 1…K−2−i. Ties count 0. Rows with zero support are skipped.

- [ ] **Step 1: Write the failing tests**

```python
import math, numpy as np
from src.utils.ordinal_metrics import confusion_matrix, odc, mean_mass_by_dist, \
    expected_calibration_error, ordinal_shape_metrics

def test_perfect_diagonal():
    cm = np.diag([10, 5, 5, 3, 2])
    assert odc(cm, True) == (1.0, 1.0) and odc(cm, False) == (1.0, 1.0)
    assert mean_mass_by_dist(cm) == [1.0, 0.0, 0.0, 0.0, 0.0]

def test_always_predict_two():
    cm = np.zeros((5, 5), int); cm[:, 2] = 10
    # full: row0 V=1, row1 V=1 (diag 0 -> 1 at d=1), row2 V=0, row3 V=1, row4 V=1 → ODC=1-4/5
    assert math.isclose(odc(cm, True)[0], 0.2)
    # offdiag: row0 walk (P01=0 -> P02=1) violation 1; row1 left/right single cells → 0; row2 0; row3 0; row4 1
    assert math.isclose(odc(cm, False)[0], 1 - 2/5)

def test_bump_far_from_diagonal():
    cm = np.array([[8, 0, 2, 0, 0]] + [[0]*i + [1] + [0]*(4-i) for i in range(1, 5)])
    full, frac = odc(cm, True)
    assert math.isclose(full, 1 - 0.2/5) and math.isclose(frac, 0.8)

def test_ties_are_not_violations():
    cm = np.diag([1]*5); cm[0] = [4, 2, 2, 2, 0]
    assert odc(cm, True)[0] == 1.0

def test_odc_skips_empty_rows():
    cm = np.diag([5, 0, 5, 5, 5]).astype(int)
    v, frac = odc(cm, True)
    assert v == 1.0 and frac == 1.0 and not math.isnan(v)
    assert all(not math.isnan(x) for x in mean_mass_by_dist(cm))

def test_confusion_and_bundle():
    t = np.array([0, 1, 2, 3, 4, 4]); p = np.array([0, 1, 2, 3, 3, 4])
    cm = confusion_matrix(t, p, 5); assert cm[4, 3] == 1 and cm.sum() == 6
    probs = np.eye(5)[p]
    out = ordinal_shape_metrics(t, p, probs, 5)
    assert set(out) == {"confusion_matrix", "odc_full", "odc_offdiag", "monotone_row_frac_full",
                        "monotone_row_frac_offdiag", "mean_mass_by_dist", "ece"}
    assert math.isclose(expected_calibration_error(probs, t), 1/6, rel_tol=1e-6)
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest tests/test_ordinal_metrics.py -q`. Expected: ImportError.
- [ ] **Step 3: Implement** `src/utils/ordinal_metrics.py` (numpy only).
- [ ] **Step 4: Run to verify pass.** Expected: `6 passed`.
- [ ] **Step 5: Commit** — `git add src/utils/ordinal_metrics.py tests/test_ordinal_metrics.py` then `git commit -m "feat: add ODC ordinal confusion-matrix metrics and ECE"`.

---

### Task 4: Runner wiring (module, loss weights, geometry, CLAP λ, param groups, renorm, metrics, diagnostics)

**Files:**
- Modify: `src/models/runner.py` (learner selection §3, SMS §4, loss weights §5, `forward`, `training_step`, `on_train_start` (new), `on_train_batch_end` (new), `on_train_epoch_end`, `_shared_eval_epoch_end`, `configure_optimizers`)
- Test: `tests/test_runner_direction1.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `resolve_loss_weights(loss_cfg) -> dict[str, float]` (module-level in `runner.py`) with keys `w_ce, w_clipdr, w_l2r, w_sms, w_ladder, w_path, w_anchor`.
    - `w_clipdr` = `loss_cfg.w_clipdr` if present, else `lambda_rank`, else 0.
    - `w_l2r` likewise from `mu_pairwise`.
    - `w_ce` and `w_sms` default to 1; the geometry weights default to 0.
  - `population_grade_counts(manifest_path: str, num_ranks: int) -> list[int]` (module-level): counts of rows with `split == "train"`, per grade 0…K−1.
  - Learner name `"ordinal_geometry_embeddings"` with `prompt_learner` keys:
    - `init_mode` (default `"icdr_clinical"`)
    - `anchor_weighting ∈ {"uniform","inverse_sqrt_freq","clap"}` (default `"uniform"`)
    - `path_mode` (default `"both"`)
    - `ladder_margin` (default 0.05)
    - `renorm_after_step` (default true)
  - `self.geometry_history: list[dict]`: one `diagnostics()` dict per train epoch, with `"epoch"` added.
  - `latest_metrics` gains every key of `ordinal_shape_metrics` (Task 3). `confusion_matrix` and `mean_mass_by_dist` are lists.
  - Logged scalars: `train_l_ce, train_l_clipdr, train_l_l2r, train_l_ladder, train_l_smooth, train_l_align, train_l_anchor` (direct module only for the geometry ones), and `{prefix}_odc_full`, `{prefix}_odc_offdiag`, `{prefix}_ece`.

Behaviour to implement:
- **forward:** if `getattr(self.prompt_learner, "direct_prototypes", False)`, then `text_features = self.prompt_learner()`. Otherwise use the existing prompt → `text_encoder` path, unchanged.
- **SMS** is built only if `cfg.model.sms.enabled and weights["w_sms"] != 0`.
- **training_step:** `loss = w_ce·CE + w_clipdr·rank + w_l2r·pairwise (+ geometry if direct)`.
  - A term with weight 0 is not computed; log 0 instead.
  - The geometry losses take `t_hat = text_features.float()`. `path_loss` uses `path_mode`, but both `l_smooth` and `l_align` are always logged.
- **on_train_start:**
  - If `anchor_weighting == "inverse_sqrt_freq"`, set λ from `inverse_sqrt_freq_lambdas(population_grade_counts(cfg.data.manifest, K))`.
  - If `"clap"`: under `no_grad` with `self.eval()`, run the support loader (`self.trainer.datamodule.train_dataloader()`), compute softmax(logit_scale·f̂T̂ᵀ), and set λ_k = the mean of p[:, k] over images with target k. A grade with no images keeps λ = 1. Then call `self.train()`.
- **on_train_batch_end:** if direct and `renorm_after_step`, call `self.prompt_learner.renormalize_()`.
- **on_train_epoch_end:** keep the SMS update. If direct, append `{"epoch": self.current_epoch, **diagnostics()}` to `geometry_history`.
- **configure_optimizers:** if direct, use two param groups: `[{"params": other_trainable}, {"params": [prompt_learner.t], "lr": cfg.train.get("text_lr", lr), "weight_decay": 0.0}]`. The scheduler is unchanged; it applies to both groups. Non-direct learners keep one group.

- [ ] **Step 1: Write the failing tests** (pure functions plus the renorm mechanics; no CLIP needed)

```python
import torch
from omegaconf import OmegaConf
from src.models.runner import resolve_loss_weights, population_grade_counts
from src.models.prompt_learner.ordinal_geometry import OrdinalGeometryEmbeddings

def test_resolve_loss_weights_backcompat():
    w = resolve_loss_weights(OmegaConf.create({"lambda_rank": 1.0, "mu_pairwise": 0.5, "tau": 1.0}))
    assert w == {"w_ce": 1.0, "w_clipdr": 1.0, "w_l2r": 0.5, "w_sms": 1.0,
                 "w_ladder": 0.0, "w_path": 0.0, "w_anchor": 0.0}

def test_resolve_loss_weights_new_keys_win():
    w = resolve_loss_weights(OmegaConf.create({"lambda_rank": 1.0, "w_clipdr": 0.0, "w_ladder": 2.0, "w_sms": 0}))
    assert w["w_clipdr"] == 0.0 and w["w_ladder"] == 2.0 and w["w_sms"] == 0.0

def test_population_grade_counts(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,modality,grade,lesion,lr,split\na,CFP,0,,0,train\nb,CFP,0,,0,train\n"
                 "c,CFP,4,,0,train\nd,CFP,1,,0,test\n")
    assert population_grade_counts(str(p), 5) == [2, 0, 0, 0, 1]

def test_renorm_after_optimizer_step():
    m = OrdinalGeometryEmbeddings(torch.nn.functional.normalize(torch.randn(5, 16), dim=-1))
    opt = torch.optim.AdamW([{"params": [m.t], "weight_decay": 0.0}], lr=0.5)
    (m() @ torch.randn(16)).sum().backward(); opt.step(); m.renormalize_()
    assert m.t.requires_grad and torch.allclose(m.t.norm(dim=-1), torch.ones(5), atol=1e-5)
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest tests/test_runner_direction1.py -q`. Expected: ImportError on `resolve_loss_weights`.
- [ ] **Step 3: Implement** the runner changes above. With `zero_division`-style guards, a grade with 0 population count makes `inverse_sqrt_freq_lambdas` raise, which is the intended loud failure.
- [ ] **Step 4: Run the new tests plus the whole suite** — `& $PY -m pytest tests -q`. Expected: all pass, including the pre-existing `test_sms`, `test_pairwise_loss`, `test_rank_loss` and `test_ordinality`.
- [ ] **Step 5: Commit** — `git add src/models/runner.py tests/test_runner_direction1.py` then `git commit -m "feat: wire direct ordinal-geometry embeddings, modular loss weights and ODC into runner"`.

---

### Task 5: Training script and sweep (recursive defaults, k = 0, JSON stats, geometry.json, overrides)

**Files:**
- Modify: `scripts/train.py` (`load_config`, `main`)
- Modify: `scripts/sweep.py` (`parse_args`, `main`)
- Test: `tests/test_train_config.py`

**Interfaces:**
- Produces:
  - `load_config(config_path, overrides=None)` resolves `defaults` **recursively**, by calling itself on each default path, and merges in order. The top-level file is applied last, then CLI overrides.
  - `train.py`, when `cfg.data.k_shot == 0`, skips `trainer.fit` and runs `trainer.test(model, datamodule=datamodule)` on the untrained model.
  - `test_stats.json` values: floats stay floats; lists and nested lists are written as-is.
  - If `model.geometry_history` is non-empty, write `<out_dir>/geometry.json` (a list of dicts).
  - `sweep.py` gains `--extra` (a string of space-separated OmegaConf overrides, split with `shlex.split(posix=False)`). They are appended to every run's command.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("train", Path("scripts/train.py"))
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

def test_load_config_recursive_defaults(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: {p: 1, q: 2}\n")
    (tmp_path / "b.yaml").write_text("defaults:\n  - a\ny: {q: 3}\n")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "c.yaml").write_text("defaults:\n  - ../b\nz: 9\n")
    cfg = train.load_config(str(sub / "c.yaml"), ["y.p=7"])
    assert cfg.x == 1 and cfg.y.p == 7 and cfg.y.q == 3 and cfg.z == 9 and "defaults" not in cfg
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest tests/test_train_config.py -q`. Expected: FAIL (`cfg.x` missing, or `defaults` key present).
- [ ] **Step 3: Implement** the `train.py` and `sweep.py` changes.
- [ ] **Step 4: Run to verify pass**, then `& $PY -m pytest tests -q` (all pass).
- [ ] **Step 5: Commit** — `git add scripts/train.py scripts/sweep.py tests/test_train_config.py` then `git commit -m "feat: recursive config defaults, untrained k=0 eval, JSON-safe stats and sweep overrides"`.

---

### Task 6: Configs (main method, ablations, matched and native baselines)

**Files:**
- Create in `configs/direction1/`, each with `defaults: [../deepdrid_cfp]`. Unique stems are needed because results go to `results/<stem>/`.
  - `d1_full.yaml`: `prompt_learner: {name: ordinal_geometry_embeddings, init_mode: icdr_clinical, anchor_weighting: uniform, path_mode: both, ladder_margin: 0.05, renorm_after_step: true}`, `sms.enabled: true`, `loss: {w_ce: 1, w_clipdr: 1, w_l2r: 1, w_sms: 1, w_ladder: 1, w_path: 1, w_anchor: 1}`. This is the main method.
  - `d1_ce_geometry.yaml`: defaults `[d1_full]`, `loss: {w_clipdr: 0, w_l2r: 0, w_sms: 0}`.
  - `d1_no_geometry.yaml`: defaults `[d1_full]`, `loss: {w_ladder: 0, w_path: 0, w_anchor: 0}`.
  - `d1_no_ladder.yaml`, `d1_no_path.yaml`, `d1_no_anchor.yaml`: defaults `[d1_full]`, the one weight set to 0.
  - `d1_init_plain.yaml`, `d1_init_random.yaml`: defaults `[d1_full]`, `init_mode: plain` / `random_ortho`.
  - `d1_anchor_invsqrt.yaml`: defaults `[d1_full]`, `anchor_weighting: inverse_sqrt_freq`.
  - `d1_rankformer.yaml`: defaults `[../deepdrid_cfp]` with nothing changed. This is "Ours (Full)", re-run on the current code.
- Create in `configs/baselines/`:
  - `clipdr_matched.yaml`: defaults `[../deepdrid_cfp]`, `lora.enabled: true`, `prompt_learner.name: ordinalclip_rank_prompt_learner`, `sms.enabled: true`, `loss: {lambda_rank: 1.0, mu_pairwise: 0.0}`.
  - `l2rclip_matched.yaml`: defaults `[../deepdrid_cfp]`, `lora.enabled: true`, `prompt_learner.name: rankformer_prompt_learner`, `sms.enabled: false`, `loss: {lambda_rank: 0.0, mu_pairwise: 0.5}`. The weight 0.5 is kept from the existing `l2rclip_stock.yaml`.
  - `clap_matched.yaml`: defaults `[../deepdrid_cfp]`, `lora.enabled: true`, `prompt_learner: {name: ordinal_geometry_embeddings, init_mode: plain, anchor_weighting: clap}`, `sms.enabled: false`, `loss: {w_ce: 1, w_clipdr: 0, w_l2r: 0, w_ladder: 0, w_path: 0, w_anchor: 1}`.
  - `clap_native.yaml`: defaults `[clap_matched]`, `lora.enabled: false`, `train: {optimizer: sgd, lr: 0.1, epochs: 300, scheduler: cosine, warmup_epochs: 0, weight_decay: 0.0}`.
- Test: `tests/test_configs_direction1.py`

- [ ] **Step 1: Write the failing test**

```python
import glob, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("train", Path("scripts/train.py"))
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
FILES = sorted(glob.glob("configs/direction1/*.yaml")) + [f"configs/baselines/{n}.yaml" for n in
         ("clipdr_matched", "l2rclip_matched", "clap_matched", "clap_native")]

def test_all_configs_load_with_full_keys():
    assert len(FILES) == 14
    for f in FILES:
        c = train.load_config(f)
        assert c.model.prompt_learner.num_ranks == 5 and "DeepDRiD" in c.data.manifest, f
        assert c.train.epochs >= 100 and c.model.image_encoder.name == "RN50", f

def test_key_switches():
    full = train.load_config("configs/direction1/d1_full.yaml")
    assert full.model.prompt_learner.init_mode == "icdr_clinical" and full.loss.w_ladder == 1
    assert train.load_config("configs/direction1/d1_no_path.yaml").loss.w_path == 0
    native = train.load_config("configs/baselines/clap_native.yaml")
    assert native.train.optimizer == "sgd" and native.model.image_encoder.lora.enabled is False
    assert native.model.prompt_learner.anchor_weighting == "clap"
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest tests/test_configs_direction1.py -q`. Expected: FAIL (no files).
- [ ] **Step 3: Create the 14 config files** listed above.
- [ ] **Step 4: Run to verify pass.** Expected: `2 passed`.
- [ ] **Step 5: Commit** — `git add configs/direction1 configs/baselines/clipdr_matched.yaml configs/baselines/l2rclip_matched.yaml configs/baselines/clap_matched.yaml configs/baselines/clap_native.yaml tests/test_configs_direction1.py` then `git commit -m "feat: add Direction 1 method, ablation and matched/native baseline configs"`.

---

### Task 7: Smoke runs and README note

**Files:**
- Modify: `README.md` (add a short "Direction 1 (branch `direction1-ordinal-geometry`)" section: what the module is, the config names, and the sweep command)

- [ ] **Step 1: Zero-shot smoke run** — `& $PY scripts/train.py --config configs/direction1/d1_full.yaml data.k_shot=0 data.seed=1 <data overrides>`. Expected:
  - it finishes without training;
  - `results/d1_full/CFP_0shot_s1/test_stats.json` has `odc_full`, `confusion_matrix` and `ece`;
  - QWK is printed (no target value; it is the first real zero-shot number).
- [ ] **Step 2: Few-shot smoke run** — `... --config configs/direction1/d1_full.yaml data.k_shot=1 data.seed=1 train.epochs=3 <data overrides>`. Expected:
  - it finishes;
  - `geometry.json` has 3 entries, each with `anchor_drift > 0` after epoch 1;
  - the log shows `train_l_ladder`, `train_l_smooth`, `train_l_align` and `train_l_anchor`.
- [ ] **Step 3: Baseline smoke runs** — the same 3-epoch, 1-shot command for `clap_matched` and `clipdr_matched`. Expected: both finish, and `clap_matched` logs λ values ≠ 1 (print them once in `on_train_start`).
- [ ] **Step 4: Full test suite** — `& $PY -m pytest tests -q`. Expected: all pass.
- [ ] **Step 5: README note and commit** — `git add README.md` then `git commit -m "docs: describe Direction 1 configs and sweep"`. Do not commit `results/` (git-ignored).
