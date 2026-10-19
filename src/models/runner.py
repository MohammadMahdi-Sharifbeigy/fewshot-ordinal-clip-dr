import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score, roc_auc_score, accuracy_score

from .clip_backbone import load_clip_model, TextEncoder
from .prompt_learner.rankformer import RankFormerPromptLearner
from .prompt_learner.ordinalclip_head import OrdinalCLIPHead
from .prompt_learner.ordinal_geometry import OrdinalGeometryEmbeddings, build_init_embeddings
from .prompt_learner.text_lora_prototypes import TextLoRAPrototypes
from .text_lora import attach_text_lora, text_lora_parameters
from .text_losses import description_order_loss
from .losses import RankingAwareLoss, CrossModalPairwiseLoss, OrdinalCLIPKLLoss
from .geometry_losses import anchor_loss, inverse_sqrt_freq_lambdas, ladder_loss, path_losses
from .sms import SMS
from .lora_adapter import attach_lora, set_trainable
from ..utils.ordinal_metrics import ordinal_shape_metrics


def resolve_loss_weights(loss_cfg) -> dict[str, float]:
    """Loss weights from the config.

    Old keys still work: lambda_rank -> w_clipdr; mu_pairwise or w_l2r -> w_pairwise
    (our distance-weighted pairwise loss, inspired by L2RCLIP; not part of the main models).
    """
    def get(key, fallback_key=None, default=0.0):
        if key in loss_cfg:
            return float(loss_cfg[key])
        if fallback_key is not None and fallback_key in loss_cfg:
            return float(loss_cfg[fallback_key])
        return float(default)

    return {
        "w_ce": get("w_ce", default=1.0),
        "w_clipdr": get("w_clipdr", "lambda_rank"),
        "w_pairwise": get("w_pairwise", "w_l2r" if "w_l2r" in loss_cfg else "mu_pairwise"),
        "w_sms": get("w_sms", default=1.0),
        "w_ladder": get("w_ladder"),
        "w_path": get("w_path"),
        "w_anchor": get("w_anchor"),
        "w_kl": get("w_kl"),
        "w_desc": get("w_desc"),
    }


def population_grade_counts(manifest_path: str, num_ranks: int) -> list[int]:
    """Grade counts of the full training split (manifest rows with split == "train")."""
    import pandas as pd

    df = pd.read_csv(manifest_path)
    grades = df.loc[df["split"] == "train", "grade"].astype(int)
    return [int((grades == k).sum()) for k in range(num_ranks)]


def clap_lambdas(probs: torch.Tensor, targets: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """CLAP λ init: mean zero-shot probability of grade k over the support images of grade k.

    A grade without support images keeps λ = 1.
    """
    lam = torch.ones(num_ranks)
    for k in range(num_ranks):
        mask = targets == k
        if mask.any():
            lam[k] = probs[mask, k].float().mean()
    return lam


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """(image, target) from a tuple/list batch (image, grade, lesion, path) or a dict batch."""
    if isinstance(batch, dict):
        return batch["image"], batch["target"]
    return batch[0], batch[1]


class FewShotOrdinalRunner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.num_ranks = cfg.model.prompt_learner.num_ranks

        # 1. Load CLIP (cast to float32 to prevent AMP GradScaler crash on static FP16 grads)
        self.clip_model = load_clip_model(cfg.model.image_encoder.name)
        self.text_encoder = TextEncoder(self.clip_model)

        # Direct grade embeddings are initialised from the frozen text encoder
        # before LoRA wraps the visual tower (encode_text reads visual.conv1's dtype)
        pl_cfg = cfg.model.prompt_learner
        init_embeddings = None
        if pl_cfg.name == "ordinal_geometry_embeddings":
            init_embeddings = build_init_embeddings(
                self.clip_model,
                init_mode=pl_cfg.get("init_mode", "icdr_clinical"),
                num_ranks=self.num_ranks,
                seed=cfg.data.seed,
            )

        # 2. Attach LoRA to visual encoder
        if cfg.model.image_encoder.lora.enabled:
            # ViT: attention projections; RN50: bottleneck convolutions
            target_modules = (
                ["v_proj", "q_proj"]
                if "ViT" in cfg.model.image_encoder.name
                else ["conv1", "conv2", "conv3"]
            )
            self.clip_model = attach_lora(
                self.clip_model,
                r=cfg.model.image_encoder.lora.r,
                target_modules=target_modules,
            )

        # 3. Prompt Learner
        learner_name = pl_cfg.name
        # Shared prompt options (OrdinalCLIP: text-initialised context, "<ctx> <rank>." sentence)
        prompt_kwargs = dict(
            init_context=pl_cfg.get("init_context", None),
            full_stop=pl_cfg.get("full_stop", False),
            interpolation=pl_cfg.get("interpolation", "linear"),
        )
        if learner_name == "ordinal_geometry_embeddings":
            self.prompt_learner = OrdinalGeometryEmbeddings(init_embeddings)
        elif learner_name == "text_lora_prototypes":
            # Direction 2: description ensemble through a LoRA-adapted text encoder
            tl_cfg = pl_cfg.get("text_lora", {})
            if tl_cfg.get("enabled", True):
                attach_text_lora(
                    self.clip_model,
                    r=int(tl_cfg.get("r", 2)),
                    alpha=float(tl_cfg.get("alpha", 1.0)),
                    dropout=float(tl_cfg.get("dropout", 0.25)),
                    targets=list(tl_cfg.get("targets", ["q", "k", "v"])),
                )
            self.prompt_learner = TextLoRAPrototypes(
                self.clip_model, self.text_encoder,
                prompt_set=pl_cfg.get("prompt_set", "icdr_ensemble"),
                num_ranks=self.num_ranks,
            )
        elif learner_name == "rankformer_prompt_learner":
            self.prompt_learner = RankFormerPromptLearner(
                self.clip_model,
                num_ranks=self.num_ranks,
                num_context_tokens=pl_cfg.num_tokens_for_context,
                attn_heads=pl_cfg.attn_heads,
                rank_param=pl_cfg.get("rank_param", "interpolated"),
                num_base=pl_cfg.get("num_base", 3),
                block=pl_cfg.get("rankformer_block", "paper"),
                alpha=pl_cfg.get("rankformer_alpha", 0.5),
                **prompt_kwargs,
            )
        elif learner_name == "ordinalclip_rank_prompt_learner":
            self.prompt_learner = OrdinalCLIPHead(
                self.clip_model,
                num_ranks=self.num_ranks,
                num_context_tokens=pl_cfg.num_tokens_for_context,
                num_base=pl_cfg.get("num_base", 3),
                **prompt_kwargs,
            )
        else:
            raise ValueError(f"Unknown prompt learner: {learner_name}")

        self.direct = bool(getattr(self.prompt_learner, "direct_prototypes", False))
        self.weights = resolve_loss_weights(cfg.loss)
        self.anchor_weighting = pl_cfg.get("anchor_weighting", "uniform")
        self.path_mode = pl_cfg.get("path_mode", "both")
        self.ladder_margin = float(pl_cfg.get("ladder_margin", 0.05))
        self.renorm_after_step = bool(pl_cfg.get("renorm_after_step", True))

        # 4. SMS calibration (training-time, CLIP-DR / FDS form); loss.w_sms = 0 also disables it
        sms_cfg = cfg.model.sms
        self.sms = (
            SMS(
                num_ranks=self.num_ranks,
                kernel_size=sms_cfg.get("kernel_size", 5),
                sigma=sms_cfg.get("sigma", 2.0),
                momentum=sms_cfg.get("momentum", 0.9),
            )
            if sms_cfg.enabled and self.weights["w_sms"] != 0
            else None
        )

        # 5. Losses
        self.criterion_main = nn.CrossEntropyLoss()
        self.criterion_rank = RankingAwareLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        self.criterion_pairwise = CrossModalPairwiseLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        self.criterion_kl = OrdinalCLIPKLLoss(num_ranks=self.num_ranks)

        # 6. Freeze / unfreeze
        set_trainable(
            self.clip_model,
            self.text_encoder,
            self.prompt_learner,
            self.sms,
            train_prompts=True,
            train_lora=cfg.model.image_encoder.lora.enabled,
            train_sms=cfg.model.sms.enabled,
            full_finetune=cfg.model.image_encoder.get("full_finetune", False),
            train_logit_scale=not cfg.model.get("freeze_logit_scale", True),
        )
        # Text-LoRA weights sit inside the (otherwise frozen) text transformer
        for p in text_lora_parameters(self.clip_model):
            p.requires_grad = True

        # Metric accumulators — cleared after every eval epoch
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.val_probs: list[torch.Tensor] = []
        self.latest_metrics: dict = {}
        # Per-epoch geometry diagnostics of the direct grade embeddings
        self.geometry_history: list[dict] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image.type(self.clip_model.dtype))
        image_features = F.normalize(image_features, dim=-1)

        if self.direct:
            # Grade embeddings live in the joint space; the text encoder is not called
            text_features = self.prompt_learner()
        else:
            prompts, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        if not self.direct:
            return
        if self.anchor_weighting == "inverse_sqrt_freq":
            counts = population_grade_counts(self.cfg.data.manifest, self.num_ranks)
            self.prompt_learner.set_anchor_lambdas(inverse_sqrt_freq_lambdas(counts))
        elif self.anchor_weighting == "clap":
            self.prompt_learner.set_anchor_lambdas(self._clap_zero_shot_lambdas())
        elif self.anchor_weighting != "uniform":
            raise ValueError(f"Unknown anchor_weighting: {self.anchor_weighting}")
        print(f"Anchor lambdas ({self.anchor_weighting}): "
              f"{[round(x, 4) for x in self.prompt_learner.anchor_lambdas.tolist()]}")

    @torch.no_grad()
    def _clap_zero_shot_lambdas(self) -> torch.Tensor:
        """Zero-shot softmax on the support set with the initial embeddings (CLAP's λ init)."""
        was_training = self.training
        self.eval()
        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
        t_hat = self.prompt_learner()
        probs, targets = [], []
        # Clean support images (test transform, fixed order), as in CLAP's zero-shot init
        for batch in self.trainer.datamodule.support_eval_dataloader():
            image, target = unpack_batch(batch)
            image = image.to(self.device)
            f = F.normalize(self.clip_model.visual(image.type(self.clip_model.dtype)), dim=-1)
            probs.append(F.softmax((logit_scale * f @ t_hat.T).float(), dim=-1).cpu())
            targets.append(torch.as_tensor(target).long().cpu())
        if was_training:
            self.train()
        return clap_lambdas(torch.cat(probs), torch.cat(targets), self.num_ranks)

    def on_train_epoch_start(self) -> None:
        if self.sms is not None:
            self.sms.reset_stats()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Project the raw parameter back onto the sphere after every optimiser step
        if self.direct and self.renorm_after_step and hasattr(self.prompt_learner, "renormalize_"):
            self.prompt_learner.renormalize_()

    def training_step(self, batch, batch_idx):
        image, target = unpack_batch(batch)
        image_features, text_features = self(image)

        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100.0) # Prevent FP16 NaN explosion
        sim = logit_scale * image_features @ text_features.T  # [B, K]

        # Cast to float32 before all loss computations.
        # AMP may keep sim in BF16/FP16; logsigmoid/cross-entropy on large
        # logit_scale values can still overflow in reduced precision.
        sim = sim.float()
        target = target.long()

        if self.sms is not None:
            # Collect this epoch's statistics from the raw similarities, then
            # calibrate with the statistics frozen at the end of the last epoch
            self.sms.update_stats(sim, target)
            sim = self.sms(sim, target)

        w = self.weights
        zero = sim.new_tensor(0.0)
        terms = {
            "ce": self.criterion_main(sim, target) if w["w_ce"] != 0 else zero,
            "clipdr": self.criterion_rank(sim, target) if w["w_clipdr"] != 0 else zero,
            "pairwise": (
                self.criterion_pairwise(
                    image_features.float(), text_features.float(), target, logit_scale=logit_scale.float()
                )
                if w["w_pairwise"] != 0
                else zero
            ),
        }
        loss = w["w_ce"] * terms["ce"] + w["w_clipdr"] * terms["clipdr"] + w["w_pairwise"] * terms["pairwise"]

        if w["w_kl"] != 0:
            # OrdinalCLIP's text-to-image KL loss
            terms["kl"] = self.criterion_kl(sim, target)
            loss = loss + w["w_kl"] * terms["kl"]

        if self.direct:
            t_hat = text_features.float()
            l_smooth, l_align = path_losses(t_hat)
            l_path = {"smooth": l_smooth, "align": l_align, "both": l_smooth + l_align}[self.path_mode]
            terms["ladder"] = ladder_loss(t_hat, self.ladder_margin)
            terms["smooth"], terms["align"] = l_smooth, l_align
            terms["anchor"] = anchor_loss(t_hat, self.prompt_learner.anchors, self.prompt_learner.anchor_lambdas)
            if w["w_desc"] != 0 and getattr(self.prompt_learner, "last_desc", None) is not None:
                # Direction 2: keep every grade's description cloud separated and ordered
                terms["desc"] = description_order_loss(
                    self.prompt_learner.last_desc.float(), self.prompt_learner.desc_labels,
                    t_hat, logit_scale.float(), self.criterion_rank,
                )
                loss = loss + w["w_desc"] * terms["desc"]
            loss = (
                loss
                + w["w_ladder"] * terms["ladder"]
                + w["w_path"] * l_path
                + w["w_anchor"] * terms["anchor"]
            )

        for name, value in terms.items():
            self.log(f"train_l_{name}", value.detach(), on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.sms is not None:
            self.sms.compute_smoothed_stats()
        if self.direct:
            self.geometry_history.append({"epoch": int(self.current_epoch), **self.prompt_learner.diagnostics()})

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch)

    def on_validation_epoch_end(self):
        self._shared_eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch)

    def on_test_epoch_end(self):
        self._shared_eval_epoch_end("test")

    @torch.no_grad()
    def _shared_eval_step(self, batch):
        image, target = unpack_batch(batch)
        image_features, text_features = self(image)

        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100.0) # Prevent FP16 NaN explosion
        sim = logit_scale * image_features @ text_features.T

        # SMS needs ground-truth grades, so it only acts during training
        probs = F.softmax(sim, dim=-1)
        preds = probs.argmax(dim=-1)

        self.val_preds.append(preds.cpu())
        self.val_targets.append(target.cpu())
        self.val_probs.append(probs.cpu())

    def _shared_eval_epoch_end(self, prefix: str) -> None:
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
        probs = torch.cat(self.val_probs).numpy()

        qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        mae = mean_absolute_error(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        acc = accuracy_score(targets, preds)

        try:
            if self.num_ranks == 2:
                auc = roc_auc_score(targets, probs[:, 1])
            else:
                auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0  # Fails when not all classes are present in small batches

        # Local Ordinality Score (LOS):
        # Fraction of samples where the predicted rank is a local peak, i.e.
        # prob[pred] > prob[pred-1]  (if pred > 0)  AND
        # prob[pred] > prob[pred+1]  (if pred < K-1).
        # This is defined for ALL samples including boundary grades.
        K = self.num_ranks
        left_ok = np.ones(len(preds), dtype=bool)
        right_ok = np.ones(len(preds), dtype=bool)

        non_zero = preds > 0
        left_ok[non_zero] = probs[non_zero, preds[non_zero]] > probs[non_zero, preds[non_zero] - 1]

        non_last = preds < K - 1
        right_ok[non_last] = probs[non_last, preds[non_last]] > probs[non_last, preds[non_last] + 1]

        los = float((left_ok & right_ok).mean())

        self.log(f"{prefix}_qwk", qwk, prog_bar=True)
        self.log(f"{prefix}_mae", mae, prog_bar=True)
        self.log(f"{prefix}_f1", macro_f1)
        self.log(f"{prefix}_auc", auc)
        self.log(f"{prefix}_acc", acc)
        self.log(f"{prefix}_los", los)

        shape = ordinal_shape_metrics(targets, preds, probs, K)
        self.log(f"{prefix}_odc_full", shape["odc_full"])
        self.log(f"{prefix}_odc_offdiag", shape["odc_offdiag"])
        self.log(f"{prefix}_ece", shape["ece"])

        # Proper scoring rules for the predicted grade distribution
        p_true = probs[np.arange(len(targets)), targets]
        nll = float(-np.log(np.clip(p_true, 1e-12, None)).mean())
        brier = float(((probs - np.eye(K)[targets]) ** 2).sum(axis=1).mean())
        self.log(f"{prefix}_nll", nll)
        self.log(f"{prefix}_brier", brier)

        self.latest_metrics = {
            "qwk": qwk, "mae": mae, "f1": macro_f1,
            "auc": auc, "acc": acc, "los": los,
            "nll": nll, "brier": brier,
            **shape,
        }

        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()

    # ------------------------------------------------------------------
    # Optimizer + LR Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer_name = self.cfg.train.get("optimizer", "adam").lower()
        lr = self.cfg.train.lr
        wd = self.cfg.train.get("weight_decay", 1e-4)

        text_lora = [p for p in text_lora_parameters(self.clip_model) if p.requires_grad]
        if self.direct and hasattr(self.prompt_learner, "t"):
            # Grade embeddings: own lr, no weight decay (decay on a sphere-projected
            # parameter only shrinks its norm and changes the effective step size)
            text_param = self.prompt_learner.t
            others = [p for p in self.parameters() if p.requires_grad and p is not text_param]
            params = [
                {"params": others},
                {"params": [text_param], "lr": self.cfg.train.get("text_lr", lr), "weight_decay": 0.0},
            ]
        elif text_lora:
            # Direction 2: text-LoRA weights get their own learning rate (train.text_lr)
            lora_ids = {id(p) for p in text_lora}
            others = [p for p in self.parameters() if p.requires_grad and id(p) not in lora_ids]
            params = [{"params": others}, {"params": text_lora, "lr": self.cfg.train.get("text_lr", lr)}]
            params = [g for g in params if g["params"]]
        else:
            params = [p for p in self.parameters() if p.requires_grad]

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif optimizer_name == "radam":
            optimizer = torch.optim.RAdam(params, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)

        scheduler_name = self.cfg.train.get("scheduler", None)
        if scheduler_name == "cosine":
            warmup_epochs = self.cfg.train.get("warmup_epochs", 0)
            total_epochs = self.cfg.train.epochs

            if warmup_epochs > 0:
                # Linear warmup then cosine decay
                def lr_lambda(epoch: int) -> float:
                    if epoch < warmup_epochs:
                        return float(epoch + 1) / float(warmup_epochs)
                    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                    return 0.5 * (1.0 + np.cos(np.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs, eta_min=lr * 1e-2
                )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer
