import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score, roc_auc_score, accuracy_score

from .clip_backbone import load_clip_model, TextEncoder
from .prompt_learner.rankformer import RankFormerPromptLearner
from .prompt_learner.ordinalclip_head import OrdinalCLIPHead
from .losses import RankingAwareLoss, CrossModalPairwiseLoss, OrdinalCLIPKLLoss
from .sms import SMS
from .lora_adapter import attach_lora, set_trainable
from ..utils.ordinal_metrics import ordinal_shape_metrics


class FewShotOrdinalRunner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.num_ranks = cfg.model.prompt_learner.num_ranks

        # 1. Load CLIP (cast to float32 to prevent AMP GradScaler crash on static FP16 grads)
        self.clip_model = load_clip_model(cfg.model.image_encoder.name)
        self.text_encoder = TextEncoder(self.clip_model)

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
        learner_name = cfg.model.prompt_learner.name
        pl_cfg = cfg.model.prompt_learner
        # Shared prompt options (OrdinalCLIP: text-initialised context, "<ctx> <rank>." sentence)
        prompt_kwargs = dict(
            init_context=pl_cfg.get("init_context", None),
            full_stop=pl_cfg.get("full_stop", False),
            interpolation=pl_cfg.get("interpolation", "linear"),
        )
        if learner_name == "rankformer_prompt_learner":
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

        # 4. SMS calibration (training-time, CLIP-DR / FDS form)
        sms_cfg = cfg.model.sms
        self.sms = (
            SMS(
                num_ranks=self.num_ranks,
                kernel_size=sms_cfg.get("kernel_size", 5),
                sigma=sms_cfg.get("sigma", 2.0),
                momentum=sms_cfg.get("momentum", 0.9),
            )
            if sms_cfg.enabled
            else None
        )

        # 5. Losses
        self.criterion_main = nn.CrossEntropyLoss()
        self.criterion_rank = RankingAwareLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        self.criterion_pairwise = CrossModalPairwiseLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        self.criterion_kl = OrdinalCLIPKLLoss(num_ranks=self.num_ranks)

        self.lambda_rank = cfg.loss.lambda_rank
        self.mu_pairwise = cfg.loss.mu_pairwise
        self.w_kl = cfg.loss.get("w_kl", 0.0)

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

        # Metric accumulators — cleared after every eval epoch
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.val_probs: list[torch.Tensor] = []
        self.latest_metrics: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image.type(self.clip_model.dtype))
        image_features = F.normalize(image_features, dim=-1)

        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if self.sms is not None:
            self.sms.reset_stats()

    def training_step(self, batch, batch_idx):
        image, target, _, _ = batch
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

        loss_main = self.criterion_main(sim, target)
        loss_rank = (
            self.criterion_rank(sim, target)
            if self.lambda_rank > 0
            else sim.new_tensor(0.0)
        )
        loss_pairwise = (
            self.criterion_pairwise(
                image_features.float(), text_features.float(), target, logit_scale=logit_scale.float()
            )
            if self.mu_pairwise > 0
            else sim.new_tensor(0.0)
        )

        loss_kl = self.criterion_kl(sim, target) if self.w_kl > 0 else sim.new_tensor(0.0)

        loss = (loss_main + self.lambda_rank * loss_rank + self.mu_pairwise * loss_pairwise
                + self.w_kl * loss_kl)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.sms is not None:
            self.sms.compute_smoothed_stats()

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
        image, target, _, _ = batch
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
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer_name = self.cfg.train.get("optimizer", "adam").lower()
        lr = self.cfg.train.lr
        wd = self.cfg.train.get("weight_decay", 1e-4)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        elif optimizer_name == "radam":
            optimizer = torch.optim.RAdam(trainable_params, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=wd, momentum=0.9)

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
