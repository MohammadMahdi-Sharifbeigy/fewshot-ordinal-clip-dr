import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score, roc_auc_score, accuracy_score

from .clip_backbone import load_clip_model, TextEncoder
from .prompt_learner.rankformer import RankFormerPromptLearner
from .prompt_learner.ordinalclip_head import OrdinalCLIPHead
from .losses import RankingAwareLoss, CrossModalPairwiseLoss
from .sms import SMS
from .lora_adapter import attach_lora, set_trainable

class FewShotOrdinalRunner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.num_ranks = cfg.model.prompt_learner.num_ranks
        
        # 1. Load CLIP
        self.clip_model = load_clip_model(cfg.model.image_encoder.name)
        self.text_encoder = TextEncoder(self.clip_model)
        
        # 2. Attach LoRA
        if cfg.model.image_encoder.lora.enabled:
            # RN50 uses 'conv1', 'conv2', 'conv3' in its bottleneck blocks
            # ViT uses 'v_proj', 'q_proj' etc.
            # We use standard target modules for peft
            target_modules = ["v_proj", "q_proj"] if "ViT" in cfg.model.image_encoder.name else ["conv1", "conv2", "conv3"]
            self.clip_model = attach_lora(
                self.clip_model, 
                r=cfg.model.image_encoder.lora.r, 
                target_modules=target_modules
            )
            
        # 3. Prompt Learner
        learner_name = cfg.model.prompt_learner.name
        if learner_name == 'rankformer_prompt_learner':
            self.prompt_learner = RankFormerPromptLearner(
                self.clip_model, 
                num_ranks=self.num_ranks,
                num_context_tokens=cfg.model.prompt_learner.num_tokens_for_context,
                attn_heads=cfg.model.prompt_learner.attn_heads
            )
        elif learner_name == 'ordinalclip_rank_prompt_learner':
            self.prompt_learner = OrdinalCLIPHead(
                self.clip_model,
                num_ranks=self.num_ranks,
                num_context_tokens=cfg.model.prompt_learner.num_tokens_for_context,
                num_base=3
            )
        else:
            raise ValueError(f"Unknown prompt learner: {learner_name}")
            
        # 4. SMS
        self.sms = None
        if cfg.model.sms.enabled:
            self.sms = SMS(num_ranks=self.num_ranks)
            
        # 5. Losses
        self.criterion_main = nn.CrossEntropyLoss()
        self.criterion_rank = RankingAwareLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        self.criterion_pairwise = CrossModalPairwiseLoss(num_ranks=self.num_ranks, tau=cfg.loss.tau)
        
        self.lambda_rank = cfg.loss.lambda_rank
        self.mu_pairwise = cfg.loss.mu_pairwise
        self.tau = cfg.loss.tau
        
        # Set trainable
        set_trainable(
            self.clip_model, 
            self.text_encoder, 
            self.prompt_learner, 
            self.sms,
            train_prompts=True,
            train_lora=cfg.model.image_encoder.lora.enabled,
            train_sms=cfg.model.sms.enabled
        )
        
        # Logic for metrics collection
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []

    def forward(self, image):
        image_features = self.clip_model.visual(image.type(self.clip_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features

    def on_train_epoch_start(self):
        if self.sms is not None:
            self.sms.reset_stats()

    def training_step(self, batch, batch_idx):
        image, target, _, _ = batch
        image_features, text_features = self(image)
        
        # logit scale
        logit_scale = self.clip_model.logit_scale.exp()
        sim = logit_scale * image_features @ text_features.T
        
        if self.sms is not None:
            self.sms.update_stats(sim, target)
            sim = self.sms(sim)
            
        loss_main = self.criterion_main(sim, target)
        loss_rank = self.criterion_rank(sim, target) if self.lambda_rank > 0 else torch.tensor(0.0).to(self.device)
        loss_pairwise = self.criterion_pairwise(image_features, text_features, target) if self.mu_pairwise > 0 else torch.tensor(0.0).to(self.device)
        
        loss = loss_main + self.lambda_rank * loss_rank + self.mu_pairwise * loss_pairwise
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.sms is not None:
            self.sms.compute_smoothed_stats()

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self._shared_eval_epoch_end('val')

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self._shared_eval_epoch_end('test')

    def _shared_eval_step(self, batch, batch_idx):
        image, target, _, _ = batch
        image_features, text_features = self(image)
        
        logit_scale = self.clip_model.logit_scale.exp()
        sim = logit_scale * image_features @ text_features.T
        
        if self.sms is not None:
            sim = self.sms(sim)
            
        probs = F.softmax(sim, dim=-1)
        preds = probs.argmax(dim=-1)
        
        self.val_preds.append(preds.cpu())
        self.val_targets.append(target.cpu())
        self.val_probs.append(probs.cpu())

    def _shared_eval_epoch_end(self, prefix):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
        probs = torch.cat(self.val_probs).numpy()
        
        qwk = cohen_kappa_score(targets, preds, weights='quadratic')
        mae = mean_absolute_error(targets, preds)
        macro_f1 = f1_score(targets, preds, average='macro')
        acc = accuracy_score(targets, preds)
        
        # Calculate AUC (Handling multiclass logic)
        try:
            if self.num_ranks == 2:
                auc = roc_auc_score(targets, probs[:, 1])
            else:
                auc = roc_auc_score(targets, probs, multi_class='ovr')
        except ValueError:
            auc = 0.0 # Sometimes AUC fails if not all classes are present in test set
            
        # Local Ordinality Score (LOS) - Simplified 
        # Check monotonic decay of probabilities from predicted class
        los_count = 0
        total = 0
        for i in range(len(preds)):
            pred = preds[i]
            p = probs[i]
            # Check if adjacent ranks have monotonic drop
            if pred > 0 and pred < self.num_ranks - 1:
                total += 1
                if p[pred-1] > p[pred-2] if pred-2 >=0 else True:
                    los_count += 1
        los = los_count / max(total, 1)

        self.log(f'{prefix}_qwk', qwk)
        self.log(f'{prefix}_mae', mae)
        self.log(f'{prefix}_f1', macro_f1)
        self.log(f'{prefix}_auc', auc)
        self.log(f'{prefix}_acc', acc)
        self.log(f'{prefix}_los', los)
        
        # Also store for scripts
        self.latest_metrics = {
            'qwk': qwk, 'mae': mae, 'f1': macro_f1, 'auc': auc, 'acc': acc, 'los': los
        }
        
        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()

    def configure_optimizers(self):
        # We only pass trainable parameters to optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        if self.cfg.train.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        else:
            optimizer = torch.optim.SGD(trainable_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
            
        return optimizer
