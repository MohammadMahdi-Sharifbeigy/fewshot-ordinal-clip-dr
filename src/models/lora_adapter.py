from peft import LoraConfig, get_peft_model
import torch.nn as nn

def attach_lora(model, r=16, target_modules=("v_proj", "q_proj"), lora_alpha=16, lora_dropout=0.1):
    """
    Attach LoRA to the vision encoder of CLIP model.
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # In CLIP RN50, the attention layers might have different names, 
    # e.g., 'attn', 'qkv'. Peft might need specific target module names.
    # We apply it to model.visual
    model.visual = get_peft_model(model.visual, config)
    return model

def set_trainable(model, text_encoder, prompt_learner, sms=None, train_prompts=True, train_lora=True, train_sms=True):
    """
    Freezes CLIP backbone, unfreezes specified components.
    """
    # Freeze all in text_encoder and model.visual
    for name, param in text_encoder.named_parameters():
        param.requires_grad = False
        
    if not train_lora:
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
            
    # Unfreeze prompt learner
    if train_prompts:
        for name, param in prompt_learner.named_parameters():
            param.requires_grad = True
            
    # Unfreeze SMS if exists
    if sms is not None and train_sms:
        for name, param in sms.named_parameters():
            param.requires_grad = True
            
    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
    if sms:
        trainable += sum(p.numel() for p in sms.parameters() if p.requires_grad)
        
    print(f"Trainable parameters: {trainable}")
