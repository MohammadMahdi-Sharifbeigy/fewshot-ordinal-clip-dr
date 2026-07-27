import torch
import torch.nn as nn
import clip

def load_clip_model(name="RN50", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads CLIP and returns the visual encoder and the model (which contains the text encoder).
    """
    model, _ = clip.load(name, device=device)
    
    # Cast to float32 to prevent PyTorch Lightning's AMP GradScaler from
    # crashing when attempting to unscale static FP16 gradients.
    model = model.float()
    
    # We will use model.visual as image_encoder
    # and the rest of model (with modified forward) as text_encoder
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        prompts: [batch_size, n_ctx, d_model]
        tokenized_prompts: [batch_size, n_ctx]
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
