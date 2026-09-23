"""
model.py -- MedSigLIP-based classifier for CXR deepfake detection.

Unlike the RAD-DINO baseline (image-only), this model encodes BOTH the image
and the paired text report using MedSigLIP's dual encoders, concatenates the
two embeddings, and feeds them through a small linear classification head.

NOTE: MedSigLIP (google/medsiglip-448) is a SigLIP-family dual encoder model.
This code assumes it exposes `get_image_features()` / `get_text_features()`
and `vision_model.encoder.layers` / `text_model.encoder.layers`, matching the
standard HF SigLIP API. Verify against your installed `transformers` version
before running -- adjust attribute paths if they differ.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class MedSiglipDeepfakeClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "google/medsiglip-448",
        num_unfrozen_blocks: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        backbone_name: HF model id for MedSigLIP
        num_unfrozen_blocks: number of trailing transformer blocks (in BOTH the
            vision and text towers) to unfreeze for fine-tuning. 0 = fully frozen
            backbone, linear-probe only.
        hidden_dim: width of the hidden layer in the classification head
        dropout: dropout probability in the classification head
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, token = "")

        # Freeze the whole backbone by default; unfreeze_last_blocks() (or the
        # num_unfrozen_blocks arg) opens up the last N blocks of each tower.
        for p in self.backbone.parameters():
            p.requires_grad = False

        vision_dim = self.backbone.config.vision_config.hidden_size
        text_dim = self.backbone.config.text_config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.num_unfrozen_blocks = 0
        if num_unfrozen_blocks > 0:
            self.unfreeze_last_blocks(num_unfrozen_blocks)

    def unfreeze_last_blocks(self, n: int):
        """Unfreeze the last n encoder blocks of both the vision and text towers."""
        vision_layers = self.backbone.vision_model.encoder.layers
        text_layers = self.backbone.text_model.encoder.layers

        for layer in list(vision_layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for layer in list(text_layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.num_unfrozen_blocks = n

    @staticmethod
    def _pooled_embedding(model_output, attention_mask=None):
        """
        Extract a single pooled embedding per example from a HF model output.
        Prefers `.pooler_output`; falls back to mean-pooling `.last_hidden_state`
        (mask-aware if an attention_mask is given) if no pooler head exists.
        """
        if isinstance(model_output, torch.Tensor):
            return model_output

        pooled = getattr(model_output, "pooler_output", None)
        if pooled is not None:
            return pooled

        hidden = model_output.last_hidden_state  # (batch, seq_len, dim)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq_len, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        return hidden.mean(dim=1)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        vision_out = self.backbone.vision_model(pixel_values=pixel_values)
        text_out = self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask)

        image_embeds = self._pooled_embedding(vision_out)
        text_embeds = self._pooled_embedding(text_out, attention_mask=attention_mask)

        combined = torch.cat([image_embeds, text_embeds], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits