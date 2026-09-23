"""
model.py -- RAD-DINO backbone (frozen or partially unfrozen) + small MLP head
for binary real-vs-deepfake classification.
"""
import torch.nn as nn
from transformers import AutoModel


class RadDinoDeepfakeClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "microsoft/rad-dino",
        num_unfrozen_blocks: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        embed_dim = self.backbone.config.hidden_size

        self.freeze_backbone()
        if num_unfrozen_blocks > 0:
            self.unfreeze_last_blocks(num_unfrozen_blocks)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n: int):
        """Unfreeze the last n transformer blocks (DINOv2 encoder layers) for fine-tuning."""
        blocks = self.backbone.encoder.layer
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]  # RAD-DINO's CLS/global token
        logits = self.head(cls_token).squeeze(-1)
        return logits