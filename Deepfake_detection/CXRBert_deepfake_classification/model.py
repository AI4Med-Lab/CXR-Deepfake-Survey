"""
model.py -- CXR-BERT-based classifier for radiology report text.

Backbone: microsoft/BiomedVLP-CXR-BERT-specialized (a BERT-family text
encoder pretrained on radiology reports). This is a text-only model, so
there is no image tower here -- classification runs on the pooled text
embedding alone.

NOTE: CXR-BERT requires `trust_remote_code=True` (it ships a custom modeling
file with a `get_projected_text_embeddings` method for contrastive use, on
top of the standard BERT forward pass). This code uses the standard forward
pass + pooled/CLS embedding rather than the projected contrastive embedding,
since that's more appropriate for a from-scratch classification head. Verify
attribute paths (`encoder.layer`, `config.hidden_size`) against your
installed `transformers` version -- if `_get_encoder_layers()` raises,
print(self.backbone) once and adjust the path.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class CXRBertClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        num_unfrozen_layers: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        backbone_name: HF model id for CXR-BERT
        num_unfrozen_layers: number of trailing transformer layers to unfreeze
            for fine-tuning. 0 = fully frozen backbone, linear-probe only.
        hidden_dim: width of the hidden layer in the classification head
        dropout: dropout probability in the classification head
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)

        # Freeze the whole backbone by default; unfreeze_last_layers() (or the
        # num_unfrozen_layers arg) opens up the last N transformer layers.
        for p in self.backbone.parameters():
            p.requires_grad = False

        text_dim = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.num_unfrozen_layers = 0
        if num_unfrozen_layers > 0:
            self.unfreeze_last_layers(num_unfrozen_layers)

    def _get_encoder_layers(self):
        """Locate the stack of transformer layers on the backbone (BERT-style)."""
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        if hasattr(self.backbone, "bert") and hasattr(self.backbone.bert, "encoder"):
            return self.backbone.bert.encoder.layer
        raise AttributeError(
            "Could not locate encoder layers on the CXR-BERT backbone. "
            "Run print(self.backbone) to inspect its structure and update "
            "_get_encoder_layers() with the correct attribute path."
        )

    def unfreeze_last_layers(self, n: int):
        """Unfreeze the last n transformer layers of the text encoder."""
        layers = self._get_encoder_layers()
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.num_unfrozen_layers = n

    @staticmethod
    def _pooled_embedding(outputs):
        """
        Extract a single pooled embedding per example. Prefers `.pooler_output`
        (BERT's [CLS] -> dense -> tanh pooler); falls back to the raw [CLS]
        token from `.last_hidden_state` if no pooler head is present.
        """
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeds = self._pooled_embedding(outputs)
        logits = self.head(embeds).squeeze(-1)
        return logits