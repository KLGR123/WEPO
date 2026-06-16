"""
DeBERTa-based element ranking model.

A thin wrapper around DeBERTa that adds a scalar classification head,
producing a single relevance logit per (context, element) pair.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ElementRanker(nn.Module):
    """
    Encodes (context, element) token pairs with DeBERTa and produces a
    single relevance logit via a linear head on the [CLS] representation.
    """

    def __init__(self, base_model_name: str = "microsoft/deberta-v3-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            token_type_ids: (B, L) or None
        Returns:
            logits: (B,)  – raw relevance scores (no sigmoid/softmax applied)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(self.dropout(cls_repr)).squeeze(-1)  # (B,)
        return logits

    def save(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), f"{output_dir}/classifier.pt")
        torch.save({"dropout": self.dropout.p}, f"{output_dir}/config.pt")

    @classmethod
    def load(cls, output_dir: str) -> "ElementRanker":
        cfg = torch.load(f"{output_dir}/config.pt", map_location="cpu", weights_only=True)
        model = cls(base_model_name=output_dir, dropout=cfg["dropout"])
        model.classifier.load_state_dict(
            torch.load(f"{output_dir}/classifier.pt", map_location="cpu", weights_only=True)
        )
        return model
