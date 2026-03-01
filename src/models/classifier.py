import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    """Transformer encoder for tabular binary classification.

    Each of the input features is treated as a "token": projected from a scalar
    to a d_model-dimensional embedding via a per-feature linear layer. A learnable
    CLS token is prepended and its final representation drives the binary output.
    """

    def __init__(
        self,
        num_features: int = 33,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-feature linear projection: scalar -> d_model
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.position_embedding = nn.Embedding(num_features + 1, d_model)  # +1 for CLS
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features) - scaled feature values
        Returns:
            logits: (batch, 1) - raw logits for binary classification
        """
        batch_size = x.size(0)

        # Project each feature scalar to d_model dimensions
        embeddings = []
        for i, proj in enumerate(self.feature_embeddings):
            feat_val = x[:, i : i + 1]  # (batch, 1)
            embeddings.append(proj(feat_val))  # (batch, d_model)
        token_embeddings = torch.stack(embeddings, dim=1)  # (batch, num_features, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)

        # Add positional encodings (CLS at position 0, features at 1..num_features)
        positions = torch.arange(self.num_features + 1, device=x.device)
        token_embeddings = token_embeddings + self.position_embedding(positions)
        token_embeddings = self.dropout(token_embeddings)

        encoded = self.transformer_encoder(token_embeddings)
        cls_output = encoded[:, 0, :]  # (batch, d_model)
        return self.classifier(cls_output)  # (batch, 1)
