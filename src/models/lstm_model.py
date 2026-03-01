import torch
import torch.nn as nn


class FraudLSTM(nn.Module):
    """LSTM for sequence-based fraud prediction.

    Takes a window of recent transaction feature vectors and predicts whether
    the next transaction is fraudulent.
    """

    def __init__(
        self,
        input_size: int = 33,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) - window of feature vectors
        Returns:
            logits: (batch, 1) - raw logits for binary classification
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.classifier(last_hidden)  # (batch, 1)
