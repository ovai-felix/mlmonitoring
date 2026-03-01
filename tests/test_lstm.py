import torch
import pytest

from src.models.lstm_model import FraudLSTM


class TestFraudLSTM:
    def test_forward_shape(self):
        model = FraudLSTM(input_size=33, hidden_size=32, num_layers=1)
        x = torch.randn(4, 32, 33)  # (batch, seq_len, features)
        logits = model(x)
        assert logits.shape == (4, 1)

    def test_different_window_sizes(self):
        model = FraudLSTM(input_size=33, hidden_size=32, num_layers=2)
        for seq_len in [16, 32, 48]:
            x = torch.randn(2, seq_len, 33)
            out = model(x)
            assert out.shape == (2, 1)

    def test_gradient_flow(self):
        model = FraudLSTM(input_size=33, hidden_size=32, num_layers=1)
        x = torch.randn(4, 32, 33)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        logits = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
