import torch
import pytest

from src.models.classifier import TabularTransformer


class TestTabularTransformer:
    def test_forward_shape(self):
        model = TabularTransformer(num_features=33, d_model=32, nhead=2, num_layers=1)
        x = torch.randn(4, 33)
        logits = model(x)
        assert logits.shape == (4, 1)

    def test_output_dtype(self):
        model = TabularTransformer(num_features=33, d_model=32, nhead=2, num_layers=1)
        x = torch.randn(8, 33)
        logits = model(x)
        assert logits.dtype == torch.float32

    def test_different_configs(self):
        for d_model, nhead in [(64, 4), (128, 4), (64, 2)]:
            model = TabularTransformer(num_features=33, d_model=d_model, nhead=nhead)
            x = torch.randn(2, 33)
            out = model(x)
            assert out.shape == (2, 1)

    def test_gradient_flow(self):
        model = TabularTransformer(num_features=33, d_model=32, nhead=2, num_layers=1)
        x = torch.randn(4, 33)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        logits = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
