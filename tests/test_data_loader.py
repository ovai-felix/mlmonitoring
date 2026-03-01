import numpy as np
import pytest

from src.services.data_loader import FraudDataset, SequenceDataset


class TestFraudDataset:
    def test_len_and_getitem(self):
        X = np.random.randn(50, 33).astype(np.float32)
        y = np.array([0] * 48 + [1] * 2, dtype=np.float32)
        ds = FraudDataset(X, y)
        assert len(ds) == 50
        features, label = ds[0]
        assert features.shape == (33,)
        assert label.shape == ()

    def test_label_values(self):
        X = np.zeros((5, 33), dtype=np.float32)
        y = np.array([0, 1, 0, 1, 0], dtype=np.float32)
        ds = FraudDataset(X, y)
        _, label = ds[1]
        assert float(label) == 1.0


class TestSequenceDataset:
    def test_len(self):
        X = np.random.randn(100, 33).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        ds = SequenceDataset(X, y, window_size=32)
        assert len(ds) == 68  # 100 - 32

    def test_getitem_shapes(self):
        X = np.random.randn(100, 33).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        ds = SequenceDataset(X, y, window_size=32)
        x_seq, label = ds[0]
        assert x_seq.shape == (32, 33)
        assert label.shape == ()

    def test_label_is_next_step(self):
        X = np.zeros((10, 33), dtype=np.float32)
        y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        ds = SequenceDataset(X, y, window_size=4)
        _, label = ds[0]  # label is y[4] = 1
        assert float(label) == 1.0
