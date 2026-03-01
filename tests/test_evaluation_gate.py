import pytest
from unittest.mock import patch

from src.services.evaluation_service import check_promotion_gate


class TestEvaluationGate:
    @patch("src.services.evaluation_service.get_production_model_metrics")
    @patch("src.services.evaluation_service.promote_model")
    def test_no_production_model_promotes(self, mock_promote, mock_metrics):
        mock_metrics.return_value = None
        result = check_promotion_gate("classification", {"f1": 0.5}, "fraud-classifier", 1)
        assert result["decision"] == "promoted"
        assert result["promoted_to"] == "Staging"
        mock_promote.assert_called_once()

    @patch("src.services.evaluation_service.get_production_model_metrics")
    @patch("src.services.evaluation_service.promote_model")
    def test_better_model_promotes(self, mock_promote, mock_metrics):
        mock_metrics.return_value = {"f1": 0.80}
        result = check_promotion_gate("classification", {"f1": 0.82}, "fraud-classifier", 2)
        assert result["decision"] == "promoted"

    @patch("src.services.evaluation_service.get_production_model_metrics")
    def test_worse_model_rejected(self, mock_metrics):
        mock_metrics.return_value = {"f1": 0.90}
        result = check_promotion_gate("classification", {"f1": 0.85}, "fraud-classifier", 2)
        assert result["decision"] == "rejected"
        assert result["promoted_to"] is None

    @patch("src.services.evaluation_service.get_production_model_metrics")
    @patch("src.services.evaluation_service.promote_model")
    def test_within_tolerance_promotes(self, mock_promote, mock_metrics):
        mock_metrics.return_value = {"f1": 0.80}
        # 0.795 >= 0.80 - 0.01 = 0.79 → should promote
        result = check_promotion_gate("classification", {"f1": 0.795}, "fraud-classifier", 3)
        assert result["decision"] == "promoted"

    def test_anomaly_manual_review(self):
        result = check_promotion_gate("anomaly", {"f1": 0.5}, "fraud-anomaly", 1)
        assert result["decision"] == "manual_review"
        assert result["promoted_to"] is None
