import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY


def _get_metric_value(name, labels=None):
    """Helper to get a prometheus metric sample value by sample name."""
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name:
                if labels is None:
                    return sample.value
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


class TestPrometheusMiddleware:
    def test_request_counted(self, test_client):
        """GET /health increments http_requests_total."""
        test_client.get("/health")
        val = _get_metric_value(
            "http_requests_total",
            {"method": "GET", "endpoint": "/health", "status_code": "200"},
        )
        assert val is not None and val >= 1

    def test_request_duration_recorded(self, test_client):
        """GET /health records latency in http_request_duration_seconds."""
        test_client.get("/health")
        # Check the histogram count bucket
        val = _get_metric_value(
            "http_request_duration_seconds_count",
            {"method": "GET", "endpoint": "/health"},
        )
        assert val is not None and val >= 1

    def test_status_code_label(self, test_client):
        """404 responses are labeled correctly."""
        test_client.get("/nonexistent-endpoint-xyz")
        val = _get_metric_value(
            "http_requests_total",
            {"method": "GET", "endpoint": "/nonexistent-endpoint-xyz", "status_code": "404"},
        )
        assert val is not None and val >= 1

    def test_metrics_endpoint_skipped(self, test_client):
        """Requests to /metrics should not be counted by the middleware."""
        # Get current count for /metrics endpoint
        before = _get_metric_value(
            "http_requests_total",
            {"method": "GET", "endpoint": "/metrics", "status_code": "200"},
        )
        test_client.get("/metrics")
        after = _get_metric_value(
            "http_requests_total",
            {"method": "GET", "endpoint": "/metrics", "status_code": "200"},
        )
        # Should not have changed (both None or same value)
        assert before == after

    def test_in_progress_gauge_returns_to_zero(self, test_client):
        """After request completes, in-progress gauge should be back to 0."""
        test_client.get("/health")
        val = _get_metric_value(
            "http_requests_in_progress",
            {"method": "GET", "endpoint": "/health"},
        )
        assert val == 0.0
