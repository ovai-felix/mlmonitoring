from prometheus_client import Counter, Histogram, Gauge

# Counters
records_ingested = Counter(
    "records_ingested_total",
    "Total records ingested",
    ["source", "status"],
)

validation_failures = Counter(
    "validation_failures_total",
    "Total validation failures",
    ["error_type"],
)

validation_warnings = Counter(
    "validation_warnings_total",
    "Total validation warnings",
    ["feature"],
)

feedback_received = Counter(
    "feedback_received_total",
    "Total feedback records received",
)

feedback_match_errors = Counter(
    "feedback_match_errors_total",
    "Feedback records that could not be matched",
)

# Histograms
ingestion_latency = Histogram(
    "ingestion_latency_seconds",
    "Time to ingest a batch",
    ["source"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

feature_transform_latency = Histogram(
    "feature_transform_latency_seconds",
    "Time to transform features",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# Gauges
baseline_feature_mean = Gauge(
    "baseline_feature_mean",
    "Baseline mean per feature",
    ["feature"],
)

baseline_feature_std = Gauge(
    "baseline_feature_std",
    "Baseline std per feature",
    ["feature"],
)

baseline_feature_null_rate = Gauge(
    "baseline_feature_null_rate",
    "Baseline null rate per feature",
    ["feature"],
)

# Phase 3: Prediction serving metrics
predictions_total = Counter(
    "predictions_total",
    "Total predictions by class label",
    ["label"],
)

anomalies_detected = Counter(
    "anomalies_detected_total",
    "Total anomalies detected by anomaly model",
)

model_reload_total = Counter(
    "model_reload_total",
    "Total model reload attempts",
    ["status"],
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

active_model_version = Gauge(
    "active_model_version",
    "Currently active model version info",
    ["slot_color", "version_tag"],
)

# Phase 4: Monitoring infrastructure metrics

# HTTP request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    ["method", "endpoint"],
)

# Model load timing
model_load_duration_seconds = Histogram(
    "model_load_duration_seconds",
    "Time to load models into a slot",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Confidence distribution
prediction_confidence = Histogram(
    "prediction_confidence",
    "Prediction confidence distribution by label",
    ["label"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

# Rolling performance gauges
rolling_accuracy = Gauge(
    "rolling_accuracy",
    "Rolling accuracy over recent predictions",
    ["window_size"],
)

rolling_precision = Gauge(
    "rolling_precision",
    "Rolling precision over recent predictions",
    ["window_size"],
)

rolling_recall = Gauge(
    "rolling_recall",
    "Rolling recall over recent predictions",
    ["window_size"],
)

rolling_f1 = Gauge(
    "rolling_f1",
    "Rolling F1 score over recent predictions",
    ["window_size"],
)

# Rolling confusion matrix
rolling_true_positives = Gauge(
    "rolling_true_positives",
    "Rolling true positives count",
    ["window_size"],
)

rolling_false_positives = Gauge(
    "rolling_false_positives",
    "Rolling false positives count",
    ["window_size"],
)

rolling_true_negatives = Gauge(
    "rolling_true_negatives",
    "Rolling true negatives count",
    ["window_size"],
)

rolling_false_negatives = Gauge(
    "rolling_false_negatives",
    "Rolling false negatives count",
    ["window_size"],
)

# Drift metrics
feature_drift_psi = Gauge(
    "feature_drift_psi",
    "PSI drift score per feature",
    ["feature"],
)

feature_drift_ks_statistic = Gauge(
    "feature_drift_ks_statistic",
    "KS test statistic per feature",
    ["feature"],
)

feature_drift_ks_pvalue = Gauge(
    "feature_drift_ks_pvalue",
    "KS test p-value per feature",
    ["feature"],
)

drift_detected = Gauge(
    "drift_detected",
    "Whether drift was detected (1=yes, 0=no)",
)

drift_report_generated_total = Counter(
    "drift_report_generated_total",
    "Total drift reports generated",
    ["status"],
)

# Prediction drift
prediction_drift_fraud_rate = Gauge(
    "prediction_drift_fraud_rate",
    "Current fraud rate in recent predictions",
)

prediction_drift_mean_confidence = Gauge(
    "prediction_drift_mean_confidence",
    "Mean confidence of recent predictions",
)

# Data quality
data_quality_null_rate = Gauge(
    "data_quality_null_rate",
    "Null rate per feature in recent data",
    ["feature"],
)

data_quality_out_of_range_rate = Gauge(
    "data_quality_out_of_range_rate",
    "Out of range rate per feature in recent data",
    ["feature"],
)

# Business metrics
fraud_detection_rate = Gauge(
    "fraud_detection_rate",
    "Overall fraud detection rate",
)

false_positive_rate = Gauge(
    "false_positive_rate",
    "False positive rate",
    ["window_size"],
)

# Phase 5: Retraining and rollback metrics
retrain_triggered_total = Counter(
    "retrain_triggered_total",
    "Total retrain triggers by reason",
    ["reason"],
)

retrain_outcome_total = Counter(
    "retrain_outcome_total",
    "Total retrain outcomes",
    ["outcome"],
)

consecutive_retrain_failures = Gauge(
    "consecutive_retrain_failures",
    "Consecutive retrain failure streak",
)

model_swap_total = Counter(
    "model_swap_total",
    "Total model swap actions",
    ["action"],
)

model_swap_timestamp = Gauge(
    "model_swap_timestamp",
    "Epoch time of last model swap",
)

post_swap_monitoring_active = Gauge(
    "post_swap_monitoring_active",
    "Whether post-swap monitoring is active (1=yes, 0=no)",
)
