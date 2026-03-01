# Project 5: Production-Grade ML System with Monitoring

Build and deploy a complete ML system — from data ingestion to model serving — covering classification, time-series forecasting, and anomaly detection, with production-grade monitoring, drift detection, multi-channel alerting, automated retraining, and blue-green deployment. This project demonstrates the full lifecycle of ML in production.

---

## Design Decisions (Resolved)

All clarifying questions have been answered. These are the locked-in decisions driving the implementation:

### Task & Data

| # | Question | Decision | Implications |
|---|---|---|---|
| 1 | **ML task** | All three: classification + time-series forecasting + anomaly detection | Primary: tabular classification (fraud or churn). Secondary: time-series forecasting on a temporal feature. Tertiary: anomaly detection on incoming data as a monitoring signal (flag unusual inputs). Three models, one system. |
| 2 | **Data source** | Public dataset + real API/live data | Train on a public dataset (e.g., Kaggle credit-card-fraud or telco-churn). Connect to a real API for live inference data. Demonstrates both offline training and online serving. |
| 3 | **Data velocity** | Batch + real-time hybrid | Batch for training, evaluation, and drift detection (process accumulated data periodically). Real-time for serving predictions (per-request inference via API). Standard production pattern. |
| 4 | **Drift type** | Unknown — detect whatever comes | Build robust detection that catches any distributional change (gradual, sudden, or mixed) without assumptions. Use multiple statistical tests (PSI, KS-test, chi-squared) for comprehensive coverage. |
| 5 | **Feedback loop** | Yes, immediate | Ground-truth labels available quickly after prediction. Enables real-time accuracy tracking, rolling metrics, and fast retraining signals. This is the strongest monitoring setup. |

### Model & Training

| # | Question | Decision | Implications |
|---|---|---|---|
| 6 | **Model type** | Deep learning (LSTM / Transformer) | More complex to serve and monitor than classical ML — which makes the monitoring story more compelling. Use PyTorch. LSTM for time-series, Transformer encoder for classification, isolation forest for anomaly detection. |
| 7 | **Training cadence** | Triggered retrain (drift-based) | Retrain ONLY when drift is detected or accuracy drops below threshold. No scheduled retraining. Demonstrates the monitoring → action feedback loop. Cooldown: max 1 retrain per 24 hours. |
| 8 | **Training compute** | GPU preferred (RTX 3080), CPU fallback | Train on local 3080 for speed. CPU fallback for CI/CD and environments without GPU. Use PyTorch with CUDA detection. |
| 9 | **Model versions** | 2-3 (simple A/B) | Current production + 1 candidate + 1 archived. MLflow model registry with Staging/Production/Archived stages. Blue-green swap between production and candidate. |

### Monitoring & Operations

| # | Question | Decision | Implications |
|---|---|---|---|
| 10 | **Failure modes** | All three: accuracy drop + high latency + stale model | Monitor all dimensions. Accuracy drop = model degradation. High latency = infrastructure issue. Stale model = drift detected but retrain hasn't happened. Each triggers different alerts. |
| 11 | **Alert audience** | All three: ML engineers + DevOps/SRE + business stakeholders | Role-specific dashboards and alert routing. ML: drift scores, accuracy, feature importance. DevOps: uptime, latency, errors, resources. Business: prediction quality, outcome metrics. |
| 12 | **SLAs** | Relaxed: p99 < 1s, uptime > 99%, drift detection < 24hr | Focus effort on monitoring infrastructure rather than performance optimization. Achievable on local hardware. |
| 13 | **Alerting channels** | Grafana + Slack + Email | Three channels with severity-based routing. Critical → Slack + Email. Warning → Slack. Info → Grafana only. |

### Infrastructure

| # | Question | Decision | Implications |
|---|---|---|---|
| 14 | **Deployment** | Docker Compose (local) | All services run locally. Single-command setup. 6+ containers in the compose stack. |
| 15 | **Model update strategy** | Blue-green (swap) | Load new model → verify health → swap all traffic. Keep previous model in memory for instant rollback. Rollback trigger: accuracy drop or latency spike within 30 minutes of swap. |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                       │
│                                                                            │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────┐    │
│  │ Public Dataset  │   │ Live API Feed   │   │ Feedback Loop          │    │
│  │ (Kaggle/UCI)    │   │ (real-time)     │   │ (ground-truth labels)  │    │
│  └───────┬────────┘   └───────┬────────┘   └───────────┬────────────┘    │
│          │                     │                         │                 │
│          ▼                     ▼                         ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                   Ingestion Service                               │     │
│  │  - Schema validation (Pydantic)                                   │     │
│  │  - Data quality checks                                            │     │
│  │  - Log validation failures → Prometheus                           │     │
│  └──────────────────────────┬───────────────────────────────────────┘     │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │              Feature Store (Parquet + SQLite)                      │     │
│  │  - Raw features (versioned, timestamped)                          │     │
│  │  - Transformed features (scaled, encoded)                         │     │
│  │  - Baseline statistics (for drift detection)                      │     │
│  └──────────────────────────┬───────────────────────────────────────┘     │
└─────────────────────────────┼────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────┐
│                       TRAINING PIPELINE                                    │
│                                                                            │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐  │
│  │ Classification     │  │ Time-Series       │  │ Anomaly Detection    │  │
│  │ (Transformer enc.) │  │ (LSTM)            │  │ (Isolation Forest)   │  │
│  │ PyTorch + CUDA     │  │ PyTorch + CUDA    │  │ scikit-learn (CPU)   │  │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────────┘  │
│           └──────────────────────┼───────────────────────┘                 │
│                                  ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  MLflow Tracking + Model Registry                                 │     │
│  │  - Experiment tracking (params, metrics, artifacts)               │     │
│  │  - Model registry (Staging → Production → Archived)               │     │
│  │  - Evaluation gate (new model must beat production)               │     │
│  └──────────────────────────┬───────────────────────────────────────┘     │
│                              │                                             │
│  Triggered by: drift alert │ accuracy drop │ manual API call               │
└─────────────────────────────┼────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────┐
│                       SERVING LAYER                                        │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  FastAPI Model Server                                             │     │
│  │                                                                   │     │
│  │  POST /predict          ← single prediction (all 3 models)       │     │
│  │  POST /predict/batch    ← batch prediction                       │     │
│  │  GET  /health           ← liveness                                │     │
│  │  GET  /ready            ← readiness (models loaded?)              │     │
│  │  GET  /model/info       ← version, metrics, training date         │     │
│  │  POST /model/reload     ← hot-swap to new model version          │     │
│  │  POST /model/rollback   ← revert to previous model               │     │
│  │  POST /feedback         ← receive ground-truth labels             │     │
│  │  GET  /metrics          ← Prometheus endpoint                     │     │
│  │                                                                   │     │
│  │  Blue-Green: v1 (active) ←→ v2 (standby)                        │     │
│  │  Request Pipeline:                                                │     │
│  │  Input → Validate → Anomaly Check → Transform → Predict → Log   │     │
│  └──────────────────────────┬───────────────────────────────────────┘     │
└─────────────────────────────┼────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────┐
│                     MONITORING & OBSERVABILITY                              │
│                                                                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ System Metrics    │  │ Model Metrics     │  │ Data/Drift Metrics   │    │
│  │ (DevOps view)     │  │ (ML Eng view)     │  │ (ML Eng view)        │    │
│  │                   │  │                   │  │                      │    │
│  │ - Latency p50/95  │  │ - Rolling accuracy│  │ - Feature drift      │    │
│  │ - Throughput       │  │ - Rolling F1      │  │   (PSI, KS-test)   │    │
│  │ - Error rate       │  │ - Confidence dist │  │ - Prediction drift   │    │
│  │ - CPU/Memory       │  │ - Prediction dist │  │ - Anomaly rate       │    │
│  │ - GPU utilization  │  │ - Confusion matrix│  │ - Data quality       │    │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘    │
│           └──────────────────────┼───────────────────────┘                 │
│                                  ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  Prometheus ──► Grafana Dashboards (3 role-specific)             │     │
│  │              ──► Grafana Alerts ──► Slack (Critical + Warning)   │     │
│  │                                 ──► Email (Critical only)        │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  Evidently AI (Drift Detection Service)                          │     │
│  │  - Runs periodically on accumulated prediction data              │     │
│  │  - Computes per-feature drift scores                              │     │
│  │  - Generates drift reports (HTML + JSON)                          │     │
│  │  - Exports summary metrics to Prometheus                          │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────┐
│                     FEEDBACK & RETRAINING LOOP                             │
│                                                                            │
│  Drift Alert OR Accuracy Drop                                              │
│       │                                                                    │
│       ▼                                                                    │
│  Trigger Retraining (cooldown: max 1/day)                                  │
│       │                                                                    │
│       ▼                                                                    │
│  Train New Model → Evaluate → Compare vs. Production                       │
│       │                                                                    │
│       ├─► Better → Blue-Green Swap → Monitor 30 min → Confirm or Rollback │
│       └─► Worse  → Alert ("retrained model did not improve") → Keep old   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Data Pipeline & Feature Engineering (Week 1)

**Goal:** Build a reliable data ingestion pipeline with validation, feature engineering, and baseline statistics for drift detection.

1. **Select the datasets**
   - **Classification (primary):** Kaggle `credit-card-fraud` (binary classification, 284K transactions, 30 PCA features, highly imbalanced) OR `telco-customer-churn` (binary, 7K rows, mixed feature types)
   - **Time-series (secondary):** Use the temporal features from the primary dataset (transaction timestamps) OR a separate dataset like `electricity-demand`
   - **Anomaly detection (tertiary):** Train an isolation forest on the primary dataset's feature distributions — used as a monitoring signal during serving
   - **Decision:** Use a dataset with timestamps so we can simulate temporal data arrival and demonstrate chronological splits

2. **Build the data ingestion service (FastAPI)**
   - `POST /data/ingest` — accept raw data (CSV upload or JSON payload)
   - `POST /data/ingest/stream` — accept individual records (simulates real-time feed)
   - `POST /feedback` — accept ground-truth labels for previously predicted records
   - Schema validation with Pydantic:
     - Check column names, types, ranges
     - Reject malformed records, log validation failures as Prometheus metrics
     - Flag out-of-range values (don't reject, but mark for monitoring)
   - Store raw data: Parquet files partitioned by ingestion date
   - Store validated data: SQLite `predictions` table (prediction + ground-truth when available)

3. **Feature engineering pipeline**
   - scikit-learn `Pipeline` for tabular features:
     - Numerical: `StandardScaler`, `SimpleImputer` (median)
     - Categorical: `OneHotEncoder` or `OrdinalEncoder`
     - Temporal: hour-of-day, day-of-week, time-since-last-event (if applicable)
   - Serialize the fitted transformer alongside the model (pickle or joblib)
   - Version feature definitions in a config file (`features.yaml`)

4. **Feature store (lightweight)**
   - Parquet files for historical features (training data, versioned by date)
   - SQLite for inference features (latest computed features per entity)
   - API: `GET /features/training?version=2026-02-27` and `GET /features/inference/{entity_id}`

5. **Compute baseline statistics**
   - For each feature in the training set: mean, std, min, max, median, null rate, distribution histogram
   - Save as `baseline_stats.json` — used as the reference for drift detection
   - Expose a summary as a Prometheus gauge (feature_mean, feature_std per feature)

**Deliverable:** Data ingestion API, feature engineering pipeline, feature store, baseline statistics. Unit tests for validation and transformation.

---

### Phase 2: Model Training Pipeline (Week 1-2)

**Goal:** Train three models (classification, time-series, anomaly detection), track with MLflow, and set up automated evaluation gates.

1. **Set up MLflow**
   - Docker container: MLflow tracking server with SQLite backend
   - Artifact storage: local `./mlflow-artifacts/` directory
   - Model registry: Staging, Production, Archived stages
   - UI accessible at `http://localhost:5000`

2. **Classification model (Transformer encoder)**
   - Architecture: small Transformer encoder (2-4 layers, 64-128 hidden dim, 2-4 heads)
   - Input: tabular features (treat each feature as a "token" with embedding)
   - Output: binary classification (fraud/not-fraud or churn/not-churn)
   - Training: PyTorch, CUDA-accelerated on RTX 3080
   - Loss: binary cross-entropy with class weights (handle imbalance)
   - Hyperparameter tuning: Optuna (5-10 trials), tracked in MLflow
   - Log: accuracy, F1, precision, recall, AUC-ROC, confusion matrix, feature importance (attention weights)

3. **Time-series model (LSTM)**
   - Architecture: 2-layer LSTM, 64-128 hidden units
   - Input: sequence of recent feature vectors (window size = 24-48 time steps)
   - Output: forecast next N values (regression) OR predict next-period classification
   - Training: PyTorch, CUDA-accelerated
   - Log: MAE, RMSE, MAPE (if forecasting) or same classification metrics

4. **Anomaly detection model (Isolation Forest)**
   - scikit-learn `IsolationForest` — CPU-only (fast, no GPU needed)
   - Train on the "normal" class of the training data
   - Output: anomaly score per input (used during serving as a monitoring signal)
   - Log: contamination threshold, precision/recall on known anomalies

5. **Model evaluation gate**
   - Automated comparison: new model vs. current Production model (same held-out test set)
   - Promotion criteria:
     - Classification: new F1 >= production F1 - 0.01 (1% tolerance)
     - Time-series: new MAE <= production MAE * 1.05 (5% tolerance)
     - Anomaly: manual review only (no auto-promote)
   - If criteria met → promote to Staging
   - If criteria fail → log rejection reason, alert ML engineers

6. **Make training triggerable**
   - CLI: `python train.py --model classification --data-version 2026-02-27`
   - API: `POST /training/trigger` (accepts model type, data version)
   - Drift-triggered: called automatically by the drift detection service
   - Guard rails: cooldown of 24 hours between retrains, max 1 retrain per trigger type per day

**Deliverable:** Three trained models in MLflow, evaluation gate logic, triggerable training pipeline.

---

### Phase 3: Model Serving with Blue-Green Deployment (Week 2-3)

**Goal:** Serve all three models behind a production-quality API with blue-green swap and instant rollback.

1. **Build the FastAPI serving application**
   - **Prediction endpoints:**
     - `POST /predict` — full prediction pipeline:
       1. Validate input (schema check)
       2. Run anomaly detection (flag if anomaly score > threshold — don't block, just log)
       3. Transform features (using the fitted pipeline)
       4. Run classification model → return class + confidence
       5. Run time-series model → return forecast (if temporal features present)
       6. Log everything (input features, anomaly score, prediction, confidence, latency)
     - `POST /predict/batch` — batch prediction (up to 100 records)
     - Response:
       ```json
       {
         "prediction": "not_fraud",
         "confidence": 0.94,
         "anomaly_score": 0.12,
         "is_anomalous_input": false,
         "forecast": {"next_period": 0.87},
         "model_version": "v3",
         "latency_ms": 45
       }
       ```
   - **Operational endpoints:**
     - `GET /health` — liveness (server is running)
     - `GET /ready` — readiness (all models loaded and warm)
     - `GET /model/info` — all model versions, training dates, metrics
     - `POST /model/reload` — hot-swap to a new model version (blue-green)
     - `POST /model/rollback` — revert to previous model version
     - `POST /feedback` — receive ground-truth label for a previous prediction
     - `GET /metrics` — Prometheus metrics endpoint

2. **Blue-green model loading**
   - On startup: load Production model from MLflow registry → "active" slot
   - On reload: load new model → "standby" slot → warm up → swap active ↔ standby
   - Keep previous model in the standby slot → instant rollback (no reload needed)
   - Health check returns model version, so monitoring can verify the swap happened

3. **Request processing pipeline**
   ```
   Request
     → Validate Input (Pydantic schema, log validation failures)
     → Anomaly Detection (score input, log anomaly_score, flag if > threshold)
     → Feature Transform (fitted pipeline)
     → Classification Predict (Transformer encoder)
     → Time-Series Predict (LSTM, if temporal features present)
     → Post-Process (format response, add metadata)
     → Log Prediction (to SQLite for drift detection + to Prometheus for metrics)
     → Response
   ```

4. **Performance optimization**
   - Model warm-up: run 10 dummy predictions on startup to prime CUDA kernels
   - PyTorch inference mode: `torch.no_grad()` + `model.eval()`
   - Input batching: for batch endpoint, process all inputs in one forward pass
   - Async request handling: FastAPI + Uvicorn with multiple workers

5. **Graceful operations**
   - Graceful shutdown: finish in-flight requests before stopping (SIGTERM handler)
   - Health returns 503 during model loading/reloading
   - Request timeout: 5 seconds (return 504 if exceeded)
   - Blue-green swap is atomic: either the new model is fully loaded or the old one stays

**Deliverable:** FastAPI server with 3 models, blue-green swap, rollback, prediction logging, and Prometheus metrics.

---

### Phase 4: Monitoring Infrastructure (Week 3)

**Goal:** Comprehensive observability across system health, model performance, and data drift — with role-specific dashboards.

1. **System metrics (Prometheus — DevOps audience)**
   - Request count (total, by endpoint, by status code)
   - Request latency histogram (p50, p95, p99)
   - Prediction throughput (predictions/second)
   - Error rate (5xx / total)
   - CPU usage, memory usage, GPU utilization (nvidia-smi exporter)
   - Model loading time, model swap events
   - Active model version (label on all metrics)

2. **Model performance metrics (Prometheus — ML engineer audience)**
   - Prediction distribution: histogram of predicted classes over time
   - Confidence distribution: histogram of confidence scores (mean, p10, p50, p90)
   - **Real-time accuracy tracking (feedback loop):**
     - As ground-truth labels arrive via `/feedback`:
       - Compute rolling accuracy (last 100, 1000, 10000 predictions)
       - Rolling F1, precision, recall
       - Rolling confusion matrix (stored as Prometheus gauge per cell)
   - Anomaly rate: fraction of inputs flagged as anomalous (by isolation forest)
   - Feature importance stability: track attention weights across model versions

3. **Data and drift monitoring (Evidently AI)**
   - **Drift detection service:** runs as a separate Docker container
   - Trigger: runs every 1 hour (or every N predictions, whichever comes first)
   - Compares:
     - Current window (last hour's prediction inputs) vs. training baseline
     - Per-feature drift score using:
       - PSI (Population Stability Index) for numerical features
       - KS-test (Kolmogorov-Smirnov) for continuous distributions
       - Chi-squared test for categorical features
     - Aggregate drift score: fraction of features with drift > threshold
   - **Prediction drift detection:**
     - Compare recent prediction distribution vs. training prediction distribution
     - Detect: class proportion shift (more fraud predictions than expected), confidence shift
   - **Data quality monitoring:**
     - Null rate per feature (sudden increase = upstream pipeline issue)
     - Out-of-range values (values outside training min/max)
     - Schema violations (missing features, wrong types)
   - Output:
     - Evidently HTML reports (saved to `./drift-reports/`)
     - JSON summary exported to Prometheus (per-feature drift score, aggregate score)
     - Alert triggered if aggregate drift score > threshold

4. **Business metrics (Grafana — business stakeholder audience)**
   - Prediction volume over time (how many predictions are we serving?)
   - Fraud/churn detection rate (what fraction of inputs are predicted positive?)
   - False positive rate (when ground-truth arrives, how often were we wrong?)
   - Model version timeline (when did we last update the model?)

5. **Build Grafana dashboards**
   - **Dashboard 1: System Health (DevOps)**
     - Request rate, latency percentiles (p50/p95/p99), error rate
     - CPU/memory/GPU utilization, model load time
     - Active model version, uptime counter
   - **Dashboard 2: Model Performance (ML Engineers)**
     - Rolling accuracy, F1, precision, recall (with 100/1000/10000 windows)
     - Prediction confidence distribution over time
     - Confusion matrix (updated live as feedback arrives)
     - Anomaly detection rate
   - **Dashboard 3: Data & Drift (ML Engineers)**
     - Per-feature drift score heatmap (features × time)
     - Aggregate drift score with threshold line
     - Data quality violations (null rate, out-of-range, schema errors)
     - Last drift report link
   - **Dashboard 4: Business Outcomes (Stakeholders)**
     - Prediction volume, detection rate, false positive rate
     - Model version history with performance at each version
     - Retraining events timeline

**Deliverable:** Prometheus metrics exporter, Evidently drift detection service, 4 Grafana dashboards (role-specific), business metrics.

---

### Phase 5: Alerting & Automated Retraining (Week 3-4)

**Goal:** The system detects problems and takes corrective action — with severity-routed alerts to Slack and email.

1. **Alerting rules (Grafana Alerts)**

   | Alert | Condition | Severity | Channel | Action |
   |---|---|---|---|---|
   | Service Down | Health check fails for 30s | Critical | Slack + Email | Auto-restart (Docker restart policy), page on-call |
   | Error Rate Spike | 5xx rate > 5% for 2 min | Critical | Slack + Email | Investigate model/service health |
   | High Latency | p99 > 1s for 5 min | Warning | Slack | Investigate load, model complexity, GPU |
   | Accuracy Drop | Rolling accuracy drops > 5% from baseline | Critical | Slack + Email | Trigger retraining, consider rollback |
   | Data Drift Detected | Aggregate drift score > 0.3 | Warning | Slack | Trigger retraining pipeline |
   | Prediction Drift | Class distribution diverges > 20% from training | Warning | Slack | Investigate data + model |
   | High Anomaly Rate | Anomaly rate > 10% for 1 hour | Warning | Slack | Investigate upstream data quality |
   | Data Quality Failure | Null rate or schema violations > 5% | Warning | Slack | Investigate upstream pipeline |
   | Stale Model | No retrain for > 7 days despite drift signals | Warning | Email | Manual review of drift reports |
   | Retrain Failed | Training job fails or new model doesn't improve | Warning | Slack + Email | Manual investigation |

2. **Alert channel setup**
   - **Slack:** Create a `#ml-alerts` channel, configure Grafana Slack webhook
   - **Email:** Configure SMTP in Grafana for Critical alerts
   - Routing:
     - `Critical` → Slack + Email (immediate attention)
     - `Warning` → Slack only (investigate when possible)
     - `Info` → Grafana dashboard annotation only

3. **Build the retraining trigger**
   - **API:** `POST /training/trigger` (accepts `{reason: "drift", drift_score: 0.45, features_drifted: ["f1", "f5"]}`)
   - **Drift-triggered (automated):**
     - Drift detection service runs hourly
     - If aggregate drift score > threshold for 2 consecutive checks → call `/training/trigger`
     - If rolling accuracy drops > 5% → call `/training/trigger`
     - Log trigger reason with full context (which features drifted, by how much, current accuracy)
   - **Guard rails:**
     - Cooldown: max 1 retrain per 24 hours
     - If retrain was triggered but new model didn't improve → don't trigger again for same drift signal
     - If 3 consecutive retrains fail to improve → escalate alert to Critical

4. **Automated model promotion pipeline**
   ```
   Drift/Accuracy Alert Fires
     │
     ▼
   Check Cooldown (last retrain > 24hr ago?)
     │ Yes
     ▼
   Trigger Training Job
     │
     ▼
   Train New Model (all 3 models retrained)
     │
     ▼
   Evaluate on Held-Out Test Set
     │
     ▼
   Compare vs. Current Production
     │
     ├─► Classification F1 >= production F1 - 0.01?
     ├─► Time-series MAE <= production MAE * 1.05?
     │
     ├─► YES (both pass) ──► Blue-Green Swap
     │                        │
     │                        ▼
     │                    Monitor 30 Minutes
     │                        │
     │                        ├─► No degradation → Confirm swap, archive old model
     │                        └─► Degradation detected → Auto-rollback, alert
     │
     └─► NO (either fails) ──► Alert: "Retrained model did not improve"
                               Keep current production model
   ```

5. **Rollback mechanism**
   - **Auto-rollback:** If accuracy drops > 3% or p99 latency > 2s within 30 minutes of a model swap → automatically rollback
   - **Manual rollback:** `POST /model/rollback` → swap standby (previous model) back to active
   - Rollback is instant (previous model is already in memory)
   - Log all swap and rollback events with timestamps and reasons

**Deliverable:** 10 alerting rules, Slack + email integration, automated retraining pipeline, blue-green promotion, auto-rollback.

---

### Phase 6: CI/CD & Containerization (Week 4)

**Goal:** `docker-compose up` runs the entire system, and the full lifecycle is testable end-to-end.

1. **Docker Compose services**
   ```yaml
   services:
     model-server:       # FastAPI serving (port 8000)
     mlflow:             # MLflow tracking + registry (port 5000)
     prometheus:         # Metrics collection (port 9090)
     grafana:            # Dashboards + alerting (port 3000)
     drift-detector:     # Evidently AI drift detection (runs hourly)
     training-worker:    # Accepts retraining triggers (port 8001)
     redis:              # Prediction log buffer (for drift detector)
   ```
   - 7 services total
   - Shared network for inter-service communication
   - Persistent volumes for MLflow artifacts, Grafana dashboards, Prometheus data

2. **Environment configuration**
   - `.env` file (gitignored):
     ```
     # Model serving
     MODEL_TYPE=transformer
     CLASSIFICATION_MODEL_NAME=fraud-classifier
     TIMESERIES_MODEL_NAME=fraud-forecaster
     ANOMALY_MODEL_NAME=anomaly-detector

     # Drift detection
     DRIFT_CHECK_INTERVAL_MINUTES=60
     DRIFT_THRESHOLD=0.3
     DRIFT_CONSECUTIVE_REQUIRED=2

     # Retraining
     RETRAIN_COOLDOWN_HOURS=24
     EVALUATION_TOLERANCE_F1=0.01
     EVALUATION_TOLERANCE_MAE=0.05

     # Alerting
     SLACK_WEBHOOK_URL=https://hooks.slack.com/...
     SMTP_HOST=smtp.gmail.com
     SMTP_PORT=587
     SMTP_USER=...
     SMTP_PASS=...
     ALERT_EMAIL_TO=...

     # Rollback
     AUTO_ROLLBACK_WINDOW_MINUTES=30
     AUTO_ROLLBACK_ACCURACY_DROP=0.03
     AUTO_ROLLBACK_LATENCY_P99_MS=2000

     # Infrastructure
     GPU_ENABLED=true
     CUDA_VISIBLE_DEVICES=0
     ```

3. **CI/CD pipeline (GitHub Actions)**
   - **On PR:**
     - Lint (ruff/flake8)
     - Unit tests (pytest)
     - Integration tests (spin up Docker Compose, run smoke tests, tear down)
   - **On merge to main:**
     - Build Docker images
     - Push to container registry (GitHub Container Registry or DockerHub)
   - **On training trigger (manual or automated):**
     - Run training job
     - Evaluate new model
     - Conditionally promote (if evaluation gate passes)
     - Update model-server (hot-swap via API call)

4. **End-to-end lifecycle test (automated)**
   - Script: `e2e_test.py`
   - Steps:
     1. `docker-compose up` → verify all services healthy
     2. Ingest training data → verify data pipeline works
     3. Trigger training → verify models appear in MLflow
     4. Send prediction requests → verify responses are correct
     5. Send feedback (ground-truth) → verify rolling accuracy updates
     6. Verify Grafana dashboards show data
     7. Inject drifted data (modify feature distributions) → verify drift detection fires
     8. Verify retraining triggers automatically
     9. Verify new model is evaluated and promoted (or rejected)
     10. Verify blue-green swap happens
     11. Inject a bad model → verify auto-rollback fires
     12. Verify all alerts fire correctly (Slack + email)
   - This test IS the demo — it shows the entire production lifecycle

**Deliverable:** Docker Compose stack (7 services), CI/CD pipeline, end-to-end lifecycle test, README with setup instructions.

---

## Success Criteria

| Metric | Target |
|---|---|
| Model serving latency (p99) | < 1 second |
| Service uptime during testing | > 99% |
| Drift detection | Catches injected drift within 2 detection cycles (< 2 hours) |
| Automated retraining | Triggers and completes without manual intervention |
| Model promotion gate | Auto-promotes only when new model meets criteria |
| Auto-rollback | Fires within 30 minutes of degradation, completes in < 60 seconds |
| Blue-green swap | Zero-downtime model update |
| Dashboards | 4 role-specific dashboards (DevOps, ML, Drift, Business) |
| Alerts | All 10 alerts testable and functional |
| Alert routing | Critical → Slack + Email, Warning → Slack, Info → Grafana |
| Feedback loop | Ground-truth labels update rolling accuracy within 1 minute |
| End-to-end test | Full lifecycle completes automatically |
| Docker Compose | `docker-compose up` runs all 7 services |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Drift detection false positives trigger unnecessary retraining | Wasted compute, model churn | Require 2 consecutive drift signals; tune thresholds on historical data; 24hr cooldown |
| Drift detection false negatives miss real drift | Stale model, degraded predictions | Multiple statistical tests (PSI + KS + chi-squared); human review of weekly drift reports |
| Retrained model is worse but passes evaluation gate | Production degradation | Auto-rollback monitoring for 30 min post-swap; tight tolerance (F1 - 0.01); accuracy drop triggers rollback |
| Ground-truth feedback overwhelms the system | Storage bloat, slow rolling metrics | Buffer feedback in Redis; compute rolling metrics on a window (not all-time); archive old feedback |
| Transformer encoder is overkill for tabular data | Worse than XGBoost, harder to serve | Accept this for the learning value; can always add XGBoost comparison in MLflow |
| 7 Docker containers strain local resources | Slow, OOM, unstable | Set memory limits per container; monitor host resources; can reduce to 5 by combining services |
| Slack/email integration requires external setup | Demo doesn't work without credentials | Make alerting channels optional (graceful degradation); Grafana alerts work standalone |
| Anomaly detection model drifts itself | False anomaly signals over time | Retrain anomaly model alongside classification model; compare anomaly rates pre/post retrain |
| GPU contention between serving and training | Both slow down when retraining | Schedule training during low-traffic periods; or CPU fallback for training while GPU serves |
| Alert fatigue from too many alerts | Alerts get ignored | Conservative thresholds; severity routing; aggregate related alerts; weekly threshold review |

---

## Tech Stack (Locked In)

| Component | Choice | Rationale |
|---|---|---|
| Classification Model | PyTorch Transformer encoder (small) | DL for tabular — more compelling monitoring story than XGBoost |
| Time-Series Model | PyTorch LSTM (2-layer) | Standard temporal model, easy to train on RTX 3080 |
| Anomaly Detection | scikit-learn Isolation Forest | Fast CPU-only training, no GPU needed, good unsupervised baseline |
| Training Compute | RTX 3080 (GPU) with CPU fallback | Consistent with other projects |
| Experiment Tracking | MLflow (Docker) | Model registry + experiment tracking + artifact storage in one |
| Model Serving | FastAPI + Uvicorn | Async, Prometheus-compatible, blue-green support |
| Feature Store | Parquet + SQLite | Simple, portable, sufficient for project scope |
| Drift Detection | Evidently AI | Purpose-built for ML monitoring, HTML reports + JSON metrics |
| Metrics Collection | Prometheus | Industry standard, pull-based, PromQL |
| Dashboards | Grafana (4 dashboards) | Best-in-class visualization, native Prometheus + alerting |
| Alerting | Grafana Alerts → Slack + Email | Integrated with dashboards, severity routing |
| Prediction Logging | Redis (buffer) + SQLite (persistent) | Fast write buffer + durable storage for drift analysis |
| Containerization | Docker Compose (7 services) | Single-command full stack deployment |
| CI/CD | GitHub Actions | Lint, test, build, push, conditionally train/deploy |
| Deployment Strategy | Blue-green (model swap) | Zero-downtime updates, instant rollback |
