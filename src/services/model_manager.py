import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

from src.config import settings
from src.metrics import model_reload_total, active_model_version, model_load_duration_seconds
from src.models.classifier import TabularTransformer
from src.models.lstm_model import FraudLSTM
from src.models.anomaly import AnomalyDetector
from src.services.feature_engineering import load_pipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelSlot:
    classifier: TabularTransformer | None = None
    lstm: FraudLSTM | None = None
    anomaly: AnomalyDetector | None = None
    pipeline: object | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    loaded_at: str | None = None
    version_tag: str = "none"
    is_ready: bool = False

    def has_any_model(self) -> bool:
        return any([self.classifier, self.lstm, self.anomaly])


def _resolve_device() -> torch.device:
    if settings.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.device)


class ModelManager:
    def __init__(self):
        self._blue = ModelSlot()
        self._green = ModelSlot()
        self._active_color: str = "blue"
        self._lock = threading.Lock()
        self._reload_in_progress = False

    @property
    def active(self) -> ModelSlot:
        return self._blue if self._active_color == "blue" else self._green

    @property
    def standby(self) -> ModelSlot:
        return self._green if self._active_color == "blue" else self._blue

    def load_slot(
        self,
        slot: ModelSlot,
        models_dir: Path | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        load_start = time.time()
        mdir = models_dir or settings.models_dir
        adir = artifacts_dir or settings.artifacts_dir
        device = _resolve_device()
        slot.device = device
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        # Load classifier
        config_path = mdir / "classifier" / "config.pt"
        model_path = mdir / "classifier" / "model.pt"
        if config_path.exists() and model_path.exists():
            config = torch.load(config_path, map_location=device, weights_only=True)
            model = TabularTransformer(**config)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device).eval()
            slot.classifier = model
            logger.info("Loaded classifier from %s", model_path)

        # Load LSTM
        config_path = mdir / "lstm" / "config.pt"
        model_path = mdir / "lstm" / "model.pt"
        if config_path.exists() and model_path.exists():
            config = torch.load(config_path, map_location=device, weights_only=True)
            model = FraudLSTM(**config)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device).eval()
            slot.lstm = model
            logger.info("Loaded LSTM from %s", model_path)

        # Load anomaly detector
        anomaly_path = mdir / "anomaly" / "model.joblib"
        if anomaly_path.exists():
            slot.anomaly = AnomalyDetector.load(anomaly_path)
            logger.info("Loaded anomaly detector from %s", anomaly_path)

        # Load feature pipeline
        try:
            slot.pipeline = load_pipeline(adir)
            logger.info("Loaded feature pipeline from %s", adir)
        except FileNotFoundError:
            logger.warning("No feature pipeline found in %s", adir)

        slot.loaded_at = timestamp
        slot.version_tag = timestamp
        model_load_duration_seconds.observe(time.time() - load_start)

    def warm_up(self, slot: ModelSlot, n: int = 10) -> None:
        device = slot.device
        if slot.classifier:
            dummy = torch.randn(n, slot.classifier.num_features, device=device)
            with torch.no_grad():
                slot.classifier(dummy)

        if slot.lstm:
            dummy = torch.randn(n, 1, slot.lstm.lstm.input_size, device=device)
            with torch.no_grad():
                slot.lstm(dummy)

        if slot.anomaly and slot.pipeline:
            dummy = np.zeros((n, 33))
            try:
                slot.anomaly.predict(dummy)
            except Exception:
                pass

        slot.is_ready = True
        logger.info("Warm-up complete for slot (version=%s)", slot.version_tag)

    def initial_load(self) -> None:
        self.load_slot(self.active)
        if self.active.has_any_model():
            self.warm_up(self.active)
            self._update_version_gauge()
            logger.info("Initial model load complete")
        else:
            logger.warning("No models found on disk — /ready will return 503")

    def reload(self) -> dict:
        with self._lock:
            if self._reload_in_progress:
                return {"status": "error", "detail": "Reload already in progress"}
            self._reload_in_progress = True

        try:
            standby = self.standby
            # Reset standby slot
            standby.classifier = None
            standby.lstm = None
            standby.anomaly = None
            standby.pipeline = None
            standby.is_ready = False

            self.load_slot(standby)
            if not standby.has_any_model():
                model_reload_total.labels(status="failure").inc()
                return {"status": "error", "detail": "No models found on disk"}

            self.warm_up(standby)
            old_color = self._active_color
            self._active_color = "green" if old_color == "blue" else "blue"
            self._update_version_gauge()
            model_reload_total.labels(status="success").inc()
            logger.info("Reload complete: %s → %s", old_color, self._active_color)
            return {"status": "ok", "active_color": self._active_color}
        except Exception as e:
            model_reload_total.labels(status="failure").inc()
            logger.exception("Reload failed")
            return {"status": "error", "detail": str(e)}
        finally:
            with self._lock:
                self._reload_in_progress = False

    def rollback(self) -> dict:
        standby = self.standby
        if not standby.is_ready:
            return {"status": "error", "detail": "Standby slot is not ready"}

        old_color = self._active_color
        self._active_color = "green" if old_color == "blue" else "blue"
        self._update_version_gauge()
        logger.info("Rollback: %s → %s", old_color, self._active_color)
        return {"status": "ok", "active_color": self._active_color}

    def get_info(self) -> dict:
        def _slot_info(slot: ModelSlot, color: str) -> dict:
            return {
                "color": color,
                "is_ready": slot.is_ready,
                "version_tag": slot.version_tag,
                "loaded_at": slot.loaded_at,
                "has_classifier": slot.classifier is not None,
                "has_lstm": slot.lstm is not None,
                "has_anomaly": slot.anomaly is not None,
                "has_pipeline": slot.pipeline is not None,
            }

        return {
            "active_color": self._active_color,
            "blue": _slot_info(self._blue, "blue"),
            "green": _slot_info(self._green, "green"),
        }

    def _update_version_gauge(self):
        active = self.active
        active_model_version.labels(
            slot_color=self._active_color,
            version_tag=active.version_tag,
        ).set(1)


model_manager = ModelManager()
