"""CLI for triggering model training.

Usage:
    python train.py --model classification --data-version v_2026-02-28
    python train.py --model timeseries --data-version v_2026-02-28 --window-size 32
    python train.py --model anomaly --data-version v_2026-02-28
    python train.py --model all --data-version v_2026-02-28
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import init_db
from src.services.training_service import train_classifier, train_lstm, train_anomaly


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model", required=True,
        choices=["classification", "timeseries", "anomaly", "all"],
    )
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    init_db()

    results = {}

    if args.model in ("classification", "all"):
        print("=" * 60)
        print("Training Classification Model (TabularTransformer)")
        print("=" * 60)
        results["classification"] = train_classifier(args.data_version, args.n_trials)
        _print_result("classification", results["classification"])

    if args.model in ("timeseries", "all"):
        print("=" * 60)
        print("Training Time-Series Model (FraudLSTM)")
        print("=" * 60)
        results["timeseries"] = train_lstm(
            args.data_version, args.window_size, args.n_trials,
        )
        _print_result("timeseries", results["timeseries"])

    if args.model in ("anomaly", "all"):
        print("=" * 60)
        print("Training Anomaly Detection Model (IsolationForest)")
        print("=" * 60)
        results["anomaly"] = train_anomaly(args.data_version)
        _print_result("anomaly", results["anomaly"])

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model_type, result in results.items():
        status = result.get("status", "unknown")
        metrics = result.get("metrics", {})
        f1 = metrics.get("f1", "N/A")
        gate = result.get("gate_result", {}).get("decision", "N/A")
        print(f"  {model_type}: status={status}, F1={f1}, gate={gate}")


def _print_result(model_type: str, result: dict):
    print(f"\nStatus: {result.get('status')}")
    if "metrics" in result:
        print("Metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")
    if "best_params" in result:
        print("Best params:")
        for k, v in result["best_params"].items():
            print(f"  {k}: {v}")
    if "gate_result" in result:
        gate = result["gate_result"]
        print(f"Gate: {gate.get('decision')} - {gate.get('reason')}")
    print()


if __name__ == "__main__":
    main()
