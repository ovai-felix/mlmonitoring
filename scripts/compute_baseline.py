"""Fit the feature pipeline, transform data, compute and save baseline stats."""
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.services.feature_engineering import fit_and_save, transform, get_output_feature_names
from src.services.feature_store import save_training_features
from src.services.baseline_stats import compute_baseline, save_baseline


def main():
    csv_path = settings.raw_data_dir / "creditcard.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run download_dataset.py first.")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")

    # Fit and save the pipeline
    print("Fitting feature pipeline...")
    pipeline = fit_and_save(df)
    print(f"Pipeline saved to {settings.artifacts_dir}")

    # Transform the data
    print("Transforming features...")
    features_df = transform(df, pipeline=pipeline)
    print(f"Transformed shape: {features_df.shape}")
    print(f"Feature names: {list(features_df.columns)}")

    # Save training features
    print("Saving training features...")
    version = save_training_features(features_df)
    print(f"Saved as version: {version}")

    # Compute and save baseline statistics
    print("Computing baseline statistics...")
    stats = compute_baseline(features_df)
    baseline = save_baseline(stats, num_records=len(features_df), version=version)
    print(f"Baseline saved with {len(baseline.features)} features")

    # Print summary
    print("\n--- Baseline Summary ---")
    for feat in baseline.features[:5]:
        print(f"  {feat.feature_name}: mean={feat.mean:.4f}, std={feat.std:.4f}")
    print(f"  ... and {len(baseline.features) - 5} more features")


if __name__ == "__main__":
    main()
