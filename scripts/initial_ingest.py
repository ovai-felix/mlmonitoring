"""Load the credit card fraud CSV through the API in batches."""
import sys
from pathlib import Path

import httpx
import pandas as pd

BATCH_SIZE = 5000
API_URL = "http://localhost:8000"


def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "raw" / "creditcard.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run download_dataset.py first.")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records with columns: {list(df.columns)}")

    total_accepted = 0
    total_rejected = 0
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    with httpx.Client(timeout=120.0) as client:
        for i in range(0, len(df), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch_df = df.iloc[i : i + BATCH_SIZE]
            records = batch_df.to_dict(orient="records")

            print(f"Batch {batch_num}/{num_batches}: sending {len(records)} records...", end=" ")
            response = client.post(
                f"{API_URL}/data/ingest",
                json={"records": records, "source": "initial_load"},
            )
            response.raise_for_status()
            result = response.json()
            total_accepted += result["accepted"]
            total_rejected += result["rejected"]
            print(f"accepted={result['accepted']}, rejected={result['rejected']}")

    print(f"\nDone! Total accepted: {total_accepted}, rejected: {total_rejected}")


if __name__ == "__main__":
    main()
