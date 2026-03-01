"""Download the Kaggle Credit Card Fraud dataset using kagglehub."""
import shutil
from pathlib import Path

import kagglehub


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / "creditcard.csv"
    if dest.exists():
        print(f"Dataset already exists at {dest}")
        return

    print("Downloading Credit Card Fraud dataset from Kaggle...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"Downloaded to: {path}")

    # kagglehub downloads to a cache dir; copy the CSV to our data/raw/
    downloaded = Path(path)
    csv_file = None
    if downloaded.is_dir():
        for f in downloaded.rglob("creditcard.csv"):
            csv_file = f
            break
    else:
        csv_file = downloaded

    if csv_file is None:
        print(f"ERROR: Could not find creditcard.csv in {downloaded}")
        return

    shutil.copy2(csv_file, dest)
    print(f"Copied to {dest}")
    print(f"File size: {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
