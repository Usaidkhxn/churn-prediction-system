from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.ml_system.predict import score_dataframe
from src.ml_system.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Batch score churn probabilities")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    cfg = load_config()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    print(f"Scoring {len(df)} rows...")
    scored = score_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"Saved scored file to {output_path.resolve()}")


if __name__ == "__main__":
    main()
