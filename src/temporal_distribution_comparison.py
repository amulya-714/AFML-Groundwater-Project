from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PREDICTION_FILE = ROOT / "results/controlled_removal_predictions.csv"

OUTPUT_FILE = ROOT / "results/temporal_distribution_comparison.csv"
MATCHED_FILE = ROOT / "results/temporal_distribution_matched_observations.csv"

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "observation_date"

FEATURES = [
    TARGET,
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
]

def main():

    print("Loading processed dataset and predictions...")

    data = pd.read_csv(DATA_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    # Reconstruct the exact well identifier used by the model.
    data["well_id"] = (
        data["Station"].astype(str) + "_" +
        data["Latitude"].astype(str) + "_" +
        data["Longitude"].astype(str)
    )

    data[DATE] = pd.to_datetime(data[DATE], errors="coerce")
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    # Reconstruct previous groundwater level using the same
    # well-specific chronological procedure as model training.
    data = data.sort_values(
        ["well_id", DATE]
    ).copy()

    data["previous_gwl"] = (
        data.groupby("well_id")[TARGET].shift(1)
    )

    # Restrict original data to the test period.
    data = data[data[DATE] >= "2018-01-01"].copy()

    # Keep only fields required for this comparison.
    data = data[
        ["well_id", DATE] + FEATURES
    ].copy()

    # Check uniqueness before merging.
    data_duplicates = data.duplicated(
        subset=["well_id", DATE]
    ).sum()

    prediction_duplicates = predictions.duplicated(
        subset=["well_id", "date"]
    ).sum()

    print("\nJOIN VALIDATION")
    print("=" * 50)
    print(f"Duplicate keys in processed test data: {data_duplicates}")
    print(f"Duplicate keys in prediction data: {prediction_duplicates}")

    if data_duplicates > 0 or prediction_duplicates > 0:
        raise ValueError(
            "Duplicate well/date keys found. Resolve them before merging."
        )

    # Merge while retaining all prediction rows.
    merged = predictions.merge(
        data,
        left_on=["well_id", "date"],
        right_on=["well_id", DATE],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"].copy()
    unmatched = merged[merged["_merge"] == "left_only"].copy()

    print(f"Prediction observations: {len(predictions):,}")
    print(f"Successfully matched: {len(matched):,}")
    print(f"Unmatched predictions: {len(unmatched):,}")

    if len(unmatched) > 0:
        print("\nExamples of unmatched keys:")
        print(
            unmatched[["date", "well_id"]]
            .head(10)
            .to_string(index=False)
        )

    if matched.empty:
        raise ValueError("No prediction observations matched.")

    # Keep the original novelty classifications.
    # Do not calculate new thresholds from test observations.
    matched = matched[
        matched["novelty_group"] == "Familiar"
    ].copy()

    matched["year"] = matched["date"].dt.year

    matched.to_csv(MATCHED_FILE, index=False)

    # Compare distributions between 2018 and 2020.
    rows = []

    for feature in FEATURES:

        for year in [2018, 2019, 2020]:

            group = matched[matched["year"] == year][feature].dropna()

            if group.empty:
                continue

            rows.append({
                "feature": feature,
                "year": year,
                "n_observations": len(group),
                "mean": group.mean(),
                "median": group.median(),
                "std": group.std(),
                "p05": group.quantile(0.05),
                "p95": group.quantile(0.95),
                "minimum": group.min(),
                "maximum": group.max(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nFAMILIAR-INPUT DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(MATCHED_FILE)


if __name__ == "__main__":
    main()
