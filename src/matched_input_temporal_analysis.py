"""Matched-input comparison of groundwater observations across years."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "within_well_temporal_observations.csv"

OUTPUT_MATCHES = ROOT / "results" / "matched_input_temporal_pairs.csv"
OUTPUT_SUMMARY = ROOT / "results" / "matched_input_temporal_summary.csv"

FEATURES = [
    "rainfall_30d_mm_z",
    "rainfall_90d_mm_z",
    "previous_gwl_z",
    "month_sin",
    "month_cos",
]

def main():

    print("Loading observations...")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "well_id",
        "year",
        "actual_gwl_m",
        "full_model_prediction_m",
        "full_model_absolute_error_m",
    ] + FEATURES

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=required).copy()

    df["year"] = df["year"].astype(int)

    print(f"Usable observations: {len(df):,}")

    print("\nObservations by year:")
    print(df["year"].value_counts().sort_index())

    year_a = df[df["year"] == 2018].copy()
    year_b = df[df["year"] == 2020].copy()

    print("\nMatching 2018 observations with 2020 observations...")

    X_a = year_a[FEATURES].to_numpy()
    X_b = year_b[FEATURES].to_numpy()

    # Find the closest 2020 observation for every 2018 observation.
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_b)

    distances, indices = nn.kneighbors(X_a)

    matched_b = year_b.iloc[indices.flatten()].reset_index(drop=True)
    matched_a = year_a.reset_index(drop=True)

    pairs = pd.DataFrame({
        "well_id_2018": matched_a["well_id"].values,
        "well_id_2020": matched_b["well_id"].values,
        "date_2018": matched_a["date"].values,
        "date_2020": matched_b["date"].values,
        "input_distance": distances.flatten(),

        "actual_gwl_2018_m": matched_a["actual_gwl_m"].values,
        "actual_gwl_2020_m": matched_b["actual_gwl_m"].values,

        "prediction_2018_m": matched_a["full_model_prediction_m"].values,
        "prediction_2020_m": matched_b["full_model_prediction_m"].values,

        "absolute_error_2018_m": matched_a["full_model_absolute_error_m"].values,
        "absolute_error_2020_m": matched_b["full_model_absolute_error_m"].values,
    })

    pairs["actual_gwl_difference_m"] = (
        pairs["actual_gwl_2020_m"] - pairs["actual_gwl_2018_m"]
    )

    pairs["prediction_difference_m"] = (
        pairs["prediction_2020_m"] - pairs["prediction_2018_m"]
    )

    pairs["absolute_error_difference_m"] = (
        pairs["absolute_error_2020_m"] - pairs["absolute_error_2018_m"]
    )

    pairs["same_well"] = (
        pairs["well_id_2018"] == pairs["well_id_2020"]
    )

    # Save all nearest-neighbour matches.
    pairs.to_csv(OUTPUT_MATCHES, index=False)

    print("\nMATCHING SUMMARY")
    print("=" * 55)
    print(f"Total matched pairs: {len(pairs):,}")
    print(f"Same-well pairs: {pairs['same_well'].sum():,}")
    print(f"Different-well pairs: {(~pairs['same_well']).sum():,}")

    # Restrict to same-well matches for a more controlled comparison.
    same_well = pairs[pairs["same_well"]].copy()

    print("\nSAME-WELL MATCHED COMPARISON")
    print("=" * 55)

    if len(same_well) == 0:
        raise ValueError("No same-well matched pairs found.")

    summary = pd.DataFrame([{
        "n_matched_pairs": len(same_well),
        "mean_input_distance": same_well["input_distance"].mean(),
        "median_input_distance": same_well["input_distance"].median(),
        "mean_actual_gwl_difference_m": same_well[
            "actual_gwl_difference_m"
        ].mean(),
        "mean_prediction_difference_m": same_well[
            "prediction_difference_m"
        ].mean(),
        "mean_absolute_error_2018_m": same_well[
            "absolute_error_2018_m"
        ].mean(),
        "mean_absolute_error_2020_m": same_well[
            "absolute_error_2020_m"
        ].mean(),
        "mean_absolute_error_difference_m": same_well[
            "absolute_error_difference_m"
        ].mean(),
        "median_absolute_error_difference_m": same_well[
            "absolute_error_difference_m"
        ].median(),
        "fraction_higher_error_2020": (
            same_well["absolute_error_difference_m"] > 0
        ).mean(),
    }])

    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nSaved:")
    print(OUTPUT_MATCHES)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
