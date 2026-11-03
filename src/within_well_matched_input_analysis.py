"""One-to-one environmental matching within the same groundwater well."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "within_well_temporal_observations.csv"

OUTPUT_PAIRS = ROOT / "results" / "within_well_matched_input_pairs.csv"
OUTPUT_SUMMARY = ROOT / "results" / "within_well_matched_input_summary.csv"

FEATURES = [
    "rainfall_30d_mm_z",
    "rainfall_90d_mm_z",
    "previous_gwl_z",
    "month_sin",
    "month_cos",
]

# Maximum Euclidean distance in standardized feature space.
MAX_DISTANCE = 1.0


def main():

    print("Loading observations...")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "well_id",
        "year",
        "date",
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

    matches = []

    common_wells = sorted(
        set(df[df.year == 2018]["well_id"])
        & set(df[df.year == 2020]["well_id"])
    )

    print(f"Wells available in both years: {len(common_wells):,}")
    print("\nPerforming within-well one-to-one matching...")

    for well_number, well_id in enumerate(common_wells, start=1):

        data_2018 = df[
            (df["well_id"] == well_id) & (df["year"] == 2018)
        ].reset_index(drop=True)

        data_2020 = df[
            (df["well_id"] == well_id) & (df["year"] == 2020)
        ].reset_index(drop=True)

        if data_2018.empty or data_2020.empty:
            continue

        X_2018 = data_2018[FEATURES].to_numpy()
        X_2020 = data_2020[FEATURES].to_numpy()

        # Calculate all pairwise environmental distances.
        distances = np.linalg.norm(
            X_2018[:, None, :] - X_2020[None, :, :],
            axis=2
        )

        # Find a minimum-cost one-to-one assignment.
        row_indices, col_indices = linear_sum_assignment(distances)

        for i, j in zip(row_indices, col_indices):

            distance = distances[i, j]

            if distance > MAX_DISTANCE:
                continue

            a = data_2018.iloc[i]
            b = data_2020.iloc[j]

            matches.append({
                "well_id": well_id,
                "date_2018": a["date"],
                "date_2020": b["date"],
                "input_distance": distance,

                "actual_gwl_2018_m": a["actual_gwl_m"],
                "actual_gwl_2020_m": b["actual_gwl_m"],

                "prediction_2018_m": a["full_model_prediction_m"],
                "prediction_2020_m": b["full_model_prediction_m"],

                "absolute_error_2018_m": a[
                    "full_model_absolute_error_m"
                ],
                "absolute_error_2020_m": b[
                    "full_model_absolute_error_m"
                ],
            })

        if well_number % 250 == 0:
            print(f"Processed {well_number}/{len(common_wells)} wells")

    if not matches:
        raise ValueError(
            "No matches found. We may need to relax the distance threshold."
        )

    pairs = pd.DataFrame(matches)

    pairs["actual_gwl_difference_m"] = (
        pairs["actual_gwl_2020_m"] - pairs["actual_gwl_2018_m"]
    )

    pairs["prediction_difference_m"] = (
        pairs["prediction_2020_m"] - pairs["prediction_2018_m"]
    )

    pairs["absolute_error_difference_m"] = (
        pairs["absolute_error_2020_m"] - pairs["absolute_error_2018_m"]
    )

    pairs.to_csv(OUTPUT_PAIRS, index=False)

    # Aggregate pair-level differences within each well first.
    per_well = (
        pairs.groupby("well_id")
        .agg(
            n_pairs=("input_distance", "size"),
            mean_input_distance=("input_distance", "mean"),
            mean_actual_gwl_difference_m=(
                "actual_gwl_difference_m", "mean"
            ),
            mean_prediction_difference_m=(
                "prediction_difference_m", "mean"
            ),
            mean_absolute_error_difference_m=(
                "absolute_error_difference_m", "mean"
            ),
        )
        .reset_index()
    )

    summary = pd.DataFrame([{
        "n_matched_pairs": len(pairs),
        "n_matched_wells": len(per_well),
        "mean_input_distance": pairs["input_distance"].mean(),
        "median_input_distance": pairs["input_distance"].median(),
        "mean_absolute_error_2018_m": pairs[
            "absolute_error_2018_m"
        ].mean(),
        "mean_absolute_error_2020_m": pairs[
            "absolute_error_2020_m"
        ].mean(),
        "mean_absolute_error_difference_m": pairs[
            "absolute_error_difference_m"
        ].mean(),
        "median_absolute_error_difference_m": pairs[
            "absolute_error_difference_m"
        ].median(),
        "fraction_pairs_higher_error_2020": (
            pairs["absolute_error_difference_m"] > 0
        ).mean(),
        "fraction_wells_higher_mean_error_2020": (
            per_well["mean_absolute_error_difference_m"] > 0
        ).mean(),
    }])

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nWITHIN-WELL MATCHED INPUT COMPARISON")
    print("=" * 55)
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PAIRS)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
