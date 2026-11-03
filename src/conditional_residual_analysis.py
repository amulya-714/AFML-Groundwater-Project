from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PREDICTION_FILE = ROOT / "results/controlled_removal_predictions.csv"

OUTPUT_FILE = ROOT / "results/conditional_residual_analysis.csv"
DETAIL_FILE = ROOT / "results/conditional_residual_observations.csv"

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "observation_date"

FEATURES = [
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
]

N_BINS = 5


def make_well_id(df):
    return (
        df["Station"].astype(str) + "_" +
        df["Latitude"].astype(str) + "_" +
        df["Longitude"].astype(str)
    )


def main():

    print("Loading original dataset and predictions...")

    data = pd.read_csv(DATA_FILE, parse_dates=[DATE])
    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["date"],
    )

    # Reconstruct well IDs exactly as in baseline_model.py.
    data["well_id"] = make_well_id(data)

    # Reconstruct previous groundwater level exactly as in baseline.
    data = data.sort_values(
        ["well_id", DATE]
    ).copy()

    data["previous_gwl"] = (
        data.groupby("well_id")[TARGET].shift(1)
    )

    # Construct the original chronological training period.
    training = data[
        data[DATE] < "2018-01-01"
    ].copy()

    # Retain the original test predictions and familiar observations.
    familiar = predictions[
        predictions["novelty_group"] == "Familiar"
    ].copy()

    # Join the test predictions to their original input features.
    inputs = data[
        ["well_id", DATE] + FEATURES
    ].copy()

    if inputs.duplicated(["well_id", DATE]).any():
        raise ValueError("Duplicate well/date keys in original dataset.")

    if familiar.duplicated(["well_id", "date"]).any():
        raise ValueError("Duplicate well/date keys in predictions.")

    merged = familiar.merge(
        inputs,
        left_on=["well_id", "date"],
        right_on=["well_id", DATE],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (merged["_merge"] != "both").sum()

    print(f"Familiar predictions: {len(familiar):,}")
    print(f"Unmatched observations: {unmatched:,}")

    if unmatched:
        raise ValueError("Some familiar predictions could not be matched.")

    merged = merged.drop(columns=["_merge", DATE])

    merged["year"] = merged["date"].dt.year

    # Positive residual = overprediction.
    # Negative residual = underprediction.
    merged["residual_m"] = (
        merged["full_model_prediction_m"] -
        merged["actual_gwl_m"]
    )

    merged["absolute_error_m"] = merged["residual_m"].abs()

    # Define bins using TRAINING DATA ONLY.
    for feature in FEATURES:

        train_values = training[feature].dropna()

        _, edges = pd.qcut(
            train_values,
            q=N_BINS,
            retbins=True,
            duplicates="drop",
        )

        # Extend outer edges to include all test values.
        edges[0] = -np.inf
        edges[-1] = np.inf

        merged[f"{feature}_bin"] = pd.cut(
            merged[feature],
            bins=edges,
            include_lowest=True,
        )

        print(
            f"{feature}: {len(edges) - 1} training-derived bins"
        )

    # Calculate residual behaviour by year and feature bin.
    rows = []

    for feature in FEATURES:

        bin_column = f"{feature}_bin"

        for bin_value, bin_data in merged.groupby(
            bin_column,
            observed=True,
        ):

            for year, group in bin_data.groupby("year"):

                rows.append({
                    "feature": feature,
                    "input_bin": str(bin_value),
                    "year": year,
                    "n_observations": len(group),
                    "n_wells": group["well_id"].nunique(),
                    "mean_input": group[feature].mean(),
                    "mean_actual_gwl_m": group["actual_gwl_m"].mean(),
                    "mean_prediction_m": group[
                        "full_model_prediction_m"
                    ].mean(),
                    "MAE_m": group["absolute_error_m"].mean(),
                    "RMSE_m": np.sqrt(
                        np.mean(group["residual_m"] ** 2)
                    ),
                    "mean_signed_error_m": group["residual_m"].mean(),
                    "median_signed_error_m": group["residual_m"].median(),
                })

    summary = pd.DataFrame(rows)

    summary.to_csv(OUTPUT_FILE, index=False)
    merged.to_csv(DETAIL_FILE, index=False)

    print("\nCONDITIONAL RESIDUAL ANALYSIS")
    print("=" * 75)
    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(DETAIL_FILE)


if __name__ == "__main__":
    main()
