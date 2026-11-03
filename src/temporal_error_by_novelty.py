from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "controlled_removal_predictions.csv"
OUTPUT_YEARLY = ROOT / "results" / "temporal_error_by_novelty.csv"
OUTPUT_SUMMARY = ROOT / "results" / "temporal_error_summary.csv"


def calculate_metrics(group):

    residuals = group["residual_m"].to_numpy()
    absolute_errors = np.abs(residuals)

    return {
        "n_observations": len(group),
        "n_wells": group["well_id"].nunique(),
        "MAE_m": mean_absolute_error(
            group["actual_gwl_m"],
            group["full_model_prediction_m"],
        ),
        "RMSE_m": np.sqrt(
            mean_squared_error(
                group["actual_gwl_m"],
                group["full_model_prediction_m"],
            )
        ),
        "mean_signed_error_m": residuals.mean(),
        "median_absolute_error_m": np.median(absolute_errors),
    }


def main():

    print("Loading controlled-removal predictions...")

    df = pd.read_csv(INPUT, parse_dates=["date"])

    required = {
        "date",
        "well_id",
        "actual_gwl_m",
        "full_model_prediction_m",
        "novelty_group",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=list(required)
    ).copy()

    # Positive residual = overprediction.
    # Negative residual = underprediction.
    df["residual_m"] = (
        df["full_model_prediction_m"] - df["actual_gwl_m"]
    )

    df["absolute_error_m"] = df["residual_m"].abs()
    df["year"] = df["date"].dt.year

    # -------------------------------------------------
    # 1. YEARLY PERFORMANCE BY NOVELTY GROUP
    # -------------------------------------------------

    print("\nCalculating yearly metrics...")

    yearly_rows = []

    groups = [
        "Familiar",
        "Poorly represented",
        "Extreme novelty",
    ]

    for year in sorted(df["year"].unique()):

        for novelty_group in groups:

            subset = df[
                (df["year"] == year)
                & (df["novelty_group"] == novelty_group)
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(subset)

            yearly_rows.append(
                {
                    "year": year,
                    "novelty_group": novelty_group,
                    **metrics,
                }
            )

    yearly = pd.DataFrame(yearly_rows)

    # -------------------------------------------------
    # 2. OVERALL YEARLY PERFORMANCE
    # -------------------------------------------------

    overall_rows = []

    for year, subset in df.groupby("year"):

        metrics = calculate_metrics(subset)

        overall_rows.append(
            {
                "year": year,
                "novelty_group": "All observations",
                **metrics,
            }
        )

    overall = pd.DataFrame(overall_rows)

    yearly_combined = pd.concat(
        [overall, yearly],
        ignore_index=True,
    )

    # -------------------------------------------------
    # 3. SUMMARY ACROSS THE COMPLETE TEST PERIOD
    # -------------------------------------------------

    summary_rows = []

    for novelty_group in [
        "All observations",
        "Familiar",
        "Poorly represented",
        "Extreme novelty",
    ]:

        if novelty_group == "All observations":
            subset = df
        else:
            subset = df[
                df["novelty_group"] == novelty_group
            ]

        if subset.empty:
            continue

        metrics = calculate_metrics(subset)

        summary_rows.append(
            {
                "novelty_group": novelty_group,
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)

    # -------------------------------------------------
    # 4. SAVE RESULTS
    # -------------------------------------------------

    yearly_combined.to_csv(
        OUTPUT_YEARLY,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print("\nYEARLY ERROR BY NOVELTY GROUP")
    print("=" * 85)

    print(
        yearly_combined.round(4).to_string(index=False)
    )

    print("\nOVERALL TEST-PERIOD SUMMARY")
    print("=" * 85)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_YEARLY)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
