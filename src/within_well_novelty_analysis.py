from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "observation_date"

WELL_COLUMNS = ["Station", "Latitude", "Longitude"]

MODEL_FEATURES = [
    "Latitude",
    "Longitude",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]

ENVIRONMENTAL_FEATURES = [
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]


def calculate_novelty(train_features, test_features, k=5):
    """Measure environmental novelty using training-scaled nearest neighbours."""
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    neighbours = NearestNeighbors(
        n_neighbors=min(k, len(train_features)),
        n_jobs=-1,
    )

    neighbours.fit(train_scaled)
    distances, _ = neighbours.kneighbors(test_scaled)

    return distances.mean(axis=1)


def calculate_metrics(group):
    """Calculate prediction error metrics for one group."""
    actual = group["actual_gwl_m"].to_numpy()
    predicted = group["prediction_m"].to_numpy()

    return {
        "n_observations": len(group),
        "n_wells": group["well_id"].nunique(),
        "MAE_m": mean_absolute_error(actual, predicted),
        "RMSE_m": np.sqrt(mean_squared_error(actual, predicted)),
        "median_absolute_error_m": group["absolute_error_m"].median(),
    }


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare the data
    # ---------------------------------------------------------
    df = pd.read_csv(DATA)

    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Use the same well definition as the previous experiments.
    df = df.sort_values(WELL_COLUMNS + [DATE]).copy()

    df["well_id"] = (
        df["Station"].astype(str)
        + "_"
        + df["Latitude"].astype(str)
        + "_"
        + df["Longitude"].astype(str)
    )

    # Use only the preceding observation for the same well.
    df["previous_gwl"] = df.groupby(
        WELL_COLUMNS,
        dropna=False,
    )[TARGET].shift(1)

    month = df[DATE].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df = df.dropna(
        subset=MODEL_FEATURES + [TARGET, DATE, "well_id"]
    ).copy()

    train = df[df[DATE] < "2018-01-01"].copy()
    test = df[df[DATE] >= "2018-01-01"].copy()

    # ---------------------------------------------------------
    # 2. Fit the original baseline model
    # ---------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[MODEL_FEATURES], train[TARGET])

    test["prediction_m"] = model.predict(test[MODEL_FEATURES])
    test["actual_gwl_m"] = test[TARGET]
    test["absolute_error_m"] = np.abs(
        test["actual_gwl_m"] - test["prediction_m"]
    )

    # ---------------------------------------------------------
    # 3. Keep wells represented in both periods
    # ---------------------------------------------------------
    training_wells = set(train["well_id"].unique())
    test_wells = set(test["well_id"].unique())

    shared_wells = training_wells.intersection(test_wells)

    shared_test = test[test["well_id"].isin(shared_wells)].copy()

    if shared_test.empty:
        raise ValueError(
            "No wells have observations in both training and test periods."
        )

    # ---------------------------------------------------------
    # 4. Measure environmental novelty
    # ---------------------------------------------------------
    shared_test["environmental_novelty"] = calculate_novelty(
        train[ENVIRONMENTAL_FEATURES].astype(float),
        shared_test[ENVIRONMENTAL_FEATURES].astype(float),
    )

    # ---------------------------------------------------------
    # 5. Create novelty groups within each well
    # ---------------------------------------------------------
    # Each well needs at least four test observations to form quartiles.
    counts = shared_test.groupby("well_id")["well_id"].transform("size")
    eligible = shared_test[counts >= 4].copy()

    if eligible.empty:
        raise ValueError(
            "No shared wells have at least four test observations. "
            "We need to use a different within-well grouping strategy."
        )

    eligible["within_well_novelty_quartile"] = (
        eligible.groupby("well_id")["environmental_novelty"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                q=4,
                labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
            )
        )
    )

    # ---------------------------------------------------------
    # 6. Summarise the within-well comparison
    # ---------------------------------------------------------
    summary_rows = []

    for quartile, group in eligible.groupby(
        "within_well_novelty_quartile",
        observed=True,
    ):
        row = calculate_metrics(group)
        row["novelty_group"] = str(quartile)
        row["mean_environmental_novelty"] = (
            group["environmental_novelty"].mean()
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Overall error for the observations included in this analysis.
    overall = calculate_metrics(eligible)
    overall["novelty_group"] = "All eligible observations"
    overall["mean_environmental_novelty"] = (
        eligible["environmental_novelty"].mean()
    )

    summary = pd.concat(
        [summary, pd.DataFrame([overall])],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 7. Also calculate metrics per well and novelty group
    # ---------------------------------------------------------
    per_well_rows = []

    for (well_id, quartile), group in eligible.groupby(
        ["well_id", "within_well_novelty_quartile"],
        observed=True,
    ):
        row = calculate_metrics(group)
        row["well_id"] = well_id
        row["novelty_group"] = str(quartile)
        row["mean_environmental_novelty"] = (
            group["environmental_novelty"].mean()
        )
        per_well_rows.append(row)

    per_well_summary = pd.DataFrame(per_well_rows)

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------
    eligible.to_csv(
        RESULTS / "within_well_novelty_observations.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "within_well_novelty_summary.csv",
        index=False,
    )

    per_well_summary.to_csv(
        RESULTS / "within_well_novelty_per_well.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Print results
    # ---------------------------------------------------------
    print(f"Training observations: {len(train):,}")
    print(f"Test observations: {len(test):,}")
    print(f"Wells in training period: {len(training_wells):,}")
    print(f"Wells in test period: {len(test_wells):,}")
    print(f"Wells present in both periods: {len(shared_wells):,}")
    print(f"Shared-well test observations: {len(shared_test):,}")
    print(f"Eligible observations used in within-well analysis: "
          f"{len(eligible):,}")
    print(f"Eligible wells: {eligible['well_id'].nunique():,}")

    print("\nWITHIN-WELL NOVELTY RESULTS")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/within_well_novelty_observations.csv")
    print("results/within_well_novelty_summary.csv")
    print("results/within_well_novelty_per_well.csv")


if __name__ == "__main__":
    main()
