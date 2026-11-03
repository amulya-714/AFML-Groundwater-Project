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

FULL_FEATURES = [
    "Latitude",
    "Longitude",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]

RAINFALL_FEATURES = [
    "rainfall_30d_mm",
    "rainfall_90d_mm",
]

K = 5
N_BOOTSTRAPS = 1000
RANDOM_SEED = 42


def calculate_novelty(X_train, X_test, k=K):
    """
    Calculate nearest-neighbour novelty.

    The scaler and nearest-neighbour reference are fitted
    using training observations only.

    Training reference distances exclude each row's
    self-match.
    """

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    if len(X_train) <= k:
        raise ValueError(
            f"Training set must contain more than {k} rows."
        )

    nn_train = NearestNeighbors(
        n_neighbors=k + 1,
        n_jobs=-1,
    )

    nn_train.fit(train_scaled)

    train_distances, _ = nn_train.kneighbors(train_scaled)

    # Exclude the self-match.
    train_local_distance = train_distances[:, 1:].mean(axis=1)

    nn_test = NearestNeighbors(
        n_neighbors=k,
        n_jobs=-1,
    )

    nn_test.fit(train_scaled)

    test_distances, _ = nn_test.kneighbors(test_scaled)

    test_novelty = test_distances.mean(axis=1)

    # Thresholds are derived exclusively from training data.
    threshold_95 = np.quantile(train_local_distance, 0.95)
    threshold_99 = np.quantile(train_local_distance, 0.99)

    return test_novelty, threshold_95, threshold_99


def assign_novelty_group(novelty, threshold_95, threshold_99):
    """Assign test observations using training-derived thresholds."""

    return np.select(
        [
            novelty <= threshold_95,
            novelty <= threshold_99,
        ],
        [
            "Familiar",
            "Poorly represented",
        ],
        default="Extreme novelty",
    )


def cluster_bootstrap_mae(
    group_df,
    n_bootstraps=N_BOOTSTRAPS,
    seed=RANDOM_SEED,
):
    """
    Estimate a 95% confidence interval for MAE.

    Resample wells rather than individual observations.
    All observations belonging to a sampled well remain together.
    """

    wells = group_df["well_id"].unique()

    if len(wells) < 2:
        return np.nan, np.nan

    errors_by_well = {
        well: group_df.loc[
            group_df["well_id"] == well,
            "absolute_error_m",
        ].to_numpy()
        for well in wells
    }

    rng = np.random.default_rng(seed)

    bootstrap_maes = []

    for _ in range(n_bootstraps):

        sampled_wells = rng.choice(
            wells,
            size=len(wells),
            replace=True,
        )

        sampled_errors = np.concatenate(
            [errors_by_well[well] for well in sampled_wells]
        )

        bootstrap_maes.append(sampled_errors.mean())

    lower, upper = np.percentile(
        bootstrap_maes,
        [2.5, 97.5],
    )

    return lower, upper


def summarize_groups(
    observations,
    novelty_column,
    group_column,
    novelty_type,
):
    rows = []

    group_order = [
        "Familiar",
        "Poorly represented",
        "Extreme novelty",
    ]

    for group_name in group_order:

        group = observations[
            observations[group_column] == group_name
        ].copy()

        if group.empty:
            continue

        actual = group["actual_gwl_m"].to_numpy()
        predictions = group["random_forest_prediction_m"].to_numpy()

        mae = mean_absolute_error(actual, predictions)

        rmse = np.sqrt(
            mean_squared_error(actual, predictions)
        )

        ci_lower, ci_upper = cluster_bootstrap_mae(
            group,
            seed=RANDOM_SEED,
        )

        rows.append(
            {
                "novelty_type": novelty_type,
                "novelty_group": group_name,
                "n_observations": len(group),
                "n_wells": group["well_id"].nunique(),
                "mean_novelty": group[novelty_column].mean(),
                "MAE_m": mae,
                "MAE_95CI_lower_m": ci_lower,
                "MAE_95CI_upper_m": ci_upper,
                "RMSE_m": rmse,
            }
        )

    return rows


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA)

    df[DATE] = pd.to_datetime(
        df[DATE],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    # Reconstruct previous groundwater level by well.
    well_columns = [
        "Station",
        "Latitude",
        "Longitude",
    ]

    df = df.sort_values(
        well_columns + [DATE]
    ).copy()

    df["well_id"] = (
        df["Station"].astype(str)
        + "_"
        + df["Latitude"].astype(str)
        + "_"
        + df["Longitude"].astype(str)
    )

    df["previous_gwl"] = df.groupby(
        well_columns,
        dropna=False,
    )[TARGET].shift(1)

    month = df[DATE].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    all_features = list(
        dict.fromkeys(FULL_FEATURES + [TARGET, DATE, "well_id"])
    )

    df = df.dropna(
        subset=all_features
    ).copy()

    # Preserve the original chronological split.
    train = df[
        df[DATE] < "2018-01-01"
    ].copy()

    test = df[
        df[DATE] >= "2018-01-01"
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or test set is empty."
        )

    X_train = train[FULL_FEATURES].astype(float)
    X_test = test[FULL_FEATURES].astype(float)

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Store the same model's predictions for both novelty analyses.
    observations = pd.DataFrame(
        {
            "date": test[DATE].to_numpy(),
            "well_id": test["well_id"].to_numpy(),
            "actual_gwl_m": y_test.to_numpy(),
            "random_forest_prediction_m": predictions,
        }
    )

    observations["absolute_error_m"] = np.abs(
        observations["actual_gwl_m"]
        - observations["random_forest_prediction_m"]
    )

    # ---------------------------------------------------------
    # 1. FULL-FEATURE NOVELTY
    # ---------------------------------------------------------

    print("\nCalculating full-feature novelty...")

    full_novelty, full_95, full_99 = calculate_novelty(
        X_train,
        X_test,
    )

    observations["full_feature_novelty"] = full_novelty

    observations["full_feature_group"] = assign_novelty_group(
        full_novelty,
        full_95,
        full_99,
    )

    # ---------------------------------------------------------
    # 2. RAINFALL-ONLY NOVELTY
    # ---------------------------------------------------------

    print("Calculating rainfall-only novelty...")

    rainfall_novelty, rainfall_95, rainfall_99 = calculate_novelty(
        train[RAINFALL_FEATURES].astype(float),
        test[RAINFALL_FEATURES].astype(float),
    )

    observations["rainfall_novelty"] = rainfall_novelty

    observations["rainfall_group"] = assign_novelty_group(
        rainfall_novelty,
        rainfall_95,
        rainfall_99,
    )

    # ---------------------------------------------------------
    # 3. GROUP METRICS AND BOOTSTRAP CONFIDENCE INTERVALS
    # ---------------------------------------------------------

    print("\nCalculating group metrics and bootstrap intervals...")

    rows = []

    rows.extend(
        summarize_groups(
            observations,
            "full_feature_novelty",
            "full_feature_group",
            "Full-feature",
        )
    )

    rows.extend(
        summarize_groups(
            observations,
            "rainfall_novelty",
            "rainfall_group",
            "Rainfall-only",
        )
    )

    summary = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 4. SAVE OUTPUTS
    # ---------------------------------------------------------

    observations.to_csv(
        RESULTS / "input_extrapolation_observations.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "input_extrapolation_summary.csv",
        index=False,
    )

    print("\nTRAINING AND TESTING INFORMATION")
    print("=" * 55)

    print(
        f"Training rows: {len(train):,}"
        f" | Test rows: {len(test):,}"
    )

    print(
        f"Training dates: {train[DATE].min().date()}"
        f" to {train[DATE].max().date()}"
    )

    print(
        f"Test dates: {test[DATE].min().date()}"
        f" to {test[DATE].max().date()}"
    )

    print("\nNOVELTY THRESHOLDS")
    print("=" * 55)

    print(
        f"Full-feature 95th percentile: {full_95:.6f}"
    )

    print(
        f"Full-feature 99th percentile: {full_99:.6f}"
    )

    print(
        f"Rainfall-only 95th percentile: {rainfall_95:.6f}"
    )

    print(
        f"Rainfall-only 99th percentile: {rainfall_99:.6f}"
    )

    print("\nNOVELTY GROUP EVALUATION")
    print("=" * 85)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved:")
    print("results/input_extrapolation_observations.csv")
    print("results/input_extrapolation_summary.csv")


if __name__ == "__main__":
    main()