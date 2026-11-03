from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "observation_date"

FEATURES = [
    "Latitude",
    "Longitude",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]

K = 5
EXTREME_FRACTION = 0.01
RANDOM_SEED = 42

MODEL_SETTINGS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}


def calculate_training_novelty(X_train, k=K):
    """
    Calculate each training observation's local novelty.

    Scaling is fitted on the complete training set.
    Self-neighbours are excluded from training distances.
    """

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)

    if len(X_train) <= k:
        raise ValueError("Not enough training observations for kNN.")

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        n_jobs=-1,
    )
    nn.fit(train_scaled)

    distances, _ = nn.kneighbors(train_scaled)

    local_novelty = distances[:, 1:].mean(axis=1)

    return scaler, train_scaled, local_novelty


def calculate_test_novelty(
    scaler,
    train_scaled,
    X_test,
    k=K,
):
    """Calculate test novelty relative to a supplied training set."""

    test_scaled = scaler.transform(X_test)

    nn = NearestNeighbors(
        n_neighbors=k,
        n_jobs=-1,
    )
    nn.fit(train_scaled)

    distances, _ = nn.kneighbors(test_scaled)

    return distances.mean(axis=1)


def select_training_removals(
    train,
    local_novelty,
):
    """
    Select the top 1% most novel training observations.

    Construct a matched-control removal set by sampling
    the same number of non-extreme observations in each year.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    train = train.copy()

    train["training_novelty"] = local_novelty
    train["observation_year"] = train[DATE].dt.year

    n_remove = max(
        1,
        int(np.ceil(len(train) * EXTREME_FRACTION)),
    )

    extreme_indices = (
        train["training_novelty"]
        .nlargest(n_remove)
        .index
    )

    extreme_rows = train.loc[extreme_indices]

    # Match control removals to the extreme removals by year.
    control_indices = []

    for year, extreme_group in extreme_rows.groupby(
        "observation_year"
    ):

        n_controls = len(extreme_group)

        eligible_controls = train.loc[
            (train["observation_year"] == year)
            & (~train.index.isin(extreme_indices))
        ]

        if len(eligible_controls) < n_controls:
            raise ValueError(
                f"Not enough control observations in year {year}."
            )

        sampled = rng.choice(
            eligible_controls.index.to_numpy(),
            size=n_controls,
            replace=False,
        )

        control_indices.extend(sampled)

    control_indices = pd.Index(control_indices)

    if len(control_indices) != len(extreme_indices):
        raise ValueError(
            "Extreme and control removal counts do not match."
        )

    return extreme_indices, control_indices, train


def evaluate_predictions(y_true, predictions):
    return {
        "n_observations": len(y_true),
        "MAE_m": mean_absolute_error(y_true, predictions),
        "RMSE_m": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "R2": (
            r2_score(y_true, predictions)
            if len(y_true) > 1
            else np.nan
        ),
    }


def fit_and_predict(name, train_subset, X_test, y_train, y_test):
    print(f"\nTraining {name}...")

    model = RandomForestRegressor(**MODEL_SETTINGS)

    model.fit(
        train_subset[FEATURES].astype(float),
        train_subset[TARGET].astype(float),
    )

    predictions = model.predict(X_test)

    metrics = evaluate_predictions(
        y_test.to_numpy(),
        predictions,
    )

    metrics["model"] = name
    metrics["training_rows"] = len(train_subset)

    return predictions, metrics


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

    # Reconstruct previous groundwater level by well.
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

    df = df.dropna(
        subset=FEATURES + [TARGET, DATE, "well_id"]
    ).copy()

    # Preserve the original chronological split.
    train = df[
        df[DATE] < "2018-01-01"
    ].copy()

    test = df[
        df[DATE] >= "2018-01-01"
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test set is empty.")

    X_train = train[FEATURES].astype(float)
    X_test = test[FEATURES].astype(float)

    y_test = test[TARGET].astype(float)

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    # -----------------------------------------------------
    # 1. IDENTIFY EXTREME TRAINING OBSERVATIONS
    # -----------------------------------------------------

    print("\nCalculating training novelty...")

    scaler, full_train_scaled, local_novelty = (
        calculate_training_novelty(X_train)
    )

    threshold_99 = np.quantile(
        local_novelty,
        0.99,
    )

    extreme_indices, control_indices, train_with_novelty = (
        select_training_removals(
            train,
            local_novelty,
        )
    )

    extreme_removed = train.drop(
        index=extreme_indices
    ).copy()

    control_removed = train.drop(
        index=control_indices
    ).copy()

    print("\nTRAINING REMOVAL INFORMATION")
    print("=" * 60)

    print(f"Training novelty 99th percentile: {threshold_99:.6f}")

    print(f"Extreme observations removed: {len(extreme_indices):,}")
    print(f"Matched controls removed: {len(control_indices):,}")

    print(f"Full model training rows: {len(train):,}")
    print(f"Extreme-removed training rows: {len(extreme_removed):,}")
    print(f"Matched-control training rows: {len(control_removed):,}")

    print("\nExtreme removals by year:")

    print(
        train_with_novelty.loc[extreme_indices]
        .groupby("observation_year")
        .size()
        .to_string()
    )

    print("\nMatched-control removals by year:")

    print(
        train_with_novelty.loc[control_indices]
        .groupby("observation_year")
        .size()
        .to_string()
    )

    # -----------------------------------------------------
    # 2. DEFINE TEST GROUPS USING FULL TRAINING SUPPORT
    # -----------------------------------------------------

    print("\nCalculating test novelty...")

    full_test_novelty = calculate_test_novelty(
        scaler,
        full_train_scaled,
        X_test,
    )

    # The thresholds are derived from full-training data.
    train_threshold_95 = np.quantile(
        np.sort(
            NearestNeighbors(
                n_neighbors=K + 1,
                n_jobs=-1,
            ).fit(full_train_scaled).kneighbors(
                full_train_scaled
            )[0][:, 1:].mean(axis=1)
        ),
        0.95,
    )

    train_threshold_99 = threshold_99

    test_groups = np.select(
        [
            full_test_novelty <= train_threshold_95,
            full_test_novelty <= train_threshold_99,
        ],
        [
            "Familiar",
            "Poorly represented",
        ],
        default="Extreme novelty",
    )

    # -----------------------------------------------------
    # 3. MEASURE REMAINING TRAINING SUPPORT
    # -----------------------------------------------------

    print("\nMeasuring support after removals...")

    extreme_scaled = scaler.transform(
        extreme_removed[FEATURES].astype(float)
    )

    control_scaled = scaler.transform(
        control_removed[FEATURES].astype(float)
    )

    extreme_test_novelty = calculate_test_novelty(
        scaler,
        extreme_scaled,
        X_test,
    )

    control_test_novelty = calculate_test_novelty(
        scaler,
        control_scaled,
        X_test,
    )

    # -----------------------------------------------------
    # 4. TRAIN THREE CONTROLLED MODELS
    # -----------------------------------------------------

    full_predictions, full_metrics = fit_and_predict(
        "Full model",
        train,
        X_test,
        train[TARGET],
        y_test,
    )

    extreme_predictions, extreme_metrics = fit_and_predict(
        "Extreme-removed model",
        extreme_removed,
        X_test,
        extreme_removed[TARGET],
        y_test,
    )

    control_predictions, control_metrics = fit_and_predict(
        "Matched-control model",
        control_removed,
        X_test,
        control_removed[TARGET],
        y_test,
    )

    # -----------------------------------------------------
    # 5. SAVE TEST PREDICTIONS AND SUPPORT DISTANCES
    # -----------------------------------------------------

    observation_results = pd.DataFrame(
        {
            "date": test[DATE].to_numpy(),
            "well_id": test["well_id"].to_numpy(),
            "actual_gwl_m": y_test.to_numpy(),
            "full_feature_novelty": full_test_novelty,
            "novelty_group": test_groups,
            "full_model_prediction_m": full_predictions,
            "extreme_removed_prediction_m": extreme_predictions,
            "matched_control_prediction_m": control_predictions,
            "full_training_support_distance": full_test_novelty,
            "extreme_removed_support_distance": extreme_test_novelty,
            "matched_control_support_distance": control_test_novelty,
        }
    )

    for model_name, prediction_column in [
        ("Full model", "full_model_prediction_m"),
        ("Extreme-removed model", "extreme_removed_prediction_m"),
        ("Matched-control model", "matched_control_prediction_m"),
    ]:

        observation_results[
            model_name.replace(" ", "_").lower() + "_absolute_error_m"
        ] = np.abs(
            observation_results["actual_gwl_m"]
            - observation_results[prediction_column]
        )

    # -----------------------------------------------------
    # 6. METRICS BY TEST NOVELTY GROUP
    # -----------------------------------------------------

    group_rows = []

    prediction_columns = {
        "Full model": "full_model_prediction_m",
        "Extreme-removed model": "extreme_removed_prediction_m",
        "Matched-control model": "matched_control_prediction_m",
    }

    for group_name in [
        "All test observations",
        "Familiar",
        "Poorly represented",
        "Extreme novelty",
    ]:

        if group_name == "All test observations":
            group = observation_results
        else:
            group = observation_results[
                observation_results["novelty_group"] == group_name
            ]

        if group.empty:
            continue

        for model_name, prediction_column in prediction_columns.items():

            metrics = evaluate_predictions(
                group["actual_gwl_m"].to_numpy(),
                group[prediction_column].to_numpy(),
            )

            group_rows.append(
                {
                    "test_group": group_name,
                    "model": model_name,
                    **metrics,
                }
            )

    group_summary = pd.DataFrame(group_rows)

    # -----------------------------------------------------
    # 7. SUPPORT DISTANCE SUMMARY
    # -----------------------------------------------------

    support_rows = []

    support_columns = {
        "Full model": "full_training_support_distance",
        "Extreme-removed model": "extreme_removed_support_distance",
        "Matched-control model": "matched_control_support_distance",
    }

    for group_name in [
        "All test observations",
        "Familiar",
        "Poorly represented",
        "Extreme novelty",
    ]:

        if group_name == "All test observations":
            group = observation_results
        else:
            group = observation_results[
                observation_results["novelty_group"] == group_name
            ]

        if group.empty:
            continue

        for model_name, column in support_columns.items():

            support_rows.append(
                {
                    "test_group": group_name,
                    "model": model_name,
                    "n_observations": len(group),
                    "mean_support_distance": group[column].mean(),
                    "median_support_distance": group[column].median(),
                }
            )

    support_summary = pd.DataFrame(support_rows)

    # -----------------------------------------------------
    # 8. SAVE RESULTS
    # -----------------------------------------------------

    group_summary.to_csv(
        RESULTS / "controlled_removal_metrics.csv",
        index=False,
    )

    support_summary.to_csv(
        RESULTS / "controlled_removal_support.csv",
        index=False,
    )

    observation_results.to_csv(
        RESULTS / "controlled_removal_predictions.csv",
        index=False,
    )

    print("\nMODEL PERFORMANCE BY TEST GROUP")
    print("=" * 85)

    print(group_summary.round(4).to_string(index=False))

    print("\nTRAINING SUPPORT BY TEST GROUP")
    print("=" * 85)

    print(support_summary.round(6).to_string(index=False))

    print("\nSaved:")
    print("results/controlled_removal_metrics.csv")
    print("results/controlled_removal_support.csv")
    print("results/controlled_removal_predictions.csv")


if __name__ == "__main__":
    main()
