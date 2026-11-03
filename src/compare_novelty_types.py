
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


def nearest_distance(train_data, test_data, k=5):
    """Calculate mean distance to the k nearest training observations."""
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    neighbours = NearestNeighbors(
        n_neighbors=min(k, len(train_data)),
        n_jobs=-1,
    )

    neighbours.fit(train_scaled)

    distances, _ = neighbours.kneighbors(test_scaled)

    return distances.mean(axis=1)


def geographic_distance_km(training_coordinates, test_coordinates):
    """Calculate distance to the nearest training well in kilometres."""
    training_radians = np.radians(training_coordinates)
    test_radians = np.radians(test_coordinates)

    neighbours = NearestNeighbors(
        n_neighbors=1,
        algorithm="ball_tree",
        metric="haversine",
    )

    neighbours.fit(training_radians)

    distances, _ = neighbours.kneighbors(test_radians)

    # Earth's mean radius in kilometres.
    return distances[:, 0] * 6371.0


def assign_quartiles(values):
    """Assign four approximately equal-sized groups, including tied values."""
    ranks = pd.Series(values).rank(method="first")

    return pd.qcut(
        ranks,
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    )


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare the dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA)

    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.sort_values(WELL_COLUMNS + [DATE]).copy()

    # Previous groundwater observation for each well.
    df["previous_gwl"] = df.groupby(
        WELL_COLUMNS,
        dropna=False,
    )[TARGET].shift(1)

    month = df[DATE].dt.month

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    required_columns = MODEL_FEATURES + [TARGET, DATE]

    df = df.dropna(subset=required_columns).copy()

    # Preserve the original chronological split.
    train = df[df[DATE] < "2018-01-01"].copy()
    test = df[df[DATE] >= "2018-01-01"].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or test dataset is empty. Check the date split."
        )

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    # ---------------------------------------------------------
    # 2. Train the baseline Random Forest
    # ---------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[MODEL_FEATURES],
        train[TARGET],
    )

    predictions = model.predict(test[MODEL_FEATURES])

    out = pd.DataFrame({
        "date": test[DATE].to_numpy(),
        "Station": test["Station"].astype(str).to_numpy(),
        "Latitude": test["Latitude"].to_numpy(),
        "Longitude": test["Longitude"].to_numpy(),
        "actual_gwl_m": test[TARGET].to_numpy(),
        "prediction_m": predictions,
    })

    out["absolute_error_m"] = np.abs(
        out["actual_gwl_m"] - out["prediction_m"]
    )

    # ---------------------------------------------------------
    # 3. Geographic novelty
    # ---------------------------------------------------------
    training_wells = (
        train[["Latitude", "Longitude"]]
        .dropna()
        .drop_duplicates()
    )

    out["geographic_novelty_km"] = geographic_distance_km(
        training_wells.to_numpy(),
        test[["Latitude", "Longitude"]].to_numpy(),
    )

    # ---------------------------------------------------------
    # 4. Environmental novelty
    # ---------------------------------------------------------
    out["environmental_novelty"] = nearest_distance(
        train[ENVIRONMENTAL_FEATURES].astype(float),
        test[ENVIRONMENTAL_FEATURES].astype(float),
    )

    # ---------------------------------------------------------
    # 5. Combined novelty using all seven features
    # ---------------------------------------------------------
    out["combined_novelty"] = nearest_distance(
        train[MODEL_FEATURES].astype(float),
        test[MODEL_FEATURES].astype(float),
    )

    # ---------------------------------------------------------
    # 6. Compare errors across novelty quartiles
    # ---------------------------------------------------------
    novelty_columns = {
        "geographic": "geographic_novelty_km",
        "environmental": "environmental_novelty",
        "combined": "combined_novelty",
    }

    summary_rows = []

    for novelty_type, column in novelty_columns.items():
        quartile_column = f"{novelty_type}_novelty_quartile"

        out[quartile_column] = assign_quartiles(out[column])

        for quartile, group in out.groupby(
            quartile_column,
            observed=True,
        ):
            actual = group["actual_gwl_m"]
            predicted = group["prediction_m"]

            summary_rows.append({
                "novelty_type": novelty_type,
                "quartile": str(quartile),
                "n": len(group),
                "MAE_m": mean_absolute_error(actual, predicted),
                "RMSE_m": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "median_absolute_error_m": (
                    group["absolute_error_m"].median()
                ),
                "mean_novelty_score": group[column].mean(),
            })

    summary = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------
    observations_path = (
        RESULTS / "novelty_type_observations.csv"
    )

    summary_path = RESULTS / "novelty_type_summary.csv"

    out.to_csv(observations_path, index=False)
    summary.to_csv(summary_path, index=False)

    # ---------------------------------------------------------
    # 8. Display results
    # ---------------------------------------------------------
    print("\nTraining period:")
    print(f"{train[DATE].min().date()} to {train[DATE].max().date()}")

    print("\nTesting period:")
    print(f"{test[DATE].min().date()} to {test[DATE].max().date()}")

    print("\nERROR BY NOVELTY TYPE AND QUARTILE")

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved results:")
    print(observations_path)
    print(summary_path)


if __name__ == "__main__":
    main()
