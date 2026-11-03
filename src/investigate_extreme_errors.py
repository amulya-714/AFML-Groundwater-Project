from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/groundwater_rainfall_features_qc.csv")
PREDICTIONS_PATH = Path("results/baseline_predictions.csv")
OUTPUT_PATH = Path("results/extreme_error_investigation.csv")

TARGET = "Groundwater Level Quarterly Manual (meter)"

# Load data.
data = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

# Reconstruct the well identifier and previous groundwater level.
data["well_id"] = (
    data["Station"].astype(str)
    + "_"
    + data["Latitude"].astype(str)
    + "_"
    + data["Longitude"].astype(str)
)

data = data.sort_values(["well_id", "observation_date"]).copy()
data["previous_gwl"] = data.groupby("well_id")[TARGET].shift(1)

# Reconstruct the baseline model's date-based training split.
train = data[data["observation_date"] < "2018-01-01"].copy()
test = data[data["observation_date"] >= "2018-01-01"].copy()

# Get the ten largest Random Forest errors.
predictions["rf_error_m"] = (
    predictions["random_forest_prediction_m"]
    - predictions["actual_gwl_m"]
)
predictions["rf_absolute_error_m"] = predictions["rf_error_m"].abs()

worst = predictions.nlargest(10, "rf_absolute_error_m").copy()

# Join the original input features to the predictions.
features = [
    "well_id",
    "observation_date",
    TARGET,
    "previous_gwl",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "qc_suspicious_rainfall_30d",
    "qc_suspicious_rainfall_90d",
]

investigation = worst.merge(
    test[features],
    left_on=["well_id", "date"],
    right_on=["well_id", "observation_date"],
    how="left",
    validate="one_to_one",
)

# Determine whether each well has observations in the training period.
training_wells = set(train["well_id"].dropna().unique())
investigation["well_seen_in_training"] = (
    investigation["well_id"].isin(training_wells)
)

# Compare each feature with the distribution in the training data.
# These are marginal checks, not a complete multivariate extrapolation test.
for column in ["rainfall_30d_mm", "rainfall_90d_mm", "previous_gwl"]:
    reference = train[column].dropna()
    if reference.empty:
        continue

    lower = reference.min()
    upper = reference.max()
    investigation[f"{column}_outside_train_range"] = (
        (investigation[column] < lower)
        | (investigation[column] > upper)
    )

# Display a focused report.
columns_to_show = [
    "date",
    "well_id",
    "actual_gwl_m",
    "random_forest_prediction_m",
    "rf_absolute_error_m",
    "previous_gwl",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "qc_suspicious_rainfall_30d",
    "qc_suspicious_rainfall_90d",
    "well_seen_in_training",
    "rainfall_30d_mm_outside_train_range",
    "rainfall_90d_mm_outside_train_range",
    "previous_gwl_outside_train_range",
]

print("\n=== EXTREME ERROR INVESTIGATION ===")
print(investigation[columns_to_show].to_string(index=False))

investigation.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved investigation to {OUTPUT_PATH}")
print(
    "\nNote: being inside each feature's individual training range "
    "does not prove that an observation is inside the joint input "
    "distribution. These checks are preliminary."
)
