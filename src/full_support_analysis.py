import pandas as pd
import numpy as np
from pathlib import Path

data = pd.read_csv("data/processed/groundwater_rainfall_features_qc.csv",
                   parse_dates=["observation_date"])
pred = pd.read_csv("results/baseline_predictions.csv")

data["well_id"] = (
    data["Station"].astype(str) + "_" +
    data["Latitude"].astype(str) + "_" +
    data["Longitude"].astype(str)
)
# Use pre-processed observation_date column (consistent with baseline_model.py)
data["date"] = data["observation_date"]
data = data.sort_values(["well_id", "date"])
data["previous_gwl"] = data.groupby("well_id")[
    "Groundwater Level Quarterly Manual (meter)"
].shift(1)

pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
pred["well_id"] = pred["well_id"].astype(str)

target = "Groundwater Level Quarterly Manual (meter)"
train = data[data["date"] < "2018-01-01"]
test = data[data["date"] >= "2018-01-01"].copy()

df = test.merge(pred, on=["well_id", "date"], how="inner")
df["abs_error"] = (df["actual_gwl_m"] - df["random_forest_prediction_m"]).abs()

features = ["rainfall_30d_mm", "rainfall_90d_mm", "previous_gwl"]
rows = []

for feature in features:
    lo, hi = train[feature].quantile([0.01, 0.99])
    valid = df[feature].notna()
    df[f"{feature}_outside"] = valid & (
        (df[feature] < lo) | (df[feature] > hi)
    )

    for label, mask in [
        ("within_1st_99th_percentile", valid & ~df[f"{feature}_outside"]),
        ("outside_1st_99th_percentile", df[f"{feature}_outside"]),
    ]:
        group = df[mask]
        rows.append({
            "feature": feature,
            "group": label,
            "n": len(group),
            "wells": group["well_id"].nunique(),
            "MAE_m": group["abs_error"].mean(),
            "RMSE_m": np.sqrt(((group["actual_gwl_m"] - group["random_forest_prediction_m"]) ** 2).mean()),
            "lower_cutoff": lo,
            "upper_cutoff": hi,
        })

combined = df[features].notna().all(axis=1)
outside_any = df[[f"{f}_outside" for f in features]].any(axis=1)

for label, mask in [
    ("all_three_within_range", combined & ~outside_any),
    ("at_least_one_outside_range", combined & outside_any),
]:
    group = df[mask]
    rows.append({
        "feature": "combined",
        "group": label,
        "n": len(group),
        "wells": group["well_id"].nunique(),
        "MAE_m": group["abs_error"].mean(),
        "RMSE_m": np.sqrt(((group["actual_gwl_m"] - group["random_forest_prediction_m"]) ** 2).mean()),
        "lower_cutoff": np.nan,
        "upper_cutoff": np.nan,
    })

out = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
out.to_csv("results/full_support_analysis.csv", index=False)
print(out.round(3).to_string(index=False))
print("\nSaved: results/full_support_analysis.csv")
