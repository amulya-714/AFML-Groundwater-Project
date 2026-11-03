import pandas as pd
import numpy as np
from pathlib import Path

data = pd.read_csv(
    "data/processed/groundwater_rainfall_features_qc.csv",
    parse_dates=["observation_date"]
)
pred = pd.read_csv("results/baseline_predictions.csv", parse_dates=["date"])

data["well_id"] = (
    data["Station"].astype(str) + "_" +
    data["Latitude"].astype(str) + "_" +
    data["Longitude"].astype(str)
)
data = data[data["observation_date"] >= "2018-01-01"].copy()
data = data.merge(
    pred, left_on=["observation_date", "well_id"],
    right_on=["date", "well_id"], how="inner",
    validate="one_to_one"
)

train = pd.read_csv(
    "data/processed/groundwater_rainfall_features_qc.csv",
    parse_dates=["observation_date"]
)
train = train[train["observation_date"] < "2018-01-01"]

cutoff = train["rainfall_90d_mm"].quantile(0.95)
data = data.dropna(subset=["rainfall_90d_mm"]).copy()
data["extreme"] = data["rainfall_90d_mm"] >= cutoff
data["abs_error"] = (
    data["actual_gwl_m"] - data["random_forest_prediction_m"]
).abs()

counts = data.groupby(["well_id", "extreme"]).size().unstack(fill_value=0)
eligible = counts[(counts.get(False, 0) > 0) & (counts.get(True, 0) > 0)].index
paired = data[data["well_id"].isin(eligible)]

summary = paired.groupby("extreme").agg(
    rows=("abs_error", "size"),
    wells=("well_id", "nunique"),
    MAE_m=("abs_error", "mean"),
    median_abs_error_m=("abs_error", "median"),
    mean_groundwater_level_m=("actual_gwl_m", "mean"),
).reset_index()
summary["group"] = summary["extreme"].map({
    False: "Below training 95th percentile",
    True: "Top 5% training rainfall"
})

Path("results").mkdir(exist_ok=True)
summary.to_csv("results/same_well_analysis.csv", index=False)
print(f"Rainfall cutoff: {cutoff:.2f} mm")
print(f"Wells represented in both groups: {len(eligible)}")
print(summary[["group", "rows", "wells", "MAE_m",
               "median_abs_error_m", "mean_groundwater_level_m"]].to_string(index=False))
print("Saved results/same_well_analysis.csv")
