import pandas as pd
import numpy as np

d = pd.read_csv("data/processed/groundwater_rainfall_features_qc.csv",
                parse_dates=["observation_date"])
p = pd.read_csv("results/baseline_predictions.csv")

# Use pre-processed observation_date column (consistent with baseline_model.py)
d["date"] = d["observation_date"]
d["well_id"] = (
    d["Station"].astype(str) + "_" +
    d["Latitude"].astype(str) + "_" +
    d["Longitude"].astype(str)
)
d = d.sort_values(["well_id", "date"])
d["previous_gwl"] = d.groupby("well_id")[
    "Groundwater Level Quarterly Manual (meter)"
].shift(1)

p["date"] = pd.to_datetime(p["date"], errors="coerce")
df = d[d["date"] >= "2018-01-01"].merge(
    p, on=["date", "well_id"], how="inner"
)
df["abs_error"] = (
    df["actual_gwl_m"] - df["random_forest_prediction_m"]
).abs()

train = d[d["date"] < "2018-01-01"]
lo, hi = train["previous_gwl"].quantile([0.01, 0.99])
df["outside"] = (
    (df["previous_gwl"] < lo) | (df["previous_gwl"] > hi)
)

paired = df.groupby(["well_id", "outside"])["abs_error"].mean().unstack()
paired = paired.dropna(subset=[False, True])
paired["difference_m"] = paired[True] - paired[False]

print("Cutoffs:", round(lo, 3), "to", round(hi, 3), "m")
print("Wells with both groups:", len(paired))
print("Mean paired MAE difference:", round(paired["difference_m"].mean(), 3), "m")
print("Median paired MAE difference:", round(paired["difference_m"].median(), 3), "m")
print("Wells with higher error outside range:",
      int((paired["difference_m"] > 0).sum()), "/", len(paired))
print("Saved paired results to results/paired_gwl_support.csv")
paired.to_csv("results/paired_gwl_support.csv")
