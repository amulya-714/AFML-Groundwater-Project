import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("data/processed/groundwater_rainfall_features_qc.csv",
                   parse_dates=["observation_date"])
pred = pd.read_csv("results/baseline_predictions.csv", parse_dates=["date"])

target = "Groundwater Level Quarterly Manual (meter)"
data["well_id"] = (data["Station"].astype(str) + "_" +
                   data["Latitude"].astype(str) + "_" +
                   data["Longitude"].astype(str))
data = data.sort_values(["well_id", "observation_date"])
data["previous_gwl"] = data.groupby("well_id")[target].shift(1)
data = data.dropna(subset=["rainfall_30d_mm", "rainfall_90d_mm", "previous_gwl"])
data = data[data["observation_date"] >= "2018-01-01"].copy()

data = data.merge(pred, left_on=["observation_date", "well_id"],
                  right_on=["date", "well_id"], how="inner")
data["abs_error"] = (data[target] - data["random_forest_prediction_m"]).abs()
data["squared_error"] = (data[target] - data["random_forest_prediction_m"]) ** 2

# Training-only feature support; exploratory 1st–99th percentile bounds.
train = pd.read_csv("data/processed/groundwater_rainfall_features_qc.csv",
                    parse_dates=["observation_date"])
train = train[train["observation_date"] < "2018-01-01"]
bounds = {}
for col in ["rainfall_30d_mm", "rainfall_90d_mm"]:
    bounds[col] = train[col].quantile([0.01, 0.99]).to_dict()
    print(f"{col} training 1st–99th percentiles: "
          f"{bounds[col][0.01]:.2f} to {bounds[col][0.99]:.2f} mm")

data["out_of_support"] = False
for col, b in bounds.items():
    data["out_of_support"] |= (
        (data[col] < b[0.01]) | (data[col] > b[0.99])
    )

data["extreme_90d_rainfall"] = (
    data["rainfall_90d_mm"] >= train["rainfall_90d_mm"].quantile(0.95)
)

def summarize(group, label):
    if len(group) == 0:
        return {"group": label, "n": 0, "MAE_m": np.nan, "RMSE_m": np.nan}
    return {
        "group": label,
        "n": len(group),
        "MAE_m": group["abs_error"].mean(),
        "RMSE_m": np.sqrt(group["squared_error"].mean()),
    }

rows = [
    summarize(data[~data["out_of_support"]], "Within rainfall support"),
    summarize(data[data["out_of_support"]], "Outside rainfall support"),
    summarize(data[~data["extreme_90d_rainfall"]], "Below training 95th percentile rainfall"),
    summarize(data[data["extreme_90d_rainfall"]], "Top 5% training rainfall"),
]
result = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
result.to_csv("results/support_analysis.csv", index=False)
print("\n", result.to_string(index=False))
print("\nSaved results/support_analysis.csv")
print("Exploratory association only, not evidence of physical causality.")
