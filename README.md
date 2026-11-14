# Why Do Groundwater ML Models Fail Under Extreme Conditions?

## 1. Project Overview

This project investigates why machine learning models for groundwater-level prediction may perform poorly under unfamiliar or changing conditions.

We examine two possible contributors:

1. **Input extrapolation:** Test observations contain combinations of inputs that are poorly represented in the training data.
2. **Temporal deterioration:** Prediction errors increase over time, even for familiar wells and similar input conditions.

The study uses historical groundwater-level observations from Karnataka and gridded rainfall data. It adopts a diagnostic, empirical approach and does not claim to establish the physical causes of temporal deterioration.

## 2. Research Question

To what extent are groundwater prediction errors associated with unfamiliar input conditions, and do errors also increase over time under familiar or similar input conditions?

## 3. Dataset and Features

- **Groundwater data:** Central Ground Water Board (CGWB) quarterly manual groundwater-level observations for Karnataka, 1991–2020.
- **Rainfall data:** Gridded rainfall data used to construct rainfall features around groundwater observation locations.
- **Processed dataset:** 104,781 rows and 21 columns before feature construction.
- **Model-ready dataset:** 102,289 observations after lag construction.

### Model Features

The Random Forest model uses seven features:

- Latitude
- Longitude
- 30-day rainfall
- 90-day rainfall
- Previous groundwater level
- Month sine
- Month cosine

The previous groundwater measurement is used as a lagged predictor. Therefore, this is a rolling one-step-ahead prediction task, assuming the previous measurement is available.

## 4. Methodology

### Chronological Train-Test Split

To preserve temporal ordering:

- **Training:** Observations before 1 January 2018.
- **Testing:** Observations from 1 January 2018 onward.
- **Training observations:** 87,426.
- **Testing observations:** 14,863.

### Models

Two approaches are compared:

1. **Persistence baseline:** Predicts groundwater level using the previous groundwater measurement.
2. **Random Forest Regressor:** Uses 300 trees, minimum leaf size of 3 and random state 42.

### Baseline Results

| Model | Test observations | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|---:|
| Persistence | 14,863 | 2.6013 | 4.4389 | 0.6280 |
| Random Forest | 14,863 | 1.9901 | 3.4977 | 0.7690 |

Random Forest reduced MAE by approximately 23.5% and RMSE by 21.2% compared with the persistence baseline.

## 5. Experiments and Findings

### Experiment 1: Training-Referenced Input Novelty

We measure how well test observations are represented in the training data using the mean distance to their five nearest training observations.

A larger distance indicates lower representation in the training data.

Four feature spaces were examined:

| Feature space | Features | MAE: lowest to highest novelty |
|---|---|---:|
| Geographic | Latitude, longitude | 1.901 → 1.905 m |
| Environmental | 30-day rainfall, 90-day rainfall, previous groundwater level, month sine and cosine | 1.541 → 2.754 m |
| Combined | All seven features | 1.386 → 2.874 m |
| Rainfall-only | 30-day and 90-day rainfall | Non-monotonic |

**Findings:**

- Geographic novelty showed almost no change in prediction error.
- Environmental novelty was associated with a 78.8% increase in MAE from the lowest to highest novelty quartile.
- Combined-feature novelty showed a 107.3% increase.
- Rainfall-only novelty did not show a consistent monotonic increase.

These findings suggest that unfamiliar combinations of environmental inputs, particularly when considered alongside geographic information, are associated with higher prediction errors.

Input novelty is a statistical measure of training-data representation, not a direct classification of physical extreme events.

### Experiment 2: Controlled Training-Support Removal

We removed the 1% most novel training observations (874 observations) and compared the resulting model with the full-data model and a year-matched control removal.

**Results:**

- Overall MAE difference for high-novelty removal: +0.0156 m.
- 95% confidence interval: [+0.0144, +0.0169] m.
- Mean support distance increased by 8.9% overall.
- Mean support distance increased by 25.1% in the very-low-support group.

The experiment provides limited evidence that training-data support affects predictive performance.

The confidence interval captures variation across ten model seeds and does not account for all sources of sampling uncertainty.

### Experiment 3: Temporal Deterioration Under Familiar Inputs

We investigated whether prediction errors increase over time even when comparisons are restricted to familiar wells and similar input conditions.

**Results:**

- Overall annual test MAE increased from 1.804 m in 2018 to 2.291 m in 2020.
- For 1,516 wells observed in both years, MAE increased from 1.709 m in 2018 to 2.131 m in 2020.
- Same-well increase: +0.422 m.
- 95% confidence interval: [+0.303, +0.535] m.
- Matched-input increase: +0.506 m.
- 95% confidence interval: [+0.410, +0.605] m.

These results suggest that unfamiliar inputs alone do not explain all observed prediction errors. Prediction performance also deteriorated over time in comparisons involving familiar wells and similar input conditions.

## 6. Overall Findings

The study identifies two patterns:

1. Test observations that are poorly represented in the training data, particularly in environmental and combined feature spaces, tend to have higher prediction errors.
2. Prediction errors also increase over time under familiar-well and similar-input comparisons.

Therefore, input extrapolation alone is insufficient to explain all observed prediction errors.

However, the study does not establish whether temporal deterioration is caused by groundwater extraction, irrigation, land-use changes, recharge, geology or other physical processes.

Changes in observation timing and measurement protocols may also influence temporal comparisons. Identifying the underlying physical mechanisms requires further investigation and additional verified data.

## 7. Limitations

- The observational design does not establish physical causation.
- Input novelty does not directly represent physical extreme events.
- The controlled-removal confidence interval captures model-seed variation, not all sources of uncertainty.
- Changes in observation schedules and measurement protocols may influence temporal comparisons.
- Additional physical and operational variables are needed to investigate the mechanisms behind temporal deterioration.


## 8. Repository Structure

```text
AFML-Groundwater-Project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── baseline_model.py
│   ├── support_analysis.py
│   ├── full_support_analysis.py
│   ├── same_well_analysis.py
│   ├── paired_gwl_support.py
│   ├── paired_bootstrap.py
│   └── bootstrap_gwl_support.py
├── results/
│   ├── baseline_metrics.csv
│   ├── baseline_predictions.csv
│   ├── support_analysis.csv
│   ├── full_support_analysis.csv
│   ├── same_well_analysis.csv
│   ├── paired_gwl_support.csv
│   ├── paired_bootstrap.txt
│   └── figures/
├── papers/
├── report/
├── requirements.txt
└── README.md
```

## 9. Setup and Execution

Clone the repository:

```bash
git clone https://github.com/amulya-714/AFML-Groundwater-Project.git
cd AFML-Groundwater-Project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline model from the repository root:

```bash
python src/baseline_model.py
```

The analysis scripts use fixed paths relative to the repository root and depend on the corresponding processed datasets and saved baseline predictions. Refer to each script for its required inputs and outputs.

## 10. Team

- Amulya P – PES1UG24AM458
- Anagha Kaushik – PES1UG24AM459
- Thrupthi G D – PES1UG24AM476
- Dhanush R – PES1UG24AM427

## 11. Project Status

The baseline model and three main lines of analysis have been developed and evaluated. The findings are diagnostic associations and should be interpreted within the limitations described above.