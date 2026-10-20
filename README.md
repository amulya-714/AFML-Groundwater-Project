# Groundwater ML Under Extreme Conditions

## Project Overview

This project investigates why machine learning models used for groundwater prediction may perform poorly during extreme hydroclimatic conditions.

We aim to examine two possible contributors to prediction failure:

1. **Input extrapolation:** Test conditions fall outside the range of input conditions represented in training data.
2. **Predictive relationship instability:** Prediction errors increase even when inputs are within the training-supported range, suggesting that the learned predictive relationship may not generalize reliably.

We will evaluate these explanations using groundwater observations and rainfall data, subject to data availability and quality.

## Research Question

When groundwater ML models fail under extreme hydroclimatic conditions, to what extent can the degradation be associated with unfamiliar inputs versus instability in predictive performance under otherwise familiar inputs?

## Project Structure

- `data/raw/`: Original datasets
- `data/processed/`: Cleaned and merged datasets
- `notebooks/`: Exploratory analysis and experiments
- `src/`: Reusable code
- `results/figures/`: Plots and evaluation results
- `papers/`: Literature notes and references
- `report/`: Research report and presentation materials

## Data Sources

Candidate sources include:

- Central Ground Water Board (CGWB): groundwater observations
- India Meteorological Department (IMD): rainfall data

Sources, licenses, units, temporal coverage, and data-processing decisions will be documented after verification.

## Evaluation Plan

The study will compare prediction performance across normal and extreme conditions, considering whether test inputs fall within the range supported by training data.

Evaluation will use justified definitions, quantitative metrics, uncertainty estimates, and robustness checks. Results will be interpreted as evidence of predictive performance patterns, not proof of physical causation.

## Team

- Amulya P
- Anagha Kaushik 
- Dhanush R
- Thrupthi G D 

## Project Status

Initial repository setup complete. Data-source verification and experimental design are in progress.