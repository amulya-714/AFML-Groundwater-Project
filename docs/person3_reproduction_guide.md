# Person 3 Reproduction Guide

## Environment

- Python 3.15
- Dependencies: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, statsmodels
- Install: `python -m pip install -r requirements.txt`
- Additional: `python -m pip install statsmodels`

## Execution Order

All commands run from the repository root.

### Step 1: Generate baseline (prerequisite)

```bash
python src/baseline_model.py
```

### Step 2: Generate novelty and controlled-removal prerequisites

```bash
python src/input_extrapolation_analysis.py
python src/compare_novelty_types.py
python src/within_well_novelty_analysis.py
python src/within_well_paired_test.py
python src/within_well_bootstrap.py
python src/controlled_training_removal.py
python src/controlled_removal_bootstrap.py
python src/paired_gwl_support.py
python src/within_well_temporal_stability.py
python src/within_well_matched_input_analysis.py
python src/within_well_clustered_bootstrap.py
python src/multivariate_temporal_residual_test.py
python src/temporal_residual_sensitivity.py
python src/create_research_figures.py
```

### Step 3: Person 3 validation pipeline

```bash
python src/person3_robustness_validation.py
python src/person3_grouped_evaluation.py
python src/person3_bootstrap_validation.py
python src/person3_rainfall_sensitivity.py
```

### Step 4: Person 3 new experiments

```bash
python src/person3_data_validation.py
python src/validated_novelty_analysis.py
python src/repeated_controlled_removal.py      # ~15 min for 10 seeds
python src/temporal_stability_validation.py
```

### Step 5: Audit and tests

```bash
python src/person3_audit.py
python tests/test_data_integrity.py
```

## Random Seeds

- All Person 3 scripts use `RANDOM_SEED = 42`
- Repeated removal uses seeds 42–51 (10 seeds)
- Bootstrap: 2,000 repetitions, well-level cluster resampling

## Runtime Estimates

| Script | Approximate time |
|---|---|
| `baseline_model.py` | 30 seconds |
| `input_extrapolation_analysis.py` | 2 minutes |
| `controlled_training_removal.py` | 3 minutes |
| `repeated_controlled_removal.py` | 15 minutes (10 seeds × 3 models + support) |
| `validated_novelty_analysis.py` | 5 minutes |
| `temporal_stability_validation.py` | 1 minute |
| `person3_audit.py` | 5 minutes |
| Full pipeline | ~40 minutes |

## Known Limitations

1. `repeated_controlled_removal.py` uses 10 seeds (target was 20; computational constraint)
2. Rainfall provenance is UNRESOLVED — source data not in repository
3. `statsmodels` is required for temporal residual scripts (not in requirements.txt)
4. RF with `n_jobs=-1` produces small non-deterministic variations across runs

## Output Locations

- Results CSVs: `results/` and `results/person3/`
- Figures: `results/figures/` and `results/figures/person3/`
- Documentation: `docs/`
- Tests: `tests/`
