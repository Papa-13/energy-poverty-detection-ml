# Machine Learning for Energy Poverty Detection Using Smart Meter Consumption Patterns

Reproducibility materials for the manuscript (Papa Kwadwo Bona Owusu,
Southampton Solent University, supervised by Hamidreza Soltani),
assembled to satisfy the supervisor's reproducibility checklist
(environment lock file, label-construction code, feature-generation
code, training/evaluation scripts, split seeds, figure-generation
scripts, and model cards).

## Repository structure

```
data/           Source data: raw-cleaned readings and the engineered
                feature master table (see "Data" below).
preprocessing/  Reading-level cleaning (missing values, duplicates,
                extreme-outlier removal).
features/       Feature engineering: consumption statistics, temporal
                patterns, vulnerability indicators, load-profile and
                variability measures (annual + winter variants).
labels/         Vulnerability-proxy label construction (annual and
                winter-only), and the three-tier leakage-correction
                feature-set definitions (Tier 1 / 2 / 3).
models/         Annual training + bootstrap-CI evaluation across all
                tiers and models; the fixed-seed LSTM baseline; the
                minimum-reading-count sensitivity analysis.
winter/         The independently re-derived, methodologically valid
                winter-only experiment: winter-only feature engineering
                and the tuned, SMOTE-in-CV-fold winter model pipeline.
figures/        SHAP interpretability scripts (one per tier/model) and
                the chart-rendering script that produces the paper's
                SHAP bar-chart figures with plain-language feature
                labels.
model_cards/    Model cards for the primary model (XGBoost, Tier 3) and
                the secondary comparison models.
requirements.txt  Pinned package versions used to produce every result
                   reported in the paper.
```

## Data

- `data/energy_data_cleaned_final.csv` -- cleaned half-hourly readings
  (output of `preprocessing/clean_data.py`).
- `data/energy_features_master.csv` -- engineered household-level
  feature table (output of `features/build_features.py`), the direct
  input to the label-construction and modelling scripts below.

**Note on the sampling seed (Section 3.1 of the paper):** the original
random seed used to draw the 5,560-household analysis sample from the
full ~167 million-reading raw dataset was not retained during the
original project work, and cannot be exactly reproduced. The sampled
data itself is therefore included directly in this repository so that
every downstream result (features, labels, models, figures) remains
exactly reproducible starting from that sample, even though the
sample-selection step itself is not re-runnable from the full raw data.

## Reproducing the paper's results end to end

```bash
pip install -r requirements.txt

# 1. Clean the raw readings (skippable -- data/energy_data_cleaned_final.csv
#    is already the cleaned output).
python preprocessing/clean_data.py raw_readings.csv data/energy_data_cleaned_final.csv

# 2. Engineer features (skippable -- data/energy_features_master.csv is
#    already the output).
python features/build_features.py data/energy_data_cleaned_final.csv

# 3. Build the annual label and the three leakage-correction tiers.
python labels/build_annual_label.py data/energy_features_master.csv
#   -> energy_features_master_labeled.csv

# 4. Train and evaluate all 4 models x 3 tiers with bootstrap 95% CIs
#    (Table 4).
python models/train_evaluate.py energy_features_master_labeled.csv

# 5. Fixed-seed LSTM baseline (Tier 1).
python models/lstm_fixed_seed.py energy_features_master_labeled.csv

# 6. Minimum-reading-count sensitivity analysis (Table 7 / Figure 11).
python models/min_reading_sensitivity.py energy_features_master_labeled.csv

# 7. SHAP interpretability for all four reported tier/model combinations.
python figures/shap_tier1_rf.py energy_features_master_labeled.csv
python figures/shap_tier2_rf.py energy_features_master_labeled.csv
python figures/shap_tier2_xgb.py energy_features_master_labeled.csv
python figures/shap_tier3.py energy_features_master_labeled.csv
python figures/regen_shap_charts.py

# 8. The independently re-derived winter-only experiment (Section 4.3).
python winter/01_winter_feature_engineering.py data/energy_data_cleaned_final.csv
#   -> winter_features_master.csv
python labels/build_winter_label.py winter_features_master.csv
#   -> winter_features_master_labeled.csv
python winter/02_winter_valid_experiment.py
#   -> winter_valid_results.csv, winter_valid_best_params.json (Table 6 / Figure 10)
```

All scripts use fixed random seeds (`random_state=42` / `tf` seed 42
throughout) and are deterministic given the same input data and package
versions pinned in `requirements.txt`.

## Models and interpretability

See `model_cards/xgboost_tier3_model_card.md` (primary model) and
`model_cards/secondary_models_card.md` (LightGBM, Random Forest,
Logistic Regression comparisons).

## Three-tier leakage correction

`labels/build_annual_label.py` defines all three feature-set tiers used
throughout the paper (Section 4.1):

- **Tier 1** (92 features): all engineered features, label-defining
  features retained.
- **Tier 2** (81 features): the six literal label-rule inputs/
  derivatives removed.
- **Tier 3** (33 features): Tier 2 further restricted to features with
  |correlation| <= 0.5 against overall mean consumption -- the subset
  least contaminated by consumption level, and the paper's primary,
  most-credible reported tier.
