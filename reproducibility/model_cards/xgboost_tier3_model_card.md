# Model Card: XGBoost, Tier 3 (Primary Model)

## Summary

This is the primary model reported in the paper "Machine Learning for
Energy Poverty Detection Using Smart Meter Consumption Patterns." It is
an XGBoost gradient-boosted tree classifier trained to predict the
`energy_poor` vulnerability-proxy label from **33 behavioural/timing
features only** (Tier 3: the subset of the leakage-excluded, 81-feature
Tier 2 set with |correlation| <= 0.5 against overall mean consumption).
Tier 3 is the tier argued throughout the paper to be the most credible,
since it excludes both the features used to construct the label
directly (Tier 2's exclusion) and features that are themselves close
proxies for the amount of energy a household consumes (Tier 3's further
exclusion) -- the paper's central methodological concern is that a
model which can simply infer consumption level is not actually learning
behavioural vulnerability signal.

## Intended use

Research / methodological demonstration of behavioural-signal-based
energy-poverty screening from smart meter data. **Not validated against
any ground-truth fuel-poverty or vulnerability determination** -- the
label itself is a consumption-derived heuristic proxy (Section 3.2), not
an officially assessed status. Not intended for deployment in any
setting that gates access to support, tariffs, or interventions without
independent human review and validation against a recognised
fuel-poverty metric.

## Training data

- 5,560 households, UK domestic half-hourly smart meter readings
  (Section 3.1), cleaned per `preprocessing/clean_data.py`.
- Features: `features/build_features.py`, 92 engineered features,
  reduced to 33 under the Tier 3 exclusion (`labels/build_annual_label.py:leakage_tiers`).
- Label: `labels/build_annual_label.py`, 5-condition rule, >=2 of 5
  required. Prevalence: 22.4% positive.
- Split: single stratified 80/20 train/test split, random seed 42, no
  household in both sets.
- Class imbalance: SMOTE oversampling, fit on the training fold only.

## Performance (held-out test set, point estimate + 95% bootstrap CI)

See `models/train_evaluate.py` output / Table 4 of the paper. Headline
test-set F1 = 0.931 (bootstrap 95% CI reported in the paper); recall and
precision, PR-AUC and ROC-AUC are reported alongside the other
tier/model combinations in the same table.

## Interpretability

SHAP (`shap.TreeExplainer`) analysis on this model is the paper's
primary interpretability result (Section 4.2, Figure 3;
`figures/shap_tier3.py`). The top contributing features are consumption
variability and distinctness measures (`variance_to_mean`,
`cv_consumption`, `distinct_consumption_levels`), the winter-to-annual
consumption ratio, and the zero-consumption ratio -- behavioural/timing
signals, consistent with the tier's design intent.

## Known limitations

- The label is a heuristic proxy, not validated ground truth (Section
  3.2, Section 5.1).
- Original household-sampling seed from the full ~167M-reading raw
  dataset to the 5,560-household analysis sample was lost and cannot be
  exactly reproduced (Section 3.1); the sampled data itself is provided
  in this repository (`data/`) so results remain reproducible from that
  sample onward.
- A small number of households have sparse reading coverage; Section
  4.4 / `models/min_reading_sensitivity.py` shows the result is stable
  across minimum-reading-count thresholds.
- Single train/test split (not k-fold cross-validated point estimates);
  uncertainty is instead quantified via test-set bootstrap resampling.

## Reproduction

```
python preprocessing/clean_data.py raw_readings.csv energy_data_cleaned_final.csv
python features/build_features.py energy_data_cleaned_final.csv
python labels/build_annual_label.py energy_features_master.csv
python models/train_evaluate.py energy_features_master_labeled.csv
python figures/shap_tier3.py energy_features_master_labeled.csv
```
