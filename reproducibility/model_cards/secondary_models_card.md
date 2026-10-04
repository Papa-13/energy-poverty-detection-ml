# Model Card: Secondary Models (LightGBM, Random Forest, Logistic Regression)

## Summary

These three models are trained and reported alongside the primary
XGBoost Tier 3 model (see `xgboost_tier3_model_card.md`), under all
three leakage-correction tiers, to show that the paper's findings are
not an artefact of one specific algorithm. None is proposed as the
paper's primary/deployment candidate.

- **LightGBM (Tier 2, 81 features)**: the secondary deployment-candidate
  comparison referenced in Section 4.2 (Figure 4), reported alongside
  Tier 2 XGBoost as an additional check that leakage-excluded
  performance is not algorithm-specific.
- **Random Forest (Tier 2, 81 features)**: tertiary SHAP comparison
  (Section 4.2, Figure 5), used to check that feature-importance
  rankings are not an artefact of the boosted-tree model specifically.
- **Random Forest (Tier 1, 92 features)**: retained for transparency as
  the original Tier 1 analysis (Section 4.2, "Figures 6-8" in the final
  manuscript numbering) -- the model and tier combination used in the
  earliest version of this work, before the label-feature-overlap issue
  was identified and the three-tier correction (Section 4.1) was
  introduced. Superseded by Tier 3 as the paper's primary result.
- **Logistic Regression**: reported under all three tiers in the main
  results table (Table 4) as a linear baseline against the tree-based
  models.

## Training data, split, and class balancing

Identical to the primary model: see `xgboost_tier3_model_card.md`.
All four model families (Logistic Regression, Random Forest, XGBoost,
LightGBM) are trained under the same train/test split, SMOTE procedure,
and evaluated with the same bootstrap-CI methodology, by
`models/train_evaluate.py`.

## Performance

See Table 4 of the paper, or re-run `models/train_evaluate.py`, which
reports all tier x model combinations in one pass.

## Interpretability scripts

- `figures/shap_tier1_rf.py` -- Random Forest, Tier 1 (92 features).
- `figures/shap_tier2_rf.py` -- Random Forest, Tier 2 (81 features).
- `figures/shap_tier2_xgb.py` -- XGBoost, Tier 2 (81 features; the
  deployment-candidate comparison for the primary Tier 3 XGBoost model).
- `figures/regen_shap_charts.py` -- renders all SHAP bar charts (plus
  the Tier 3 primary result) with plain-language feature labels
  (`figures/feature_labels.py`).

## Known limitations

Shared with the primary model (heuristic label, lost original sampling
seed, sparse-coverage households) -- see
`xgboost_tier3_model_card.md`. Tier 1 and Tier 2 models additionally
retain some or all label-defining features in their predictor set, so
their headline performance numbers should be read as upper bounds
inflated by label-feature overlap, not as evidence of superior
real-world detection ability relative to Tier 3 (Section 4.1).
