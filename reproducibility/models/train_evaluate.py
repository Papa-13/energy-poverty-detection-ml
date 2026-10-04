"""
Annual training and evaluation pipeline: trains Logistic Regression,
Random Forest, XGBoost, and LightGBM under each of the three
leakage-correction tiers (Section 4.1 of the paper), and reports
bootstrap 95% confidence intervals for recall, precision, F1, ROC-AUC,
and PR-AUC.

Procedure (identical across tiers/models):
  1. Single stratified 80/20 train/test split (random seed 42; no
     household appears in both).
  2. SMOTE class-balancing fit on the training fold only.
  3. Fit the model on the SMOTE-resampled training fold; evaluate once
     on the untouched test fold to get point estimates.
  4. Bootstrap the FIXED test-set predictions/probabilities 1,000 times
     (resampling test-set rows with replacement) to obtain 95% CIs on
     each metric. This reflects sampling variability of the test set
     alone, not retraining variability.

Input: energy_features_master_labeled.csv (output of
labels/build_annual_label.py), or energy_features_master.csv (the label
will then be (re)built in-process).

Output: annual_bootstrap_ci_results.csv -- one row per tier x model,
with point estimates and [2.5%, 97.5%] bootstrap CIs.

Usage:
    python train_evaluate.py energy_features_master_labeled.csv
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'labels'))
from build_annual_label import build_annual_label, leakage_tiers  # noqa: E402

warnings.filterwarnings('ignore')
np.random.seed(42)

N_BOOT = 1000

MODEL_DEFS = {
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': lambda: xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    'LightGBM': lambda: lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
}


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=N_BOOT, seed=42):
    n = len(y_true)
    rng = np.random.RandomState(seed)
    boot = {k: [] for k in ['recall', 'precision', 'f1', 'roc_auc', 'pr_auc']}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        boot['recall'].append(recall_score(yt, yp, zero_division=0))
        boot['precision'].append(precision_score(yt, yp, zero_division=0))
        boot['f1'].append(f1_score(yt, yp, zero_division=0))
        try:
            boot['roc_auc'].append(roc_auc_score(yt, ypr))
            boot['pr_auc'].append(average_precision_score(yt, ypr))
        except Exception:
            pass
    return {f'{k}_ci_lo': np.percentile(v, 2.5) for k, v in boot.items()} | \
           {f'{k}_ci_hi': np.percentile(v, 97.5) for k, v in boot.items()}


def run(df: pd.DataFrame) -> pd.DataFrame:
    if 'energy_poor' not in df.columns:
        df = build_annual_label(df)
    tiers = leakage_tiers(df)

    results = []
    for tier_name, feat_cols in tiers.items():
        X, y = df[feat_cols].copy(), df['energy_poor'].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        y_test_arr = y_test.values

        for model_name, ctor in MODEL_DEFS.items():
            model = ctor()
            model.fit(X_train_res, y_train_res)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test_arr, pred).ravel()

            point = dict(
                tier=tier_name, model=model_name,
                recall=recall_score(y_test_arr, pred),
                precision=precision_score(y_test_arr, pred),
                f1=f1_score(y_test_arr, pred),
                roc_auc=roc_auc_score(y_test_arr, proba),
                pr_auc=average_precision_score(y_test_arr, proba),
                tn=tn, fp=fp, fn=fn, tp=tp,
            )
            point.update(bootstrap_ci(y_test_arr, pred, proba))
            results.append(point)
            print(f"{tier_name} | {model_name}: F1={point['f1']:.4f} "
                  f"[{point['f1_ci_lo']:.4f}, {point['f1_ci_hi']:.4f}]  "
                  f"PR-AUC={point['pr_auc']:.4f}")

    return pd.DataFrame(results)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'energy_features_master_labeled.csv'
    df = pd.read_csv(path)
    results_df = run(df)
    results_df.to_csv('annual_bootstrap_ci_results.csv', index=False)
    print("\nSaved annual_bootstrap_ci_results.csv")
