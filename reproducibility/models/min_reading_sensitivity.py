"""
Minimum-reading-count sensitivity analysis (Section 4.4, Table 7 /
Figure 11): re-trains the annual Tier 3 XGBoost model after excluding
households below each of several total-reading thresholds, to check
whether the headline Tier 3 result is an artefact of sparsely sampled
households.

Usage:
    python min_reading_sensitivity.py energy_features_master_labeled.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

import sys
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'energy_features_master_labeled.csv')
LEAK_COLS = ['self_disconnect_ratio','mean_consumption','quintile_1','quintile_2','quintile_3',
             'quintile_4','quintile_5','winter_zero_ratio','consumption_volatility','winter_avg',
             'winter_avg_consumption']
exclude_base = ['household_id','energy_poor','vulnerability_score']
tier2_cols = [c for c in df.columns if c not in exclude_base + LEAK_COLS]
mag_corr = df[tier2_cols].corrwith(df['mean_consumption']).abs()
tier3_cols = mag_corr[mag_corr <= 0.5].index.tolist()

print("total_readings distribution:")
print(df['total_readings'].describe())

thresholds = [0, 10, 20, 30, 50, 70]
results = []
for min_reads in thresholds:
    sub = df[df['total_readings'] >= min_reads].copy()
    n_hh = len(sub)
    n_pos = sub['energy_poor'].sum()
    if n_pos < 10 or (n_hh - n_pos) < 10:
        print(f"min_reads={min_reads}: skipped, insufficient class counts (n={n_hh}, pos={n_pos})")
        continue
    X = sub[tier3_cols].copy()
    y = sub['energy_poor'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1) if y_train.sum() > 1 else 1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    row = dict(
        min_reads=min_reads, n_households=n_hh, n_excluded=len(df)-n_hh,
        pct_excluded=round((len(df)-n_hh)/len(df)*100, 2),
        n_test=len(y_test),
        recall=recall_score(y_test, pred), precision=precision_score(y_test, pred),
        f1=f1_score(y_test, pred), roc_auc=roc_auc_score(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
    )
    results.append(row)
    print(f"min_reads>={min_reads:3d} | n_hh={n_hh:5d} (excl {row['pct_excluded']:5.2f}%) | "
          f"F1={row['f1']:.4f} Recall={row['recall']:.4f} Precision={row['precision']:.4f} PR-AUC={row['pr_auc']:.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv('min_reading_sensitivity.csv', index=False)
print("\nSaved min_reading_sensitivity.csv")
