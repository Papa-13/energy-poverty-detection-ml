"""
SHAP interpretability analysis: XGBoost, Tier 3 (33 behavioural/timing-
only features, |corr| <= 0.5 with mean_consumption). This is the PRIMARY
interpretability result reported in the paper (Section 4.2, Figure 3),
since Tier 3 is argued to be the most credible tier.

Usage:
    python shap_tier3.py energy_features_master_labeled.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
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
print(f"Tier 3: {len(tier3_cols)} features")
print(tier3_cols)

X = df[tier3_cols].copy()
y = df['energy_poor'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
model.fit(X_train_res, y_train_res)

explainer = shap.TreeExplainer(model)
sample_size = min(1000, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({'feature': tier3_cols, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
imp_df.to_csv('xgb_tier3_shap_importance.csv', index=False)
print(imp_df.head(20))
