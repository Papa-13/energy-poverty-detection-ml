import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "energy_features_master_labeled.csv")
LEAK_COLS = ['self_disconnect_ratio','mean_consumption','quintile_1','quintile_2','quintile_3',
             'quintile_4','quintile_5','winter_zero_ratio','consumption_volatility','winter_avg',
             'winter_avg_consumption']
exclude = ['household_id','energy_poor','vulnerability_score'] + LEAK_COLS
feat_cols = [c for c in df.columns if c not in exclude]

X = df[feat_cols].copy(); y = df['energy_poor'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
model.fit(X_train_res, y_train_res)

sample = df.sample(n=1000, random_state=42)
X_sample = sample[feat_cols]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
sv = shap_values
mean_abs_shap = np.abs(sv).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
importance.head(15).to_csv('xgb_shap_importance.csv')
print("Top 15 SHAP features (XGBoost, Tier 2 leakage-excluded, selected/deployed model):")
print(importance.head(15))
