"""
SHAP interpretability analysis: Random Forest, Tier 1 (92 features,
label-defining features retained). Retained in the paper as the
"original Tier 1 analysis, for transparency" (Section 4.2, Figures 6-8),
superseded by the Tier-3 XGBoost analysis as the primary result.

This is a clean, self-contained re-derivation of notebook
04_SHAP_Interpretability_Analysis.ipynb's original approach (which
loaded a previously pickled model/scaler); here the model is trained
in-process so the script has no external pickle dependency.

Usage:
    python shap_tier1_rf.py energy_features_master_labeled.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')


def main(path):
    df = pd.read_csv(path)
    exclude_cols = ['household_id', 'energy_poor', 'vulnerability_score']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y = df['energy_poor'].copy()

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)

    sample_size = min(1000, len(X_scaled))
    X_sample = X_scaled.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, -1]
    else:
        sv = shap_values

    importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(sv).mean(axis=0),
    }).sort_values('importance', ascending=False)

    importance.to_csv('shap_feature_importance.csv', index=False)
    print("Top 20 SHAP features (Random Forest, Tier 1, 92 features):")
    print(importance.head(20))
    print("\nSaved shap_feature_importance.csv")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'energy_features_master_labeled.csv'
    main(path)
