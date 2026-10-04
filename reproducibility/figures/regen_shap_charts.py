"""
Generate the four SHAP bar-chart figures used in Section 4.2 of the
paper (Figures 3-6 in the final numbering), with plain-language feature
labels substituted for technical column names (feature_labels.py).

Run after the corresponding SHAP-computation scripts in this directory
(shap_tier1_rf.py, shap_tier2_xgb.py, shap_tier2_rf.py, shap_tier3.py),
which produce the CSV inputs referenced below.

Usage:
    python regen_shap_charts.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from feature_labels import readable

def make_chart(csv_path, feat_col, val_col, title, outpath, topn=15, index_col=None):
    df = pd.read_csv(csv_path, index_col=index_col)
    if feat_col not in df.columns:
        df = df.reset_index()
        df.columns = [feat_col, val_col]
    df = df.sort_values(val_col, ascending=False).head(topn)
    df = df.iloc[::-1]  # for barh ascending plot
    labels = [readable(f) for f in df[feat_col]]
    # Disambiguate duplicate readable labels (near-duplicate source columns)
    # by appending the technical column name in parentheses.
    from collections import Counter
    counts = Counter(labels)
    labels = [
        f"{lab} ({tech})" if counts[lab] > 1 else lab
        for lab, tech in zip(labels, df[feat_col])
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(labels, df[val_col], color='#4472C4')
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(title, fontsize=11, wrap=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print("saved", outpath)

# Tier 2 XGBoost (Figure 2)
make_chart('xgb_shap_importance.csv', 'index', '0',
           'SHAP feature importance (mean |SHAP value|), XGBoost,\nTier 2 leakage-excluded model (81 features), top 15 features',
           'media/xgb_shap_importance.png', index_col=0)

# Tier 2 Random Forest (Figure 3)
make_chart('clean_shap_importance.csv', 'index', '0',
           'SHAP feature importance (mean |SHAP value|), Random Forest,\nTier 2 leakage-excluded model (81 features), top 15 features',
           'media/clean_shap_importance.png', index_col=0)

# Tier 1 Random Forest (Figure 4)
make_chart('shap_feature_importance.csv', 'feature', 'importance',
           'SHAP feature importance (mean |SHAP value|), Random Forest,\nTier 1 (92 features, label-defining features retained), top 20 features',
           'media/tier1_shap_corrected_title.png', topn=20)

# NEW: Tier 3 XGBoost (primary/most-credible model)
make_chart('xgb_tier3_shap_importance.csv', 'feature', 'mean_abs_shap',
           'SHAP feature importance (mean |SHAP value|), XGBoost,\nTier 3 behavioural/timing-only model (33 features), top 15 features',
           'media/xgb_tier3_shap_importance.png')
