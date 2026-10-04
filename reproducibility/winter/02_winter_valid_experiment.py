import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix, make_scorer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings, json
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('winter_features_master_labeled.csv')
LEAK_COLS = ['self_disconnect_ratio','mean_consumption','zero_consumption_ratio','consumption_volatility',
             'quintile_1','quintile_2','quintile_3','quintile_4','quintile_5']
exclude_base = ['household_id','winter_energy_poor','vulnerability_score']

tiers = {}
tiers['Winter Tier 1 (79 features)'] = [c for c in df.columns if c not in exclude_base]
tiers['Winter Tier 2 (70 features)'] = [c for c in df.columns if c not in exclude_base + LEAK_COLS]
mag_corr = df[tiers['Winter Tier 2 (70 features)']].corrwith(df['mean_consumption']).abs()
tiers['Winter Tier 3 (29 features)'] = mag_corr[mag_corr <= 0.5].index.tolist()

f1_scorer = make_scorer(f1_score)

param_distributions = {
    'Logistic Regression': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2'],
    },
    'Random Forest': {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [None, 6, 10, 20],
        'clf__min_samples_leaf': [1, 2, 4],
    },
    'XGBoost': {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [3, 4, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__subsample': [0.7, 0.85, 1.0],
    },
    'LightGBM': {
        'clf__n_estimators': [100, 200, 400],
        'clf__num_leaves': [15, 31, 63],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__subsample': [0.7, 0.85, 1.0],
    },
}

def make_estimator(name):
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=2000, random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    if name == 'XGBoost':
        return xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    if name == 'LightGBM':
        return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

N_BOOT = 1000
all_results = []
best_params_log = {}

for tier_name, feat_cols in tiers.items():
    X = df[feat_cols].copy()
    y = df['winter_energy_poor'].copy()

    # Household-level train/test split (one row per household already; stratified on label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessing (scaler) on TRAINING data only, then transform test
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', make_estimator(model_name)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_distributions[model_name],
            n_iter=15, scoring=f1_scorer, cv=cv, random_state=42, n_jobs=-1,
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        best_params_log[f'{tier_name} | {model_name}'] = search.best_params_

        pred = best_model.predict(X_test_scaled)
        proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_test_arr = y_test.values

        point = dict(
            tier=tier_name, model=model_name,
            recall=recall_score(y_test_arr, pred), precision=precision_score(y_test_arr, pred),
            f1=f1_score(y_test_arr, pred), roc_auc=roc_auc_score(y_test_arr, proba),
            pr_auc=average_precision_score(y_test_arr, proba),
        )
        cm = confusion_matrix(y_test_arr, pred)
        point['tn'], point['fp'], point['fn'], point['tp'] = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        n = len(y_test_arr)
        rng = np.random.RandomState(42)
        boot = {k: [] for k in ['recall','precision','f1','roc_auc','pr_auc']}
        for _ in range(N_BOOT):
            idx = rng.randint(0, n, n)
            yt, yp, ypr = y_test_arr[idx], pred[idx], proba[idx]
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
        for k in ['recall','precision','f1','roc_auc','pr_auc']:
            arr = np.array(boot[k])
            point[f'{k}_ci_lo'] = np.percentile(arr, 2.5)
            point[f'{k}_ci_hi'] = np.percentile(arr, 97.5)

        all_results.append(point)
        print(f"{tier_name} | {model_name}: F1={point['f1']:.4f} "
              f"[{point['f1_ci_lo']:.4f},{point['f1_ci_hi']:.4f}] "
              f"Recall={point['recall']:.4f} Precision={point['precision']:.4f} "
              f"PR-AUC={point['pr_auc']:.4f} | best_params={search.best_params_}")

pd.DataFrame(all_results).to_csv('winter_valid_results.csv', index=False)
with open('winter_valid_best_params.json', 'w') as f:
    json.dump(best_params_log, f, indent=2)
print("\nSaved winter_valid_results.csv and winter_valid_best_params.json")
