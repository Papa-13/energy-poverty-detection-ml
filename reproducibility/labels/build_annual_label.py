"""
Construct the annual vulnerability-proxy label `energy_poor` exactly as
specified in Section 3.2 of the paper.

A household is labelled vulnerable (energy_poor = 1) if it meets at least
two of the following five conditions, each defined relative to the
empirical distribution of the 5,560-household analysis sample:

  1. Self-disconnection ratio above the 80th percentile
  2. Mean consumption below the 20th percentile
  3. Winter zero-consumption ratio above the 80th percentile
  4. Consumption volatility above the 80th percentile
  5. Winter average consumption below the 20th percentile

IMPORTANT (Section 3.2 of the paper): conditions 1-5 map onto features
(`self_disconnect_ratio`, `mean_consumption`, `winter_zero_ratio`,
`consumption_volatility`, `winter_avg`) that are also retained, unmodified,
in the 92-feature predictor set (plus `quintile_1`, a one-hot re-encoding
of condition 2). This is a known, disclosed label-feature overlap, not an
oversight -- see Section 3.2 and Section 4.1 (the three-tier correction)
for how this is handled throughout the paper.

Input: the master feature file produced by features/build_features.py
(energy_features_master.csv), which must already contain the columns
referenced below.

Usage:
    python build_annual_label.py energy_features_master.csv
"""
import sys
import pandas as pd


def build_annual_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = []

    threshold_1 = df['self_disconnect_ratio'].quantile(0.80)
    conditions.append(df['self_disconnect_ratio'] > threshold_1)

    threshold_2 = df['mean_consumption'].quantile(0.20)
    conditions.append(df['mean_consumption'] < threshold_2)

    threshold_3 = df['winter_zero_ratio'].quantile(0.80)
    conditions.append(df['winter_zero_ratio'] > threshold_3)

    threshold_4 = df['consumption_volatility'].quantile(0.80)
    conditions.append(df['consumption_volatility'] > threshold_4)

    threshold_5 = df['winter_avg'].quantile(0.20)
    conditions.append(df['winter_avg'] < threshold_5)

    vulnerability_score = sum(conditions)
    df['vulnerability_score'] = vulnerability_score
    df['energy_poor'] = (vulnerability_score >= 2).astype(int)

    print(f"Total households: {len(df):,}")
    print(f"Energy poor: {df['energy_poor'].sum():,} ({df['energy_poor'].mean()*100:.2f}%)")
    print("\nVulnerability score distribution:")
    print(df['vulnerability_score'].value_counts().sort_index())

    return df


def leakage_tiers(df: pd.DataFrame):
    """
    Returns the three leakage-correction feature-set tiers used throughout
    Section 4.1 of the paper:
      - Tier 1 (92 features): all engineered features, label-defining
        features retained.
      - Tier 2 (81 features): the six literal label-rule inputs/derivatives
        removed (self_disconnect_ratio, mean_consumption, winter_zero_ratio,
        consumption_volatility, winter_avg, and the five quintile_* one-hot
        columns, of which quintile_1 is the literal re-encoding).
      - Tier 3 (33 features): Tier 2 further restricted to features with
        |correlation| <= 0.5 against mean_consumption, i.e. the features
        least contaminated by overall consumption level.
    """
    LEAK_COLS = ['self_disconnect_ratio', 'mean_consumption', 'quintile_1', 'quintile_2',
                 'quintile_3', 'quintile_4', 'quintile_5', 'winter_zero_ratio',
                 'consumption_volatility', 'winter_avg', 'winter_avg_consumption']
    exclude_base = ['household_id', 'energy_poor', 'vulnerability_score']

    tiers = {}
    tiers['tier1'] = [c for c in df.columns if c not in exclude_base]
    tiers['tier2'] = [c for c in df.columns if c not in exclude_base + LEAK_COLS]
    mag_corr = df[tiers['tier2']].corrwith(df['mean_consumption']).abs()
    tiers['tier3'] = mag_corr[mag_corr <= 0.5].index.tolist()
    return tiers


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'energy_features_master.csv'
    df = pd.read_csv(path)
    labelled = build_annual_label(df)
    labelled.to_csv('energy_features_master_labeled.csv', index=False)

    tiers = leakage_tiers(labelled)
    for name, cols in tiers.items():
        print(f"{name}: {len(cols)} features")
