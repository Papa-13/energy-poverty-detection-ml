"""
Construct the winter-only vulnerability-proxy label `winter_energy_poor`,
used in the "valid, independently re-derived winter-only experiment"
(Section 4.3 of the paper).

Starting from the annual 5-condition rule (build_annual_label.py), two
pairs of conditions collapse into numerically identical quantities once
every feature is computed from winter-only data (winter average
consumption vs. overall mean consumption; the winter zero-consumption
ratio vs. the overall zero-consumption ratio -- "winter average" and
"average" mean the same thing inside a winter-only dataset). The winter
label is therefore built from the four conditions that remain genuinely
distinct under this collapse:

  1. Self-disconnection ratio above the 80th percentile (winter-only)
  2. Mean consumption below the 20th percentile (winter-only)
  3. Zero-consumption ratio above the 80th percentile (winter-only)
  4. Consumption volatility above the 80th percentile (winter-only)

A household is labelled winter-vulnerable (winter_energy_poor = 1) if it
meets at least two of these four conditions -- the closest available
analogue to the annual rule's "at least two of five."

Input: winter_features_master.csv, the output of
winter/01_winter_feature_engineering.py.

Usage:
    python build_winter_label.py winter_features_master.csv
"""
import sys
import pandas as pd


def build_winter_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = []

    t1 = df['self_disconnect_ratio'].quantile(0.80)
    conditions.append(df['self_disconnect_ratio'] > t1)

    t2 = df['mean_consumption'].quantile(0.20)
    conditions.append(df['mean_consumption'] < t2)

    t3 = df['zero_consumption_ratio'].quantile(0.80)
    conditions.append(df['zero_consumption_ratio'] > t3)

    t4 = df['consumption_volatility'].quantile(0.80)
    conditions.append(df['consumption_volatility'] > t4)

    vulnerability_score = sum(conditions)
    df['vulnerability_score'] = vulnerability_score
    df['winter_energy_poor'] = (vulnerability_score >= 2).astype(int)

    print(f"Total households: {len(df):,}")
    print(f"Winter energy poor: {df['winter_energy_poor'].sum():,} "
          f"({df['winter_energy_poor'].mean()*100:.2f}%)")
    print("\nVulnerability score distribution:")
    print(df['vulnerability_score'].value_counts().sort_index())

    return df


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'winter_features_master.csv'
    df = pd.read_csv(path)
    labelled = build_winter_label(df)
    labelled.to_csv('winter_features_master_labeled.csv', index=False)
