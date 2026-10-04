"""
Data cleaning pipeline, exactly as performed in notebook 01
(01_energy_data_cleaning_eda.ipynb), Section 3.1 of the paper.

Steps, in order:
  1. Drop rows with missing timestamp, energy_kwh, household_id, or
     tariff_type.
  2. Drop duplicate records (same household_id + timestamp), keeping the
     first occurrence.
  3. Remove extreme-high outliers only: readings above the 99.9th
     percentile of energy_kwh. Zero and near-zero readings are
     deliberately RETAINED, since they are themselves a meaningful
     energy-poverty signal (self-disconnection), not measurement noise.
     (This is "Strategy 2" of the four outlier strategies considered in
     notebook 01; Strategies 1/3/4 -- keep-all, IQR-removal, and
     winsorization -- were considered and rejected in favour of this one.)

Input: the raw half-hourly meter-reading export (one row per
household/timestamp/energy_kwh reading, plus household_id and
tariff_type columns).

Output: energy_data_cleaned_final.csv, the cleaned reading-level file
consumed by features/build_features.py.

Usage:
    python clean_data.py raw_readings.csv [output.csv]
"""
import sys
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_shape = df.shape

    # --- 1. Missing values ---------------------------------------------
    df_cleaned = df.dropna(subset=['timestamp']).copy()
    df_cleaned = df_cleaned.dropna(subset=['energy_kwh'])
    df_cleaned = df_cleaned.dropna(subset=['household_id', 'tariff_type'])
    print(f"After dropping missing values: {df_cleaned.shape[0]:,} rows "
          f"(removed {original_shape[0] - df_cleaned.shape[0]:,})")

    # --- 2. Duplicate records -------------------------------------------
    duplicates_key = df_cleaned.duplicated(subset=['household_id', 'timestamp']).sum()
    if duplicates_key > 0:
        df_cleaned = df_cleaned.drop_duplicates(subset=['household_id', 'timestamp'], keep='first')
        print(f"Removed {duplicates_key:,} duplicate (household_id, timestamp) records")
    else:
        print("No duplicate (household_id, timestamp) records found")

    # --- 3. Extreme-high outlier removal only ---------------------------
    # Zero/near-zero consumption is retained deliberately -- see module
    # docstring. Only values above the 99.9th percentile are dropped.
    high_999 = df_cleaned['energy_kwh'].quantile(0.999)
    before = len(df_cleaned)
    df_final = df_cleaned[df_cleaned['energy_kwh'] <= high_999].copy()
    print(f"Removed {before - len(df_final):,} extreme-high outlier readings "
          f"(> 99.9th percentile = {high_999:.4f} kWh)")

    print(f"\nOriginal: {original_shape[0]:,} rows -> Final: {len(df_final):,} rows "
          f"({len(df_final) / original_shape[0] * 100:.2f}% retained)")
    print(f"Unique households: {df_final['household_id'].nunique():,}")

    return df_final


if __name__ == '__main__':
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'raw_readings.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'energy_data_cleaned_final.csv'

    df = pd.read_csv(in_path, parse_dates=['timestamp'])
    cleaned = clean_data(df)
    cleaned.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
