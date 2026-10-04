"""
Full feature-engineering pipeline producing the 92-feature annual master
feature set used throughout the paper (Section 3.4), applied to the
500,000-reading working sample (Section 3.1). This is the exact logic
used in the original analysis notebooks (01-02), reorganised into
importable functions and documented in Appendix A of the paper.

Input: a cleaned raw-readings CSV with columns:
    household_id, timestamp, energy_kwh, year, month, day, hour,
    dayofweek, quarter, is_weekend, day_name, month_name
(produced by preprocessing/clean_data.py)

Usage:
    python build_features.py energy_data_cleaned_final.csv
"""
import sys
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def create_consumption_statistics(df):
    """15 features: central tendency, spread, percentiles, distribution shape."""
    features = df.groupby('household_id')['energy_kwh'].agg([
        ('mean_consumption', 'mean'),
        ('median_consumption', 'median'),
        ('total_consumption', 'sum'),
        ('std_consumption', 'std'),
        ('min_consumption', 'min'),
        ('max_consumption', 'max'),
        ('range_consumption', lambda x: x.max() - x.min()),
        ('q25_consumption', lambda x: x.quantile(0.25)),
        ('q75_consumption', lambda x: x.quantile(0.75)),
        ('q90_consumption', lambda x: x.quantile(0.90)),
        ('q95_consumption', lambda x: x.quantile(0.95)),
        ('skewness_consumption', lambda x: skew(x)),
        ('kurtosis_consumption', lambda x: kurtosis(x)),
        ('total_readings', 'count'),
        ('non_zero_readings', lambda x: (x > 0).sum())
    ]).reset_index()
    features['zero_consumption_ratio'] = (features['total_readings'] - features['non_zero_readings']) / features['total_readings']
    features['iqr_consumption'] = features['q75_consumption'] - features['q25_consumption']
    return features


def create_temporal_features(df):
    """20 features: time-of-day, day-of-week, weekend/weekday, monthly variability."""
    def get_time_period(hour):
        if 6 <= hour < 12: return 'morning'
        elif 12 <= hour < 18: return 'afternoon'
        elif 18 <= hour < 23: return 'evening'
        else: return 'night'

    df = df.copy()
    df['time_period'] = df['hour'].apply(get_time_period)

    temporal_features = []
    for period in ['morning', 'afternoon', 'evening', 'night']:
        period_df = df[df['time_period'] == period].groupby('household_id')['energy_kwh'].mean().reset_index()
        period_df.columns = ['household_id', f'{period}_avg_consumption']
        temporal_features.append(period_df)

    features = temporal_features[0]
    for feat_df in temporal_features[1:]:
        features = features.merge(feat_df, on='household_id', how='outer')

    peak_hours = df.groupby('household_id').apply(
        lambda x: x.groupby('hour')['energy_kwh'].mean().idxmax()
    ).reset_index()
    peak_hours.columns = ['household_id', 'peak_consumption_hour']
    features = features.merge(peak_hours, on='household_id', how='left')

    weekend_avg = df[df['is_weekend'] == 1].groupby('household_id')['energy_kwh'].mean().reset_index()
    weekend_avg.columns = ['household_id', 'weekend_avg_consumption']
    weekday_avg = df[df['is_weekend'] == 0].groupby('household_id')['energy_kwh'].mean().reset_index()
    weekday_avg.columns = ['household_id', 'weekday_avg_consumption']
    features = features.merge(weekend_avg, on='household_id', how='left')
    features = features.merge(weekday_avg, on='household_id', how='left')
    features['weekend_weekday_ratio'] = features['weekend_avg_consumption'] / (features['weekday_avg_consumption'] + 1e-6)

    for day in range(7):
        day_df = df[df['dayofweek'] == day].groupby('household_id')['energy_kwh'].mean().reset_index()
        day_df.columns = ['household_id', f'day_{day}_avg_consumption']
        features = features.merge(day_df, on='household_id', how='left')

    monthly_avg = df.groupby(['household_id', 'month'])['energy_kwh'].mean().reset_index()
    monthly_std = monthly_avg.groupby('household_id')['energy_kwh'].std().reset_index()
    monthly_std.columns = ['household_id', 'monthly_consumption_variability']
    features = features.merge(monthly_std, on='household_id', how='left')
    return features


def create_vulnerability_indicators(df):
    """25 features: self-disconnection, consecutive zeros, volatility, winter stats."""
    features_list = []
    for household, household_data in df.groupby('household_id'):
        household_data = household_data.sort_values('timestamp')
        feature_dict = {'household_id': household}

        daytime_data = household_data[(household_data['hour'] >= 6) & (household_data['hour'] < 23)]
        feature_dict['self_disconnect_events'] = (daytime_data['energy_kwh'] == 0).sum()
        feature_dict['self_disconnect_ratio'] = feature_dict['self_disconnect_events'] / len(daytime_data) if len(daytime_data) > 0 else 0

        zero_mask = household_data['energy_kwh'] == 0
        zero_groups = (zero_mask != zero_mask.shift()).cumsum()
        consecutive_zeros = household_data[zero_mask].groupby(zero_groups).size()
        feature_dict['max_consecutive_zeros'] = consecutive_zeros.max() if len(consecutive_zeros) > 0 else 0
        feature_dict['avg_consecutive_zeros'] = consecutive_zeros.mean() if len(consecutive_zeros) > 0 else 0

        low_threshold = household_data['energy_kwh'].quantile(0.10)
        feature_dict['very_low_consumption_count'] = (household_data['energy_kwh'] < low_threshold).sum()
        feature_dict['very_low_consumption_ratio'] = feature_dict['very_low_consumption_count'] / len(household_data)

        evening_data = household_data[(household_data['hour'] >= 18) & (household_data['hour'] < 23)]
        if len(evening_data) > 0:
            feature_dict['evening_avg_consumption'] = evening_data['energy_kwh'].mean()
            feature_dict['evening_zero_ratio'] = (evening_data['energy_kwh'] == 0).sum() / len(evening_data)
        else:
            feature_dict['evening_avg_consumption'] = 0
            feature_dict['evening_zero_ratio'] = 0

        feature_dict['consumption_volatility'] = household_data['energy_kwh'].std() / (household_data['energy_kwh'].mean() + 1e-6)

        daily_consumption = household_data.groupby(household_data['timestamp'].dt.date)['energy_kwh'].sum()
        feature_dict['daily_consumption_std'] = daily_consumption.std()
        feature_dict['daily_consumption_cv'] = daily_consumption.std() / (daily_consumption.mean() + 1e-6)

        overall_mean = household_data['energy_kwh'].mean()
        feature_dict['below_avg_consumption_ratio'] = (household_data['energy_kwh'] < overall_mean).sum() / len(household_data)

        winter_months = household_data[household_data['month'].isin([12, 1, 2])]
        if len(winter_months) > 0:
            feature_dict['winter_avg_consumption'] = winter_months['energy_kwh'].mean()
            feature_dict['winter_min_consumption'] = winter_months['energy_kwh'].min()
            feature_dict['winter_zero_ratio'] = (winter_months['energy_kwh'] == 0).sum() / len(winter_months)
        else:
            feature_dict['winter_avg_consumption'] = 0
            feature_dict['winter_min_consumption'] = 0
            feature_dict['winter_zero_ratio'] = 0

        night_data = household_data[(household_data['hour'] >= 0) & (household_data['hour'] < 6)]
        if len(night_data) > 0:
            feature_dict['night_avg_consumption'] = night_data['energy_kwh'].mean()
            feature_dict['night_min_consumption'] = night_data['energy_kwh'].min()
        else:
            feature_dict['night_avg_consumption'] = 0
            feature_dict['night_min_consumption'] = 0

        consumption_diff = household_data['energy_kwh'].diff()
        sharp_drops = consumption_diff[consumption_diff < -0.5]
        feature_dict['sharp_drop_count'] = len(sharp_drops)
        feature_dict['sharp_drop_ratio'] = len(sharp_drops) / len(household_data)

        weekend_evening = household_data[(household_data['is_weekend'] == 1) &
                                         (household_data['hour'] >= 18) &
                                         (household_data['hour'] < 23)]
        feature_dict['weekend_evening_avg'] = weekend_evening['energy_kwh'].mean() if len(weekend_evening) > 0 else 0

        feature_dict['consumption_regularity'] = 1 / (1 + feature_dict['consumption_volatility'])

        features_list.append(feature_dict)
    return pd.DataFrame(features_list)


def create_load_profile_features(df):
    """15 features: load factor, base load, ramp rates, peak/off-peak."""
    features_list = []
    for household, household_data in df.groupby('household_id'):
        feature_dict = {'household_id': household}

        avg_load = household_data['energy_kwh'].mean()
        peak_load = household_data['energy_kwh'].max()
        feature_dict['load_factor'] = avg_load / (peak_load + 1e-6)

        non_zero_consumption = household_data[household_data['energy_kwh'] > 0]['energy_kwh']
        feature_dict['base_load'] = non_zero_consumption.min() if len(non_zero_consumption) > 0 else 0
        feature_dict['base_load_avg'] = non_zero_consumption.quantile(0.10) if len(non_zero_consumption) > 0 else 0
        feature_dict['peak_to_base_ratio'] = peak_load / (feature_dict['base_load'] + 1e-6)

        consumption_diff = household_data['energy_kwh'].diff().abs()
        feature_dict['avg_ramp_rate'] = consumption_diff.mean()
        feature_dict['max_ramp_rate'] = consumption_diff.max()
        feature_dict['ramp_rate_std'] = consumption_diff.std()

        peak_threshold = household_data['energy_kwh'].quantile(0.90)
        feature_dict['peak_demand_hours'] = (household_data['energy_kwh'] >= peak_threshold).sum()
        feature_dict['peak_demand_ratio'] = feature_dict['peak_demand_hours'] / len(household_data)

        off_peak_data = household_data[(household_data['hour'] >= 23) | (household_data['hour'] < 7)]
        if len(off_peak_data) > 0:
            feature_dict['off_peak_avg'] = off_peak_data['energy_kwh'].mean()
            feature_dict['off_peak_ratio'] = off_peak_data['energy_kwh'].sum() / household_data['energy_kwh'].sum()
        else:
            feature_dict['off_peak_avg'] = 0
            feature_dict['off_peak_ratio'] = 0

        on_peak_data = household_data[(household_data['hour'] >= 7) & (household_data['hour'] < 23)]
        feature_dict['on_peak_avg'] = on_peak_data['energy_kwh'].mean() if len(on_peak_data) > 0 else 0

        feature_dict['load_diversity'] = household_data['energy_kwh'].nunique() / len(household_data)

        features_list.append(feature_dict)
    return pd.DataFrame(features_list)


def create_variability_features(df):
    """10 features: coefficient of variation, entropy, distinct levels, mode frequency."""
    def _agg(x):
        return pd.Series({
            'cv_consumption': x['energy_kwh'].std() / (x['energy_kwh'].mean() + 1e-6),
            'iqr_normalized': (x['energy_kwh'].quantile(0.75) - x['energy_kwh'].quantile(0.25)) / (x['energy_kwh'].median() + 1e-6),
            'range_normalized': (x['energy_kwh'].max() - x['energy_kwh'].min()) / (x['energy_kwh'].mean() + 1e-6),
            'mad_consumption': (x['energy_kwh'] - x['energy_kwh'].mean()).abs().mean(),
            'median_ad_consumption': (x['energy_kwh'] - x['energy_kwh'].median()).abs().median(),
            'variance_to_mean': x['energy_kwh'].var() / (x['energy_kwh'].mean() + 1e-6),
            'consistency_score': 1 / (1 + x['energy_kwh'].std() / (x['energy_kwh'].mean() + 1e-6)),
            'consumption_entropy': stats.entropy(np.histogram(x['energy_kwh'], bins=10)[0] + 1e-6),
            'distinct_consumption_levels': x['energy_kwh'].nunique(),
            'mode_frequency': (x['energy_kwh'] == x['energy_kwh'].mode()[0]).sum() / len(x) if len(x['energy_kwh'].mode()) > 0 else 0
        })
    return df.groupby('household_id').apply(_agg).reset_index()


def create_winter_features(df):
    """10 features: December-February consumption statistics and winter/annual ratio."""
    winter_df = df[df['month'].isin([12, 1, 2])].copy()
    if len(winter_df) == 0:
        return pd.DataFrame(columns=['household_id', 'winter_avg', 'winter_std', 'winter_min', 'winter_max',
                                    'winter_zero_count', 'winter_evening_avg', 'winter_night_avg',
                                    'winter_to_annual_ratio', 'winter_volatility', 'winter_self_disconnect'])

    annual_avg = df.groupby('household_id')['energy_kwh'].mean().reset_index()
    annual_avg.columns = ['household_id', 'annual_avg']

    winter_features = winter_df.groupby('household_id')['energy_kwh'].agg([
        ('winter_avg', 'mean'),
        ('winter_std', 'std'),
        ('winter_min', 'min'),
        ('winter_max', 'max'),
        ('winter_zero_count', lambda x: (x == 0).sum())
    ]).reset_index()

    winter_evening = winter_df[(winter_df['hour'] >= 18) & (winter_df['hour'] < 23)]
    winter_evening_avg = winter_evening.groupby('household_id')['energy_kwh'].mean().reset_index()
    winter_evening_avg.columns = ['household_id', 'winter_evening_avg']

    winter_night = winter_df[(winter_df['hour'] >= 0) & (winter_df['hour'] < 6)]
    winter_night_avg = winter_night.groupby('household_id')['energy_kwh'].mean().reset_index()
    winter_night_avg.columns = ['household_id', 'winter_night_avg']

    features = winter_features.merge(winter_evening_avg, on='household_id', how='left')
    features = features.merge(winter_night_avg, on='household_id', how='left')
    features = features.merge(annual_avg, on='household_id', how='left')

    features['winter_to_annual_ratio'] = features['winter_avg'] / (features['annual_avg'] + 1e-6)
    features['winter_volatility'] = features['winter_std'] / (features['winter_avg'] + 1e-6)

    winter_daytime = winter_df[(winter_df['hour'] >= 6) & (winter_df['hour'] < 23)]
    winter_disconnect = winter_daytime.groupby('household_id').apply(
        lambda x: (x['energy_kwh'] == 0).sum() / len(x)
    ).reset_index()
    winter_disconnect.columns = ['household_id', 'winter_self_disconnect']
    features = features.merge(winter_disconnect, on='household_id', how='left')

    features = features.drop('annual_avg', axis=1)
    return features


def create_acorn_placeholder_features(df_master):
    """
    5 features: consumption-based quintile indicators, used as a documented
    PLACEHOLDER for genuine Acorn socioeconomic data, which was not available
    for this sample (Section 3.1). quintile_1 is a deterministic re-encoding
    of the label's "mean consumption below 20th percentile" condition and is
    excluded from Tiers 2-3 (Section 3.2) for exactly that reason.
    """
    df_master = df_master.copy()
    df_master['consumption_quintile'] = pd.qcut(df_master['mean_consumption'], q=5, labels=[1, 2, 3, 4, 5])
    for i in range(1, 6):
        df_master[f'quintile_{i}'] = (df_master['consumption_quintile'] == i).astype(int)
    df_master = df_master.drop('consumption_quintile', axis=1)
    return df_master


def build_features(raw_readings_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_readings_path, parse_dates=['timestamp'])

    print("Building consumption statistics...")
    f1 = create_consumption_statistics(df)
    print("Building temporal features...")
    f2 = create_temporal_features(df)
    print("Building vulnerability indicators...")
    f3 = create_vulnerability_indicators(df)
    print("Building load profile features...")
    f4 = create_load_profile_features(df)
    print("Building variability features...")
    f5 = create_variability_features(df)
    print("Building winter-specific features...")
    f6 = create_winter_features(df)

    master = f1.merge(f2, on='household_id', how='outer') \
               .merge(f3, on='household_id', how='outer') \
               .merge(f4, on='household_id', how='outer') \
               .merge(f5, on='household_id', how='outer') \
               .merge(f6, on='household_id', how='outer')

    print("Adding Acorn-placeholder quintile indicators...")
    master = create_acorn_placeholder_features(master)

    master = master.fillna(0)
    print(f"\nFinal feature master: {master.shape[0]} households, {master.shape[1] - 1} features")
    return master


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'energy_data_cleaned_final.csv'
    master = build_features(path)
    master.to_csv('energy_features_master.csv', index=False)
    print("Saved energy_features_master.csv")
