"""
Derive winter-only features independently from raw half-hourly readings,
mirroring the exact feature-engineering functions used for the annual
analysis (notebook 02), but applied strictly to December-January-February
readings for each household.

Observation window: all readings with month in {12, 1, 2}, drawn from the
same 2011-11-23 to 2014-02-28 raw sample used throughout this study
(i.e. winter months occurring within that window: Dec 2011, Jan-Feb 2012,
Dec 2012, Jan-Feb 2013, Dec 2013, Jan-Feb 2014).
"""
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Loading raw cleaned readings...")
raw = pd.read_csv('/mnt/user-data/uploads/energy_data_cleaned_final.csv', parse_dates=['timestamp'])
raw = raw.rename(columns={'dayofweek': 'day_of_week'})

winter = raw[raw['month'].isin([12, 1, 2])].copy()
print(f"Winter-only readings: {len(winter):,} (of {len(raw):,} total, "
      f"{len(winter)/len(raw)*100:.2f}%)")
print(f"Households with >=1 winter reading: {winter['household_id'].nunique()} of {raw['household_id'].nunique()}")

MIN_WINTER_READINGS = 10
counts = winter.groupby('household_id').size()
keep_households = counts[counts >= MIN_WINTER_READINGS].index
print(f"Households with >= {MIN_WINTER_READINGS} winter readings (retained): {len(keep_households)}")
print(f"Households excluded (fewer than {MIN_WINTER_READINGS} winter readings, "
      f"including zero): {raw['household_id'].nunique() - len(keep_households)}")

df = winter[winter['household_id'].isin(keep_households)].copy()

# ---------------------------------------------------------------
# 1. Consumption statistics (15 features) — identical logic to notebook 02
# ---------------------------------------------------------------
def create_consumption_statistics(df):
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
        day_df = df[df['day_of_week'] == day].groupby('household_id')['energy_kwh'].mean().reset_index()
        day_df.columns = ['household_id', f'day_{day}_avg_consumption']
        features = features.merge(day_df, on='household_id', how='left')
    monthly_avg = df.groupby(['household_id', 'month'])['energy_kwh'].mean().reset_index()
    monthly_std = monthly_avg.groupby('household_id')['energy_kwh'].std().reset_index()
    monthly_std.columns = ['household_id', 'monthly_consumption_variability']
    features = features.merge(monthly_std, on='household_id', how='left')
    return features

def create_vulnerability_indicators(df):
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
    features = df.groupby('household_id').apply(_agg).reset_index()
    return features

print("\nEngineering winter-only feature blocks (identical logic to the annual pipeline, applied to winter-only readings)...")
f1 = create_consumption_statistics(df)
print(f"  consumption statistics: {f1.shape}")
f2 = create_temporal_features(df)
print(f"  temporal: {f2.shape}")
f3 = create_vulnerability_indicators(df)
print(f"  vulnerability indicators: {f3.shape}")
f4 = create_load_profile_features(df)
print(f"  load profile: {f4.shape}")
f5 = create_variability_features(df)
print(f"  variability: {f5.shape}")

master = f1.merge(f2, on='household_id', how='outer') \
           .merge(f3, on='household_id', how='outer') \
           .merge(f4, on='household_id', how='outer') \
           .merge(f5, on='household_id', how='outer')

# Winter-equivalent quintile indicators (based on winter-only mean consumption,
# the same proxy logic used for the annual ACORN-placeholder quintiles)
master['consumption_quintile'] = pd.qcut(master['mean_consumption'], q=5, labels=[1,2,3,4,5])
for i in range(1, 6):
    master[f'quintile_{i}'] = (master['consumption_quintile'] == i).astype(int)
master = master.drop('consumption_quintile', axis=1)

master = master.fillna(0)
print(f"\nWinter feature master: {master.shape}")
master.to_csv('winter_features_master.csv', index=False)
print("Saved winter_features_master.csv")
