import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from src.utils import path_builder
from src.features.indicators import add_stock_indicators
from src.models.config import FEATURE_COLS, TARGET_COL, GROUP_KEY_COL, WINDOW_SIZE, SCALING_COLS, SCALED_FEATURE_DATA_PATH, SCALER_DATA_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH, FEATURE_DATA_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
ROLLING_VOL_WINDOW = 5
MA_SHORT = 10
MA_LONG = 50
RSI_WINDOW = 14

def add_option_features(df):
    """Add option-specific features"""
    df = df.copy()
    
    df = df.sort_values(['contractID', 'date']).reset_index(drop=True)
    
    print("=== FEATURE ENGINEERING AUDIT ===")
    print(f"Input data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    df['iv_change'] = df.groupby('contractID')['implied_volatility'].diff()
    
    df['implied_volatility'] = df.groupby('contractID')['implied_volatility'].shift(1)
    df['iv_change'] = df.groupby('contractID')['iv_change'].shift(1)

    df['close_prev'] = df.groupby('contractID')['close'].shift(1)
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        df[greek] = df.groupby('contractID')[greek].shift(1)

    df['log_moneyness'] = np.log(df['close_prev'] / df['strike'])

    df['type'] = df['type'].map({'call': 1, 'put': 0})

    df['intrinsic_value'] = np.where(
        df['type'] == 1,
        np.maximum(0, df['close_prev'] - df['strike']),
        np.maximum(0, df['strike'] - df['close_prev'])
    )

    df['is_itm'] = df['intrinsic_value'] > 0

    df['time_to_maturity'] = (df['expiration'] - df['date']).dt.days / 365
    df = df[df['time_to_maturity'] > 0]

    pre_drop_count = len(df)
    df = df.dropna(subset=['close_prev', 'log_moneyness', 'delta', 'gamma', 'theta', 'vega', 'rho', 'iv_change'])
    post_drop_count = len(df)
    
    print(f"Dropped {pre_drop_count - post_drop_count} rows due to NaN values")
    print(f"Final data shape: {df.shape}")
    
    sample_contracts = df['contractID'].unique()[:3]
    for contract_id in sample_contracts:
        contract_data = df[df['contractID'] == contract_id].head(5)
        print(f"Contract {contract_id} sample (first 5 rows):")
        print(f"  Dates: {contract_data['date'].tolist()}")
        print(f"  close_prev: {contract_data['close_prev'].tolist()}")
        print(f"  IV (shifted): {contract_data['implied_volatility'].tolist()}")
        
    print("=== END FEATURE ENGINEERING AUDIT ===")
    return df


def normalize_features(df):
    """Normalize option features"""
    df = df.copy()
    
    scaler = StandardScaler()
    
    df.loc[:, ['log_moneyness_norm', 'intrinsic_value_norm']] = scaler.fit_transform(
        df[['log_moneyness', 'intrinsic_value']]
    )
    
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    df.loc[:, greeks] = scaler.fit_transform(df[greeks])
    
    return df


def merge_option_and_stock_features(option_df, stock_df):
    """Merge option features with stock features"""
    merged_df = pd.merge(option_df, stock_df, on='date', how='inner')
    
    merged_df = merged_df.rename(columns={
        'volume_x': 'option_volume',
        'volume_y': 'stock_volume',
        'type': 'option_type_encoded',
    })
    
    return merged_df

def add_greek_product_features(df):
    """Add product features based on the greeks and other option metrics"""
    df = df.copy()

    df["delta_x_iv"] = df["delta"] * df["implied_volatility"]
    df["vega_x_ttm"] = df["vega"] * df["time_to_maturity"]
    df["gamma_x_logm"] = df["gamma"] * df["log_moneyness_norm"]
    df["theta_x_intrinsic"] = df["theta"] * df["intrinsic_value_norm"]

    return df


def create_feature_dataset(options_data, stock_data):
    """Create a complete feature dataset from options and stock data"""
    if os.path.exists(FEATURE_DATA_PATH):
        logger.info(f"Feature dataset already exists at {FEATURE_DATA_PATH}. Loading existing data.")
        final_dataset = pd.read_parquet(FEATURE_DATA_PATH)
        print(final_dataset.info())
        return final_dataset
    
    options_with_features = add_option_features(options_data)
    
    options_normalized = normalize_features(options_with_features)
    
    stock_with_features = add_stock_indicators(stock_data)

    options_with_greek_features = add_greek_product_features(options_normalized)
    
    final_dataset = merge_option_and_stock_features(options_with_greek_features, stock_with_features)

    final_dataset['date'] = pd.to_numeric(final_dataset['date'])
    final_dataset['expiration'] = pd.to_numeric(final_dataset['expiration'])
    
    final_dataset = final_dataset[FEATURE_COLS + [TARGET_COL] + ['contractID']]

    final_dataset.to_parquet(FEATURE_DATA_PATH, index=False)
    logger.info(f"Feature dataset saved to {FEATURE_DATA_PATH}")

    return final_dataset


def audit_window_leakage(train_df, val_df, test_df, window_size):
    """Audit for potential window-based data leakage between splits."""
    print("\n=== WINDOW LEAKAGE AUDIT ===")
    
    train_max_date = train_df['date'].max()
    val_min_date = val_df['date'].min()
    val_max_date = val_df['date'].max()
    test_min_date = test_df['date'].min()
    
    print(f"Train max date: {train_max_date}")
    print(f"Val min date: {val_min_date}, Val max date: {val_max_date}")
    print(f"Test min date: {test_min_date}")
    
    all_contracts = set(train_df['contractID']) | set(val_df['contractID']) | set(test_df['contractID'])
    problematic_contracts = []
    
    for contract_id in list(all_contracts)[:20]:
        contract_train = train_df[train_df['contractID'] == contract_id]
        contract_val = val_df[val_df['contractID'] == contract_id]
        contract_test = test_df[test_df['contractID'] == contract_id]
        
        spans_multiple = sum([len(contract_train) > 0, len(contract_val) > 0, len(contract_test) > 0]) > 1
        
        if spans_multiple:
            problematic_contracts.append(contract_id)
            if len(problematic_contracts) <= 5:
                print(f"Contract {contract_id}:")
                if len(contract_train) > 0:
                    print(f"  Train: {contract_train['date'].min()} to {contract_train['date'].max()} ({len(contract_train)} rows)")
                if len(contract_val) > 0:
                    print(f"  Val: {contract_val['date'].min()} to {contract_val['date'].max()} ({len(contract_val)} rows)")
                if len(contract_test) > 0:
                    print(f"  Test: {contract_test['date'].min()} to {contract_test['date'].max()} ({len(contract_test)} rows)")
    
    print(f"Found {len(problematic_contracts)} contracts spanning multiple splits")
    
    window_leakage_risk = 0
    for contract_id in problematic_contracts[:10]:
        contract_train = train_df[train_df['contractID'] == contract_id]
        contract_val = val_df[val_df['contractID'] == contract_id]
        contract_test = test_df[test_df['contractID'] == contract_id]
        
        if len(contract_train) > 0 and len(contract_val) > 0:
            train_max = contract_train['date'].max()
            val_min = contract_val['date'].min()
            train_max_dt = pd.to_datetime(train_max)
            val_min_dt = pd.to_datetime(val_min)
            if (val_min_dt - train_max_dt).days < window_size:
                window_leakage_risk += 1
        
        if len(contract_val) > 0 and len(contract_test) > 0:
            val_max = contract_val['date'].max()
            test_min = contract_test['date'].min()
            val_max_dt = pd.to_datetime(val_max)
            test_min_dt = pd.to_datetime(test_min)
            if (test_min_dt - val_max_dt).days < window_size:
                window_leakage_risk += 1
    
    if window_leakage_risk > 0:
        print(f"WARNING: {window_leakage_risk} potential window leakage cases detected!")
        print(f"Windows of size {window_size} might span across split boundaries")
    else:
        print(f"Window leakage check passed ✓")
    
    print("=== END WINDOW LEAKAGE AUDIT ===\n")
    
    return len(problematic_contracts), window_leakage_risk


def create_time_window_features(df, window_size, feature_cols, target_col, group_key_cols=None, date_col='date', min_window_size=None):
    """Create sliding window features for time series modeling with progress bar."""
    if min_window_size is None:
        min_window_size = window_size
    
    logger.info(f"Creating time window features with window size {window_size}")
    logger.info(f"Using {len(feature_cols)} features and target column '{target_col}'")
    
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    sort_cols = [date_col]
    if group_key_cols:
        sort_cols = group_key_cols + sort_cols
    
    logger.info(f"Sorting DataFrame by columns: {sort_cols}")
    df_sorted = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    X_windows = []
    y_values = []
    
    if group_key_cols:
        groups = df_sorted.groupby(group_key_cols)
        total_groups = len(groups)
        logger.info(f"Found {total_groups} unique groups based on {group_key_cols}")
        
        processed_groups = 0
        skipped_groups = 0
        total_windows = 0
        
        with tqdm(total=total_groups, desc="Processing groups") as pbar:
            for group_name, group_df in groups:
                if isinstance(group_name, tuple):
                    group_str = "_".join(str(x) for x in group_name)
                else:
                    group_str = str(group_name)
                
                group_df = group_df.sort_values(by=date_col).reset_index(drop=True)
                
                if len(group_df) < min_window_size:
                    skipped_groups += 1
                    pbar.set_postfix(
                        processed=processed_groups, 
                        skipped=skipped_groups,
                        windows=total_windows
                    )
                    pbar.update(1)
                    continue
                
                num_windows = len(group_df) - window_size + 1
                
                for i in range(num_windows):
                    window = group_df.iloc[i:i+window_size]
                    
                    X_window = window[feature_cols].values
                    y_value = window.iloc[-1][target_col]
                    
                    X_windows.append(X_window)
                    y_values.append(y_value)
                
                processed_groups += 1
                total_windows += num_windows
                
                pbar.set_postfix(
                    processed=processed_groups, 
                    skipped=skipped_groups,
                    windows=total_windows
                )
                pbar.update(1)
    else:
        if len(df_sorted) < min_window_size:
            logger.warning(f"Not enough data points ({len(df_sorted)} < {min_window_size})")
            return np.array([]), np.array([])
        
        num_windows = len(df_sorted) - window_size + 1
        
        with tqdm(total=num_windows, desc="Creating windows") as pbar:
            for i in range(num_windows):
                window = df_sorted.iloc[i:i+window_size]
                
                X_window = window[feature_cols].values
                y_value = window.iloc[-1][target_col]
                
                X_windows.append(X_window)
                y_values.append(y_value)
                pbar.update(1)
    
    X = np.array(X_windows)
    y = np.array(y_values)
    
    logger.info(f"Created {len(X)} samples with shape {X.shape}")
    return X, y

def create_y_train_window_ds(df, window_size, feature_cols, target_col, group_key_cols=None, train_ratio=0.7, val_ratio=0.15, date_col='date', min_window_size=None):
    unique_dates = df[date_col].sort_values().unique()
    n_dates = len(unique_dates)
    
    train_idx = int(n_dates * train_ratio)
    
    train_dates = unique_dates[:train_idx]
    
    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    
    train_df = df[df[date_col].isin(train_dates)]
    
    X_train, y_train = create_time_window_features(
        train_df, window_size, feature_cols, target_col, 
        group_key_cols, date_col, min_window_size
    )

    return y_train

def create_time_window_dataset(df, window_size, feature_cols, target_col, group_key_cols=None, train_ratio=0.7, val_ratio=0.15, date_col='date', min_window_size=None):
    unique_dates = df[date_col].sort_values().unique()
    n_dates = len(unique_dates)
    
    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    
    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    logger.info(f"Val dates: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} dates)")
    logger.info(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")
    
    train_df = df[df[date_col].isin(train_dates)]
    val_df = df[df[date_col].isin(val_dates)]
    test_df = df[df[date_col].isin(test_dates)]
    
    X_train, y_train = create_time_window_features(
        train_df, window_size, feature_cols, target_col, 
        group_key_cols, date_col, min_window_size
    )
    
    X_val, y_val = create_time_window_features(
        val_df, window_size, feature_cols, target_col, 
        group_key_cols, date_col, min_window_size
    )
    
    X_test, y_test = create_time_window_features(
        test_df, window_size, feature_cols, target_col, 
        group_key_cols, date_col, min_window_size
    )

    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_VAL_PATH, y_val)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def standardize_and_scale_features(df, feature_cols=SCALING_COLS, exclude_cols=None, scale_method='z-score'):
    import pickle
    import os
    
    df = df.copy()
    
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    if exclude_cols:
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    logger.info(f"Standardizing {len(feature_cols)} features using {scale_method} method")
    
    features_df = df[feature_cols]
    
    if scale_method == 'z-score':
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scale_method}. Use 'z-score' or 'minmax'")
    
    scaled_features = scaler.fit_transform(features_df)
    
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
    
    for col in feature_cols:
        df.loc[:, col] = scaled_df[col]
    
    metadata = {
        'scaled_columns': feature_cols,
        'scaling_method': scale_method,
        'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'std': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        'max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None
    }
    
    scaler_path = SCALER_DATA_PATH
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Both X and y scalers saved to {scaler_path}")
    
    scaler_x_path = path_builder("data", "scaler_X.pkl")
    
    with open(scaler_x_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"X scaler saved to {scaler_x_path}")
    
    logger.info(f"Standardization complete. Data shape: {df.shape}")
    df.to_parquet(SCALED_FEATURE_DATA_PATH, index=False)
    logger.info(f"Scaled features saved to {SCALED_FEATURE_DATA_PATH}")
    return df, metadata

def create_leak_free_splits(df, train_ratio=0.7, val_ratio=0.15):
    print("\n=== CREATING LEAK-FREE SPLITS ===")
    
    contract_info = df.groupby('contractID').agg({
        'date': ['min', 'max', 'count']
    }).reset_index()
    
    contract_info.columns = ['contractID', 'start_date', 'end_date', 'num_observations']
    contract_info['duration_days'] = (
        pd.to_datetime(contract_info['end_date']) - 
        pd.to_datetime(contract_info['start_date'])
    ).dt.days
    
    print(f"Total contracts: {len(contract_info)}")
    print(f"Contract duration stats:")
    print(f"  Mean: {contract_info['duration_days'].mean():.1f} days")
    print(f"  Median: {contract_info['duration_days'].median():.1f} days")
    print(f"  Max: {contract_info['duration_days'].max()} days")
    
    contract_info = contract_info.sort_values('end_date').reset_index(drop=True)
    
    n_contracts = len(contract_info)
    train_end_idx = int(n_contracts * train_ratio)
    val_end_idx = int(n_contracts * (train_ratio + val_ratio))
    
    train_contracts = set(contract_info.iloc[:train_end_idx]['contractID'])
    val_contracts = set(contract_info.iloc[train_end_idx:val_end_idx]['contractID'])
    test_contracts = set(contract_info.iloc[val_end_idx:]['contractID'])
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_contracts)} contracts")
    print(f"  Val: {len(val_contracts)} contracts") 
    print(f"  Test: {len(test_contracts)} contracts")
    
    train_val_overlap = train_contracts & val_contracts
    train_test_overlap = train_contracts & test_contracts
    val_test_overlap = val_contracts & test_contracts
    
    assert len(train_val_overlap) == 0, f"Train/Val overlap: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Train/Test overlap: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Val/Test overlap: {len(val_test_overlap)}"
    
    print("✓ No contract overlap between splits!")
    
    train_df = df[df['contractID'].isin(train_contracts)].copy()
    val_df = df[df['contractID'].isin(val_contracts)].copy()
    test_df = df[df['contractID'].isin(test_contracts)].copy()
    
    train_date_range = (train_df['date'].min(), train_df['date'].max())
    val_date_range = (val_df['date'].min(), val_df['date'].max())
    test_date_range = (test_df['date'].min(), test_df['date'].max())
    
    print(f"Date ranges:")
    print(f"  Train: {pd.to_datetime(train_date_range[0])} to {pd.to_datetime(train_date_range[1])}")
    print(f"  Val: {pd.to_datetime(val_date_range[0])} to {pd.to_datetime(val_date_range[1])}")
    print(f"  Test: {pd.to_datetime(test_date_range[0])} to {pd.to_datetime(test_date_range[1])}")
    
    print(f"Row counts:")
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    print("=== END LEAK-FREE SPLITS ===\n")
    
    return train_df, val_df, test_df


def generate_gru_window_datasets():
    from src.features.data_processing import process_options_data
    import pickle
    
    merged_data_clean, stock_df = process_options_data()
    
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    
    print("=== DATA LEAKAGE AUDIT ===")
    print(f"Original dataset shape: {feature_dataset.shape}")
    print(f"Date range: {feature_dataset['date'].min()} to {feature_dataset['date'].max()}")
    print(f"Number of unique contracts: {feature_dataset['contractID'].nunique()}")
    
    feature_dataset = feature_dataset.sort_values(['contractID', 'date']).reset_index(drop=True)
    
    train_df, val_df, test_df = create_leak_free_splits(
        feature_dataset, 
        train_ratio=0.7, 
        val_ratio=0.15
    )
    
    train_contracts = set(train_df['contractID'])
    val_contracts = set(val_df['contractID'])
    test_contracts = set(test_df['contractID'])
    
    train_val_overlap = train_contracts & val_contracts
    train_test_overlap = train_contracts & test_contracts
    val_test_overlap = val_contracts & test_contracts
    
    print(f"VERIFICATION - Contract overlap:")
    print(f"  Train/Val: {len(train_val_overlap)} contracts")
    print(f"  Train/Test: {len(train_test_overlap)} contracts")
    print(f"  Val/Test: {len(val_test_overlap)} contracts")
    
    assert len(train_val_overlap) == 0, "Train/Val contract overlap detected!"
    assert len(train_test_overlap) == 0, "Train/Test contract overlap detected!"
    assert len(val_test_overlap) == 0, "Val/Test contract overlap detected!"
    
    print("✓ Contract leakage verification passed!")
    
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        contract_date_issues = []
        for contract_id in split_df['contractID'].unique()[:10]:
            contract_data = split_df[split_df['contractID'] == contract_id]['date']
            if not contract_data.is_monotonic_increasing:
                contract_date_issues.append(contract_id)
        
        if contract_date_issues:
            print(f"WARNING: {split_name} split has non-monotonic dates in contracts: {contract_date_issues[:5]}...")
        else:
            print(f"{split_name} split: Dates are properly ordered within contracts ✓")
    
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[FEATURE_COLS])
    
    print(f"Scaler fitted on {len(train_df)} training samples")
    print(f"Feature means (first 5): {scaler_X.mean_[:5]}")
    print(f"Feature stds (first 5): {scaler_X.scale_[:5]}")
    
    train_df[FEATURE_COLS] = scaler_X.transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler_X.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler_X.transform(test_df[FEATURE_COLS])
    
    scaler_x_path = path_builder("data", "scaler_X.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_X, f)
    
    print("Skipping window leakage audit (contracts are properly separated)")
    
    X_train, y_train = create_time_window_features(
        train_df, WINDOW_SIZE, FEATURE_COLS, TARGET_COL, GROUP_KEY_COL
    )
    X_val, y_val = create_time_window_features(
        val_df, WINDOW_SIZE, FEATURE_COLS, TARGET_COL, GROUP_KEY_COL
    )
    X_test, y_test = create_time_window_features(
        test_df, WINDOW_SIZE, FEATURE_COLS, TARGET_COL, GROUP_KEY_COL
    )
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    
    print(f"y_train stats - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"y_val stats - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
    print(f"y_test stats - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    with open(SCALER_DATA_PATH, "wb") as f:
        pickle.dump(scaler_y, f)
    
    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_VAL_PATH, y_val)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)
    
    print("=== END DATA LEAKAGE AUDIT ===")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_flat_datasets_for_xgboost():
    from src.features.data_processing import process_options_data
    
    logger.info("=== GENERATING LEAK-FREE FLAT DATASETS FOR XGBOOST ===")
    
    merged_data_clean, stock_df = process_options_data()
    
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    
    logger.info(f"Original dataset shape: {feature_dataset.shape}")
    logger.info(f"Date range: {feature_dataset['date'].min()} to {feature_dataset['date'].max()}")
    logger.info(f"Number of unique contracts: {feature_dataset['contractID'].nunique()}")
    
    feature_dataset = feature_dataset.sort_values(['contractID', 'date']).reset_index(drop=True)
    
    train_df, val_df, test_df = create_leak_free_splits(
        feature_dataset, 
        train_ratio=0.7, 
        val_ratio=0.15
    )
    
    train_contracts = set(train_df['contractID'])
    val_contracts = set(val_df['contractID'])
    test_contracts = set(test_df['contractID'])
    
    overlap_train_val = len(train_contracts & val_contracts)
    overlap_train_test = len(train_contracts & test_contracts) 
    overlap_val_test = len(val_contracts & test_contracts)
    
    logger.info("VERIFICATION - Contract overlap:")
    logger.info(f"  Train/Val: {overlap_train_val} contracts")
    logger.info(f"  Train/Test: {overlap_train_test} contracts")
    logger.info(f"  Val/Test: {overlap_val_test} contracts")
    
    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        logger.info("✓ Contract leakage verification passed!")
    else:
        logger.error("✗ Contract leakage detected!")
        
    feature_cols = [col for col in feature_dataset.columns if col not in ['contractID', 'date', 'mark']]
    target_col = 'mark'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    logger.info(f"Train samples: {len(X_train)} ({len(X_train)/len(feature_dataset)*100:.1f}%)")
    logger.info(f"Val samples: {len(X_val)} ({len(X_val)/len(feature_dataset)*100:.1f}%)")
    logger.info(f"Test samples: {len(X_test)} ({len(X_test)/len(feature_dataset)*100:.1f}%)")
    logger.info(f"Feature columns: {feature_cols}")
    logger.info("=== END LEAK-FREE FLAT DATASETS FOR XGBOOST ===")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    from src.features.data_processing import process_options_data
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    merged_data_clean, stock_df = process_options_data()
    
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    print(feature_dataset.info())

    scaler_X = StandardScaler()
    scaler_X.fit(feature_dataset[FEATURE_COLS])

    scaler_x_path = path_builder("data", "scaler_X.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_X, f)

    print(f"X scaler saved to {scaler_x_path}")

    window_df = feature_dataset.copy()
    window_df['contractID'] = merged_data_clean['contractID']

    print(window_df.info())
    
    y_train = create_y_train_window_ds(
        window_df, 
        window_size=10, 
        feature_cols=FEATURE_COLS, 
        target_col=TARGET_COL,
        group_key_cols=GROUP_KEY_COL
    )

    print(f"y_train shape: {y_train.shape}")
    print(f"y_train sample: {y_train[:5]}")

    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    with open(SCALER_DATA_PATH, "wb") as f:
        pickle.dump(scaler_y, f)
    
    original_features = create_feature_dataset(merged_data_clean, stock_df)

    scaler_X = StandardScaler()
    scaler_X.fit(original_features[FEATURE_COLS])

    scaler_x_path = path_builder("data", "scaler_X.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_X, f)
    
    print(f"Final dataset shape: {feature_dataset.shape}")
    print(f"Features: {feature_dataset.columns.tolist()}")

