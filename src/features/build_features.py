import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from src.utils import path_builder
from src.features.indicators import add_stock_indicators
from src.model.config import FEATURE_COLS, TARGET_COL, GROUP_KEY_COL, WINDOW_SIZE, SCALING_COLS, SCALED_FEATURE_DATA_PATH, SCALER_DATA_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
ROLLING_VOL_WINDOW = 5
MA_SHORT = 10
MA_LONG = 50
RSI_WINDOW = 14

features_data_path = path_builder("data", "features_data.parquet")
X_train_path = path_builder("src\\neural", "X_train.npy")
y_train_path = path_builder("src\\neural", "y_train.npy")
X_val_path = path_builder("src\\neural", "X_val.npy")
y_val_path = path_builder("src\\neural", "y_val.npy")
X_test_path = path_builder("src\\neural", "X_test.npy")
y_test_path = path_builder("src\\neural", "y_test.npy")

def add_option_features(df):
    """
    Add option-specific features
    
    Args:
        df (pd.DataFrame): Clean options data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    df = df.copy()
    # Shift implied volatility by one day within each contract
    df['implied_volatility'] = df.groupby('contractID')['implied_volatility'].shift(1)

    # Add IV shift
    df['iv_change'] = df.groupby('contractID')['implied_volatility'].diff()

    # Shift close and greeks by one day within each contract
    df['close_prev'] = df.groupby('contractID')['close'].shift(1)
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        df[greek] = df.groupby('contractID')[greek].shift(1)

    # Calculate log moneyness using previous day's close
    df['log_moneyness'] = np.log(df['close_prev'] / df['strike'])

    # Encode option type
    df['type'] = df['type'].map({'call': 1, 'put': 0})

    # Intrinsic value and ITM flag using previous close
    df['intrinsic_value'] = np.where(
        df['type'] == 1,
        np.maximum(0, df['close_prev'] - df['strike']),
        np.maximum(0, df['strike'] - df['close_prev'])
    )

    df['is_itm'] = df['intrinsic_value'] > 0

    # Time to maturity (safe)
    df['time_to_maturity'] = (df['expiration'] - df['date']).dt.days / 365
    df = df[df['time_to_maturity'] > 0]

    # Drop rows with NaNs (first row of each contract)
    df = df.dropna(subset=['close_prev', 'log_moneyness', 'delta', 'gamma', 'theta', 'vega', 'rho', 'iv_change'])
    return df


def normalize_features(df):
    """
    Normalize option features
    
    Args:
        df (pd.DataFrame): DataFrame with option features
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    # Create an explicit copy to avoid the SettingWithCopyWarning
    df = df.copy()
    
    scaler = StandardScaler()
    
    # Normalize the moneyness and greeks using .loc
    df.loc[:, ['log_moneyness_norm', 'intrinsic_value_norm']] = scaler.fit_transform(
        df[['log_moneyness', 'intrinsic_value']]
    )
    
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    df.loc[:, greeks] = scaler.fit_transform(df[greeks])
    
    return df


def merge_option_and_stock_features(option_df, stock_df):
    """
    Merge option features with stock features
    
    Args:
        option_df (pd.DataFrame): Options data with features
        stock_df (pd.DataFrame): Stock data with features
        
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    merged_df = pd.merge(option_df, stock_df, on='date', how='inner')
    
    # Clean up column names for clarity
    merged_df = merged_df.rename(columns={
        'volume_x': 'option_volume',
        'volume_y': 'stock_volume',
        'type': 'option_type_encoded',
    })
    
    return merged_df

def add_greek_product_features(df):
    """
    Add product features based on the greeks and other option metrics
    """
    df = df.copy()

    # Calculate product features
    df["delta_x_iv"] = df["delta"] * df["implied_volatility"]
    df["vega_x_ttm"] = df["vega"] * df["time_to_maturity"]
    df["gamma_x_logm"] = df["gamma"] * df["log_moneyness_norm"]
    df["theta_x_intrinsic"] = df["theta"] * df["intrinsic_value_norm"]

    return df


def create_feature_dataset(options_data, stock_data):
    """
    Create a complete feature dataset from options and stock data
    
    Args:
        options_data (pd.DataFrame): Clean options data
        stock_data (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: Complete feature dataset
    """
    if os.path.exists(features_data_path):
        logger.info(f"Feature dataset already exists at {features_data_path}. Loading existing data.")
        final_dataset = pd.read_parquet(features_data_path)
        print(final_dataset.info())
        return final_dataset
    

    # Add option features
    options_with_features = add_option_features(options_data)
    
    # Normalize option features
    options_normalized = normalize_features(options_with_features)
    
    # Add stock features
    stock_with_features = add_stock_indicators(stock_data)

    # Add product features based on greeks
    options_with_greek_features = add_greek_product_features(options_normalized)
    
    # Merge options and stock features
    final_dataset = merge_option_and_stock_features(options_with_greek_features, stock_with_features)

    # Ensure numeric types for date and expiration
    final_dataset['date'] = pd.to_numeric(final_dataset['date'])
    final_dataset['expiration'] = pd.to_numeric(final_dataset['expiration'])
    # final_dataset.drop('contractID', axis=1, inplace=True)
    # final_dataset['symbol'] = final_dataset['symbol'].astype('category')
    final_dataset = final_dataset[FEATURE_COLS + [TARGET_COL]]
    # final_dataset.drop('symbol', axis=1, inplace=True)

    final_dataset.to_parquet(features_data_path, index=False)
    logger.info(f"Feature dataset saved to {features_data_path}")
    
    return final_dataset


def create_time_window_features(
    df, 
    window_size, 
    feature_cols, 
    target_col, 
    group_key_cols=None, 
    date_col='date',
    min_window_size=None
):
    """
    Create sliding window features for time series modeling with progress bar.
    
    Args:
        df (pd.DataFrame): Time-ordered DataFrame with all engineered features
        window_size (int): Number of lookback steps per sample
        feature_cols (list): List of column names to use as features
        target_col (str): Column name for the target variable
        group_key_cols (list, optional): Columns that identify unique entities 
                                         (e.g., contractID, option type, strike)
        date_col (str, optional): Column name for the date field
        min_window_size (int, optional): Minimum window size to accept (default: window_size)
                                         
    Returns:
        tuple: (X, y) where:
            - X is a 3D numpy array with shape [num_samples, window_size, num_features]
            - y is a 1D numpy array with shape [num_samples] containing target values
    """
    if min_window_size is None:
        min_window_size = window_size
    
    logger.info(f"Creating time window features with window size {window_size}")
    logger.info(f"Using {len(feature_cols)} features and target column '{target_col}'")
    
    # Check if the required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    # Ensure df is sorted by date and group keys
    sort_cols = [date_col]
    if group_key_cols:
        sort_cols = group_key_cols + sort_cols
    
    logger.info(f"Sorting DataFrame by columns: {sort_cols}")
    df_sorted = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Initialize lists to store the results
    X_windows = []
    y_values = []
    
    # If we're grouping by contract/option
    if group_key_cols:
        # Get unique groups
        groups = df_sorted.groupby(group_key_cols)
        total_groups = len(groups)
        logger.info(f"Found {total_groups} unique groups based on {group_key_cols}")
        
        # Initialize counters for progress tracking
        processed_groups = 0
        skipped_groups = 0
        total_windows = 0
        
        # Create a progress bar for groups
        with tqdm(total=total_groups, desc="Processing groups") as pbar:
            for group_name, group_df in groups:
                if isinstance(group_name, tuple):
                    group_str = "_".join(str(x) for x in group_name)
                else:
                    group_str = str(group_name)
                
                # Sort by date within group
                group_df = group_df.sort_values(by=date_col).reset_index(drop=True)
                
                # Skip if not enough data points
                if len(group_df) < min_window_size:
                    skipped_groups += 1
                    pbar.set_postfix(
                        processed=processed_groups, 
                        skipped=skipped_groups,
                        windows=total_windows
                    )
                    pbar.update(1)
                    continue
                
                # Calculate number of windows for this group
                num_windows = len(group_df) - window_size + 1
                
                # Create sliding windows
                for i in range(num_windows):
                    window = group_df.iloc[i:i+window_size]
                    
                    # Extract features and target
                    X_window = window[feature_cols].values
                    y_value = window.iloc[-1][target_col]
                    
                    X_windows.append(X_window)
                    y_values.append(y_value)
                
                processed_groups += 1
                total_windows += num_windows
                
                # Update progress bar
                pbar.set_postfix(
                    processed=processed_groups, 
                    skipped=skipped_groups,
                    windows=total_windows
                )
                pbar.update(1)
    else:
        # Process the entire dataset as one time series
        # Skip if not enough data points
        if len(df_sorted) < min_window_size:
            logger.warning(f"Not enough data points ({len(df_sorted)} < {min_window_size})")
            return np.array([]), np.array([])
        
        # Calculate total number of windows
        num_windows = len(df_sorted) - window_size + 1
        
        # Create a progress bar for windows
        with tqdm(total=num_windows, desc="Creating windows") as pbar:
            for i in range(num_windows):
                window = df_sorted.iloc[i:i+window_size]
                
                # Extract features and target
                X_window = window[feature_cols].values
                y_value = window.iloc[-1][target_col]
                
                X_windows.append(X_window)
                y_values.append(y_value)
                pbar.update(1)
    
    # Convert lists to numpy arrays
    X = np.array(X_windows)
    y = np.array(y_values)
    
    logger.info(f"Created {len(X)} samples with shape {X.shape}")
    return X, y

### THROWAWAY FUNC

def create_y_train_window_ds(df, 
    window_size, 
    feature_cols, 
    target_col, 
    group_key_cols=None,
    train_ratio=0.7,
    val_ratio=0.15,
    date_col='date',
    min_window_size=None):

    # Get unique dates in chronological order
    unique_dates = df[date_col].sort_values().unique()
    n_dates = len(unique_dates)
    
    # Calculate split indices
    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = unique_dates[:train_idx]
    
    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    
    # Filter data by date ranges
    train_df = df[df[date_col].isin(train_dates)]
    
    # Create window features for each split
    X_train, y_train = create_time_window_features(
        train_df, window_size, feature_cols, target_col, 
        group_key_cols, date_col, min_window_size
    )

    return y_train


###############

def create_time_window_dataset(
    df, 
    window_size, 
    feature_cols, 
    target_col, 
    group_key_cols=None,
    train_ratio=0.7,
    val_ratio=0.15,
    date_col='date',
    min_window_size=None
):
    """
    Create train/validation/test splits for time series modeling.
    
    Args:
        df (pd.DataFrame): Time-ordered DataFrame with all engineered features
        window_size (int): Number of lookback steps per sample
        feature_cols (list): List of column names to use as features
        target_col (str): Column name for the target variable
        group_key_cols (list, optional): Columns that identify unique entities
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        date_col (str): Column name for the date field
        min_window_size (int, optional): Minimum window size to accept
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Get unique dates in chronological order
    unique_dates = df[date_col].sort_values().unique()
    n_dates = len(unique_dates)
    
    # Calculate split indices
    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    
    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    logger.info(f"Val dates: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} dates)")
    logger.info(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")
    
    # Filter data by date ranges
    train_df = df[df[date_col].isin(train_dates)]
    val_df = df[df[date_col].isin(val_dates)]
    test_df = df[df[date_col].isin(test_dates)]
    
    # Create window features for each split
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

    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(X_val_path, X_val)
    np.save(y_val_path, y_val)
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def standardize_and_scale_features(df, feature_cols=SCALING_COLS, exclude_cols=None, scale_method='z-score'):
    """
    Standardize and scale features using Z-score normalization or Min-Max scaling.
    
    Args:
        df (pd.DataFrame): DataFrame with features to normalize
        feature_cols (list, optional): List of columns to normalize. If None, normalizes all numeric columns.
        exclude_cols (list, optional): List of columns to exclude from normalization
        scale_method (str, optional): Method to use for scaling. Options: 'z-score', 'minmax'
        
    Returns:
        pd.DataFrame: DataFrame with standardized features
    """
    import pickle
    import os
    
    df = df.copy()
    
    # If no specific feature columns are provided, use all numeric columns
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Exclude any columns specified
    if exclude_cols:
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    logger.info(f"Standardizing {len(feature_cols)} features using {scale_method} method")
    
    # Create a DataFrame with just the features to standardize
    features_df = df[feature_cols]
    
    if scale_method == 'z-score':
        # Use StandardScaler for Z-score normalization (μ=0, σ=1)
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        # Use MinMaxScaler for scaling to [0, 1] range
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scale_method}. Use 'z-score' or 'minmax'")
    
    # Fit and transform the features
    scaled_features = scaler.fit_transform(features_df)
    
    # Create a new DataFrame with the same column names
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
    
    # Replace the original columns with the scaled versions
    for col in feature_cols:
        df.loc[:, col] = scaled_df[col]
    
    # Add information about what was scaled
    metadata = {
        'scaled_columns': feature_cols,
        'scaling_method': scale_method,
        'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'std': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        'max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None
    }
    
    # Save the scaler to a pickle file
    scaler_path = SCALER_DATA_PATH
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Both X and y scalers saved to {scaler_path}")
    
    # Save the X scaler to a separate pickle file
    scaler_x_path = path_builder("data", "scaler_X.pkl")
    
    with open(scaler_x_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"X scaler saved to {scaler_x_path}")
    
    logger.info(f"Standardization complete. Data shape: {df.shape}")
    df.to_parquet(SCALED_FEATURE_DATA_PATH, index=False)
    logger.info(f"Scaled features saved to {SCALED_FEATURE_DATA_PATH}")
    return df, metadata

def generate_gru_window_datasets():
    from src.data_processing import process_options_data
    
    # Get clean data
    merged_data_clean, stock_df = process_options_data()
    
    # Create feature dataset
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    scaled_features, scaling_metadata = standardize_and_scale_features(
        feature_dataset, 
        feature_cols=FEATURE_COLS, 
        scale_method='z-score'
    )

    logger.info(f"Feature scaling metadata: {scaling_metadata}")

    window_df = scaled_features.copy()
    window_df['contractID'] = merged_data_clean['contractID']

    print(window_df.info())

    X_train, y_train, X_val, y_val, X_test, y_test = create_time_window_dataset(
        window_df,
        window_size=10, 
        feature_cols=FEATURE_COLS, 
        target_col=TARGET_COL,
        group_key_cols=GROUP_KEY_COL
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == "__main__":
    # This section would only run when this file is executed directly
    from src.data_processing import process_options_data
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    # Get clean data
    merged_data_clean, stock_df = process_options_data()
    
    # Create feature dataset
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    # scaled_features, scaling_metadata = standardize_and_scale_features(
    # feature_dataset, 
    # feature_cols=FEATURE_COLS, 
    # exclude_cols=['contractID', 'date', 'expiration'], 
    # scale_method='z-score'
    # )
    print(feature_dataset.info())

    # Create and fit X scaler on original features
    scaler_X = StandardScaler()
    scaler_X.fit(feature_dataset[FEATURE_COLS])

    # Save X scaler to separate file
    scaler_x_path = path_builder("data", "scaler_X.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_X, f)

    print(f"X scaler saved to {scaler_x_path}")
    
    # print(scaling_metadata)

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

    # Save to disk
    with open(SCALER_DATA_PATH, "wb") as f:
        pickle.dump(scaler_y, f)
    
    # Load the non-standardized feature dataset
    original_features = create_feature_dataset(merged_data_clean, stock_df)

    # Create and fit X scaler on original features
    scaler_X = StandardScaler()
    scaler_X.fit(original_features[FEATURE_COLS])

    # Save X scaler to separate file
    scaler_x_path = path_builder("data", "scaler_X.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_X, f)
    
    print(f"Final dataset shape: {feature_dataset.shape}")
    print(f"Features: {feature_dataset.columns.tolist()}")

    # X, y = create_time_window_features(
    #     window_df, 
    #     window_size=10, 
    #     feature_cols=FEATURE_COLS, 
    #     target_col=TARGET_COL,
    #     group_key_cols=GROUP_KEY_COL
    # )

