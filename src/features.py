import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from utils import path_builder

logger = logging.getLogger(__name__)

# Constants
ROLLING_VOL_WINDOW = 5
MA_SHORT = 10
MA_LONG = 50
RSI_WINDOW = 14

features_data_path = path_builder("data", "features_data.parquet")

def add_option_features(df):
    """
    Add option-specific features
    
    Args:
        df (pd.DataFrame): Clean options data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    # Create an explicit copy to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Calculating log moneyness
    df.loc[:, 'log_moneyness'] = np.log(df['close'] / df['strike'])
    
    # Encoding the option type
    df.loc[:, 'type'] = df['type'].map({'call': 1, 'put': 0})
    
    # Adding column for intrinsic value
    df.loc[:, 'intrinsic_value'] = np.where(
        (df['type'] == 1),
        np.maximum(0, df['close'] - df['strike']),
        np.maximum(0, df['strike'] - df['close'])
    )
    
    # Binary flag for in-the-money options
    df.loc[:, 'is_itm'] = np.where(
        df['intrinsic_value'] > 0,
        True,
        False
    )
    
    # Time to maturity in years
    df.loc[:, 'time_to_maturity'] = (df['expiration'] - df['date']).dt.days / 365
    df = df[df['time_to_maturity'] > 0]
    
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


def add_stock_features(stock_df):
    """
    Add stock-specific features
    
    Args:
        stock_df (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: DataFrame with added stock features
    """
    # Calculate log returns
    stock_df['log_return'] = np.log(stock_df['close'] / stock_df['close'].shift(1))
    
    # Rolling volatility
    stock_df[f'vol_{ROLLING_VOL_WINDOW}d'] = stock_df['log_return'].rolling(ROLLING_VOL_WINDOW).std()
    
    # Moving averages
    stock_df['ma_short'] = stock_df['close'].rolling(MA_SHORT).mean()
    stock_df['ma_long'] = stock_df['close'].rolling(MA_LONG).mean()
    
    # Crossover signal
    stock_df['ma_signal'] = (stock_df['ma_short'] > stock_df['ma_long']).astype(int)
    
    # RSI
    stock_df[f'rsi_{RSI_WINDOW}'] = compute_rsi(stock_df)
    
    # RSI signals
    stock_df['rsi_overbought'] = (stock_df[f'rsi_{RSI_WINDOW}'] > 70).astype(int)
    stock_df['rsi_oversold'] = (stock_df[f'rsi_{RSI_WINDOW}'] < 30).astype(int)
    
    # Normalize RSI
    stock_df[f'rsi_{RSI_WINDOW}_norm'] = MinMaxScaler().fit_transform(stock_df[[f'rsi_{RSI_WINDOW}']])
    
    # Drop rows with NaN values (from rolling calculations)
    stock_df.dropna(inplace=True)
    
    return stock_df


def compute_rsi(df, window=RSI_WINDOW):
    """
    Compute Relative Strength Index
    
    Args:
        df (pd.DataFrame): DataFrame with close prices
        window (int): RSI window period
        
    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = df['close'].diff()
    
    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Rolling averages
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # RS and RSI
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


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
        return pd.read_parquet(features_data_path)
    

    # Add option features
    options_with_features = add_option_features(options_data)
    
    # Normalize option features
    options_normalized = normalize_features(options_with_features)
    
    # Add stock features
    stock_with_features = add_stock_features(stock_data)
    
    # Merge options and stock features
    final_dataset = merge_option_and_stock_features(options_normalized, stock_with_features)

    final_dataset.to_parquet(features_data_path, index=False)
    logger.info(f"Feature dataset saved to {features_data_path}")
    
    return final_dataset


if __name__ == "__main__":
    # This section would only run when this file is executed directly
    from data_processing import process_options_data
    
    # Get clean data
    merged_data_clean, stock_df = process_options_data()
    
    # Create feature dataset
    feature_dataset = create_feature_dataset(merged_data_clean, stock_df)
    
    print(f"Final dataset shape: {feature_dataset.shape}")
    print(f"Features: {feature_dataset.columns.tolist()}")