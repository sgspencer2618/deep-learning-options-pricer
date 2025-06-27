import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from src.utils import path_builder

logger = logging.getLogger(__name__)

# Constants
ROLLING_VOL_WINDOW = 5
MA_SHORT = 10
MA_LONG = 50
RSI_WINDOW = 14

def add_stock_indicators(stock_df):
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