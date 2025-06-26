import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from dataIngest.helpers.s3_helper import S3Uploader
from utils import path_builder

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Build the absolute path to the combined data file
combined_data_file_path = path_builder("data", "combined_options_data.parquet")

def load_all_parquet_files(prefix="options-data/AAPL/"):
    """
    Load all parquet files from S3 with the given prefix into a single DataFrame
    
    Args:
        prefix (str): The S3 prefix to search for parquet files
        
    Returns:
        pd.DataFrame: Combined DataFrame of all parquet files
    """
    if os.path.exists(combined_data_file_path):
        logger.info(f"Combined data file already exists at {combined_data_file_path}. Loading from file.")
        return pd.read_parquet(combined_data_file_path)
    
    uploader = S3Uploader()
    
    # List all objects with the given prefix
    response = uploader.s3_client.list_objects_v2(
        Bucket=uploader.bucket_name,
        Prefix=prefix
    )
    
    # Filter for parquet files
    all_keys = []
    if 'Contents' in response:
        all_keys = [item['Key'] for item in response['Contents'] 
                   if item['Key'].endswith('.parquet')]
    
    logger.info(f"Found {len(all_keys)} parquet files")
    
    # Load each file and append to a list
    dataframes = []
    for i, key in enumerate(all_keys):
        logger.info(f"Loading file {i+1}/{len(all_keys)}: {key}")
        try:
            df = uploader.read_parquet_from_s3(key)
            if df is not None and not df.empty:
                dataframes.append(df)
        except Exception as e:
            logger.error(f"Error loading {key}: {str(e)}")
    
    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined DataFrame has {len(combined_df)} rows")
        combined_df.to_parquet(combined_data_file_path, index=False)
        return combined_df
    else:
        logger.warning("No data was loaded")
        return pd.DataFrame()


def load_stock_data(file_path):
    """
    Load stock data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with stock data
    """
    if os.path.exists(file_path):
        stock_data = pd.read_csv(file_path, parse_dates=['timestamp'])
        stock_data = stock_data.rename(columns={'timestamp': 'date'})
        return stock_data
    else:
        logger.warning(f"File not found: {file_path}. Skipping stock data load.")
        return None


def merge_options_with_stock_data(options_data, stock_data):
    """
    Merge options data with stock data
    
    Args:
        options_data (pd.DataFrame): Options data
        stock_data (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: Merged data
    """
    if stock_data is not None and not options_data.empty:
        options_data['date'] = pd.to_datetime(options_data['date'], errors='coerce')
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        merged_data = pd.merge(options_data, stock_data, on='date', how='inner')
        logger.info(f"Merged DataFrame has {len(merged_data)} rows")
        return merged_data
    else:
        logger.warning("Cannot merge data: missing options or stock data")
        return pd.DataFrame()


def clean_options_data(merged_data):
    """
    Clean the options data by handling missing values and formatting
    
    Args:
        merged_data (pd.DataFrame): Merged options and stock data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Convert expiration to datetime
    merged_data['expiration'] = pd.to_datetime(merged_data['expiration'], errors='coerce')
    
    # Drop rows with missing important values
    clean_data = merged_data.dropna(subset=['date', 'type', 'expiration', 'strike', 
                                            'implied_volatility', 'delta', 'gamma', 
                                            'theta', 'vega', 'rho'])
    
    # Drop problematic columns if they exist
    if '{' in clean_data.columns:
        clean_data = clean_data.drop(columns=['{'])
    
    return clean_data


# Build the absolute path to the data file
stock_file_path = path_builder("data", "AAPL_OHLC.csv")

def process_options_data(options_prefix="options-data/AAPL/", stock_file_path=stock_file_path):
    """
    Main function to process options data
    
    Args:
        options_prefix (str): S3 prefix for options data
        stock_file_path (str): Path to stock data file
        
    Returns:
        pd.DataFrame: Processed options data
    """
    # Load options data
    options_data = load_all_parquet_files(prefix=options_prefix)
    
    # Load stock data
    stock_data = load_stock_data(stock_file_path)
    
    # Merge data
    merged_data = merge_options_with_stock_data(options_data, stock_data)
    
    # Clean data
    clean_data = clean_options_data(merged_data)
    
    return clean_data, stock_data


if __name__ == "__main__":
    # Example usage
    merged_data_clean, stock_df = process_options_data()
    print(f"Processed data shape: {merged_data_clean.shape}")