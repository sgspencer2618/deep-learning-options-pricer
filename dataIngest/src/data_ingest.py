import sys
import os
# Add parent directory to path to access scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_historical import fetch_historical_options_data, get_last_n_trading_days, get_last_n_trading_days_starting
import pandas as pd
from helpers.s3_helper import S3Uploader

def parse_options_data(symbol: str, day: str):
    """
    Parse options data for a given stock symbol and day.
    Args:
        symbol (str): The stock symbol to parse data for.
        day (str): The date for which to parse the data in 'YYYY-MM-DD' format.
    """
    success = fetch_historical_options_data(symbol, day)
    if not success:
        print(f"Failed to fetch data for {symbol} on {day}")
        return
    
    options_df = pd.read_csv(f'data/{symbol}_{day}_options.csv')
    return options_df


def upload_options_data_to_s3(symbol: str, days: int) -> bool:
    """
    Upload options data DataFrame to S3 with organized folder structure.
    Args:
        symbol (str): Stock symbol
        days (int): Number of trading days to fetch data for (n-previous trading days)
        df (pd.DataFrame): Options data DataFrame"""
    uploader = S3Uploader()
    
    # get trading days from today
    # trading_days = get_last_n_trading_days(days)

    # get trading days from a specific date
    trading_days = get_last_n_trading_days_starting(days, end_date="2025-04-07")

    if trading_days is None or len(trading_days) == 0:
        print(f"No trading days available for days = {days}.")
        return False

    for date in trading_days:
        try:
            # Create organized S3 key structure
            s3_key = f"options-data/{symbol}/{symbol}_{date}_options.parquet"

            # Fetch and parse options data for the given date
            df = parse_options_data(symbol, date)
            success = uploader.upload_dataframe_as_parquet(df, s3_key)
        except Exception as e:
            print(f"Error processing {symbol} on {date}: {str(e)}")
            return False
        
    df_test = uploader.read_parquet_from_s3(f"options-data/{symbol}/{symbol}_2025-06-09_options.parquet")
    print(f"Test read from S3 successful: {df_test is not None}")
    if df_test is None:
        print("Failed to read test data from S3.")
        return False

    df_test.to_csv('filename.csv')

    return True


# df = pd.concat([
#     pd.read_csv(f"s3://your-bucket/daily/data_{d}.csv", storage_options=creds)
#     for d in dates
# ], ignore_index=True)


if __name__ == "__main__":
    symbol = "AAPL"
    days = 8
    success = upload_options_data_to_s3(symbol, days)
    print(f"Upload success code: {success}")