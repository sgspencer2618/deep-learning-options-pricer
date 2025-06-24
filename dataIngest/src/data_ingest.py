import sys
import os
import logging
# Add parent directory to path to access scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_historical import fetch_historical_options_data, get_last_n_trading_days, get_last_n_trading_days_starting, fetch_historical_OHLC_data
import pandas as pd
from helpers.s3_helper import S3Uploader
from helpers.metadata_service import update_ingestion_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'scheduler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        return None
    
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
    
    # Try to read existing metadata to get the most recent date
    metadata_key = f"options-data/{symbol}/{symbol}_ingestion_metadata.json"
    end_date = "2025-01-26"  # Default fallback date
    
    try:
        # Check if metadata exists and get latest date
        metadata = uploader.read_json_from_s3(metadata_key)
        if metadata and metadata.get("latest_date"):
            # Use the latest date from metadata
            end_date = metadata["latest_date"]
            logger.info(f"Using latest date from metadata: {end_date}")
    except Exception as e:
        logger.info(f"No existing metadata found, using default date: {end_date}")
    
    # Get trading days starting from the end_date
    trading_days = get_last_n_trading_days_starting(days, end_date=end_date)

    if trading_days is None or len(trading_days) == 0:
        logger.info(f"No trading days available for days = {days}.")
        return False

    # Track successful uploads for metadata update
    successfully_uploaded_dates = []

    for date in trading_days:
        try:
            # Create organized S3 key structure
            logger.debug(f"Processing {symbol} for date {date}")
            s3_key = f"options-data/{symbol}/{symbol}_{date}_options.parquet"

            # Fetch and parse options data for the given date
            df = parse_options_data(symbol, date)
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol} on {date}. Skipping upload.")
                continue
            logger.debug(f"DataFrame for {symbol} on {date}:\n{df.head()}")
            success = uploader.upload_dataframe_as_parquet(df, s3_key)
            
            if success:
                successfully_uploaded_dates.append(date)
        except Exception as e:
            logger.error(f"Error processing {symbol} on {date}: {str(e)}")
            return False
    
    # Update metadata with the dates we successfully processed
    if successfully_uploaded_dates:
        try:
            update_ingestion_metadata(uploader, symbol, successfully_uploaded_dates)
            logger.info(f"Updated metadata with {len(successfully_uploaded_dates)} new dates")
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
    
    # Verify using the most recent file we uploaded, or fall back to the test file
    verify_key = f"options-data/{symbol}/{symbol}_2025-06-09_options.parquet"
    if successfully_uploaded_dates:
        # Use the most recent uploaded file for verification
        latest_date = sorted(successfully_uploaded_dates)[-1]
        verify_key = f"options-data/{symbol}/{symbol}_{latest_date}_options.parquet"
        
    df_test = uploader.read_parquet_from_s3(verify_key)
    logger.info(f"Test read from S3 successful: {df_test is not None}")
    if df_test is None:
        logger.error("Failed to read test data from S3.")
        return False

    df_test.to_csv('filename.csv')

    return True


# df = pd.concat([
#     pd.read_csv(f"s3://your-bucket/daily/data_{d}.csv", storage_options=creds)
#     for d in dates
# ], ignore_index=True)


if __name__ == "__main__":
    # symbol = "AAPL"
    # days = 5
    # success = upload_options_data_to_s3(symbol, days)
    # print(f"Upload success code: {success}")

    fetch_historical_OHLC_data("AAPL")