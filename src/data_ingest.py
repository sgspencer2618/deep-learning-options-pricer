import sys
import os
# Add parent directory to path to access scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_historical import fetch_historical_options_data
import pandas as pd

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

if __name__ == "__main__":
    symbol = "AAPL"
    day = "2017-11-15"
    options_data = parse_options_data(symbol, day)
    
    if options_data is not None:
        print(options_data.head())  # Display the first few rows of the options data
    else:
        print("No data available.")