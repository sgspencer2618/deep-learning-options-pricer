import os
from dotenv import load_dotenv
import requests
import json

# This script fetches historical options data for a given stock symbol using the Alpha Vantage API.
load_dotenv()
AV_API_KEY = os.getenv("AV_API_KEY")

def fetch_historical_options_data(symbol: str, day: str):
    """
    Fetch historical options data for a given stock symbol and day using the Alpha Vantage API.
    Args:
        symbol (str): The stock symbol to fetch data for.
        day (str): The date for which to fetch the data in 'YYYY-MM-DD' format.
    """
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={day}&apikey={AV_API_KEY}&datatype=csv'
    r = requests.get(url)
    # Check if the request was successful
    if r.status_code == 200:
        # Create the data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save the response content to a CSV file
        filename = f'data/{symbol}_{day}_options.csv'
        with open(filename, 'w') as file:
            file.write(r.text)
        print(f"Data saved to {filename}")
        return True
    else:
        print(f"Error: {r.status_code}")
        return False