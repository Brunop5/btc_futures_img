import requests
import pandas as pd
from datetime import datetime

def fetch_btc_futures_ohlc(interval='1h', limit=500):
    """
    Fetch OHLC data for BTC/USDT futures from Binance
    
    Parameters:
    interval (str): Timeframe interval. Valid values: 
                   '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    limit (int): Number of candles to fetch (max 1500)
    
    Returns:
    pandas.DataFrame: DataFrame with OHLC data
    """
    
    # Binance Futures API endpoint
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"
    
    # Parameters for the request
    params = {
        'symbol': 'BTCUSDT',
        'interval': interval,
        'limit': limit
    }
    
    try:
        # Make the request
        response = requests.get(f"{base_url}{endpoint}", params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Convert response to DataFrame
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                       'volume', 'close_time', 'quote_volume', 'trades',
                                       'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string values to float for price and volume columns
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Keep only the relevant columns
        df = df[['open', 'high', 'low', 'close']]
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    # Fetch last 100 hourly candles
    df = fetch_btc_futures_ohlc(interval='1h', limit=200)
    df.to_csv("1h_test.csv")
    df = fetch_btc_futures_ohlc(interval='4h', limit=200)
    df.to_csv("4h_test.csv")
    df = fetch_btc_futures_ohlc(interval='15m', limit=200)
    df.to_csv("15m_test.csv")
    df = fetch_btc_futures_ohlc(interval='1d', limit=100)
    df.to_csv("1d_test.csv")