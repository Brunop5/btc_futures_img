import requests
import pandas as pd
from datetime import datetime

def fetch_btc_futures_ohlc(interval='1h', start_time=None, end_time=None):
    """
    Fetch OHLC data for BTC/USDT futures from Binance
    
    Parameters:
    interval (str): Timeframe interval
    start_time (int): Start time in milliseconds timestamp
    end_time (int): End time in milliseconds timestamp
    
    Returns:
    pandas.DataFrame: DataFrame with OHLC data
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': interval,
        'limit': 1000  # Maximum allowed by Binance
    }
    
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params)
        response.raise_for_status()
        
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

def fetch_historical_data(interval='1h', months_back=12):
    """
    Fetch several months of historical data by making multiple API calls
    
    Parameters:
    interval (str): Timeframe interval
    months_back (int): Number of months of historical data to fetch
    
    Returns:
    pandas.DataFrame: Concatenated DataFrame with all historical data
    """
    end_time = int(datetime.now().timestamp() * 1000)
    # Calculate start time (months_back months ago)
    start_time = int((datetime.now().timestamp() - (months_back * 30 * 24 * 60 * 60)) * 1000)
    
    all_df = []
    current_start = start_time
    
    while current_start < end_time:
        # Calculate next end time (1000 candles ahead)
        if interval == '1h':
            next_end = current_start + (1000 * 60 * 60 * 1000)  # 1000 hours in milliseconds
        elif interval == '4h':
            next_end = current_start + (1000 * 4 * 60 * 60 * 1000)
        elif interval == '1d':
            next_end = current_start + (1000 * 24 * 60 * 60 * 1000)
        else:
            raise ValueError("Unsupported interval")
            
        next_end = min(next_end, end_time)
        
        print(f"Fetching data from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(next_end/1000)}")
        
        df = fetch_btc_futures_ohlc(interval=interval, start_time=current_start, end_time=next_end)
        if df is not None and not df.empty:
            all_df.append(df)
        
        current_start = next_end
        
    if all_df:
        final_df = pd.concat(all_df)
        final_df = final_df[~final_df.index.duplicated(keep='first')]  # Remove any duplicates
        final_df.sort_index(inplace=True)
        return final_df
    return None

# Example usage:
if __name__ == "__main__":
    # Fetch 12 months of hourly data
    df = fetch_historical_data(interval='1h', months_back=100)
    if df is not None:
        print(f"Fetched {len(df)} candles")
        df.to_csv("1h_test2.csv")