import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
# Add this line at the top with your other imports
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names*')


def image_to_edge_features(image_path):
    """Extracts edge features using Canny filter"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (700, 700))  # Resize for consistency
    edges = cv2.Canny(img, threshold1=100, threshold2=200)  # Apply edge detection
    return edges.flatten()


def reconstruct_image_from_edges(edge_features, img_size=(700, 700)):
    """
    Reconstructs an image from flattened edge features.
    
    Parameters:
    - edge_features: Flattened edge features (1D array)
    - img_size: Tuple (height, width) of the original image

    Returns:
    - Reconstructed image as a NumPy array
    """
    # Reshape the flattened array back into the 2D image format
    edge_image = np.reshape(edge_features, img_size)

    # Display the reconstructed image
    plt.imshow(edge_image, cmap='gray')
    plt.title("Reconstructed Image from Edges")
    plt.axis("off")
    plt.show()

    return edge_image


def save_chart(data, filename="chart.png", size=512):
    """
    Generates and saves a square candlestick chart from OHLC data.
    
    Parameters:
    - data: OHLC price data
    - filename: Output filename
    - size: Size in pixels for both width and height (default 512)
    """
    # Create a custom style with solid colors and visible edges
    mc = mpf.make_marketcolors(up='g',
                              down='r',
                              edge='inherit',
                              wick='inherit',
                              volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, 
                          gridstyle='',
                          y_on_right=False)
    
    # Create the figure with square dimensions
    fig, ax = mpf.plot(data, 
                      type='candle',
                      style=s,
                      volume=False,
                      returnfig=True,
                      figsize=(8, 8),  # Square figure
                      tight_layout=True)
    
    # Remove all axes, labels, and borders
    ax[0].set_axis_off()
    
    # Save with exact pixel dimensions
    plt.savefig(filename, 
                bbox_inches='tight', 
                pad_inches=0,
                dpi=size/8,  # Adjust DPI to get desired pixel size
                format='png')
    plt.close()
    
    # Verify the output size
    img = cv2.imread(filename)
    height, width, channels = img.shape
    print(f"Image size: {width}x{height} pixels")
    
    return width, height


def ohlc_to_edge_features(data, size=(700, 700)):
    """
    Converts OHLC data directly to edge features without generating an image
    
    Parameters:
    - data: DataFrame with OHLC data
    - size: Tuple of (height, width) for the feature matrix
    
    Returns:
    - Flattened edge features array
    """
    # Normalize price data to 0-1 range
    min_price = min(data['low'].min(), data['open'].min(), data['close'].min())
    max_price = max(data['high'].max(), data['open'].max(), data['close'].max())
    price_range = max_price - min_price
    
    # Create empty canvas
    canvas = np.zeros(size)
    
    # Calculate scaling factors
    time_scale = size[1] / len(data)
    price_scale = size[0] / price_range
    
    # Calculate candle width (make bodies thicker than wicks)
    candle_width = max(int(time_scale / 2), 1)  # Body width
    wick_width = max(int(candle_width / 2), 1)  # Wick width
    
    for i, row in enumerate(data.itertuples()):
        # Calculate x position (center of candle)
        x = int(i * time_scale)
        
        # Calculate y positions for OHLC (flip the y-coordinates)
        open_y = size[0] - int((row.open - min_price) * price_scale)
        high_y = size[0] - int((row.high - min_price) * price_scale)
        low_y = size[0] - int((row.low - min_price) * price_scale)
        close_y = size[0] - int((row.close - min_price) * price_scale)
        
        # Draw thicker candle body
        start_y = min(open_y, close_y)
        end_y = max(open_y, close_y)
        x_start = max(x - candle_width // 2, 0)
        x_end = min(x + candle_width // 2 + 1, size[1])
        canvas[start_y:end_y+1, x_start:x_end] = 255
        
        # Draw thinner wicks
        x_wick_start = max(x - wick_width // 2, 0)
        x_wick_end = min(x + wick_width // 2 + 1, size[1])
        
        # Upper wick (from top of body to high)
        canvas[high_y:end_y, x_wick_start:x_wick_end] = 255
        
        # Lower wick (from bottom of body to low)
        canvas[start_y:low_y, x_wick_start:x_wick_end] = 255
    
    # Apply edge detection
    edges = cv2.Canny(canvas.astype(np.uint8), threshold1=100, threshold2=200)
    
    return edges.flatten()


def add_target_column(df, bars_forward):
    df = df.copy()
    df['target'] = 0
    
    for i in range(len(df) - bars_forward):
        current_close = df.iloc[i]['close']
        next_highs = df.iloc[i+1:i+bars_forward+1]['high'].max()
        next_lows = df.iloc[i+1:i+bars_forward+1]['low'].min()
        
        if next_highs > current_close * 1.02:
            df.iloc[i, df.columns.get_loc('target')] = 1
        elif next_lows < current_close * 0.98:
            df.iloc[i, df.columns.get_loc('target')] = -1
    
    return df


def test(df, tp, model, lookback=100, size=(700, 700)):
    """
    Backtests the trading strategy using model predictions
    
    Parameters:
    - df: DataFrame with OHLC data
    - tp: Take profit percentage
    - model: Trained model for predictions
    - lookback: Number of candles to look back for feature generation
    - size: Size of the edge features image
    """
    usdt = 1000
    fee = 0.0005
    long = False
    short = False
    longs = 0
    shorts = 0
    wins = 0
    losses = 0
    equity_curve = []
    previous_usdt = usdt

    # We need at least lookback periods before we can start trading
    for i in range(lookback, len(df)):
        # Generate features for current candle
        window_data = df.iloc[i-lookback:i]
        edge_features = ohlc_to_edge_features(window_data, size=size)
        
        # Reshape features for prediction
        features = edge_features.reshape(1, -1)
        
        # Get model prediction
        prediction = model.predict(features)[0]
        
        current_row = df.iloc[i]
        
        # Check existing positions
        if long:
            if current_row['high'] >= long_entry*(1+tp):
                usdt += 100*(1+tp) - 100*(1+tp) * fee
                long = False
                wins += 1
            elif current_row['low'] <= long_entry*(1-tp/2):
                usdt += 100*(1-tp/2) - 100*(1-tp/2)*fee
                long = False
                losses += 1

        if short:
            if current_row['low'] <= short_entry*(1-tp):
                usdt += 100*(1+tp) - 100*(1+tp)*fee
                short = False
                wins += 1
            elif current_row['high'] >= short_entry*(1+tp/2):
                usdt += 100*(1-tp/2) - 100*(1-tp/2)*fee
                short = False
                losses += 1

        # Enter new positions based on model predictions
        if prediction == 1 and not long and usdt > 100:
            longs += 1
            long_entry = current_row['close']
            long = True
            usdt -= 100 + 100*fee

        if prediction == -1 and not short and usdt > 100:
            shorts += 1
            short_entry = current_row['close']
            short = True
            usdt -= 100 + 100*fee

        # Track equity
        if usdt != previous_usdt:
            equity_curve.append(usdt)
            print(f"Equity: ${usdt:.2f}")
            previous_usdt = usdt

    print(f"\nFinal Results:")
    print(f"Shorts: {shorts}")
    print(f"Longs: {longs}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Winrate: {(wins/(wins+losses))*100:.2f}%")
    print(f"Final Equity: ${usdt:.2f}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Candles')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.show()

    return usdt


def final_df(df, lookback=100, size=(700, 700)):
    """Creates a dataframe with edge features and target labels"""
    # Pre-allocate numpy array for edge features
    num_samples = len(df) - lookback
    num_features = size[0] * size[1]  # Total pixels in the edge detection output
    edge_features_array = np.zeros((num_samples, num_features))
    
    # Add target column first
    df = add_target_column(df, 5)
    
    for i in range(lookback, len(df)):
        # Get the last 100 candles
        window_data = df.iloc[i-lookback:i]
        # Generate edge features
        edge_features = ohlc_to_edge_features(window_data, size=size)
        edge_features_array[i-lookback] = edge_features
        print(i)
    
    # Create final dataframe with features
    final_df = pd.DataFrame(
        edge_features_array,
        columns=[f'feature_{i}' for i in range(num_features)]
    )
    print("done")
    
    # Add target column
    final_df['target'] = df['target'].iloc[lookback:].reset_index(drop=True)
    print("done")
    
    return final_df


def test_random_forest(df, test_size=0.2, n_estimators=100):
    """Creates, trains and tests a Random Forest model with balanced class weights"""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and train the model with balanced class weights
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',  # This helps with imbalanced classes
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = rf.predict(X_test)
    print(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return rf