import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential, save_model
from keras.layers import Dense, Input

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


def ohlc_to_pixel_image(data, size=(64, 64), style='candlestick'):
    """
    Converts OHLC data directly to pixel images for neural network training
    
    Parameters:
    - data: DataFrame with OHLC data
    - size: Tuple of (height, width) for the image
    - style: 'candlestick' or 'line' chart style
    
    Returns:
    - 2D numpy array representing the image pixels
    """
    # Normalize price data to 0-1 range
    min_price = min(data['low'].min(), data['open'].min(), data['close'].min())
    max_price = max(data['high'].max(), data['open'].max(), data['close'].max())
    price_range = max_price - min_price
    
    # Create empty canvas (white background)
    canvas = np.ones(size, dtype=np.uint8) * 255
    
    # Calculate scaling factors
    time_scale = size[1] / len(data)
    price_scale = (size[0] - 10) / price_range  # Leave some margin
    
    if style == 'candlestick':
        # Calculate candle width
        candle_width = max(int(time_scale * 0.8), 1)
        
        for i, row in enumerate(data.itertuples()):
            # Calculate x position (center of candle)
            x = int(i * time_scale + time_scale/2)
            
            # Calculate y positions for OHLC (flip the y-coordinates)
            open_y = size[0] - 5 - int((row.open - min_price) * price_scale)
            high_y = size[0] - 5 - int((row.high - min_price) * price_scale)
            low_y = size[0] - 5 - int((row.low - min_price) * price_scale)
            close_y = size[0] - 5 - int((row.close - min_price) * price_scale)
            
            # Determine if candle is bullish or bearish
            is_bullish = row.close > row.open
            
            # Draw wicks (thin lines) - BLACK on white background
            x_wick = max(x - 1, 0)
            x_wick_end = min(x + 1, size[1])
            canvas[high_y:low_y+1, x_wick:x_wick_end] = 0  # Black wicks
            
            # Draw candle body - BLACK on white background
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            x_start = max(x - candle_width//2, 0)
            x_end = min(x + candle_width//2, size[1])
            
            # Both bullish and bearish candles are black on white background
            canvas[body_top:body_bottom+1, x_start:x_end] = 0  # Black body
                
    elif style == 'line':
        # Draw line chart - BLACK line on white background
        for i in range(len(data) - 1):
            x1 = int(i * time_scale)
            x2 = int((i + 1) * time_scale)
            
            y1 = size[0] - 5 - int((data.iloc[i]['close'] - min_price) * price_scale)
            y2 = size[0] - 5 - int((data.iloc[i+1]['close'] - min_price) * price_scale)
            
            # Draw line between points - BLACK on white background
            if x2 < size[1] and y1 >= 0 and y2 >= 0 and y1 < size[0] and y2 < size[0]:
                # Simple line drawing
                canvas[y1, x1] = 0  # Black pixel
                canvas[y2, x2] = 0  # Black pixel
                # Fill in between
                if abs(y2 - y1) > 1:
                    for y in range(min(y1, y2), max(y1, y2) + 1):
                        if 0 <= y < size[0]:
                            canvas[y, x1] = 0  # Black pixel
    
    return canvas


def visualize_pixel_image(pixel_image, title="Pixel Image"):
    """
    Visualizes the generated pixel image
    
    Parameters:
    - pixel_image: 2D numpy array representing the image
    - title: Title for the plot
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(pixel_image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()


def create_pixel_dataframe(df, lookback=100, image_size=(256, 256), style='candlestick'):
    """
    Creates a DataFrame with pixel features for neural network training
    
    Parameters:
    - df: DataFrame with OHLC data
    - lookback: Number of candles to look back for each image
    - image_size: Size of the generated images (height, width)
    - style: Chart style ('candlestick' or 'line')
    
    Returns:
    - DataFrame where each row contains flattened pixel values and target
    """
    # Pre-allocate numpy array for pixel features
    num_samples = len(df) - lookback
    num_pixels = image_size[0] * image_size[1]
    pixel_features_array = np.zeros((num_samples, num_pixels))
    
    # Add target column first
    df_with_target = add_target_column(df, 5)
    
    print(f"Generating {num_samples} pixel images...")
    print(f"Each image: {image_size[0]}x{image_size[1]} = {image_size[0]*image_size[1]:,} pixels")
    print(f"Total pixels to process: {num_samples * image_size[0] * image_size[1]:,}")
    
    start_time = time.time()
    
    for i in range(lookback, len(df)):
        # Get the window of data for this image
        window_data = df.iloc[i-lookback:i]
        
        # Generate pixel image
        pixel_image = ohlc_to_pixel_image(window_data, size=image_size, style=style)
        
        # Flatten the image to 1D array
        pixel_features_array[i-lookback] = pixel_image.flatten()
        
        # Progress tracking
        current_progress = i - lookback + 1
        if current_progress % 10 == 0 or current_progress == num_samples:
            elapsed = time.time() - start_time
            rate = current_progress / elapsed if elapsed > 0 else 0
            eta = (num_samples - current_progress) / rate if rate > 0 else 0
            
            print(f"Progress: {current_progress}/{num_samples} ({current_progress/num_samples*100:.1f}%) - "
                  f"Rate: {rate:.1f} images/sec - ETA: {eta:.1f}s")
    
    # Create final dataframe with pixel features
    pixel_columns = [f'pixel_{i}' for i in range(num_pixels)]
    final_df = pd.DataFrame(pixel_features_array, columns=pixel_columns)
    
    # Add target column at the START
    target_values = df_with_target['target'].iloc[lookback:].reset_index(drop=True)
    final_df.insert(0, 'target', target_values)
    
    print(f"Done! Created DataFrame with {len(final_df)} samples and {num_pixels} pixel features each.")
    

    return final_df


def preprocess_data(df: pd.DataFrame, image_size=(256, 256)):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # Convert to numeric and handle any errors
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)  # Fill NaN values with 0
    
    # Convert to numpy array and normalize
    X = X.values / 255.0
    
    # Print pixel statistics
    non_white_pixels = np.count_nonzero(X[0] != 1.0)
    print(f"Pixels in first row that are not white (1.0): {non_white_pixels}")
    
    # Reshape to image format (height, width, channels)
    X = X.reshape(-1, image_size[0], image_size[1], 1)

    # Print target distribution BEFORE categorical conversion
    y_original = df.iloc[:, 0].values
    y_n1_count = np.count_nonzero(y_original == -1)
    y_0_count = np.count_nonzero(y_original == 0)
    y_1_count = np.count_nonzero(y_original == 1)
    print(f"Number of times y is -1: {y_n1_count}")
    print(f"Number of times y is 0: {y_0_count}")
    print(f"Number of times y is 1: {y_1_count}")
    
    # Convert target to categorical (3 classes: -1, 0, 1)
    y = to_categorical(y + 1, num_classes=3)  # Shift -1,0,1 to 0,1,2

    return X, y


def build_model(image_size):
    model = Sequential([
        Input(image_size[0], image_size[1], 1),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def fit_model(model, X, y):
    