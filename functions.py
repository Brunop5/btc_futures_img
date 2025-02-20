import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mplfinance as mpf


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


if __name__ == "__main__":
    data = pd.read_csv("1d_test.csv", index_col=0, parse_dates=True)
    data = data.tail(100)
    save_chart(data)
    edge_features2 = ohlc_to_edge_features(data)
    rec_img2 = reconstruct_image_from_edges(edge_features2)
