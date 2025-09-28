#!/usr/bin/env python3
"""
Test script to demonstrate pixel-based image generation for neural network training
"""

import pandas as pd
import numpy as np
from functions import (
    ohlc_to_pixel_image, 
    visualize_pixel_image, 
    create_pixel_dataframe,
    ohlc_to_edge_features,
    reconstruct_image_from_edges
)

# Configuration - Change this to experiment with different image sizes
IMAGE_SIZE = (256, 256)  # (height, width) - Try: (64, 64), (128, 128), (256, 256)
ORIGINAL_EDGE_SIZE = (700, 700)  # Original edge detection resolution

def test_pixel_image_generation():
    """Test the pixel image generation functions"""
    
    # Load sample data
    print("Loading sample data...")
    data = pd.read_csv("1h_test.csv", index_col=0, parse_dates=True)
    
    # Take a small sample for testing
    sample_data = data.head(200)
    print(f"Using {len(sample_data)} candles for testing")
    
    # Test 1: Generate a single pixel image and edge features
    print("\n=== Test 1: Single Image Generation ===")
    window_data = sample_data.head(100)  # Use first 100 candles
    
    # Generate candlestick pixel image
    pixel_image_candle = ohlc_to_pixel_image(window_data, size=IMAGE_SIZE, style='candlestick')
    print(f"Candlestick pixel image shape: {pixel_image_candle.shape}")
    print(f"Pixel values range: {pixel_image_candle.min()} to {pixel_image_candle.max()}")
    
    # Generate edge features at original resolution
    edge_features_original = ohlc_to_edge_features(window_data, size=ORIGINAL_EDGE_SIZE)
    edge_image_original = edge_features_original.reshape(ORIGINAL_EDGE_SIZE)
    
    # Generate edge features at target resolution for comparison
    edge_features_resized = ohlc_to_edge_features(window_data, size=IMAGE_SIZE)
    edge_image_resized = edge_features_resized.reshape(IMAGE_SIZE)
    
    print(f"Edge features original shape: {edge_features_original.shape}")
    print(f"Edge features resized shape: {edge_features_resized.shape}")
    print(f"Edge values range (original): {edge_features_original.min()} to {edge_features_original.max()}")
    print(f"Edge values range (resized): {edge_features_resized.min()} to {edge_features_resized.max()}")
    
    # Calculate data loss
    original_nonzero = np.count_nonzero(edge_features_original)
    resized_nonzero = np.count_nonzero(edge_features_resized)
    data_loss_percent = (1 - resized_nonzero / original_nonzero) * 100 if original_nonzero > 0 else 0
    
    print(f"Original edge features non-zero pixels: {original_nonzero}")
    print(f"Resized edge features non-zero pixels: {resized_nonzero}")
    print(f"Data loss: {data_loss_percent:.2f}%")
    
    # Visualize all images side by side
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original pixel image
    axes[0, 0].imshow(pixel_image_candle, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f"Pixel Image {IMAGE_SIZE}")
    axes[0, 0].axis('off')
    
    # Edge features original resolution
    axes[0, 1].imshow(edge_image_original, cmap='gray')
    axes[0, 1].set_title(f"Edge Features {ORIGINAL_EDGE_SIZE} - Original")
    axes[0, 1].axis('off')
    
    # Edge features resized
    axes[1, 0].imshow(edge_image_resized, cmap='gray')
    axes[1, 0].set_title(f"Edge Features {IMAGE_SIZE} - Resized")
    axes[1, 0].axis('off')
    
    # Generate line chart pixel image for comparison
    pixel_image_line = ohlc_to_pixel_image(window_data, size=IMAGE_SIZE, style='line')
    axes[1, 1].imshow(pixel_image_line, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f"Line Chart Pixel {IMAGE_SIZE}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Line chart image shape: {pixel_image_line.shape}")
    
    # Test 2: Generate DataFrame with pixel features
    print("\n=== Test 2: Pixel DataFrame Generation ===")
    
    # Create pixel dataframe (smaller for testing)
    pixel_df = create_pixel_dataframe(
        sample_data, 
        lookback=50,  # Smaller lookback for testing
        image_size=IMAGE_SIZE,  # Use the configured image size
        style='candlestick'
    )
    
    print(f"Pixel DataFrame shape: {pixel_df.shape}")
    print(f"Columns: {list(pixel_df.columns[:5])}... (showing first 5)")
    print(f"Target distribution:")
    print(pixel_df['target'].value_counts())
    
    # Test 3: Show how to use for neural network training
    print("\n=== Test 3: Neural Network Ready Data ===")
    
    # Separate features and targets
    X = pixel_df.drop('target', axis=1).values
    y = pixel_df['target'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target array shape: {y.shape}")
    print(f"Feature matrix dtype: {X.dtype}")
    print(f"Target array dtype: {y.dtype}")
    
    # Show sample of pixel values
    print(f"Sample pixel values (first 10 pixels of first image): {X[0][:10]}")
    
    return pixel_df

def compare_edge_vs_pixel():
    """Compare edge features vs pixel features"""
    
    print("\n=== Comparison: Edge Features vs Pixel Features ===")
    
    # Load data
    data = pd.read_csv("1h_test.csv", index_col=0, parse_dates=True)
    sample_data = data.head(200)
    
    # Generate edge features at different resolutions
    window_data = sample_data.head(100)
    
    # Original resolution
    edge_features_original = ohlc_to_edge_features(window_data, size=ORIGINAL_EDGE_SIZE)
    edge_image_original = edge_features_original.reshape(ORIGINAL_EDGE_SIZE)
    
    # Target resolution
    edge_features_resized = ohlc_to_edge_features(window_data, size=IMAGE_SIZE)
    edge_image_resized = edge_features_resized.reshape(IMAGE_SIZE)
    
    # Generate pixel features (new method)
    pixel_image = ohlc_to_pixel_image(window_data, size=IMAGE_SIZE, style='candlestick')
    pixel_features = pixel_image.flatten()
    
    print(f"Edge features original shape: {edge_features_original.shape}")
    print(f"Edge features resized shape: {edge_features_resized.shape}")
    print(f"Pixel features shape: {pixel_features.shape}")
    print(f"Edge features unique values (original): {len(np.unique(edge_features_original))}")
    print(f"Edge features unique values (resized): {len(np.unique(edge_features_resized))}")
    print(f"Pixel features unique values: {len(np.unique(pixel_features))}")
    
    # Calculate data loss from resizing
    original_nonzero = np.count_nonzero(edge_features_original)
    resized_nonzero = np.count_nonzero(edge_features_resized)
    data_loss_percent = (1 - resized_nonzero / original_nonzero) * 100 if original_nonzero > 0 else 0
    
    print(f"\nData Loss Analysis:")
    print(f"Original edge features non-zero pixels: {original_nonzero}")
    print(f"Resized edge features non-zero pixels: {resized_nonzero}")
    print(f"Data loss from {ORIGINAL_EDGE_SIZE} to {IMAGE_SIZE}: {data_loss_percent:.2f}%")
    print(f"Pixel features non-zero pixels: {np.count_nonzero(pixel_features)}")
    
    # Visualize comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original pixel image
    axes[0, 0].imshow(pixel_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f"Pixel Image {IMAGE_SIZE}")
    axes[0, 0].axis('off')
    
    # Edge features original resolution
    axes[0, 1].imshow(edge_image_original, cmap='gray')
    axes[0, 1].set_title(f"Edge Features {ORIGINAL_EDGE_SIZE}")
    axes[0, 1].axis('off')
    
    # Edge features resized
    axes[0, 2].imshow(edge_image_resized, cmap='gray')
    axes[0, 2].set_title(f"Edge Features {IMAGE_SIZE}")
    axes[0, 2].axis('off')
    
    # Histogram of pixel values
    axes[1, 0].hist(pixel_features, bins=50, alpha=0.7, color='blue', label='Pixel Features')
    axes[1, 0].set_title("Pixel Features Distribution")
    axes[1, 0].set_xlabel("Pixel Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    
    # Histogram of edge values (original)
    axes[1, 1].hist(edge_features_original, bins=50, alpha=0.7, color='red', label=f'Edge Features {ORIGINAL_EDGE_SIZE}')
    axes[1, 1].set_title(f"Edge Features Distribution {ORIGINAL_EDGE_SIZE}")
    axes[1, 1].set_xlabel("Edge Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    
    # Histogram of edge values (resized)
    axes[1, 2].hist(edge_features_resized, bins=50, alpha=0.7, color='orange', label=f'Edge Features {IMAGE_SIZE}')
    axes[1, 2].set_title(f"Edge Features Distribution {IMAGE_SIZE}")
    axes[1, 2].set_xlabel("Edge Value")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Show information content comparison
    print(f"\nInformation Content Analysis:")
    print(f"Edge features sparsity {ORIGINAL_EDGE_SIZE}: {np.count_nonzero(edge_features_original) / len(edge_features_original) * 100:.2f}%")
    print(f"Edge features sparsity {IMAGE_SIZE}: {np.count_nonzero(edge_features_resized) / len(edge_features_resized) * 100:.2f}%")
    print(f"Pixel features sparsity: {np.count_nonzero(pixel_features) / len(pixel_features) * 100:.2f}%")
    print(f"Edge features variance {ORIGINAL_EDGE_SIZE}: {np.var(edge_features_original):.2f}")
    print(f"Edge features variance {IMAGE_SIZE}: {np.var(edge_features_resized):.2f}")
    print(f"Pixel features variance: {np.var(pixel_features):.2f}")
    
    # Resolution comparison
    original_pixels = ORIGINAL_EDGE_SIZE[0] * ORIGINAL_EDGE_SIZE[1]
    target_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    print(f"\nResolution Comparison:")
    print(f"{ORIGINAL_EDGE_SIZE} total pixels: {original_pixels:,}")
    print(f"{IMAGE_SIZE} total pixels: {target_pixels:,}")
    print(f"Resolution reduction: {(1 - target_pixels/original_pixels)*100:.1f}%")
    print(f"Data compression ratio: {original_pixels/target_pixels:.1f}:1")

if __name__ == "__main__":
    print("Testing Pixel Image Generation for Neural Network Training")
    print("=" * 60)
    print(f"Using image size: {IMAGE_SIZE}")
    print(f"Original edge size: {ORIGINAL_EDGE_SIZE}")
    print("=" * 60)
    
    # Run tests
    pixel_df = test_pixel_image_generation()
    compare_edge_vs_pixel()
    
    print("\n" + "=" * 60)
    print("Test completed! You can now use create_pixel_dataframe() for neural network training.")
    print("The function returns a DataFrame where each row contains:")
    print("- pixel_0, pixel_1, ..., pixel_N: Flattened pixel values")
    print("- target: The prediction target (-1, 0, 1)")
    print(f"\nTo change image size, modify IMAGE_SIZE = {IMAGE_SIZE} at the top of this file")
