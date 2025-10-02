import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import math

# Force CPU backend and safer CPU kernels BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure TensorFlow threading and disable XLA JIT for stability
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

import warnings
# Add this line at the top with your other imports
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names*')


def add_target_column(df, bars_forward):
    df = df.copy()
    df['target'] = 1
    
    for i in range(len(df) - bars_forward):
        current_close = df.iloc[i]['close']
        next_highs = df.iloc[i+1:i+bars_forward+1]['high'].max()
        next_lows = df.iloc[i+1:i+bars_forward+1]['low'].min()
        
        if next_highs > current_close * 1.007:
            df.iloc[i, df.columns.get_loc('target')] = 2
        elif next_lows < current_close * 0.993:
            df.iloc[i, df.columns.get_loc('target')] = 0
    
    return df


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
    df_with_target = add_target_column(df, 3)
    
    print("Target distribution:")
    print(df_with_target['target'].value_counts().sort_index())
    print(f"Total samples: {len(df_with_target)}")
    
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
    
    # Create final dataframe with pixel features
    pixel_columns = [f'pixel_{i}' for i in range(num_pixels)]
    final_df = pd.DataFrame(pixel_features_array, columns=pixel_columns)
    
    # Add target column at the START
    target_values = df_with_target['target'].iloc[lookback:].reset_index(drop=True)
    final_df.insert(0, 'target', target_values)

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
    y_n1_count = np.count_nonzero(y_original == 0)
    y_0_count = np.count_nonzero(y_original == 1)
    y_1_count = np.count_nonzero(y_original == 2)
    print(f"Number of times y is 0: {y_n1_count}")
    print(f"Number of times y is 1: {y_0_count}")
    print(f"Number of times y is 2: {y_1_count}")

    return X, y


def build_model(image_size):
    inputs = keras.Input(shape=(image_size[0], image_size[1], 1))

    def se_block(x, reduction=16):
        channels = x.shape[-1]
        squeeze = keras.layers.GlobalAveragePooling2D()(x)
        squeeze = keras.layers.Dense(max(int(channels) // reduction, 8), activation="relu")(squeeze)
        excite = keras.layers.Dense(int(channels), activation="sigmoid")(squeeze)
        excite = keras.layers.Reshape((1, 1, int(channels)))(excite)
        return keras.layers.Multiply()([x, excite])

    def res_block(x, filters, stride=1, use_se=True, drop_rate=0.0):
        shortcut = x
        y = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(y)
        y = keras.layers.BatchNormalization()(y)
        if use_se:
            y = se_block(y)
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        y = keras.layers.Add()([shortcut, y])
        y = keras.layers.Activation("relu")(y)
        if drop_rate > 0:
            y = keras.layers.Dropout(drop_rate)(y)
        return y

    # Stem
    x = keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Stages
    cfg = [
        (64,  3, 1, 0.10),
        (128, 4, 2, 0.15),
        (256, 6, 2, 0.20),
        (512, 3, 2, 0.25),
    ]
    for filters, blocks, stride, drop in cfg:
        x = res_block(x, filters, stride=stride, use_se=True, drop_rate=drop)
        for _ in range(blocks - 1):
            x = res_block(x, filters, stride=1, use_se=True, drop_rate=drop)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_training_history(history):
    """
    Plot training and validation accuracy and loss from model training history.
    
    Parameters:
    - history: History object returned by model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


class ImageDataGenerator(Sequence):
    """Generator for lazy loading of image data from NPZ files"""

    def __init__(self, data_dir="picture_data", batch_size=4, image_size=(64, 64), shuffle=True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        self.npz_files = list(self.data_dir.glob("*.npz"))
        self.n_samples = self._count_total_samples()

        # Build flat index: (file_idx, sample_idx_in_file)
        self.index_map = []
        for fi, npz_file in enumerate(self.npz_files):
            with np.load(npz_file) as data:
                n = len(data['X'])
            self.index_map.extend([(fi, si) for si in range(n)])

        # indices for simple batching
        self.indices = np.arange(len(self.index_map))
        self.on_epoch_end()

    def _count_total_samples(self):
        total = 0
        for npz_file in self.npz_files:
            with np.load(npz_file) as data:
                total += len(data['X'])
        return total

    def __len__(self):
        return int(math.ceil(len(self.index_map) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        idx_slice = self.indices[start:end]

        # gather pairs for this batch
        selected_pairs = [self.index_map[i] for i in idx_slice]

        # group by file to minimize I/O
        by_file = {}
        for fi, si in selected_pairs:
            by_file.setdefault(int(fi), []).append(int(si))

        X_batches, y_batches = [], []
        for fi, sample_ids in by_file.items():
            with np.load(self.npz_files[fi]) as data:
                X_chunk = data['X']  # (N, H, W) or (N, H, W, 1)
                y_chunk = data['y']  # (N,) or one-hot
                X_sel = X_chunk[sample_ids]
                y_sel = y_chunk[sample_ids]
                if X_sel.ndim == 3:
                    X_sel = X_sel.reshape((-1, self.image_size[0], self.image_size[1], 1))
                if y_sel.ndim == 2 and y_sel.shape[1] == 3:
                    y_sel = np.argmax(y_sel, axis=1)
                X_batches.append(X_sel)
                y_batches.append(y_sel)

        X = np.concatenate(X_batches, axis=0).astype(np.float32)
        y = np.concatenate(y_batches, axis=0).astype(np.int32)
        y = y.astype(np.int32).reshape(-1)

        # optional shuffle inside batch for randomness
        order = np.arange(len(y))
        np.random.shuffle(order)
        return X[order], y[order]
    # Keep for compatibility if you use it elsewhere
    def __call__(self):
        # Fallback single-batch sampler (unchanged behavior)
        return self.__getitem__(0)

    def get_all_data(self):
        all_X, all_y = [], []
        for npz_file in self.npz_files:
            with np.load(npz_file) as data:
                all_X.append(data['X'])
                all_y.append(data['y'])
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # Normalize shapes
        if X.ndim == 3:
            X = X.reshape((-1, self.image_size[0], self.image_size[1], 1))
        if y.ndim == 2 and y.shape[1] == 3:
            y = np.argmax(y, axis=1)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def save_data_to_npz(X, y, filename):
    """Save preprocessed data to NPZ file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, X=X, y=y)
    print(f"Saved {len(X)} samples to {filename}")


def preprocess_and_save_chunks(data, chunk_size=1000, image_size=(64, 64), data_dir="picture_data"):
    """Preprocess data in chunks and save to NPZ files"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        
        # Skip chunks smaller than lookback
        if len(chunk) < 100:  # Default lookback
            print(f"Skipping chunk {i//chunk_size + 1}: only {len(chunk)} samples (need 100+)")
            continue
            
        print(f"Processing chunk {i//chunk_size + 1}: samples {i} to {min(i+chunk_size, len(data))}")
        
        df = create_pixel_dataframe(chunk, image_size=image_size)
        X, y = preprocess_data(df)
        
        filename = data_dir / f"chunk_{i//chunk_size:03d}.npz"
        save_data_to_npz(X, y, filename)
    
    print(f"All chunks saved to {data_dir}")


def split_and_get_generators(img_size, chunk_size, batch_size):
    # Check if data is already preprocessed
    data_dir = Path("picture_data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    train_files = list(train_dir.glob("*.npz"))
     
    if not train_files:
        print("Preprocessing data...")
        data = pd.read_csv("1h_test2.csv", index_col=0, parse_dates=True)
        train_data = data
        
        # Split data first
        train_chunk, temp_chunk = train_test_split(train_data, test_size=0.3, random_state=42, shuffle=False)
        val_chunk, test_chunk = train_test_split(temp_chunk, test_size=0.5, random_state=42, shuffle=False)
        
        print(f"Train: {len(train_chunk)}, Val: {len(val_chunk)}, Test: {len(test_chunk)}")
        
        # Preprocess each split separately
        preprocess_and_save_chunks(train_chunk, chunk_size=chunk_size, 
                                  image_size=img_size, data_dir="picture_data/train")
        preprocess_and_save_chunks(val_chunk, chunk_size=chunk_size, 
                                  image_size=img_size, data_dir="picture_data/val")
        preprocess_and_save_chunks(test_chunk, chunk_size=chunk_size, 
                                  image_size=img_size, data_dir="picture_data/test")
    else:
        print(f"Found {len(train_files)} existing train files, skipping preprocessing")
    
    # Create generators for each split
    train_generator = ImageDataGenerator(data_dir="picture_data/train", 
                                       batch_size=batch_size, image_size=img_size, shuffle=True)
    val_generator = ImageDataGenerator(data_dir="picture_data/val", 
                                     batch_size=batch_size, image_size=img_size, shuffle=False)

    test_generator = ImageDataGenerator(data_dir="picture_data/test", 
                                     batch_size=batch_size, image_size=img_size, shuffle=False)
    
    return train_generator, val_generator, test_generator


def train_with_validation(model, train_generator, val_generator, epochs, batch_size):
    """Train model with validation data (CPU-safe)."""
    # Compute class weights without loading all features into RAM
    counts = np.zeros(3, dtype=np.int64)
    for npz_path in train_generator.npz_files:
        with np.load(npz_path) as data:
            y_part = data['y']
            if y_part.ndim == 2 and y_part.shape[1] == 3:
                y_part = np.argmax(y_part, axis=1)
            binc = np.bincount(y_part.astype(np.int64), minlength=3)
            counts += binc
    total = int(counts.sum()) if counts.sum() > 0 else 1
    weights = {i: float(total / (3.0 * max(1, int(counts[i])))) for i in range(3)}

    # Initialize final layer bias to log class priors to avoid initial single-class predictions
    priors = (counts / max(1, counts.sum())).astype(np.float32)
    priors = np.clip(priors, 1e-8, 1.0)
    log_priors = np.log(priors)
    try:
        final_dense = model.layers[-1]
        w, b = final_dense.get_weights()
        if b.shape[0] == 3:
            final_dense.set_weights([w, log_priors])
    except Exception as _:
        pass

    callbacks = [
        #EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss"),
        #ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        verbose=1,
        steps_per_epoch=100,
        validation_steps=30,
        class_weight=weights,
    )

    return history


def plot_training_history(history):
    """
    Plot training and validation accuracy and loss from model training history.
    
    Parameters:
    - history: History object returned by model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_model(model, test_generator, model_name="Model"):
    """
    Test a trained model on test data and provide comprehensive evaluation metrics.
    
    Parameters:
    - model: Trained Keras model
    - test_generator: ImageDataGenerator for test data
    - model_name: Name for display purposes
    
    Returns:
    - Dictionary with test results
    """
    print(f"\n=== Testing {model_name} ===")
    
    # Load all test data
    X_test, y_test = test_generator.get_all_data()
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    unique, counts = np.unique(y_pred, return_counts=True)
    print("Prediction counts:")
    for pred, count in zip(unique, counts):
        print(f"  Class {pred}: {count} predictions")
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Samples: {len(y_test)}")
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"\nTest Data Class Distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print("     Predicted")
    print("     0  1  2")
    for i, row in enumerate(cm):
        print(f"{i}   {row}")
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Down (-1)', 'Neutral (0)', 'Up (1)']))
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i in range(3):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"  Class {i}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    # Return results dictionary
    results = {
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return results


def compare_models(models_dict, test_generator):
    """
    Compare multiple models on the same test data.
    
    Parameters:
    - models_dict: Dictionary with model names as keys and models as values
    - test_generator: ImageDataGenerator for test data
    
    Returns:
    - Dictionary with comparison results
    """
    print(f"\n=== Model Comparison ===")
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\nTesting {name}...")
        results[name] = test_model(model, test_generator, name)
    
    # Summary comparison
    print(f"\n=== Summary Comparison ===")
    print(f"{'Model':<20} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 40)
    
    for name, result in results.items():
        print(f"{name:<20} {result['accuracy']:<10.4f} {len(result['y_true']):<10}")
    
    return results

