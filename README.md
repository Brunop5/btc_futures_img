# Bitcoin Futures Price Predictor

A personal project built to predict Bitcoin futures price movement on Binance using **chart images** from multiple timeframes and feeding them into a **convolutional neural network**.

---

## 🧠 Idea & Purpose

I used to try day trading BTC futures and found myself **somewhat profitable** — but without a solid, defined strategy. Most of my trading decisions came from how the chart *looked* — I recognized certain patterns visually, but I couldn’t translate that “feel” into consistent, data-driven rules.

So I thought: _“What if I just use the images directly?”_

This project is my first serious attempt to automate my intuition. The idea:

1. Convert OHLC (Open–High–Low–Close) data into **256×256 grayscale candlestick chart images**  
2. Feed those images into a **deep convolutional neural network** built with TensorFlow/Keras  
3. Predict whether price will **go up, stay neutral, or go down** within the next few candles

Data comes from the Binance public API. After processing, it’s split into train/validation/test sets and fed into a residual CNN trained entirely on **image pixels**.

> This project is no longer maintained. I quickly realized I had bitten off more than I could chew at my skill level, and I plan to return to it once I’ve learned more about deep learning, time series, and feature engineering.

---

## ✨ Features

- `get_binance_data.py` — Fetches BTC futures OHLC data from Binance for any timeframe and time range, and exports to CSV.  
- `functions.py` — Core logic:
  - `add_target_column` — Labels each row as **down (0)**, **neutral (1)**, or **up (2)** based on future price movement.
  - `ohlc_to_pixel_image` — Converts raw OHLC data into **candlestick or line chart pixel images**.
  - `visualize_pixel_image` — Quick helper to preview generated images.
  - `create_pixel_dataframe` — Generates flattened pixel feature DataFrames from historical data.
  - `preprocess_data` — Normalizes and reshapes image data for CNN input.
  - `build_model` — Builds a deep **Residual CNN with Squeeze-and-Excitation blocks**, outputting 3-class predictions.
  - `split_data` — Splits dataset into train/validation/test sets.
  - `plot_training_history` — Plots accuracy and loss curves for model training.
  - `ImageDataGenerator` — Custom Keras Sequence for lazy loading of large NPZ datasets without loading everything into RAM.
  - `preprocess_and_save_chunks` — Converts raw OHLC data into pixel images in chunks and saves them as `.npz` datasets.
  - `train_with_validation` — Trains the model with class weighting and checkpointing.
  - `test_model` — Evaluates trained models on test data with accuracy, confusion matrix, and per-class metrics.
  - `compare_models` — Tests and compares multiple trained models on the same dataset.

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras** — neural network model
- **Pandas** — data handling
- **Matplotlib** — visualization
- **OpenCV** — image generation
- **scikit-learn** — data splitting & metrics
- **mplfinance** — candlestick chart rendering

---

## 📂 Additional Info

- Includes CSV files with raw OHLC Binance data.
- All model-ready image datasets are generated on the fly and stored as compressed `.npz` files for training.

---

## TL;DR

This was a wild, exploratory project born from the frustration of trading "by feel" and a desire to turn chart visuals into code. It’s far from perfect, but I learned a lot — and I plan to revisit it with more ML experience.

---

## 🌐 Other Projects

Check out my other project: [Mint Machina](https://mintmachina.com) — a website for creating cryptocurrency easily.
