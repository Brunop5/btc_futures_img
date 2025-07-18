# Bitcoin Futures Price Predictor

A personal project built to predict Bitcoin futures price movement on Binance using images from multiple timeframes and feeding them into a machine learning model.

---

## Idea & Purpose

I used to try day trading BTC futures and found myself **somewhat profitable** — but without a solid, defined strategy. I was mostly trading based on how the chart *looked* (visual patterns). The problem was: I couldn’t translate those visual cues into consistent logic or data-driven rules.

So I thought: _What if I just use images?_

This project is my first amateur attempt at creating a bot that trades BTC based on chart visuals. My idea was to:

1. Convert OHLC (Open-High-Low-Close) data into images  
3. Train a machine learning model to recognize patterns and predict price direction

After some Googling, I came across **edge features** as a possible way to extract structured information from chart images. To validate this approach, I also wrote a script that recreates an image from its extracted edge features — just to check if the transformation makes sense.

For data, I used the Binance public API, which I was already familiar with. The raw OHLC data is converted into edge features and fed into a **Random Forest Classifier**.

I chose `scikit-learn` and `RandomForestClassifier` mostly due to my prior experience and its simplicity. I know my limited ML knowledge likely held this back from reaching its potential — but that was part of the point: to learn.

> This project is no longer maintained. I quickly realized I had bitten off more than I could chew at my skill level, and I plan to return to it once I’ve learned more about machine learning, time series modeling, and image processing.

---

## Features

- `get_binance_data.py` — Fetches BTC futures OHLC data from Binance for a given timeframe and time range. Exports as CSV.
- `functions.py` — Core logic:
  - `ohlc_to_edge_features` — Converts OHLC to edge features. Feels like black magic now.
  - `add_target_column` — Adds a label (+1 or -1) depending on whether price went up or down in the next N candles.
  - `final_df` — Combines features and labels into a training-ready DataFrame.
  - `test_random_forest` — Trains and tests a Random Forest model (outputs accuracy).
  - `test` — Simulates fake trades using model predictions (outputs basic results).
  - Plus other helper functions.
- `main.py` — Puts it all together — loads data, processes it, trains the model, and tests it.

---

## Tech Stack

- **Python**
- **Libraries used:**
  - `requests` — for data fetching from Binance API
  - `pandas` — for data processing and CSV handling
  - `cv2` (OpenCV) — for edge detection
  - `matplotlib` — for image and result visualization
  - `mplfinance` (`mpf`) — for candlestick chart image generation
  - `scikit-learn` — for ML model training and evaluation

---

## Additional Info

- Includes CSV files with raw OHLC Binance data.
- Final model-ready DataFrames are created on the fly, not stored in the repo.

---

## TL;DR

This was a wild, exploratory project born from the frustration of trading "by feel" and a desire to turn chart visuals into code.  
It’s far from perfect, but I learned a lot — and I plan to revisit it with more ML experience.
